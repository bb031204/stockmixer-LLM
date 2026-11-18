import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from news_fusion.inputs import normalize_article
from news_fusion.io import digest,read_records
from news_fusion.upstream import industry_impact,scope_claims,make_reviews,build_review,market_owners
from news_fusion.routing import checked_reviews,publish_routes,load_routes
from news_fusion.relations import resolve_relations,build_relations,load_relations
from news_fusion.pipeline import Pipeline,DEFAULTS,validate_settings
from tests_news.test_historical_relations import security,axis,article
from tests_news.test_pipeline import make_axis,FakeBackend
from news_fusion.encoder import encode_articles
from news_fusion.panel import build_panel
from news_fusion.dataset import NewsPanel


def scoped(scope='NASDAQ_LISTED'):
    row=normalize_article({'source_id':'scoped','title':'Market policy','summary':'New policy changes listing requirements for the affected equity market.',
        'language':'en','market':'US','published_at':'2020-03-05T18:00:00Z','impact_scope':['market'],
        'publication_precision':'second','source_record_hash':'a'*64,
        'market_scope':[scope],'scope_evidence':[{'scope':scope,'quote':'New policy changes listing requirements',
            'mechanism':'The policy changes obligations of listed companies.','confidence':.95}],
        'upstream_analysis':{'record_hash':'a'*64,'audit':{'record_hash_matches':True,'sha256':'b'*64}},
        'qwen_model_revision':'fixture_not_a_real_model','prompt_version':'fixture',
        'llm_inferred_at':'2020-03-05T19:00:00Z','availability_basis':'observed',
        'first_seen_at':'2020-03-05T18:10:00Z','fetched_at':'2020-03-05T18:20:00Z',
        'feature_available_at':'2020-03-05T19:30:00Z'})
    return row


class UpstreamMappingTests(unittest.TestCase):
    def test_real_producer_industry_aliases_preserve_unknown_confidence(self):
        raw={'industry_name_original':'Financial Services','quote':'monetary policy influences inflation',
             'confidence':None,'evidence_type':'inferred','mechanism':'Policy changes financial conditions.'}
        result=industry_impact(raw)
        self.assertEqual(result['industry_name'],'Financial Services')
        self.assertEqual(result['evidence'],raw['quote']);self.assertIsNone(result['confidence'])
        self.assertNotIn('industry_name',raw)

    def test_conflicting_schema_aliases_are_not_silently_selected(self):
        with self.assertRaises(ValueError):industry_impact({'industry_name':'Banks','industry_name_original':'Oil'})
        with self.assertRaises(ValueError):industry_impact({'industry_name_original':42})

    def test_scope_requires_quote_confidence_and_bound_audit(self):
        self.assertEqual(scope_claims(scoped())[0][0]['scope'],'NASDAQ_LISTED')
        for mutate in [lambda a:a.pop('scope_evidence'),lambda a:a['scope_evidence'][0].update(confidence=None),
                       lambda a:a['scope_evidence'][0].update(quote='Invented policy source quote'),
                       lambda a:a['upstream_analysis'].update(record_hash='c'*64),
                       lambda a:a.update(qwen_model_revision=None)]:
            a=scoped();mutate(a);self.assertFalse(scope_claims(a)[0])

    def test_unknown_and_conflicting_scope_remain_pending(self):
        self.assertFalse(scope_claims(scoped('GLOBAL'))[0])
        a=scoped();a['affected_markets']=['NYSE_LISTED']
        self.assertEqual(scope_claims(a)[1],'conflicting_market_scope_fields')
        a=scoped();a.pop('market_scope');a.pop('scope_evidence')
        self.assertEqual(make_reviews([a],'NASDAQ')[0]['decision'],'pending')

    def test_model_confidence_is_not_calibrated_mapping_probability(self):
        claim=scope_claims(scoped())[0][0]
        self.assertEqual(claim['source_confidence'],.95)
        self.assertIsNone(claim['mapping_confidence'])

    def test_review_replay_detects_scope_tampering(self):
        a=scoped();row=make_reviews([a],'SP500')[0]
        routes,_=checked_reviews([a],[row],True,'SP500')
        self.assertEqual(routes[0]['scope_claims'][0]['scope'],'NASDAQ_LISTED')
        row['scope_claims'][0]['scope']='US_EQUITY'
        with self.assertRaises(ValueError):checked_reviews([a],[row],True,'SP500')

    def test_exchange_identity_not_inferred_from_dataset_name_and_date_end_exclusive(self):
        a=axis();s=security();other=security(stock_uid='NASDAQ:XYZ',array_index=1,ticker='XYZ',exchange='NYSE')
        self.assertEqual([i for i,_ in market_owners('NYSE_LISTED',[s,other],a,'2014-01-02')],[1])
        self.assertFalse(market_owners('NASDAQ_LISTED',[s],a,'2015-01-01'))
        self.assertEqual(len(market_owners('US_EQUITY',[],a,'2014-01-02')),2)

    def test_new_industry_policy_rejects_missing_source_confidence(self):
        a=article({'name':'Alpha','ticker':'ABC'},industry_impacts=[{
            'industry_name_original':'Banking','quote':'Alpha announces revenue growth','evidence_type':'inferred',
            'mechanism':'Business improves bank lending conditions.','confidence':None}])
        links,_,pending,_=resolve_relations(axis(),[a],[security()],[],[],policy={'industry_min_confidence':.85})
        self.assertEqual([l['channel'] for l in links],['company'])
        self.assertEqual(pending[0]['reason'],'missing_or_low_industry_source_confidence')

    def test_explicit_industry_evidence_does_not_require_invented_model_confidence(self):
        a=article({'name':'Alpha'},summary='Alpha announces revenue growth in the Banking industry.',industry_impacts=[{
            'industry_name_original':'Banking','quote':'growth in the Banking industry','evidence_type':'explicit','confidence':None}])
        alias=dict(security(),alias='Banking',industry_standard='SIC',industry_version='1987',industry_code='60')
        links,_,pending,_=resolve_relations(axis(),[a],[security()],[],[alias],policy={'industry_min_confidence':.85})
        industry=[l for l in links if l['channel']=='industry']
        self.assertEqual(len(industry),1);self.assertIsNone(industry[0]['source_confidence'])
        self.assertFalse(pending)

    def test_explicit_industry_quote_must_mention_that_industry(self):
        a=article({'name':'Alpha'},summary='Alpha announces revenue growth in the Banking industry.',industry_impacts=[{
            'industry_name_original':'Banking','quote':'Alpha announces revenue growth','evidence_type':'explicit','confidence':.99}])
        alias=dict(security(),alias='Banking',industry_standard='SIC',industry_version='1987',industry_code='60')
        links,_,pending,_=resolve_relations(axis(),[a],[security()],[],[alias],policy={'industry_min_confidence':.85})
        self.assertFalse([l for l in links if l['channel']=='industry'])
        self.assertEqual(pending[0]['reason'],'invalid_industry_evidence_type_or_mechanism')

    def test_conflicting_security_id_does_not_fall_back_to_name(self):
        a=article({'name':'Alpha','ticker':'ABC','security_id':'CIK:WRONG'})
        links,_,_,_=resolve_relations(axis(),[a],[security()],[],[],policy={'industry_min_confidence':.85})
        self.assertFalse(links)

    def test_pipeline_config_forbids_second_qwen_pass(self):
        settings=copy.deepcopy(DEFAULTS);validate_settings(settings)
        self.assertEqual(settings['analysis_source'],'upstream_only')
        settings['infer_industries']=True
        with self.assertRaisesRegex(ValueError,'forbids'):validate_settings(settings)

    def test_review_step_never_constructs_ollama_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=object.__new__(Pipeline);p.project=Path(tmp);p.root=Path(tmp)
            p.settings=copy.deepcopy(DEFAULTS);p.axes={'NASDAQ':None};p.config={'review':{'min_confidence':.85}}
            p.state={'artifacts':{'NASDAQ':{}}};p.save=lambda:None;p.pin_artifact=lambda path:None
            provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64}
            p.articles=lambda:([scoped()],{},provenance)
            with patch('news_fusion.auto_review.OllamaReviewer',side_effect=AssertionError('No Qwen calls')):
                p.step_review()
            self.assertEqual(json.loads((Path(p.state['artifacts']['NASDAQ']['review'])/'manifest.json').read_text())['model_calls'],0)

    def test_scoped_market_panel_reaches_only_verified_exchange_stock(self):
        from news_fusion.io import write_records
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,schedule=make_axis(root);news=scoped()
            provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64}
            securities=[dict(security(),dataset='TEST',stock_uid='TEST:A',ticker='A',dataset_ticker='A',
                             valid_from='2020-03-01',valid_to='2021-01-01'),
                        dict(security(),dataset='TEST',stock_uid='TEST:B',array_index=1,ticker='B',dataset_ticker='B',
                             exchange='NYSE',valid_from='2020-03-01',valid_to='2021-01-01')]
            write_records(root/'securities.jsonl',securities)
            build_relations(a,[news],[root/'securities.jsonl',None,None],root/'relations',provenance,
                            policy={'version':'evidence_mapping_v2','industry_min_confidence':.85})
            links,members,_=load_relations(root/'relations',a)
            review=make_reviews([news],'TEST')
            publish_routes([news],review,'TEST',root/'routes',provenance,True)
            load_routes(root/'routes','TEST')
            encode_articles([news],FakeBackend(),root/'features',root/'cache',input_provenance=provenance)
            build_panel(a,schedule,root/'features',links,members,root/'panel',projection='random_projection',components=2,
                        decision_lookback=2,market_routes=root/'routes',relations_artifact=root/'relations')
            panel=NewsPanel(root/'panel')
            self.assertEqual(panel.features['market'].shape,(2,8,10))
            self.assertTrue(panel.masks['market'][0].any());self.assertFalse(panel.masks['market'][1].any())
            self.assertEqual(panel.decision(0)['market_news'].shape,(2,10))
            panel.assert_market_routing()
            del panel

if __name__=='__main__':unittest.main()
