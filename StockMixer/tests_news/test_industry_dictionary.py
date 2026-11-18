import copy,json,tempfile,unittest
from pathlib import Path
from news_fusion.industry_dictionary import IndustryDictionary
from news_fusion.relations import resolve_relations,build_relations,load_relations
from news_fusion.io import digest,write_records,write_json,file_manifest
from news_fusion.upstream import industry_impact
from tests_news.test_historical_relations import axis,security,article

ROOT=Path(__file__).resolve().parents[1]

class IndustryDictionaryTests(unittest.TestCase):
    def setUp(self):self.dictionary=IndustryDictionary(ROOT/'news_assets/industry_dictionary')
    def news(self):
        quote='Semiconductor industry faces a widespread capacity constraint.'
        raw={'industry_key':'SEC-SIC:3674','industry_name_original':'Semiconductor','industry_standard':'SEC-SIC',
            'industry_version':'EDGAR-SIC','industry_code':'3674','dictionary_sha256':self.dictionary.sha256,
            'evidence_type':'inferred','confidence':.95,'quote':quote,'mechanism':'Capacity constraints reduce supply available to industry customers.',
            'original_evidence_verified':True,'original_evidence_quote':quote,
            'evidence_basis':'original_article_quote_verified_before_raw_cleanup'}
        a=article({'name':'Alpha'},industry_impacts=[industry_impact(raw)],source_record_hash='a'*64,
            upstream_analysis={'record_hash':'a'*64,'content_hash':'c'*64,
                'audit':{'record_hash_matches':True,'sha256':'b'*64},
                'industry_dictionary':{'sha256':self.dictionary.sha256},'raw_industry_impacts':[raw]})
        return a
    def test_original_quote_can_be_bound_without_changing_encoded_summary(self):
        a=self.news();before=a['summary']
        links,_,pending,_=resolve_relations(axis(),[a],[security()],[],[],policy={'industry_min_confidence':.85},dictionary=self.dictionary)
        industries=[l for l in links if l['channel']=='industry']
        self.assertEqual(len(industries),1);self.assertEqual(industries[0]['industry_code'],'3674')
        self.assertEqual(a['summary'],before);self.assertFalse(pending)
    def test_a_boolean_original_verified_flag_without_matching_audit_is_insufficient(self):
        a=self.news();a['upstream_analysis']['audit']['record_hash_matches']=False
        links,_,_,_=resolve_relations(axis(),[a],[security()],[],[],policy={'industry_min_confidence':.85},dictionary=self.dictionary)
        self.assertFalse([l for l in links if l['channel']=='industry'])
    def test_dictionary_version_or_code_conflict_is_rejected(self):
        for field,value in [('dictionary_sha256','d'*64),('industry_code','9999')]:
            a=self.news();a['industry_impacts'][0][field]=value
            links,_,missing,_=resolve_relations(axis(),[a],[security()],[],[],policy={'industry_min_confidence':.85},dictionary=self.dictionary)
            self.assertFalse([l for l in links if l['channel']=='industry'])
            self.assertEqual(missing[0]['reason'],'industry_dictionary_version_or_code_mismatch')
    def test_coded_industry_relation_does_not_create_stock_memberships(self):
        links,members,_,_=resolve_relations(axis(),[self.news()],[security()],[],[],policy={'industry_min_confidence':.85},dictionary=self.dictionary)
        self.assertTrue([l for l in links if l['channel']=='industry']);self.assertFalse(members)
    def test_frozen_dictionary_relation_replays_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write_records(p/'securities.jsonl',[security()])
            a=axis();a.stock_hash=digest(a.stock_uids);a.date_hash=digest(['2014-01-02'])
            build_relations(a,[self.news()],[p/'securities.jsonl',None,None],p/'relations',
                {'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64},
                policy={'industry_min_confidence':.85},dictionary=self.dictionary)
            links,_,manifest=load_relations(p/'relations',a)
            self.assertEqual(manifest['industry_dictionary_sha256'],self.dictionary.sha256)
            self.assertEqual(len([l for l in links if l['channel']=='industry']),1)
            self.assertTrue(manifest['dictionary_sources_embedded'])
            self.assertTrue((p/'relations/sec_sic_source.html').exists())

    def test_approved_industry_cannot_become_available_before_inference_and_review(self):
        news=article({'name':'Alpha'},summary='Alpha operates in the Banking industry.',industry_impacts=[{
            'industry_name':'Banking','evidence':'Alpha operates in the Banking industry.', 'evidence_type':'explicit'}],
            llm_inferred_at='2020-02-01T00:00:00Z',effective_feature_available_at='2020-03-01T00:00:00Z',
            industry_scope_review={'decision':'include','reviewed_at':'2020-04-01T00:00:00Z'})
        alias=dict(security(),alias='Banking',industry_standard='SIC',industry_version='1987',industry_code='60')
        links,_,_,_=resolve_relations(axis(),[news],[security()],[],[alias],policy={'version':'evidence_mapping_v3','industry_min_confidence':.85})
        row=next(l for l in links if l['channel']=='industry')
        self.assertEqual(row['known_from'],'2020-04-01T00:00:00+00:00')
        late=copy.deepcopy(news);late['feature_available_at']='2020-05-01T00:00:00Z'
        late_links,_,_,_=resolve_relations(axis(),[late],[security()],[],[alias],policy={'version':'evidence_mapping_v3','industry_min_confidence':.85})
        self.assertEqual(next(l for l in late_links if l['channel']=='industry')['known_from'],'2020-05-01T00:00:00+00:00')
        old,_,_,_=resolve_relations(axis(),[news],[security()],[],[alias],policy={'version':'evidence_mapping_v2','industry_min_confidence':.85})
        self.assertEqual(next(l for l in old if l['channel']=='industry')['known_from'],alias['known_from'])

    def test_observed_pit_industry_requires_approval_time_aware_relation_version(self):
        from news_fusion.relations import require_industry_timing
        old={'mapping_policy':{'version':'evidence_mapping_v2'}}
        with self.assertRaisesRegex(ValueError,'evidence_mapping_v3'):require_industry_timing(old,'observed_pit',['industry'])
        require_industry_timing(old,'retrospective',['industry'])
        require_industry_timing(old,'observed_pit',['market'])
        require_industry_timing({'mapping_policy':{'version':'evidence_mapping_v3'}},'observed_pit',['industry'])

    def test_industry_vector_reaches_only_the_verified_historical_member(self):
        from tests_news.test_pipeline import make_axis,FakeBackend
        from news_fusion.inputs import normalize_article
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        from news_fusion.dataset import NewsPanel
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a,schedule=make_axis(p)
            news=normalize_article(dict(self.news(),language='en',market='US',published_at='2020-03-05T18:00:00Z'))
            owner=security(dataset='TEST',stock_uid='TEST:A',dataset_ticker='A',ticker='A',valid_from='2020-01-01',valid_to='2021-01-01')
            member=dict(owner,industry_standard='SEC-SIC',industry_version='EDGAR-SIC',industry_code='3674',
                industry_name='Semiconductors and Related Devices',exposure_weight=1.0)
            write_records(p/'securities.jsonl',[owner]);write_records(p/'members.jsonl',[member])
            provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64}
            build_relations(a,[news],[p/'securities.jsonl',p/'members.jsonl',None],p/'relations',provenance,
                policy={'industry_min_confidence':.85},dictionary=self.dictionary)
            links,members,_=load_relations(p/'relations',a)
            encode_articles([news],FakeBackend(),p/'features',p/'cache',input_provenance=provenance)
            build_panel(a,schedule,p/'features',links,members,p/'panel',projection='random_projection',components=2,
                decision_lookback=2,relations_artifact=p/'relations')
            panel=NewsPanel(p/'panel')
            self.assertTrue(panel.masks['industry'][0].any());self.assertFalse(panel.masks['industry'][1].any())
            self.assertEqual(panel.decision(0)['industry_news'].shape,(2,10))
            del panel

    def test_reference_table_cannot_promote_an_unsupported_membership(self):
        from news_fusion.industry_mapping import verified_mapping
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=axis();a.stock_hash=digest(a.stock_uids);a.date_hash=digest(['2014-01-02'])
            write_records(p/'stock_industry_index.jsonl',[{'dataset':a.dataset,'stock_uid':uid,'array_index':i,
                'reference_industry_key':'SEC-SIC:3674'} for i,uid in enumerate(a.stock_uids)])
            write_records(p/'verified_security_master.jsonl',[security()])
            write_records(p/'verified_industry_memberships.jsonl',[dict(security(),industry_code='3674')])
            write_json(p/'manifest.json',{'kind':'dataset_industry_axis_mapping','dictionary_sha256':self.dictionary.sha256,
                'undated_references_enabled_for_training':False,'datasets':{a.dataset:{'stock_axis_hash':a.stock_hash,'date_axis_hash':a.date_hash}},
                'files':file_manifest(p)})
            with self.assertRaisesRegex(ValueError,'not independently supported'):
                verified_mapping(p,self.dictionary,{a.dataset:a},[security()],[])

if __name__=='__main__':unittest.main()
