import copy
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from news_fusion.relations import resolve_relations, validate_master
from news_fusion.pipeline import validate_settings, DEFAULTS, settings_fingerprint,validate_feature_settings
from news_fusion.industry_review import parse_impacts
from news_fusion.io import digest


def axis():
    return SimpleNamespace(dataset='NASDAQ', stock_uids=['NASDAQ:ABC', 'NASDAQ:XYZ'], source={})


def security(**updates):
    row={'dataset':'NASDAQ','stock_uid':'NASDAQ:ABC','array_index':0,'ticker':'ABC','dataset_ticker':'ABC',
         'exchange':'NASDAQ','security_id':'CIK:123','company_name':'Alpha Inc.','aliases':['Alpha'],
         'verification_status':'verified','mapping_source':'https://issuer.example/2013/filing',
         'evidence':'Historical company listing evidence','reviewed_by':'fixture',
         'valid_from':'2013-01-01','valid_to':'2015-01-01','known_from':'2020-01-01T00:00:00Z'}
    row.update(updates)
    return row


def article(entity, **updates):
    row={'source_id':'fixture','published_at':'2014-01-02','title':'Alpha financial report',
         'summary':'Alpha announces revenue growth in its financial report.',
         'input_text_hash':'test-hash','record_version':'test-version','entities':[entity], 'industry_impacts':[]}
    row.update(updates)
    row['input_text_hash']=digest([row.get('title') or '',row.get('summary') or ''])
    return row


class HistoricalRelationsTests(unittest.TestCase):
    def test_exact_code_and_company_use_dataset_index(self):
        links,_,unresolved,_=resolve_relations(axis(),[article({'名称':'Alpha','股票代码':'ABC','交易所':'Nasdaq'})],[security()],[],[])
        self.assertEqual(links[0]['array_index'],0)
        self.assertEqual(links[0]['stock_uid'],'NASDAQ:ABC')
        self.assertFalse(unresolved)

    def test_conflicting_code_does_not_fall_back_to_company_name(self):
        links,_,missing,_=resolve_relations(axis(),[article({'name':'Alpha','ticker':'XYZ'})],[security()],[],[])
        self.assertFalse(links)
        self.assertEqual(missing[0]['reason'],'unverified_or_unmatched_identity')

    def test_wrong_exchange_is_not_silently_ignored(self):
        links,_,_,_=resolve_relations(axis(),[article({'name':'Alpha','ticker':'ABC','exchange':'NYSE'})],[security()],[],[])
        self.assertFalse(links)

    def test_metadata_entity_not_mentioned_in_text_does_not_link(self):
        a=article({'name':'Alpha','ticker':'ABC'},title='Other company report',summary='Other company announces a new product.')
        links,_,_,_=resolve_relations(axis(),[a],[security()],[],[])
        self.assertFalse(links)

    def test_historical_interval_and_pending_master_are_excluded(self):
        for owner in (security(valid_to='2014-01-02'),security(verification_status='pending')):
            links,_,_,_=resolve_relations(axis(),[article({'name':'Alpha','ticker':'ABC'})],[owner],[],[])
            self.assertFalse(links)

    def test_reused_code_requires_identity_disambiguation(self):
        other=security(stock_uid='NASDAQ:XYZ',array_index=1,dataset_ticker='XYZ',security_id='CIK:999',company_name='Other Inc.',aliases=[])
        a=article({'ticker':'ABC'},summary='NASDAQ: ABC reports earnings.')
        with self.assertRaisesRegex(ValueError,'Conflicting historical ticker'):
            resolve_relations(axis(),[a],[security(),other],[],[])

    def test_company_name_ambiguity_is_not_resolved_by_first_row(self):
        other=security(stock_uid='NASDAQ:XYZ',array_index=1,dataset_ticker='XYZ',ticker='XYZ',security_id='CIK:999')
        links,_,missing,_=resolve_relations(axis(),[article({'name':'Alpha'})],[security(),other],[],[])
        self.assertFalse(links);self.assertEqual(missing[0]['reason'],'ambiguous_identity')

    def test_ticker_reuse_with_nonoverlapping_intervals_resolves_correct_row(self):
        other=security(stock_uid='NASDAQ:XYZ',array_index=1,dataset_ticker='XYZ',security_id='CIK:999',valid_from='2015-01-01',valid_to=None)
        links,_,_,_=resolve_relations(axis(),[article({'name':'Alpha','ticker':'ABC'},published_at='2016-01-02')],[security(),other],[],[])
        self.assertEqual(links[0]['stock_uid'],'NASDAQ:XYZ');self.assertEqual(links[0]['array_index'],1)

    def test_numeric_code_is_not_an_array_position(self):
        links,_,_,_=resolve_relations(axis(),[article({'ticker':0,'name':'Alpha'})],[security()],[],[])
        # numeric metadata is invalid, including zero; it must never become a name-only match
        self.assertFalse(links)

    def test_incorrect_array_index_fails(self):
        with self.assertRaisesRegex(ValueError,'index mismatch'):
            validate_master(axis(),[security(array_index=1)],[],[])

    def test_membership_requires_matching_historical_security(self):
        member=security(industry_standard='SIC',industry_version='1987',industry_code='3571',industry_name='Computers',security_id='CIK:999')
        with self.assertRaisesRegex(ValueError,'no matching'):
            validate_master(axis(),[security()],[member],[])

    def test_unsubstantiated_industry_inference_is_not_promoted(self):
        a=article({'name':'Alpha'},industry_impacts=[{'industry_name':'Computers','evidence_type':'inferred','evidence':'invented text','mechanism':'Growth'}])
        alias=security(alias='Computers',industry_standard='SIC',industry_version='1987',industry_code='3571')
        links,_,unresolved,_=resolve_relations(axis(),[a],[security()],[],[alias])
        self.assertEqual([r['channel'] for r in links],['company'])
        self.assertEqual(unresolved[0]['channel'],'industry')

    def test_pipeline_supports_explicit_original_reproduction(self):
        c=copy.deepcopy(DEFAULTS);validate_settings(c)
        c['data_protocol']='original'
        validate_settings(c)
        c['data_protocol']='invalid'
        with self.assertRaisesRegex(ValueError,'strict or original'):validate_settings(c)

    def test_parameters_are_in_preparation_fingerprint(self):
        a={'panel':{'mode':'retrospective'},'fusion':{'enabled_channels':['market']}}
        b=copy.deepcopy(a);b['panel']['mode']='observed_pit'
        self.assertNotEqual(settings_fingerprint(a),settings_fingerprint(b))

    def test_cached_features_cannot_ignore_changed_max_length_or_model(self):
        names=('config.json','model.safetensors','tokenizer_config.json','special_tokens_map.json','vocab.txt')
        hashes={name:'a'*64 for name in names}
        installed={'revision':'fixed-revision','files':hashes}
        config={'finbert':{'model_id':'ProsusAI/finbert','max_length':512,'pooling':'cls_last_hidden'},'pipeline':{'model_revision':'fixed-revision'}}
        manifest={'encoder':{'model_id':'ProsusAI/finbert','max_length':512,'embedding_method':'cls_last_hidden','local_file_sha256':hashes}}
        validate_feature_settings(manifest,config,installed)
        config['finbert']['max_length']=256
        with self.assertRaisesRegex(ValueError,'length/pooling'):validate_feature_settings(manifest,config,installed)
        config['finbert']['max_length']=512
        installed['files']=dict(hashes,**{'model.safetensors':'b'*64})
        with self.assertRaisesRegex(ValueError,'weight/tokenizer'):validate_feature_settings(manifest,config,installed)

    def test_industry_schema_requires_exact_quote_and_allowed_taxonomy(self):
        impact={'industry_name':'Computers','evidence_type':'inferred','evidence':'revenue growth',
                'mechanism':'Greater demand for business equipment','direction':'positive_conditional','confidence':0.9,'event_type':'demand','impact_role':'potential_beneficiary','importance':0.8,'horizon':'short'}
        response={'done':True,'message':{'content':json.dumps({'impacts':[impact]})}}
        self.assertEqual(len(parse_impacts(response,article({}),[{'alias':'Computers'}],3)),1)
        impact['evidence']='Invented passage absent from source'
        response['message']['content']=json.dumps({'impacts':[impact]})
        with self.assertRaisesRegex(ValueError,'exact source'):parse_impacts(response,article({}),[{'alias':'Computers'}],3)

    def test_industry_confidence_percentage_and_hallucinated_code_are_rejected(self):
        impact={'industry_name':'Computers','evidence_type':'explicit','evidence':'revenue growth',
                'mechanism':'Industry demand changes','direction':'unknown','confidence':95,'event_type':'unknown','impact_role':'unknown','importance':0.5,'horizon':'unknown'}
        response={'done':True,'message':{'content':json.dumps({'impacts':[impact]})}}
        with self.assertRaises(ValueError):parse_impacts(response,article({}),[{'alias':'Computers'}],3)
        impact['confidence']=0.95;impact['industry_name']='Invented industry'
        response['message']['content']=json.dumps({'impacts':[impact]})
        with self.assertRaisesRegex(ValueError,'verified taxonomy'):parse_impacts(response,article({}),[{'alias':'Computers'}],3)

    def test_relation_proof_is_independently_resolved_and_tamper_detected(self):
        from news_fusion.relations import build_relations,load_relations
        from news_fusion.io import write_records
        a=axis();a.stock_hash='stock-axis-fixture';a.date_hash='date-axis-fixture'
        provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'a'*64}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);master=root/'master.jsonl';write_records(master,[security()])
            out=root/'relations'
            build_relations(a,[article({'name':'Alpha','ticker':'ABC'})],[master,None,None],out,provenance)
            links,members,manifest=load_relations(out,a)
            self.assertEqual(links[0]['stock_uid'],'NASDAQ:ABC')
            path=out/'security_master.jsonl'
            path.write_text(path.read_text(encoding='utf-8').replace('CIK:123','CIK:999'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_relations(out,a)
