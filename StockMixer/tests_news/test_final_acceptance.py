import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from news_fusion.final_acceptance import valid_windows,summarize_final
from news_fusion.sample_review import claim_pin,apply_review,market_proof
from news_fusion.semantic_review import apply_sample
from news_fusion.routing import article_pin
from news_fusion.io import digest,file_manifest,write_json,write_records,now


class FinalInputTests(unittest.TestCase):
    def setUp(self):
        self.axis=SimpleNamespace(dataset='TEST',stock_uids=['TEST:A','TEST:B'],dates=list(range(40)),train_end=24,valid_end=32)
        self.config={'panel':{'decision_lookback':16},'pipeline':{'data_protocol':'original','minimum_training_cells':1},
            'fusion':{'enabled_channels':['market']},'market_channel':{'gates':{
                'min_events':dict(train=1,valid=1,test=1),'min_active_days':dict(train=1,valid=1,test=1),'max_event_type_share':None}}}

    def masks(self):
        return {ch:np.zeros((2,40),dtype=bool) for ch in ('company','industry','market')}

    def test_price_window_includes_target_label(self):
        mask=np.ones((2,40));mask[0,16]=0
        data=SimpleNamespace(arrays={'mask':mask},split=SimpleNamespace(lookback=16,steps=1))
        valid=valid_windows(data,16)
        self.assertFalse(valid[0,15]);self.assertTrue(valid[1,15])
        self.assertFalse(valid[:,-1].any());self.assertFalse(valid[:,:15].any())

    def test_price_window_includes_first_input_day(self):
        mask=np.ones((2,40));mask[0,0]=0
        valid=valid_windows(SimpleNamespace(arrays={'mask':mask},split=SimpleNamespace(lookback=16,steps=1)),16)
        self.assertFalse(valid[0,15]);self.assertTrue(valid[0,16])

    def test_same_event_counts_once_after_two_reports(self):
        masks=self.masks();masks['market'][0,16]=True
        ledger=[{'channel':'market','stock_uid':'TEST:A','decision_idx':16,'source_id':sid,
                 'input_text_hash':sid,'market_event_id':'same_event'} for sid in ['a','b']]
        r=summarize_final(self.axis,masks,ledger,np.ones((2,40),dtype=bool),self.config)
        self.assertEqual(r['coverage']['market']['train']['articles'],2)
        self.assertEqual(r['coverage']['market']['train']['events'],1)

    def test_invalid_prices_remove_events_and_days_before_gate(self):
        masks=self.masks();masks['market'][0,16]=True
        ledger=[{'channel':'market','stock_uid':'TEST:A','decision_idx':16,'source_id':'a','market_event_id':'e'}]
        valid=np.ones((2,40),dtype=bool);valid[0,16]=False
        r=summarize_final(self.axis,masks,ledger,valid,self.config)
        self.assertEqual(r['coverage']['market']['train']['panel_active_mask_cells'],1)
        self.assertEqual(r['coverage']['market']['train']['valid_price_label_cells'],0)
        self.assertEqual(r['market_quality_after_price_filter']['splits']['train']['events'],0)
        self.assertTrue(any('train events=0' in b for b in r['blockers']))

    def test_original_overlap_not_counted_as_nominal_training(self):
        masks=self.masks();masks['industry'][0,23]=True
        ledger=[{'channel':'industry','stock_uid':'TEST:A','decision_idx':23,'source_id':'a'}]
        r=summarize_final(self.axis,masks,ledger,np.ones((2,40),dtype=bool),self.config)
        self.assertEqual(r['coverage']['industry']['train']['articles'],0)
        self.assertEqual(r['coverage']['industry']['valid']['articles'],1)
        self.assertEqual(r['coverage']['industry']['original_train_candidates']['articles'],1)


class SourceBoundSampleTests(unittest.TestCase):
    def setUp(self):
        self.a={'source_id':'a','source_record_hash':'raw','input_text_hash':digest(['Title','A company appoints a new lawyer.']),
            'title':'Title','summary':'A company appoints a new lawyer.','impact_scope':['company','industry','market'],
            'industry_impacts':[{'industry_key':'GOOD'},{'industry_key':'BAD'}],'market_scope':['US_EQUITY'],'scope_evidence':[]}
        self.row={'source_id':'a','claim_pin':claim_pin(self.a),'review_method':'assistant_secondary_review','reviewed_by':'Codex',
            'reviewed_at':now(),'rationale':'A company appointment does not establish a whole-market event.',
            'evidence_quote':'A company appoints a new lawyer.','verdict':'restrict','exclude_channels':['market'],
            'exclude_industry_keys':['BAD']}

    def artifact(self,d,row):
        write_records(d/'reviews.jsonl',[row]);write_json(d/'manifest.json',{'kind':'source_bound_semantic_sample_v1',
            'sample_count':1,'files':file_manifest(d)})

    def test_keep_supported_industry_and_preserve_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,self.row);out,s=apply_review([self.a],d)
            self.assertEqual(out[0]['industry_impacts'],[{'industry_key':'GOOD'}])
            self.assertEqual(out[0]['summary'],self.a['summary'])
            self.assertEqual(s['human_confirmed_samples'],0)
            self.assertEqual(len(self.a['industry_impacts']),2)

    def test_changed_text_or_claims_cannot_inherit_old_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,self.row)
            for key,value in [('input_text_hash','changed'),('source_record_hash','changed'),('market_scope',[])]:
                a=copy.deepcopy(self.a);a[key]=value
                with self.assertRaisesRegex(ValueError,'stale'):apply_review([a],d)

    def test_assistant_cannot_claim_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,dict(self.row,review_method='human_confirmation'))
            with self.assertRaisesRegex(ValueError,'human'):apply_review([self.a],d)

    def test_unknown_industry_key_cannot_be_silently_restricted(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,dict(self.row,exclude_industry_keys=['NOT_A_CLAIM']))
            with self.assertRaisesRegex(ValueError,'absent'):apply_review([self.a],d)

    def test_bound_quote_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,dict(self.row,evidence_quote='A fabricated quote not in the article.'))
            with self.assertRaisesRegex(ValueError,'bound'):apply_review([self.a],d)

    def test_market_exclusion_replays_existing_semantic_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);self.artifact(d,self.row);a=apply_review([self.a],d)[0][0]
            reviews=[{'source_id':'a','article_pin':article_pin(a),'decision':'include','evidence_quote':self.row['evidence_quote'],
                'rationale':'Prior automatic market approval.','reviewed_by':'upstream','reviewed_at':now()}]
            proof=market_proof([a],reviews,manifest_sha='0'*64)
            result,summary=apply_sample([a],reviews,proof['rows'],proof['confirmation'])
            self.assertEqual(result[0]['decision'],'exclude');self.assertEqual(summary['human_confirmed_samples'],0)


if __name__=='__main__':unittest.main()
