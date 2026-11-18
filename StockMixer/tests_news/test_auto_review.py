"""Synthetic local-review protocol tests; no real model service required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from news_fusion.auto_review import auto_review,parse_review,OllamaReviewer
from news_fusion.configuration import REVIEW_DEFAULTS,validate_review_settings
from news_fusion.routing import checked_reviews,publish_routes,load_routes
from news_fusion.io import read_records,verify_files
from tests_news.test_pipeline import article


def response(decision='include',scope='us_broad_market',confidence=.95,quote='Revenue improves for business'):
    result={'decision':decision,'scope':scope,'confidence':confidence,'evidence_quote':quote,
        'rationale':'SYNTHETIC fixture reasoning for contract validation only.',
        'impact_mechanism':'SYNTHETIC fixture transmission mechanism, not a real news assessment.'}
    return {'model':'SYNTHETIC_TEST_ONLY','done':True,'done_reason':'stop','message':{'content':json.dumps(result)}}


class FixtureBackend:
    def __init__(self,answers=None,digest='a'*64):
        self.calls=0;self.answers=list(answers or []);self.corrections=[]
        self.metadata={'provider':'SYNTHETIC_TEST_ONLY','model':REVIEW_DEFAULTS['model'],'model_digest':digest,
            'request_options':{k:REVIEW_DEFAULTS[k] for k in ('temperature','seed','num_ctx','num_predict')}}
    def assert_current(self):pass
    def review(self,article,correction=None):
        self.calls+=1;self.corrections.append(correction)
        value=self.answers.pop(0) if self.answers else response()
        if isinstance(value,BaseException):raise value
        return value


class AutoReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.articles=[article('one'),article('two')]
        self.settings=copy.deepcopy(REVIEW_DEFAULTS)
        self.provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'b'*64,'synthetic_fixture':True}
    def tearDown(self):self.temp.cleanup()
    def run_review(self,name,backend,**kwargs):
        return auto_review(self.articles,kwargs.pop('dataset','NASDAQ'),self.root/name,self.root/'cache',backend,
            kwargs.pop('settings',self.settings),self.provenance,**kwargs)

    def test_valid_reviews_are_bound_to_dataset_and_publishable_as_machine_reviews(self):
        backend=FixtureBackend();manifest=self.run_review('first',backend)
        rows=read_records(self.root/'first/market_review.jsonl')
        self.assertEqual(backend.calls,2);self.assertEqual(manifest['decisions'],{'include':2})
        self.assertTrue(all(row['review_method']=='local_llm' and row['dataset']=='NASDAQ' for row in rows))
        self.assertFalse(manifest['human_sample_checked'])
        verify_files(self.root/'first',manifest,required=('audit.jsonl','market_review.jsonl','prompt.txt'))
        checked_reviews(self.articles,rows,dataset='NASDAQ')
        publish_routes(self.articles,rows,'NASDAQ',self.root/'routes',self.provenance)
        self.assertEqual(len(load_routes(self.root/'routes','NASDAQ')[0]),2)

    def test_other_us_datasets_reuse_cache_without_additional_inference(self):
        self.run_review('first',FixtureBackend())
        backend=FixtureBackend();manifest=self.run_review('second',backend,dataset='NYSE')
        self.assertEqual(backend.calls,0);self.assertEqual(manifest['stats']['cache_hits'],2)
        self.assertFalse(manifest['model_inference_run'])
        self.assertTrue(all(row['dataset']=='NYSE' for row in read_records(self.root/'second/market_review.jsonl')))

    def test_model_version_change_invalidates_cache(self):
        self.run_review('first',FixtureBackend())
        backend=FixtureBackend(digest='c'*64);self.run_review('second',backend)
        self.assertEqual(backend.calls,2)

    def test_low_confidence_remains_pending_and_threshold_reapplies_to_cached_response(self):
        self.run_review('first',FixtureBackend([response(confidence=.4)]),limit=1)
        rows=read_records(self.root/'first/market_review.jsonl')
        self.assertEqual(rows[0]['decision'],'pending');self.assertEqual(rows[0]['auto_review_status'],'low_confidence')
        settings=copy.deepcopy(self.settings);settings['min_confidence']=.3
        backend=FixtureBackend();self.run_review('second',backend,limit=1,settings=settings)
        self.assertEqual(backend.calls,0)
        self.assertEqual(read_records(self.root/'second/market_review.jsonl')[0]['decision'],'include')

    def test_bad_evidence_retries_and_retains_raw_attempts(self):
        backend=FixtureBackend([response(quote='Fabricated quote absent from article'),response()])
        manifest=self.run_review('review',backend,limit=1)
        audit=read_records(self.root/'review/audit.jsonl')
        self.assertEqual(backend.calls,2);self.assertEqual(manifest['stats']['failed_attempts'],1)
        self.assertIn('verbatim',backend.corrections[1]['error'])
        self.assertIsNotNone(backend.corrections[1]['previous_content'])
        self.assertIn('response',audit[0]);self.assertEqual(audit[0]['event'],'failed_attempt')
        self.assertEqual(read_records(self.root/'review/market_review.jsonl')[0]['decision'],'include')

    def test_consecutive_failures_stop_and_preserve_pending_output(self):
        settings=copy.deepcopy(self.settings);settings['max_consecutive_failures']=1
        backend=FixtureBackend([OSError('fixture unavailable'),OSError('fixture unavailable')])
        with self.assertRaisesRegex(ValueError,'Consecutive'):
            self.run_review('failed',backend,settings=settings)
        state=json.loads((self.root/'failed/run_status.json').read_text())
        self.assertEqual(state['status'],'interrupted_or_failed')
        self.assertTrue(all(row['decision']=='pending' for row in read_records(self.root/'failed/market_review.jsonl')))
        self.assertFalse((self.root/'failed/manifest.json').exists())

    def test_restart_after_interruption_reuses_completed_cache_and_actual_time(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_review('interrupted',FixtureBackend([response(),KeyboardInterrupt()]))
        previous=read_records(self.root/'interrupted/market_review.jsonl')[0]['reviewed_at']
        backend=FixtureBackend();self.run_review('resumed_new_version',backend)
        self.assertEqual(backend.calls,1)
        self.assertEqual(read_records(self.root/'resumed_new_version/market_review.jsonl')[0]['reviewed_at'],previous)

    def test_invalid_cache_is_not_trusted(self):
        self.run_review('first',FixtureBackend())
        cached=next((self.root/'cache').glob('*.json'));cached.write_text('{corrupt',encoding='utf-8')
        backend=FixtureBackend();self.run_review('second',backend)
        self.assertEqual(backend.calls,1)
        self.assertIn('invalid_cache',[row['event'] for row in read_records(self.root/'second/audit.jsonl')])

    def test_explicit_subset_keeps_all_other_rows_pending_and_never_truncates_text(self):
        original=copy.deepcopy(self.articles)
        self.run_review('subset',FixtureBackend(),source_ids=['two'])
        rows=read_records(self.root/'subset/market_review.jsonl')
        self.assertEqual([row['decision'] for row in rows],['pending','include'])
        self.assertEqual(self.articles,original)
        settings=copy.deepcopy(self.settings);settings['max_input_chars']=128
        self.articles[0]['summary']='Long text '*100
        backend=FixtureBackend();self.run_review('oversized',backend,limit=1,settings=settings)
        self.assertEqual(backend.calls,0)
        self.assertEqual(read_records(self.root/'oversized/market_review.jsonl')[0]['auto_review_status'],'input_too_long')

    def test_json_schema_truncation_nonfinite_and_scope_contradictions_fail(self):
        for value in (response(scope='company'),response(confidence=True),response(confidence=float('nan')),
                      {'done':False,'message':{'content':'{}'}},dict(response(),done_reason='length')):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_review(value,self.articles[0])

    def test_external_endpoint_and_invalid_settings_fail_before_contact(self):
        settings=copy.deepcopy(self.settings);settings['endpoint']='https://example.com'
        with self.assertRaisesRegex(ValueError,'loopback'):OllamaReviewer(settings)
        for key,value in [('min_confidence',1.1),('max_attempts',0),('think','false'),('expected_model_digest','bad')]:
            settings=copy.deepcopy(self.settings);settings[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_review_settings(settings)

    def test_existing_outputs_and_overlapping_cache_are_not_overwritten(self):
        self.run_review('first',FixtureBackend())
        with self.assertRaises(FileExistsError):self.run_review('first',FixtureBackend())
        with self.assertRaisesRegex(ValueError,'separate'):
            auto_review(self.articles,'NASDAQ',self.root/'cache',self.root/'cache',FixtureBackend(),self.settings,self.provenance)

if __name__=='__main__':unittest.main()