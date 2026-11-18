import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import torch
import yaml
from news_fusion.pipeline import Pipeline,validate_panel_settings,checked_state,settings_fingerprint,main
from news_fusion.configuration import REVIEW_DEFAULTS,compile_config,read_yaml
from news_fusion.io import read_records,write_records,write_json,digest_file
from news_fusion.industry_review import enrich_industries,parse_impacts
from news_fusion.review_reuse import validate_review_artifact
from news_fusion.auto_review import auto_review
from news_fusion.protocol import PROJECT
from news_fusion import experiment
from tests_news.test_auto_review import FixtureBackend
from tests_news.test_historical_relations import security,article as identity_article
from tests_news.test_pipeline import article


def industry_response():
    impact={'industry_name':'Computers','evidence_type':'inferred','evidence':'revenue growth',
            'mechanism':'Greater demand for business equipment','direction':'positive_conditional','confidence':.9,
            'event_type':'demand','impact_role':'potential_beneficiary','importance':.8,'horizon':'short'}
    return {'done':True,'message':{'content':json.dumps({'impacts':[impact]})}}


class IndustryBackend(FixtureBackend):
    def request(self,path,payload=None):
        self.calls+=1
        response=self.answers.pop(0)
        if isinstance(response,BaseException):raise response
        return response


class PipelineRecheckTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=copy.deepcopy(REVIEW_DEFAULTS)
        self.provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'b'*64,'synthetic_fixture':True}
    def tearDown(self):self.temp.cleanup()

    def test_direct_review_reuse_rejects_changed_confidence_threshold(self):
        articles=[article('one')]
        auto_review(articles,'NASDAQ',self.root/'review',self.root/'cache',FixtureBackend(),self.settings,self.provenance)
        changed=dict(self.settings,min_confidence=.99)
        with self.assertRaisesRegex(ValueError,'different inference settings'):
            validate_review_artifact(self.root/'review',articles,'NASDAQ',self.provenance,changed)

    def test_completed_step_is_revalidated_instead_of_skipped(self):
        pipeline=object.__new__(Pipeline)
        pipeline.state={'steps':{'review':'completed'}}
        pipeline.verify_completed=Mock(side_effect=ValueError('stale artifact'))
        pipeline.save=Mock()
        with self.assertRaisesRegex(ValueError,'stale artifact'):pipeline.run('review')
        pipeline.save.assert_called_once()
        self.assertTrue(pipeline.state['last_error']['cache_recheck'])

    def test_mutated_article_binding_is_not_accepted_with_a_stale_text_hash(self):
        from news_fusion.relations import resolve_relations
        from tests_news.test_historical_relations import axis
        a=identity_article({'name':'Alpha','ticker':'ABC'})
        a['summary']+=' Added text outside the frozen version.'
        with self.assertRaisesRegex(ValueError,'input_text_hash'):
            resolve_relations(axis(),[a],[security()],[],[])

    def test_accepted_manifest_change_is_detected_before_resume(self):
        outputs=self.root/'outputs';outputs.mkdir()
        snapshot=self.root/'snapshot';snapshot.mkdir();write_json(snapshot/'manifest.json',{'fixture':True})
        artifact=self.root/'artifact';artifact.mkdir();write_json(artifact/'manifest.json',{'fixture':1})
        config={'pipeline':{'state':str(outputs/'state.json')},'adapted_snapshot':str(snapshot)}
        state={'settings_sha256':settings_fingerprint(config),'snapshot_manifest_sha256':digest_file(snapshot/'manifest.json'),
               'accepted_manifests':{str(artifact):digest_file(artifact/'manifest.json')}}
        write_json(outputs/'state.json',state)
        write_json(artifact/'manifest.json',{'fixture':2})
        with self.assertRaisesRegex(ValueError,'manifest changed'):checked_state(self.root,config)

    def test_non_boolean_main_switch_cannot_launch_a_workflow(self):
        base=read_yaml(PROJECT/'configs/stockmixer.yaml');base['llm']['enabled']='false'
        path=self.root/'bad.yaml';path.write_text(yaml.safe_dump(base),encoding='utf-8')
        with patch('news_fusion.workflow.run_workflow') as run:
            with self.assertRaisesRegex(ValueError,'boolean'):main(['--config',str(path)])
        run.assert_not_called()

    def test_completed_review_rechecks_snapshot_without_model_calls(self):
        articles=[article('one')];backend=FixtureBackend()
        auto_review(articles,'NASDAQ',self.root/'review',self.root/'cache',backend,self.settings,self.provenance)
        pipeline=object.__new__(Pipeline)
        pipeline.config={'review':self.settings};pipeline.state={'artifacts':{'NASDAQ':{'review':str(self.root/'review')}}}
        pipeline.articles=lambda:(articles,{},dict(self.provenance,snapshot_manifest_sha256='c'*64))
        with self.assertRaisesRegex(ValueError,'another snapshot'):pipeline.verify_completed('review')
        self.assertEqual(backend.calls,1)

    def test_encoder_cpu_threads_are_restored_on_error(self):
        original=torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            pipeline=object.__new__(Pipeline);pipeline.settings={'finbert_threads':1}
            def fail():
                self.assertEqual(torch.get_num_threads(),1)
                raise RuntimeError('fixture failure')
            pipeline._encode=fail
            with self.assertRaises(RuntimeError):pipeline.step_encode()
            self.assertEqual(torch.get_num_threads(),2)
        finally:torch.set_num_threads(original)

    def industry_args(self):
        alias=security(alias='Computers',industry_standard='SIC',industry_version='1987',industry_code='3571')
        a=identity_article({'name':'Alpha'},industry_impacts=[{'legacy':'must not survive failed new inference'}])
        a['title']=a['title'] or ''
        options={'industry_max_impacts':3,'industry_min_confidence':.85}
        review=dict(self.settings,max_attempts=1)
        return a,alias,options,review

    def test_failed_industry_inference_does_not_reuse_old_impacts(self):
        a,alias,options,review=self.industry_args()
        backend=IndustryBackend([OSError('fixture failure')])
        rows,stats=enrich_industries([a],[alias],review,options,self.root/'cache',self.root/'industry',backend)
        self.assertFalse(rows[0]['industry_impacts']);self.assertEqual(rows[0]['industry_inference_status'],'failed')
        self.assertEqual(stats['failed_articles'],1)
        self.assertTrue(read_records(self.root/'industry/audit.jsonl')[0]['error'])

    def test_consecutive_industry_failure_retains_audit_and_partial_status(self):
        a,alias,options,review=self.industry_args();review['max_consecutive_failures']=1
        with self.assertRaisesRegex(ValueError,'Repeated industry'):
            enrich_industries([a],[alias],review,options,self.root/'cache',self.root/'industry',IndustryBackend([OSError('retained error')]))
        self.assertTrue((self.root/'industry/audit.jsonl').exists())
        self.assertTrue((self.root/'industry/prompt.txt').exists())
        self.assertFalse((self.root/'industry/manifest.json').exists())
        self.assertEqual(json.loads((self.root/'industry/run_status.json').read_text())['status'],'interrupted_or_failed')

    def test_successful_industry_response_and_all_semantic_fields_are_preserved(self):
        a,alias,options,review=self.industry_args()
        rows,_=enrich_industries([a],[alias],review,options,self.root/'cache',self.root/'industry',IndustryBackend([industry_response()]))
        impact=rows[0]['industry_impacts'][0]
        self.assertEqual(impact['impact_role'],'potential_beneficiary');self.assertEqual(impact['horizon'],'short')
        self.assertEqual(read_records(self.root/'industry/audit.jsonl')[0]['response'],industry_response())

    def test_industry_confidence_reapplies_to_cache_without_another_call(self):
        a,alias,options,review=self.industry_args()
        rows,_=enrich_industries([a],[alias],review,dict(options,industry_min_confidence=.95),self.root/'cache',self.root/'first',IndustryBackend([industry_response()]))
        self.assertFalse(rows[0]['industry_impacts'])
        backend=IndustryBackend([])
        rows,_=enrich_industries([a],[alias],review,options,self.root/'cache',self.root/'second',backend)
        self.assertEqual(backend.calls,0);self.assertEqual(len(rows[0]['industry_impacts']),1)

    def test_panel_cannot_silently_use_other_aggregation_or_encoder_settings(self):
        config={'panel':{'projection_components':16,'lookback_sessions':5,'simulated_latency_seconds':300,
                        'decay_tau_sessions':3.,'projection':'pca','projection_seed':1234},
                'fusion':{'news_dim':24},'finbert':{'model_id':'ProsusAI/finbert','max_length':512,'pooling':'cls_last_hidden'}}
        manifest={'feature_dim':24,'lookback_sessions':5,'simulated_latency_seconds':300,'decay_tau_sessions':3.,
                  'transform':{'fit_mode':'train_only_pca','components':16},
                  'article_encoder':{'model_id':'ProsusAI/finbert','max_length':512,'embedding_method':'cls_last_hidden'}}
        validate_panel_settings(SimpleNamespace(manifest=manifest),config)
        for key,value in [('lookback_sessions',3),('decay_tau_sessions',9.)]:
            other=copy.deepcopy(manifest);other[key]=value
            with self.assertRaisesRegex(ValueError,'configured'):validate_panel_settings(SimpleNamespace(manifest=other),config)

    def test_explicit_panel_or_baseline_path_is_never_overwritten_by_automatic_resolution(self):
        base=read_yaml(PROJECT/'configs/stockmixer.yaml');base['market_name']='NYSE';base['llm']['enabled']=True
        llm=read_yaml(PROJECT/'configs/llm.yaml')
        from news_fusion.pipeline import baseline_signature
        for manual in ('panel','baseline'):
            c=copy.deepcopy(llm)
            explicit=str(self.root/('manual_'+manual))
            if manual=='panel':c['panels']['NYSE']=explicit
            else:c['baseline_checkpoint']=explicit
            with patch('news_fusion.configuration.load_llm_config',return_value=c),\
                 patch('news_fusion.pipeline.prepared_inputs',return_value={'panel':'auto_panel','baseline_checkpoint':'auto_baseline'}),\
                 patch('news_fusion.pipeline.checked_state',return_value={'baseline_parameters_sha256':baseline_signature(base)}):
                result=compile_config(PROJECT,PROJECT/'configs/stockmixer.yaml',base)
            actual=result['news']['panel'] if manual=='panel' else result['workflow']['baseline_checkpoint']
            self.assertEqual(actual,explicit)

    def test_news_on_invalid_price_windows_does_not_count_as_trainable(self):
        masks=np.zeros((2,20),dtype=np.uint8);masks[1]=1
        news=np.zeros((2,20),dtype=np.uint8);news[0,15:17]=1
        panel=SimpleNamespace(manifest={'feature_dim':24,'article_encoder':{'model_id':'ProsusAI/finbert'}},
            masks={'company':news},assert_experiment=lambda *a,**k:None,assert_historical_relations=lambda:None)
        data=SimpleNamespace(axis=object(),arrays={'mask':masks},
            split=SimpleNamespace(lookback=16,targets=lambda p:np.array([16,17])))
        config={'fusion':{'time_mode':'retrospective','news_dim':24,'enabled_channels':['company']}}
        with patch('news_fusion.dataset.NewsPanel',return_value=panel):
            with self.assertRaisesRegex(ValueError,'price/label'):experiment.load_panel('fixture',data,config)
            news[1,15]=1
            self.assertIs(experiment.load_panel('fixture',data,config),panel)
