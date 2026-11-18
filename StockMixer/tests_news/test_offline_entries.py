"""Contract fixtures only: no real model inference or real review approval."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from news_fusion.offline_inputs import encoding_inputs
from news_fusion.routing import review_template,publish_routes,load_routes,feature_routes,article_pin
from news_fusion.panel import build_panel,load_features,relationship_valid
from news_fusion.dataset import NewsPanel
from news_fusion.encoder import encode_articles
from news_fusion.io import write_records,write_json,file_manifest,digest_file
from tests_news.test_pipeline import article,make_axis,FakeBackend


class OfflineInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.snapshot=self.root/'frozen';self.snapshot.mkdir()
        write_records(self.snapshot/'articles.jsonl',[article()])
        write_json(self.snapshot/'manifest.json',{'kind':'frozen_current_selection','files':{'articles.jsonl':digest_file(self.snapshot/'articles.jsonl')}})
        self.config={'adapted_snapshot':'frozen','news_root':'raw'}

    def tearDown(self):self.temp.cleanup()

    def test_default_reads_verified_snapshot_and_does_not_scan_raw(self):
        with patch('news_fusion.snapshot.validate_snapshot') as verify:
            articles,counts,provenance=encoding_inputs(self.root,self.config)
        verify.assert_called_once_with(self.snapshot)
        self.assertEqual(counts,{'ready':1})
        self.assertEqual(articles[0]['source_id'],'one')
        self.assertEqual(provenance['mode'],'verified_snapshot')
        self.assertEqual(provenance['snapshot_manifest_sha256'],digest_file(self.snapshot/'manifest.json'))

    def test_unknown_snapshot_and_mid_read_edits_fail_without_raw_fallback(self):
        write_json(self.snapshot/'manifest.json',{'kind':'unknown'})
        with self.assertRaisesRegex(ValueError,'verified frozen/adapted'):
            encoding_inputs(self.root,self.config)
        write_json(self.snapshot/'manifest.json',{'kind':'frozen_current_selection','files':{'articles.jsonl':digest_file(self.snapshot/'articles.jsonl')}})
        def edit(*args):write_json(self.snapshot/'manifest.json',{'kind':'changed'})
        with patch('news_fusion.snapshot.validate_snapshot',side_effect=edit),self.assertRaisesRegex(ValueError,'changed'):
            encoding_inputs(self.root,self.config)

    def test_explicit_files_are_project_relative_and_override_missing_snapshot(self):
        a=article();write_records(self.root/'input.jsonl',[a])
        articles,_,provenance=encoding_inputs(self.root,{'adapted_snapshot':'missing'},['input.jsonl'])
        self.assertEqual(len(articles),1);self.assertEqual(provenance['mode'],'explicit_files')
        a['language']='unknown';write_records(self.root/'input.jsonl',[a])
        with self.assertRaisesRegex(ValueError,'No encoding-ready'):
            encoding_inputs(self.root,self.config,['input.jsonl'])

    def test_cli_check_input_never_loads_a_backend(self):
        from news_fusion import __main__ as cli
        config=dict(self.config,finbert={'model_dir':None})
        with patch.object(cli,'load_llm_config',return_value=config),patch('news_fusion.snapshot.validate_snapshot'),\
             patch('news_fusion.encoder.LocalFinBert') as backend,contextlib.redirect_stdout(io.StringIO()) as output:
            cli.main(['--project-root',str(self.root),'encode','--check-input'])
        backend.assert_not_called()
        result=json.loads(output.getvalue());self.assertEqual(result['selected_articles'],1)
        self.assertFalse(result['model_inference_run'])


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.axis,self.schedule=make_axis(self.root)
        self.articles=[article('approved',scopes=['market']),article('excluded',scopes=['market'])]
        self.provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'a'*64,'synthetic_unit_fixture':True}
        review_template(self.articles,'TEST',self.root/'template',self.provenance)
        self.reviews=[json.loads(line) for line in (self.root/'template/market_review.jsonl').read_text().splitlines()]

    def tearDown(self):self.temp.cleanup()

    def resolve_reviews(self):
        reviews=copy.deepcopy(self.reviews)
        for row in reviews:
            row.update(decision='include' if row['source_id']=='approved' else 'exclude',
                reviewed_at='2020-01-01T00:00:00Z',reviewed_by='SYNTHETIC UNIT REVIEWER',
                rationale='Synthetic fixture only: market impact is assessed explicitly.',evidence_quote='Revenue improves for business')
        return reviews

    def publish(self,reviews=None,name='routes',partial=False):
        publish_routes(self.articles,reviews or self.resolve_reviews(),'TEST',self.root/name,self.provenance,partial)
        return self.root/name

    def test_pending_template_cannot_be_published(self):
        self.assertTrue(all(row['decision']=='pending' for row in self.reviews))
        with self.assertRaisesRegex(ValueError,'pending'):
            self.publish(self.reviews)
        self.assertFalse((self.root/'routes').exists())

    def test_complete_review_roundtrip_and_real_publication_time(self):
        routes,bindings,manifest=load_routes(self.publish(),'TEST')
        self.assertEqual([r['source_id'] for r in routes],['approved'])
        self.assertEqual(manifest['decisions'],{'include':1,'exclude':1})
        self.assertEqual(routes[0]['article_pin'],article_pin(self.articles[0]))
        self.assertGreater(routes[0]['known_from'],'2020-01-01')
        self.assertFalse(relationship_valid(routes[0],'2020-03-06','2020-03-05',self.schedule.cutoffs[1],'observed_pit'))

    def test_partial_review_requires_explicit_opt_in(self):
        rows=self.resolve_reviews();rows[1]=self.reviews[1]
        with self.assertRaisesRegex(ValueError,'pending'):self.publish(rows)
        _,_,manifest=load_routes(self.publish(rows,name='partial',partial=True),'TEST')
        self.assertEqual(manifest['decisions']['pending'],1)

    def test_wrong_versions_missing_evidence_reviewer_and_identity_are_rejected(self):
        for field,value in [('article_pin','b'*64),('evidence_quote','not present in the article'),('reviewed_by',''),('rationale','')]:
            rows=self.resolve_reviews();rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.publish(rows)
        with self.assertRaises(ValueError):self.publish(self.resolve_reviews()[:1])
        rows=self.resolve_reviews();rows[1]=rows[0]
        with self.assertRaises(ValueError):self.publish(rows)

    def test_dataset_mismatch_and_rehashed_route_edits_are_rejected(self):
        routes=self.publish()
        with self.assertRaisesRegex(ValueError,'dataset'):load_routes(routes,'SP500')
        rows=json.loads((routes/'market_routes.jsonl').read_text().splitlines()[0]);rows['reviewed_by']='changed'
        write_records(routes/'market_routes.jsonl',[rows])
        manifest=json.loads((routes/'manifest.json').read_text());manifest['files']=file_manifest(routes);write_json(routes/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'review decisions'):load_routes(routes,'TEST')

    def test_publication_cannot_retarget_review_or_change_pinned_text(self):
        with self.assertRaisesRegex(ValueError,'dataset'):
            publish_routes(self.articles,self.resolve_reviews(),'SP500',self.root/'wrong_dataset',self.provenance)
        edited=copy.deepcopy(self.articles);edited[0]['summary']='Fabricated new text with unchanged old input hash'
        rows=self.resolve_reviews();rows[0]['summary']=edited[0]['summary']
        with self.assertRaisesRegex(ValueError,'content hash'):
            publish_routes(edited,rows,'TEST',self.root/'wrong_text',self.provenance)

    def test_features_must_match_reviewed_source_versions(self):
        routes=self.publish()
        encode_articles(self.articles,FakeBackend(),self.root/'features',self.root/'cache',input_provenance=self.provenance)
        articles,_,manifest=load_features(self.root/'features')
        accepted,_=feature_routes(routes,self.axis,articles,manifest)
        self.assertEqual(len(accepted),1)
        articles[0]['input_text_hash']='b'*64
        with self.assertRaisesRegex(ValueError,'version'):feature_routes(routes,self.axis,articles,manifest)

    def test_reviewed_panel_publishes_proof_and_excludes_unapproved_news(self):
        routes=self.publish()
        encode_articles(self.articles,FakeBackend(),self.root/'features',self.root/'cache',input_provenance=self.provenance)
        manifest=build_panel(self.axis,self.schedule,self.root/'features',[],[],self.root/'panel',
            market_routes=routes,projection='random_projection',decision_lookback=2)
        self.assertTrue(manifest['market_routing_verified'])
        panel=NewsPanel(self.root/'panel');panel.assert_market_routing()
        ledger=[json.loads(line) for line in (self.root/'panel/provenance.jsonl').read_text().splitlines()]
        self.assertEqual({row['source_id'] for row in ledger},{'approved'})
        self.assertGreater(panel.masks['market'].sum(),0)
        self.assertEqual(panel.masks['company'].sum(),0)

    def test_boolean_alone_cannot_make_an_unreviewed_panel_trainable(self):
        encode_articles(self.articles,FakeBackend(),self.root/'features',self.root/'cache')
        build_panel(self.axis,self.schedule,self.root/'features',[],[],self.root/'panel',projection='random_projection',decision_lookback=2)
        path=self.root/'panel/manifest.json';manifest=json.loads(path.read_text());manifest['market_routing_verified']=True;write_json(path,manifest)
        with self.assertRaisesRegex(ValueError,'pinned review artifact'):NewsPanel(self.root/'panel')

    def test_unreviewed_market_links_cannot_bypass_artifact(self):
        encode_articles(self.articles,FakeBackend(),self.root/'features',self.root/'cache')
        with self.assertRaisesRegex(ValueError,'published market-route'):
            build_panel(self.axis,self.schedule,self.root/'features',[{'source_id':'approved','channel':'market'}],[],
                self.root/'panel',projection='random_projection',decision_lookback=2)

if __name__=='__main__':unittest.main()
