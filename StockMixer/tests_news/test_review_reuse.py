import copy
import json
import tempfile
import unittest
from pathlib import Path
from news_fusion.auto_review import auto_review
from news_fusion.configuration import REVIEW_DEFAULTS
from news_fusion.review_reuse import rebind_review,validate_rebound
from news_fusion.io import read_records,write_records,write_json,file_manifest
from tests_news.test_auto_review import FixtureBackend,response
from tests_news.test_pipeline import article


class ReviewReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.articles=[article('one'),article('two')]
        self.settings=copy.deepcopy(REVIEW_DEFAULTS)
        self.provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'b'*64,'synthetic_fixture':True}

    def tearDown(self):self.temp.cleanup()

    def create(self,limit=None):
        return auto_review(self.articles,'NASDAQ',self.root/'source',self.root/'cache',
            FixtureBackend([response(),response(confidence=0.4)]),self.settings,self.provenance,limit=limit)

    def test_rebind_preserves_decisions_pending_and_actual_completion_time(self):
        source=self.create()
        clone=rebind_review(self.root/'source',self.articles,'NYSE',self.root/'clone',self.provenance,self.settings)
        original=read_records(self.root/'source/market_review.jsonl')
        rows=read_records(self.root/'clone/market_review.jsonl')
        self.assertEqual(rows,[dict(r,dataset='NYSE') for r in original])
        self.assertEqual(clone['decisions'],source['decisions'])
        self.assertEqual(clone['stats']['model_calls'],0)
        self.assertFalse(clone['human_sample_checked'])
        self.assertEqual(rows[1]['decision'],'pending')
        validate_rebound(self.root/'clone',self.articles,self.provenance)

    def test_sample_cannot_be_rebound_as_a_full_review(self):
        self.create(limit=1)
        with self.assertRaisesRegex(ValueError,'full-snapshot'):
            rebind_review(self.root/'source',self.articles,'NYSE',self.root/'clone',self.provenance,self.settings)

    def test_independent_recheck_detects_changed_review_time_even_if_hash_is_refreshed(self):
        self.create()
        rebind_review(self.root/'source',self.articles,'SP500',self.root/'clone',self.provenance,self.settings)
        path=self.root/'clone/market_review.jsonl'
        rows=read_records(path);rows[0]['reviewed_at']='2020-01-01T00:00:00Z';write_records(path,rows)
        manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        manifest['files']=file_manifest(path.parent);write_json(path.parent/'manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'decision, evidence or review timestamp'):
            validate_rebound(path.parent,self.articles,self.provenance)

    def test_prompt_configuration_change_requires_new_review(self):
        self.create();settings=copy.deepcopy(self.settings);settings['seed']+=1
        with self.assertRaisesRegex(ValueError,'different inference settings'):
            rebind_review(self.root/'source',self.articles,'NYSE',self.root/'clone',self.provenance,settings)
