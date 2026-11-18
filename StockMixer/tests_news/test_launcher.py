import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import yaml

from news_fusion.configuration import read_yaml, load_llm_config
from news_fusion.io import digest_file, write_json
from news_fusion.launcher import Launcher, PREPARATION_STEPS, completed_retrace
from news_fusion.pipeline import Pipeline, STEPS
from news_fusion.protocol import PROJECT
from news_fusion.source_adapter import validate_adapted
from tests_news import test_retrace_adapter


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'configs').mkdir()
        self.batch = self.root/'upstream'
        test_retrace_adapter.RetraceAdapterTests().fixture(self.batch/'adapted')
        write_json(self.batch/'progress.json', {'pending': 0, 'total': 1, 'processed': 1})
        write_json(self.batch/'live_progress.json', {'pending': 0})
        write_json(self.batch/'acceptance.json', {'passed': True})
        self.seal_source()
        self.base = read_yaml(PROJECT/'configs/stockmixer.yaml')
        self.base['llm']['config'] = 'llm.yaml'
        self.base['data']['axis_dir'] = str(PROJECT/'news_assets/axes')
        self.config = load_llm_config(PROJECT/'configs/llm.yaml')
        self.config['pipeline'].update(python=sys.executable, retrace_artifact=str(self.batch), datasets=['NASDAQ'],
            launch_state='outputs/artifacts/news/launch_state.json', auto_import_retrace=True)
        self.write_configs()

    def write_configs(self):
        self.base_path = self.root/'configs/stockmixer.yaml'
        self.llm_path = self.root/'configs/llm.yaml'
        self.base_path.write_text(yaml.safe_dump(self.base, sort_keys=False), encoding='utf-8')
        self.llm_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding='utf-8')

    def seal_source(self):
        files = {f.relative_to(self.batch).as_posix(): digest_file(f)
                 for f in self.batch.rglob('*') if f.is_file() and f.name != 'manifest.json'}
        write_json(self.batch/'manifest.json', {'classification_contract': 'dictionary-scope-replacement-1', 'files': files})

    def launcher(self):
        return Launcher(self.root)

    def test_running_source_blocks_without_activation_or_training(self):
        original = self.llm_path.read_bytes()
        write_json(self.batch/'live_progress.json', {'pending': 2})
        with patch('news_fusion.launcher.Pipeline') as pipeline, patch('news_fusion.workflow.run_workflow') as train:
            with self.assertRaisesRegex(ValueError, '尚未完成'):
                self.launcher().run()
        self.assertEqual(original, self.llm_path.read_bytes())
        pipeline.assert_not_called()
        train.assert_not_called()

    def test_failed_acceptance_and_changed_file_are_rejected(self):
        write_json(self.batch/'acceptance.json', {'passed': False})
        with self.assertRaisesRegex(ValueError, 'acceptance'):
            completed_retrace(self.batch)
        write_json(self.batch/'acceptance.json', {'passed': True})
        self.seal_source()
        (self.batch/'adapted/data/US/article.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            completed_retrace(self.batch)

    def test_manifest_cannot_escape_source_root(self):
        write_json(self.batch/'manifest.json', {'classification_contract': 'dictionary-scope-replacement-1', 'files': {'../secret': 'a'*64}})
        with self.assertRaisesRegex(ValueError, 'escaped'):
            completed_retrace(self.batch)

    def test_real_import_preserves_source_and_invalidates_only_old_products(self):
        before = {f: digest_file(f) for f in self.batch.rglob('*') if f.is_file()}
        old = load_llm_config(self.llm_path)
        result = self.launcher().run('import')
        new = load_llm_config(self.llm_path)
        self.assertEqual(result['status'], 'imported')
        self.assertFalse(result['source']['partial'])
        self.assertTrue(validate_adapted(result['snapshot'])['valid'])
        self.assertEqual({f: digest_file(f) for f in before}, before)
        for key in ('panel', 'fusion', 'training', 'finbert', 'baseline_checkpoint'):
            self.assertEqual(new[key], old[key])
        self.assertEqual(new['pipeline'].get('sample_review'), old['pipeline'].get('sample_review'))
        self.assertNotEqual(new['pipeline']['state'], old['pipeline']['state'])
        self.assertTrue(Path(result['config_backup']).exists())
        again = self.launcher().run('import')
        self.assertEqual(result['snapshot'], again['snapshot'])
        self.assertEqual(len(list((self.root/'outputs/artifacts/news/adapted').iterdir())), 1)

    def test_snapshot_mutation_is_not_silently_reimported(self):
        result = self.launcher().run('import')
        (Path(result['snapshot'])/'articles.jsonl').write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.launcher().run('import')

    def fake_pipeline(self, fail=None):
        pipeline = Mock()
        pipeline.state = {'artifacts': {'NASDAQ': {}}, 'readiness': {'NASDAQ': {'ready': True}}}
        pipeline.root = self.root/'preparation'
        if fail:
            def run(step):
                if step == fail:
                    raise ValueError('Coverage gate failure')
            pipeline.run.side_effect = run
        return pipeline

    def test_all_orders_validation_before_baseline_and_training(self):
        pipeline = self.fake_pipeline()
        with patch('news_fusion.launcher.Pipeline', return_value=pipeline):
            result = self.launcher().run('all')
        self.assertEqual([c.args[0] for c in pipeline.run.call_args_list], list(STEPS))
        self.assertEqual(result['status'], 'completed')
        pipeline.run.assert_any_call('train')

    def test_coverage_failure_never_reaches_baseline_or_training(self):
        pipeline = self.fake_pipeline('validate')
        with patch('news_fusion.launcher.Pipeline', return_value=pipeline):
            with self.assertRaisesRegex(ValueError, 'Coverage'):
                self.launcher().run('all')
        called = [c.args[0] for c in pipeline.run.call_args_list]
        self.assertNotIn('baseline', called)
        self.assertNotIn('train', called)
        pointer = self.root/'outputs/artifacts/news/launch_state.json'
        self.assertEqual(json.loads(pointer.read_text())['last_error']['phase'], 'validate')

    def test_prepare_stops_before_fusion_and_train_stage_runs_only_train(self):
        pipeline = self.fake_pipeline()
        with patch('news_fusion.launcher.Pipeline', return_value=pipeline):
            self.launcher().run('prepare')
        self.assertEqual([c.args[0] for c in pipeline.run.call_args_list], list(PREPARATION_STEPS))
        pipeline.reset_mock()
        with patch('news_fusion.launcher.Pipeline', return_value=pipeline):
            self.launcher().run('train')
        pipeline.run.assert_called_once_with('train')

    def test_status_is_read_only_and_disabled_baseline_does_not_load_news(self):
        before = set(self.root.rglob('*'))
        self.launcher().run('status')
        self.assertEqual(before, set(self.root.rglob('*')))
        self.base['llm'] = {'enabled': False, 'config': 'missing.yaml'}
        self.write_configs()
        with patch('news_fusion.configuration.load_llm_config', side_effect=AssertionError('No news')), \
                patch('news_fusion.workflow.run_workflow', return_value={'status': 'completed'}) as run:
            self.launcher().run('all')
        run.assert_called_once_with(self.base_path, self.root)

    def test_manual_import_option_does_not_train_on_unregistered_data(self):
        self.config['pipeline']['auto_import_retrace'] = False
        self.write_configs()
        with patch('news_fusion.launcher.Pipeline') as pipeline:
            with self.assertRaisesRegex(ValueError, '先运行'):
                self.launcher().run('all')
        pipeline.assert_not_called()


class PerMarketBaselineTests(unittest.TestCase):
    def test_null_market_reference_trains_new_e0_instead_of_reusing_other_market(self):
        pipeline = object.__new__(Pipeline)
        pipeline.config = {'baseline_checkpoint': 'NASDAQ-only.pt', 'pipeline': {'baseline_checkpoints': {'NYSE': None}}}
        pipeline.state = {'artifacts': {'NYSE': {}}}
        pipeline.require = Mock()
        pipeline.save = Mock()
        pipeline.baseline_config = lambda name: {'synthetic': name}
        pipeline.project = Path('.')
        with tempfile.TemporaryDirectory() as temp:
            pipeline.root = Path(temp)
            with patch('news_fusion.workflow.run_workflow', return_value={'best_checkpoint': 'NYSE-best.pt'}) as run:
                pipeline.step_baseline()
        run.assert_called_once()
        self.assertEqual(pipeline.state['artifacts']['NYSE']['baseline_checkpoint'], 'NYSE-best.pt')


if __name__ == '__main__':
    unittest.main()
