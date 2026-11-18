import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from news_fusion.pipeline import (DEFAULTS, Pipeline, validate_baseline_source,
                                 validate_news_protocol, validate_settings)


class OriginalNewsPreparationTests(unittest.TestCase):
    def test_strict_remains_default_and_unknown_protocol_rejected(self):
        settings = copy.deepcopy(DEFAULTS)
        self.assertEqual(settings['data_protocol'], 'strict')
        validate_settings(settings)
        settings['data_protocol'] = 'anything'
        with self.assertRaises(ValueError):
            validate_settings(settings)

    def test_original_is_explicit_retrospective_only(self):
        config = {'pipeline': {'data_protocol': 'original'}, 'panel': {'mode': 'retrospective'}}
        validate_news_protocol(config)
        for mode in ('observed_pit', 'simulated_pit'):
            config['panel']['mode'] = mode
            with self.assertRaisesRegex(ValueError, 'retrospective'):
                validate_news_protocol(config)

    def test_strict_retains_its_time_modes(self):
        for mode in ('retrospective', 'observed_pit', 'simulated_pit'):
            validate_news_protocol({'pipeline': {'data_protocol': 'strict'}, 'panel': {'mode': mode}})

    def test_baseline_config_uses_selected_protocol(self):
        for protocol in ('original', 'strict'):
            pipeline = object.__new__(Pipeline)
            pipeline.base = {'llm': {'enabled': True}, 'data': {'protocol': 'original'},
                             'output': {'directory': 'old'}}
            pipeline.settings = {'data_protocol': protocol}
            pipeline.project = Path('.')
            pipeline.base_path = Path('config.yaml')
            with patch('news_fusion.configuration.compile_config', side_effect=lambda p,f,b,n:b):
                resolved = pipeline.baseline_config('NASDAQ')
            self.assertEqual(resolved['data']['protocol'], protocol)
            self.assertFalse(resolved['llm']['enabled'])
            self.assertTrue(pipeline.base['llm']['enabled'])

    def parent(self):
        training = dict(epochs=100, learning_rate=.001, alpha=.1, numpy_seed=123456789,
                        torch_seed=12345678, selection_metric='val_loss', test_policy='per_epoch',
                        metric_protocol='stockmixer_original')
        config = {'data': {'preprocessing': 'original'}, 'training': training}
        return {'mode': 'baseline', 'smoke_only': False, 'protocol': 'original_reproduction',
                'config': config}, copy.deepcopy(config)

    @patch('news_fusion.experiment.validate_checkpoint')
    def test_baseline_rejects_parameter_drift_and_mixed_protocol(self, checked):
        parent, expected = self.parent()
        validate_baseline_source(parent, object(), expected)
        for field, value in [('epochs', 1), ('alpha', 1.0), ('learning_rate', .02), ('torch_seed', 99)]:
            changed = copy.deepcopy(expected)
            changed['training'][field] = value
            with self.assertRaisesRegex(ValueError, field):
                validate_baseline_source(parent, object(), changed)
        expected['data']['preprocessing'] = 'train_max_ratio'
        with self.assertRaisesRegex(ValueError, 'preprocessing'):
            validate_baseline_source(parent, object(), expected)

    @patch('news_fusion.experiment.validate_checkpoint')
    def test_smoke_or_fusion_cannot_be_imported_as_baseline(self, checked):
        for field, value in [('smoke_only', True), ('mode', 'fusion')]:
            parent, expected = self.parent()
            parent[field] = value
            with self.assertRaisesRegex(ValueError, 'non-smoke E0'):
                validate_baseline_source(parent, object(), expected)

    def test_importing_baseline_never_calls_training(self):
        parent, expected = self.parent()
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'best_model.pt'
            source.write_bytes(b'fixture')
            pipeline = object.__new__(Pipeline)
            pipeline.project = Path(temp)
            pipeline.config = {'baseline_checkpoint': str(source)}
            pipeline.state = {'artifacts': {'NASDAQ': {}}}
            pipeline.require = lambda step: None
            pipeline.resolve = lambda value: Path(value)
            pipeline.save = lambda: None
            expected['workflow'] = {'prepared_data': {'NASDAQ': str(Path(temp)/'data')}}
            pipeline.baseline_config = lambda name: expected
            with patch('news_fusion.experiment.load_run_checkpoint', return_value=parent), \
                 patch('news_fusion.protocol.load_market', return_value=object()), \
                 patch('news_fusion.experiment.validate_checkpoint'), \
                 patch('news_fusion.workflow.run_workflow', side_effect=AssertionError('Training must not start')) as train:
                pipeline.step_baseline()
            train.assert_not_called()
            self.assertEqual(pipeline.state['artifacts']['NASDAQ']['baseline_checkpoint'], str(source))


if __name__ == '__main__':
    unittest.main()
