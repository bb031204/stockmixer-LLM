import contextlib
import gc
import copy
import csv
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from news_fusion import experiment, workflow
from news_fusion.outputs import verify_bundle, bundle_files
from news_fusion.io import digest_file, write_json
from news_fusion.protocol import validate_config
from tests_news.baseline_fixture import configuration
from tests_news.test_review_regressions import make_market


@contextlib.contextmanager
def temporary_project():
    # Mock call records can form cycles retaining Windows memory-mapped inputs.
    with tempfile.TemporaryDirectory() as temporary:
        try:
            yield temporary
        finally:
            gc.collect()


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value=torch.nn.Parameter(torch.tensor(1.))

    def forward(self,x):
        return x[:,-1,-1:]*self.value


def setup_project(root):
    _,config=configuration(dataset="NYSE")
    config=copy.deepcopy(config)
    config['data']['preprocessing']='train_max_ratio'
    config['training']['epochs']=2
    config['training']['device']='cpu'
    config['workflow']['smoke']=True
    config['workflow']['prepare_if_missing']=False
    config['workflow']['output_directory']='outputs/20261116_120000_NYSE_E0_smoke_unit_run'
    config['workflow']['prepared_data']['NYSE']='outputs/data'
    (root/'configs').mkdir()
    (root/'outputs').mkdir()
    make_market(root/'outputs/data')
    return config


def save_config(root,config):
    path=root/'configs/stockmixer.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    return path


class WorkflowTests(unittest.TestCase):
    def test_no_argument_entry_uses_workflow_defaults(self):
        with patch.object(workflow,'run_workflow') as run:
            workflow.main([])
            run.assert_called_once_with(None)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            workflow.main(['train','--epochs','1'])

    def test_historical_checkpoint_config_without_workflow_remains_valid(self):
        _,config=configuration(dataset="NYSE")
        del config['workflow']
        validate_config(config)

    def test_invalid_workflow_and_unquoted_dates_are_rejected(self):
        _,config=configuration(dataset="NYSE")
        changes=[('smoke','false'),('prepare_if_missing',1),('mode','other'),
                 ('output_directory',''),('prepared_data',{'NYSE':'a'}),('unknown',True)]
        for key,value in changes:
            with self.subTest(key=key):
                bad=copy.deepcopy(config);bad['workflow'][key]=value
                with self.assertRaises(ValueError):
                    validate_config(bad)
        for value in (date(2020,1,1),'2020-02-30','20200101'):
            bad=copy.deepcopy(config);bad['workflow']['prediction']['as_of']=value
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_fusion_requires_real_inputs_and_baseline_mode_rejects_unused_panel(self):
        _,config=configuration(dataset="NYSE")
        config['workflow']['mode']='fusion'
        with self.assertRaisesRegex(ValueError,'Fusion mode requires'):
            validate_config(config)
        config['workflow']['mode']='baseline';config['news']['panel']='unused'
        with self.assertRaisesRegex(ValueError,'Baseline mode'):
            validate_config(config)

    def test_one_call_trains_tests_each_epoch_and_predicts_from_selected_best(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root);save_config(root,config)
            original=experiment.evaluate;phases=[]
            original_predict=experiment.predict
            preserved=[]
            def track_predict(checkpoint,output,*args,**kwargs):
                keep=['checkpoints/best_model.pt','checkpoints/last_model.pt','configs/config.json','tables/metrics.json',
                      'tables/history.json','_artifact/manifest.json','logs/train.log']
                before={name:digest_file(Path(output)/name) for name in keep}
                answer=original_predict(checkpoint,output,*args,**kwargs)
                self.assertEqual(before,{name:digest_file(Path(output)/name) for name in keep})
                preserved.append(True)
                return answer
            def track(model,data,targets,*args):
                phases.append('test' if targets[0]==data.axis.valid_end else 'validation')
                return original(model,data,targets,*args)
            with patch.object(experiment,'make_model',side_effect=lambda *a:ToyModel()),\
                 patch.object(experiment,'evaluate',side_effect=track),\
                 patch.object(experiment,'predict',side_effect=track_predict),contextlib.redirect_stdout(io.StringIO()):
                result=workflow.run_workflow(project=root)
            self.assertEqual(phases,['validation','test','validation','test'])
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['training_completed'] and result['prediction_completed'])
            self.assertEqual(result['prediction_rows'],12)
            verify_bundle(root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run')
            bundle=verify_bundle(root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run')
            self.assertIn('prediction_artifact_manifest_sha256',bundle)
            self.assertEqual(result['training_directory'],result['prediction_directory'])
            self.assertFalse((root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run_prediction').exists())
            prediction=json.loads((root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run/_prediction_artifact/manifest.json').read_text())
            self.assertEqual(prediction['checkpoint_sha256'],result['best_checkpoint_sha256'])
            self.assertEqual(prediction['checkpoint_code_compatibility'],'exact_code')
            source=json.loads((root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run/configs/config.json').read_text())
            self.assertEqual(source,config)
            with (root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run/predictions/predictions.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),12)
            self.assertEqual(preserved,[True])
            run=Path(result['training_directory'])
            self.assertEqual(verify_bundle(run)['kind'],'training_bundle')
            with self.assertRaises(FileExistsError):
                original_predict(result['best_checkpoint'],run,attach_to_training=True,device='cpu')
            # Rehashing the displayed forecast cannot replace its immutable original.
            with (run/'predictions/predictions.csv').open('a',encoding='utf-8') as stream:
                stream.write('tampered\n')
            bundle=json.loads((run/'manifest.json').read_text())
            bundle['files']=bundle_files(run);write_json(run/'manifest.json',bundle)
            with self.assertRaisesRegex(ValueError,'immutable original'):
                verify_bundle(run)
            # Completed reports point to the exact selected best, never last_model.pt.
            self.assertTrue(result['best_checkpoint'].endswith('checkpoints\\best_model.pt') or result['best_checkpoint'].endswith('checkpoints/best_model.pt'))

    def test_bad_prediction_cutoff_fails_before_training(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root)
            config['workflow']['prediction']['as_of']='2020-01-27';save_config(root,config)
            with patch.object(experiment,'train') as train,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'training/selection'):
                    workflow.run_workflow(project=root)
                train.assert_not_called()
            state=json.loads((root/'outputs/reports/20261116_120000_NYSE_E0_smoke_unit_run_workflow/workflow.json').read_text())
            self.assertEqual(state['status'],'interrupted_or_failed')
            self.assertFalse((root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run').exists())

    def test_failed_training_never_starts_prediction(self):
        with temporary_project() as temporary:
            root=Path(temporary);save_config(root,setup_project(root))
            with patch.object(experiment,'train',side_effect=RuntimeError('simulated training failure')),\
                 patch.object(experiment,'predict') as predict,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'simulated'):
                    workflow.run_workflow(project=root)
                predict.assert_not_called()
            state=json.loads((root/'outputs/reports/20261116_120000_NYSE_E0_smoke_unit_run_workflow/workflow.json').read_text())
            self.assertFalse(state['training_completed'])
            self.assertEqual(state['phase'],'training')

    def test_prediction_failure_retains_completed_training_and_diagnostic_report(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root);save_config(root,config)
            with patch.object(experiment,'make_model',side_effect=lambda *a:ToyModel()),\
                 patch.object(experiment,'predict',side_effect=ValueError('simulated prediction failure')),\
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'simulated'):
                    workflow.run_workflow(project=root)
            verify_bundle(root/'outputs/20261116_120000_NYSE_E0_smoke_unit_run')
            state=json.loads((root/'outputs/reports/20261116_120000_NYSE_E0_smoke_unit_run_workflow/workflow.json').read_text())
            self.assertTrue(state['training_completed'])
            self.assertFalse(state['prediction_completed'])
            self.assertEqual(state['phase'],'predicting')

    def test_config_disabled_prediction_is_explicit_in_status(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root)
            config['workflow']['prediction']['enabled']=False;save_config(root,config)
            with patch.object(experiment,'make_model',side_effect=lambda *a:ToyModel()),\
                 patch.object(experiment,'predict') as predict,contextlib.redirect_stdout(io.StringIO()):
                result=workflow.run_workflow(project=root)
                predict.assert_not_called()
            self.assertEqual(result['status'],'completed')
            self.assertTrue(result['training_completed'])
            self.assertFalse(result['prediction_enabled'])
            self.assertIsNone(result['prediction_directory'])

    def test_missing_prepared_data_does_not_overwrite_an_existing_incomplete_directory(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root)
            config['workflow']['prepared_data']['NYSE']='outputs/incomplete'
            config['workflow']['prepare_if_missing']=True
            (root/'outputs/incomplete').mkdir();save_config(root,config)
            with patch.object(workflow,'prepare_market') as prepare,\
                 patch.object(experiment,'train') as train,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    workflow.run_workflow(project=root)
                prepare.assert_not_called();train.assert_not_called()


if __name__=='__main__':
    unittest.main()

