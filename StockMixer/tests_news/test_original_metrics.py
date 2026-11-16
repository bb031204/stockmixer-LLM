import contextlib
import copy
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from news_fusion import experiment as e
from news_fusion.protocol import configuration,validate_config
from tests_news.test_workflow import temporary_project,setup_project,ToyModel


class OfficialMetricTests(unittest.TestCase):
    def test_official_masked_ic_includes_zeroed_invalid_rows(self):
        prediction=np.array([[5.,2.],[3.,4.],[2.,3.],[4.,8.]])
        truth=np.array([[2.,4.],[3.,2.],[1.,1.],[4.,3.]])
        mask=np.array([[1,1],[1,1],[1,1],[0,0]])
        original=e.original_metrics(prediction,truth,mask)
        daily=[np.corrcoef(prediction[:,i]*mask[:,i],truth[:,i]*mask[:,i])[0,1] for i in range(2)]
        self.assertAlmostEqual(original['IC'],np.mean(daily))
        self.assertAlmostEqual(original['RIC'],np.mean(daily)/np.std(daily))
        self.assertNotAlmostEqual(original['IC'],e.metrics(prediction,truth,mask)['IC'])

    def test_all_five_values_match_original_source_with_masks_and_ties(self):
        source=Path('D:/finance/baseline/StockMixer-master/src/evaluator.py')
        spec=importlib.util.spec_from_file_location('official_reference_evaluator',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.RandomState(31)
        prediction=np.round(rng.randn(24,7),1)
        truth=rng.randn(24,7)*.01
        mask=(rng.rand(24,7)>.2).astype(float)
        expected=module.evaluate(prediction,truth,mask)
        actual=e.original_metrics(prediction,truth,mask)
        self.assertEqual(set(actual),{'mse','IC','RIC','prec_10','sharpe5'})
        for key in expected:
            self.assertEqual(actual[key],expected[key])
        vendored=Path(e.__file__).parent/'original_evaluator.py'
        self.assertEqual(source.read_bytes(),vendored.read_bytes())

    def test_decimal_console_fields_and_loss_decomposition(self):
        perf={'loss':.12,'mse_loss':.1,'rank_loss':.2,'mse':.1,'IC':.02,'RIC':.3,'prec_10':.5,'sharpe5':1.2}
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            e.print_original_epoch((.12,.1,.2),perf,perf)
        text=output.getvalue()
        for label in ('Train : loss:0.12  =  0.1 + alpha*0.2','Valid : loss:','Test: loss:',
                      'Valid performance:\n','Test performance:\n','mse:0.1, IC:.02, RIC:.3, prec@10:.5, SR:1.2'):
            self.assertIn(label,text)
        self.assertNotIn('RankIC',text)
        self.assertNotIn('val_ICIR',text)

    def test_undefined_metrics_are_explicit_json_null(self):
        result=e.original_metrics(np.ones((12,3)),np.ones((12,3)),np.ones((12,3)))
        self.assertIsNone(result['IC'])
        self.assertIsNone(result['RIC'])
        json.dumps(result,allow_nan=False)


class EpochEvaluationTests(unittest.TestCase):
    def test_high_test_metric_cannot_override_validation_selected_epoch_or_arrays(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root)
            def perf(loss,ic):
                return {'loss':loss,'mse_loss':loss,'rank_loss':0.,'mse':loss,'IC':ic,'RIC':ic,'prec_10':.5,'sharpe5':ic}
            arr1=(np.ones((12,2)),np.zeros((12,2)),np.ones((12,2)))
            arr2=(np.full((12,2),2.),np.zeros((12,2)),np.ones((12,2)))
            evaluated=[(perf(1.,.1),arr1),(perf(4.,-.9),arr1),(perf(2.,.2),arr2),(perf(3.,.99),arr2)]
            with patch.object(e,'make_model',side_effect=lambda *a:ToyModel()),patch.object(e,'evaluate',side_effect=evaluated),contextlib.redirect_stdout(io.StringIO()):
                manifest=e.train(root,config,root/'outputs/data',root/'outputs/selected',epochs=2,smoke=True)
            self.assertEqual(manifest['best_epoch'],1)
            self.assertEqual(manifest['test']['IC'],-.9)
            self.assertEqual(manifest['test_evaluations'],2)
            self.assertEqual(manifest['final_test_evaluations'],0)
            selected=np.load(root/'outputs/selected/predictions/test_predictions.npz')
            self.assertTrue(np.array_equal(selected['prediction'],arr1[0]));selected.close()
            checkpoint=e.load_run_checkpoint(root/'outputs/selected/checkpoints/best_model.pt')
            self.assertEqual(checkpoint['epoch'],1)
            last=e.load_run_checkpoint(root/'outputs/selected/checkpoints/last_model.pt')
            self.assertEqual(last['epoch'],2)
            history=json.loads((root/'outputs/selected/tables/history.json').read_text())
            self.assertEqual([row['test_ic'] for row in history],[-.9,.99])
            self.assertTrue((root/'outputs/selected/figures/sr_curve.png').exists())

    def test_monitoring_test_does_not_change_trained_weights(self):
        with temporary_project() as temporary:
            root=Path(temporary);config=setup_project(root)
            with patch.object(e,'make_model',side_effect=lambda *a:ToyModel()),contextlib.redirect_stdout(io.StringIO()):
                e.train(root,config,root/'outputs/data',root/'outputs/per_epoch',epochs=2,smoke=True)
                legacy=copy.deepcopy(config);legacy['training']['test_policy']='final_best_once'
                manifest=e.train(root,legacy,root/'outputs/data',root/'outputs/final_only',epochs=2,smoke=True)
            self.assertEqual(manifest['final_test_evaluations'],1)
            self.assertEqual(manifest['test_evaluations'],1)
            first=e.load_run_checkpoint(root/'outputs/per_epoch/checkpoints/last_model.pt')
            second=e.load_run_checkpoint(root/'outputs/final_only/checkpoints/last_model.pt')
            self.assertTrue(all(torch.equal(first['model_state_dict'][key],second['model_state_dict'][key]) for key in first['model_state_dict']))

    def test_unknown_metric_and_test_policy_rejected(self):
        _,config=configuration()
        for key in ('metric_protocol','test_policy'):
            bad=copy.deepcopy(config);bad['training'][key]='unknown'
            with self.assertRaises(ValueError):
                validate_config(bad)


if __name__=='__main__':
    unittest.main()
