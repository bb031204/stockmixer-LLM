import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from news_fusion import experiment
from news_fusion.io import file_manifest, write_json
from news_fusion.outputs import DIRECTORIES, bundle_files, destination, publish_training, timestamp, verify_bundle, run_destination


def fixture(root):
    archive=root/"_artifact";archive.mkdir(parents=True)
    (archive/"best_model.pt").write_bytes(b"SYNTHETIC_UNIT_TEST_BEST")
    (archive/"last_model.pt").write_bytes(b"SYNTHETIC_UNIT_TEST_LAST")
    write_json(archive/"config.json",{"synthetic_fixture_only":True})
    (archive/"model.txt").write_text("synthetic fixture",encoding="utf-8")
    write_json(archive/"history.json",[{"epoch":1,"train_loss":.2,"val_loss":.1,"val_IC":.03,"val_RankIC":.02}])
    write_json(archive/"sample_ledger.json",[])
    write_json(archive/"run_status.json",{"status":"completed"})
    stats={"loss":.1,"mse":.09,"IC":.03,"RankIC":.02,"ICIR":.1}
    write_json(archive/"test_metrics.json",stats)
    manifest={"kind":"strict_training_run","dataset":"TEST","best_epoch":1,"validation":stats,"test":stats,
              "final_test_evaluations":1,"smoke_only":True,"files":file_manifest(archive)}
    write_json(archive/"manifest.json",manifest)
    return archive


class OutputLayoutTests(unittest.TestCase):
    def test_default_paths_match_stockmixer_outputs_and_taipei_timestamp(self):
        self.assertEqual(timestamp("2026-11-13T05:32:14+00:00"),"20261113_133214")
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary)
            with patch("news_fusion.outputs.timestamp",return_value="20261113_133214"):
                self.assertEqual(destination(project,dataset="NYSE",kind="E0"),project/"outputs/20261113_133214_NYSE_E0")
                (project/"outputs/20261113_133214_NYSE_E0").mkdir(parents=True)
                self.assertEqual(destination(project,dataset="NYSE",kind="E0").name,"20261113_133214_NYSE_E0_01")

    def test_explicit_outputs_are_project_relative_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary)
            self.assertEqual(destination(project,"outputs/a"),project/"outputs/a")
            self.assertEqual(destination(project,"a"),project/"outputs/a")
            with self.assertWarns(FutureWarning):
                self.assertEqual(destination(project,"runs/a"),project/"outputs/a")
            for invalid in ("../elsewhere",project/"data",project/"outputs"):
                with self.assertRaises(ValueError):
                    destination(project,invalid)

    def test_custom_experiment_labels_gain_timestamp_dataset_and_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch("news_fusion.outputs.timestamp",return_value="20261116_120000"):
                self.assertEqual(run_destination(root,"outputs/my_run",dataset="NASDAQ",kind="E0").name,
                    "20261116_120000_NASDAQ_E0_my_run")
                name="20261116_120000_NYSE_fusion_smoke"
                self.assertEqual(run_destination(root,name,dataset="NYSE",kind="fusion_smoke").name,name)
                with self.assertRaisesRegex(ValueError,"dataset differs"):
                    run_destination(root,name,dataset="SP500")
                for bad in ("../elsewhere",root/"data"):
                    with self.assertRaises(ValueError):
                        run_destination(root,bad,dataset="NYSE")

    def test_training_export_has_required_directories_tables_figures_and_exact_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"output"; archive=fixture(root)
            original={p.name:p.read_bytes() for p in archive.iterdir()}
            publish_training(archive,root)
            self.assertTrue(all((root/d).is_dir() for d in DIRECTORIES))
            for name in ("configs/config.yaml","tables/history.csv","tables/metrics.json","figures/loss_curve.png","figures/ic_curve.png","logs/train.log"):
                self.assertTrue((root/name).is_file(),name)
            self.assertEqual((root/"checkpoints/best_model.pt").read_bytes(),original["best_model.pt"])
            self.assertEqual(original,{p.name:p.read_bytes() for p in archive.iterdir()})
            self.assertEqual(verify_bundle(root)["dataset"],"TEST")

    def test_rehashed_display_weight_cannot_replace_immutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/"output"; archive=fixture(root)
            publish_training(archive,root)
            (root/"checkpoints/best_model.pt").write_bytes(b"CHANGED_DISPLAY_WEIGHT")
            manifest=json.loads((root/"manifest.json").read_text())
            manifest["files"]=bundle_files(root);write_json(root/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"immutable original"):
                verify_bundle(root)

    def test_cli_train_and_prediction_do_not_require_an_explicit_output(self):
        with contextlib.redirect_stdout(io.StringIO()),patch.object(experiment,"train",return_value={}) as train:
            experiment.main(["--dataset","NYSE","train","--data","prepared"])
            output=train.call_args.args[3]
            self.assertEqual(output.parent.name,"outputs")
            self.assertIn("NYSE_E0",output.name)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(experiment,"predict",return_value={}) as predict:
            experiment.main(["predict","--checkpoint","a"])
            self.assertIsNone(predict.call_args.args[1])

    def test_audited_formal_baseline_fingerprint_is_explicit_and_distinct(self):
        self.assertEqual(len(experiment.AUDITED_REVIEW_BASELINE),8)
        self.assertEqual(experiment.AUDITED_REVIEW_BASELINE["experiment.py"],"9d4d9767c78b8c039fd15facc2eed9be8b3209cf8c2071c8773e205964427de7")
        self.assertNotEqual(experiment.AUDITED_REVIEW_BASELINE,experiment.AUDITED_PHASE1_BASELINE)


if __name__=="__main__":
    unittest.main()
