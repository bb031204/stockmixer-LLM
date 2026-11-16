import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from news_fusion import experiment
from news_fusion.io import write_json
from news_fusion.panel import relationship_valid
from news_fusion.dataset import NewsPanel
from news_fusion.protocol import MarketData, TargetSplit, configuration, positive_train_scale
from news_fusion.time_axis import Axis, Schedule


class StrictProtocolTests(unittest.TestCase):
    def test_all_market_target_boundaries(self):
        for days, train, valid, count in ((1245, 756, 1008, 740), (1611, 1006, 1259, 990)):
            split = TargetSplit(days, train, valid)
            self.assertEqual(len(split.targets("train")), count)
            self.assertEqual(split.offsets("train")[-1], count-1)
            all_sets = [set(split.targets(p)) for p in ("train", "validation", "test")]
            self.assertFalse(all_sets[0] & all_sets[1] or all_sets[1] & all_sets[2])
            self.assertEqual(split.targets("train")[-1], train-1)
            self.assertEqual(split.targets("validation")[0], train)
            self.assertEqual(split.targets("test")[0], valid)

    def test_train_scale_independent_of_future_and_old_global_divisor(self):
        raw = np.arange(1, 1+2*8*5, dtype=float).reshape(2, 8, 5)
        missing = np.zeros_like(raw, dtype=bool)
        a, _ = positive_train_scale(raw, missing, 4, shared_close=True)
        future = raw.copy(); future[:, 4:] *= 1000
        b, _ = positive_train_scale(future, missing, 4, shared_close=True)
        np.testing.assert_array_equal(a[:, :4], b[:, :4])
        prior_global_scale = np.array([3., 1000.])[:, None, None]
        rebased, _ = positive_train_scale(raw/prior_global_scale, missing, 4, shared_close=True)
        np.testing.assert_array_equal(a, rebased)
        np.testing.assert_allclose(a[:, 1:, -1]/a[:, :-1, -1], raw[:, 1:, -1]/raw[:, :-1, -1], rtol=2e-7)

    def test_missing_sentinels_do_not_fit_scale_and_remain_fixed(self):
        raw = np.ones((1, 8, 5)); raw[:, 3] = 2
        missing = np.zeros_like(raw, dtype=bool)
        raw[:, 0] = -1234; missing[:, 0] = True
        scaled, scale = positive_train_scale(raw, missing, 4, shared_close=True)
        np.testing.assert_array_equal(scale, np.full((1, 5), 2.))
        np.testing.assert_array_equal(scaled[:, 0], np.full((1, 5), np.float32(1.1)))
        with self.assertRaises(ValueError):
            positive_train_scale(raw, np.ones_like(missing), 4)

    def test_strict_config_rejects_unsafe_scaler_and_model_migration(self):
        project, config = configuration()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"bad.yaml"
            import yaml
            config["data"]["preprocessing"] = "standard"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                configuration(path)

    def test_metrics_distinguish_rank_ic_from_original_ric(self):
        p = np.array([[1, 1], [2, 2], [100, 3]], dtype=float)
        y = np.array([[1, 1], [2, 3], [3, 2]], dtype=float)
        result = experiment.metrics(p, y, np.ones_like(p))
        self.assertNotAlmostEqual(result["IC"], result["RankIC"])
        self.assertEqual(result["RIC"], result["ICIR"])

    def test_nonnull_metrics_roundtrip_with_pytorch_safe_loader(self):
        rng=np.random.RandomState(42)
        p=rng.randn(12,4); y=rng.randn(12,4)*.01
        result=experiment.metrics(p,y,np.ones_like(p))
        self.assertIsInstance(result["sharpe5"],float)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"metrics.pt"
            torch.save(result,path)
            loaded=torch.load(path,weights_only=True)
            self.assertEqual(loaded,result)

    def test_cutoff_equality_excluded_in_every_mode(self):
        axis = Axis("TEST", ["A"], ["2020-03-05", "2020-03-06", "2020-03-09"], 1, 2, {})
        schedule = Schedule(axis, [{"date": d, "decision_cutoff_at": d+"T21:00:00Z"} for d in axis.dates])
        article = {"published_at": "2020-03-05T21:00:00Z"}
        self.assertEqual(schedule.first_session(article, "retrospective"), 1)
        self.assertEqual(schedule.first_session(article, "simulated_pit", 0), 1)

    def test_native_date_mapping_and_exclusive_end(self):
        link = {"verification_status": "verified", "mapping_source": "test",
                "valid_from": date(2020, 1, 1), "valid_to": date(2021, 1, 1)}
        self.assertTrue(relationship_valid(link, "2020-03-05", "2020-03-05", None, "retrospective"))
        self.assertFalse(relationship_valid(link, "2021-01-01", "2020-03-05", None, "retrospective"))

    def test_latest_decision_needs_no_future_label(self):
        market = MarketData.__new__(MarketData)
        market.axis = Axis("TEST", ["A"], [str(i) for i in range(20)], 17, 19, {})
        market.split = TargetSplit(20, 17, 19)
        market.arrays = {"eod": np.ones((1,20,5)), "price": np.ones((1,20)),
                         "mask": np.ones((1,20)), "returns": np.zeros((1,20))}
        self.assertEqual(market.decision(19)[0].shape, (1,16,5))
        with self.assertRaises(IndexError):
            market.window(20)


class TrainingIsolationTests(unittest.TestCase):
    def test_interruption_retains_partial_weights_but_is_not_a_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"run"
            with self.assertRaises(KeyboardInterrupt):
                with experiment.training_directory(output) as directory:
                    torch.save({"partial":True},directory/"last_model.pt")
                    raise KeyboardInterrupt()
            self.assertTrue((output/"last_model.pt").exists())
            status=json.loads((output/"run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"],"interrupted_or_failed")
            with self.assertRaises(OSError):
                experiment.load_run_checkpoint(output/"last_model.pt")

    def test_test_evaluated_each_epoch_after_validation(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.value = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, x):
                return x[:, -1, -1:] * self.value
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/"manifest.json", {"fixture": True})
            dates = __import__("pandas").date_range("2020-01-01", periods=22).strftime("%Y-%m-%d").tolist()
            market = MarketData.__new__(MarketData)
            market.directory = root
            market.axis = Axis("NYSE", ["NYSE:S"+str(i) for i in range(12)], dates, 18, 20, {"market_num":25})
            market.split = TargetSplit(22,18,20)
            market.arrays = {"eod": np.ones((12,22,5), np.float32), "price": np.ones((12,22), np.float32),
                "returns": np.tile(np.linspace(0,.01,12,dtype=np.float32)[:,None], (1,22)), "mask": np.ones((12,22), np.float32)}
            _, config = configuration(dataset="NYSE")
            config = copy.deepcopy(config); config["training"]["device"] = "cpu"; config["data"]["preprocessing"]="train_max_ratio"
            original_evaluate, phases = experiment.evaluate, []
            def tracked(model, data, targets, *args):
                phases.append("test" if targets[0] == 20 else "validation")
                return original_evaluate(model, data, targets, *args)
            with patch.object(experiment, "MarketData", return_value=market), \
                 patch.object(experiment, "make_model", side_effect=lambda *a: ToyModel()), \
                 patch.object(experiment, "evaluate", side_effect=tracked):
                manifest = experiment.train(root, config, root, root/"run", epochs=2, smoke=True)
                self.assertEqual(phases, ["validation", "test", "validation", "test"])
                self.assertEqual(manifest["final_test_evaluations"], 0)
                self.assertEqual(manifest["test_evaluations"], 2)
                checkpoint = experiment.load_run_checkpoint(root/"run/checkpoints/best_model.pt")
                self.assertTrue(checkpoint["smoke_only"])
                self.assertEqual(checkpoint["epoch"], manifest["best_epoch"])
                with self.assertRaisesRegex(ValueError, "training/selection"):
                    experiment.predict(root/"run/checkpoints/best_model.pt", root/"bad-pred", dates[17])


class PanelIntegrationTests(unittest.TestCase):
    def test_float_embedding_indices_and_split_reuse_guard(self):
        # Parquet nullable integer columns can roundtrip as 0.0, 1.0, None.
        from tests_news.test_pipeline import make_axis, article, FakeBackend, company
        from news_fusion.encoder import encode_articles
        from news_fusion import panel as panel_module
        from news_fusion.io import read_records
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            axis, schedule=make_axis(root)
            encode_articles([article()], FakeBackend(), root/"features", root/"cache")
            feature_manifest=json.loads((root/"features/manifest.json").read_text(encoding="utf-8"))
            rows=read_records(root/"features"/feature_manifest["record_table"])
            rows[0]["embedding_row"]=0.0
            with patch.object(panel_module,"read_records",return_value=rows):
                records, vectors, _=panel_module.load_features(root/"features")
                self.assertIsInstance(records[0]["embedding_row"],int)
                del vectors
            panel_module.build_panel(axis,schedule,root/"features",[company()],[],root/"panel",projection="random_projection",decision_lookback=2)
            panel=NewsPanel(root/"panel")
            panel.assert_experiment(axis,"retrospective",lookback=2)
            self.assertEqual(panel.decision(7)["decision_idx"],7)
            changed=copy.copy(axis); changed.train_end=3
            with self.assertRaisesRegex(ValueError,"split"):
                panel.assert_experiment(changed,"retrospective",lookback=2)
            with self.assertRaisesRegex(ValueError,"time mode"):
                panel.assert_experiment(axis,"observed_pit",lookback=2)
            del panel

    def test_unverified_market_panel_and_fake_encoder_not_trainable(self):
        from tests_news.test_pipeline import make_axis, article, FakeBackend
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        import types
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); axis,schedule=make_axis(root)
            encode_articles([article(scopes=["market"])],FakeBackend(),root/"features",root/"cache")
            build_panel(axis,schedule,root/"features",[],[],root/"panel",projection="random_projection",decision_lookback=2)
            data=types.SimpleNamespace(axis=axis,split=TargetSplit(8,4,6,lookback=2))
            _,config=configuration()
            with self.assertRaisesRegex(ValueError,"real, pinned FinBERT"):
                experiment.load_panel(root/"panel",data,config)


if __name__ == "__main__":
    unittest.main()
