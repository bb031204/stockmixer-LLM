"""Regression cases for contract, time boundary, CLI and durable checkpoint failures."""
import contextlib
import copy
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import torch

from news_fusion import experiment
from news_fusion.io import digest, digest_file, file_manifest, verify_files, write_json
from news_fusion.protocol import MarketData, TargetSplit, backbone_contract, positive_train_scale, validate_config
from tests_news.baseline_fixture import configuration
from news_fusion.time_axis import Axis, Schedule


def make_market(root):
    root.mkdir()
    n,t,end,valid=12,22,18,20
    dates=pd.bdate_range("2020-01-02",periods=t).strftime("%Y-%m-%d").tolist()
    raw_axis={"dataset":"NYSE","stock_uids":["NYSE:S"+str(i) for i in range(n)],"trading_dates":dates,
        "train_end_idx":end,"valid_end_idx":valid,"market_num":25,"source_sha256":{}}
    write_json(root/"axis.json",raw_axis)
    axis=Axis.load(root/"axis.json")
    close=np.arange(1,n+1)[:,None]*(1+np.arange(t)[None,:]*.001)
    raw=np.repeat(close[:,:,None],5,axis=2)
    eod,scale=positive_train_scale(raw,np.zeros_like(raw,dtype=bool),end,shared_close=True)
    returns=np.zeros((n,t),dtype=np.float32)
    returns[:,1:]=(close[:,1:]-close[:,:-1])/close[:,:-1]
    for name,array in (("eod",eod),("price",eod[:,:,-1]),("returns",returns),("mask",np.ones((n,t),np.float32)),("scale",scale)):
        np.save(root/(name+".npy"),array,allow_pickle=False)
    write_json(root/"source_hashes.json",{})
    with (root/"stock_identity.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["dataset","array_index","stock_uid","ticker"])
        writer.writeheader()
        writer.writerows({"dataset":"NYSE","array_index":i,"stock_uid":uid,"ticker":"S"+str(i)} for i,uid in enumerate(axis.stock_uids))
    manifest={"schema_version":"stockmixer_strict_v1","kind":"strict_market_data","dataset":"NYSE",
        "stock_axis_hash":axis.stock_hash,"date_axis_hash":axis.date_hash,"backbone":backbone_contract(),
        "train_end_idx":end,"valid_end_idx":valid,"lookback":16,"steps":1,"shape":[n,t,5],
        "split":TargetSplit(t,end,valid).describe(dates),"preprocessing":{"mode":"train_max_ratio","fit_end_exclusive":end},
        "files":file_manifest(root)}
    write_json(root/"manifest.json",manifest)
    return root


class ContractRegressionTests(unittest.TestCase):
    def test_nonfinite_unknown_and_wrong_type_config_rejected_before_training(self):
        _,config=configuration(dataset="NYSE")
        changes=[("training","alpha",float("nan")),("training","learning_rate",float("inf")),
            ("training","epochs",1.5),("training","epochs",True),("training","numpy_seed",-1),
            ("model","scale_argument",1),("fusion","enabled_channels",["market","market"]),
            ("fusion","dropout",float("nan")),("training","weight_decay",.01)]
        for section,key,value in changes:
            with self.subTest(key=key,value=value):
                bad=copy.deepcopy(config); bad[section][key]=value
                with self.assertRaises(ValueError):
                    validate_config(bad)

    def test_fractional_split_and_nonfinite_latency_rejected(self):
        with self.assertRaises(ValueError):
            TargetSplit(30,18.5,25)
        axis=Axis("TEST",["A"],["2020-01-02","2020-01-03","2020-01-06"],1,2,{})
        schedule=Schedule(axis,[{"date":d,"decision_cutoff_at":d+"T21:00:00Z"} for d in axis.dates])
        for latency in (float("nan"),float("inf"),-1,True):
            with self.assertRaises(ValueError):
                schedule.first_session({"published_at":"2020-01-02T18:00:00Z"},"simulated_pit",latency)

    def test_manifest_cannot_omit_required_data_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=make_market(Path(temporary)/"data")
            manifest=json.loads((root/"manifest.json").read_text())
            del manifest["files"]["eod.npy"]
            write_json(root/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"required"):
                MarketData(root)

    def test_hash_consistency_does_not_hide_wrong_prices_labels_scales_or_identities(self):
        for problem in ("price","returns","scale","identity","fit_boundary"):
            with self.subTest(problem=problem),tempfile.TemporaryDirectory() as temporary:
                root=make_market(Path(temporary)/"data")
                data=MarketData(root); del data
                manifest=json.loads((root/"manifest.json").read_text())
                if problem in ("price","returns","scale"):
                    path=root/(problem+".npy")
                    array=np.load(path)
                    if problem=="scale":
                        array[0,0]=-1
                    else:
                        array[0,5]+=.25
                    np.save(path,array,allow_pickle=False)
                elif problem=="identity":
                    path=root/"stock_identity.csv"
                    text=path.read_text(); path.write_text(text.replace("NYSE:S0","NYSE:WRONG"))
                else:
                    manifest["preprocessing"]["fit_end_exclusive"]=19
                manifest["files"]=file_manifest(root); write_json(root/"manifest.json",manifest)
                with self.assertRaises(ValueError):
                    MarketData(root)

    def test_bad_calendar_date_rejected_and_float_sample_index_not_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=make_market(Path(temporary)/"data")
            data=MarketData(root)
            with self.assertRaises(ValueError):
                data.window(16.9)
            del data
            axis=json.loads((root/"axis.json").read_text())
            axis["trading_dates"][0]="2020-01-00"
            write_json(root/"axis.json",axis)
            with self.assertRaisesRegex(ValueError,"calendar"):
                Axis.load(root/"axis.json")

    def test_metrics_reject_nan_masks_and_invalid_truth(self):
        prediction=np.ones((12,3))
        mask=np.ones_like(prediction); mask[0,0]=float("nan")
        with self.assertRaises(ValueError):
            experiment.metrics(prediction,prediction,mask)
        truth=prediction.copy(); truth[0,0]=float("nan")
        with self.assertRaises(ValueError):
            experiment.metrics(prediction,truth,np.ones_like(prediction))


class PCARegressionTests(unittest.TestCase):
    def test_fit_excludes_early_articles_that_never_reach_a_training_decision(self):
        from tests_news.test_pipeline import article, FakeBackend
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            dates=pd.bdate_range("2020-01-02",periods=35).strftime("%Y-%m-%d").tolist()
            write_json(root/"axis.json",{"dataset":"TEST","stock_uids":["TEST:A","TEST:B"],"trading_dates":dates,
                "train_end_idx":25,"valid_end_idx":30,"source_sha256":{}})
            axis=Axis.load(root/"axis.json")
            schedule=Schedule(axis,[{"date":d,"decision_cutoff_at":d+"T21:00:00Z"} for d in dates])
            articles=[article("early",dates[0]+"T18:00:00Z",["market"])]
            articles += [article("train"+str(i),dates[16]+"T18:00:00Z",["market"]) for i in range(35)]
            articles += [article("validation",dates[25]+"T18:00:00Z",["market"])]
            encode_articles(articles,FakeBackend(),root/"features",root/"cache")
            manifest=build_panel(axis,schedule,root/"features",[],[],root/"panel",projection="pca",decision_lookback=16)
            transform=manifest["transform"]
            self.assertEqual(transform["training_decision_start_inclusive"],15)
            self.assertEqual(transform["training_decision_end_exclusive"],24)
            self.assertEqual(transform["fitted_articles"],35)
            self.assertEqual(transform["fit_source_ids_hash"],digest(sorted("train"+str(i) for i in range(35))))


class StructuredEvidenceTests(unittest.TestCase):
    def test_feature_count_mismatch_is_rejected(self):
        from tests_news.test_pipeline import article, FakeBackend
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import load_features
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            encode_articles([article()],FakeBackend(),root/"features",root/"cache")
            path=root/"features/manifest.json"
            manifest=json.loads(path.read_text()); manifest["successful"]=0
            write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,"counts"):
                load_features(root/"features")
    def test_enrichment_evidence_survives_encoding_without_changing_text(self):
        from tests_news.test_pipeline import article, FakeBackend
        from news_fusion.inputs import normalize_article
        from news_fusion.encoder import encode_articles
        from news_fusion.io import read_records
        base=article()
        enriched=dict(base,entities=[{"name":"Verified issuer","security_id":"source-provided"}],
            events=[{"type":"policy","evidence":"Explicit source statement"}],
            industry_impacts=np.array([{"direction":"conditional_negative","confidence":np.float32(.8)}],dtype=object))
        normalized=normalize_article(enriched)
        self.assertEqual(normalized["input_text_hash"],base["input_text_hash"])
        self.assertEqual(normalized["events"],enriched["events"])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=encode_articles([normalized],FakeBackend(),root/"features",root/"cache")
            row=read_records(root/"features"/manifest["record_table"])[0]
            self.assertEqual(row["entities"],enriched["entities"])
            self.assertEqual(row["industry_impacts"][0]["direction"],"conditional_negative")

    def test_same_id_different_enrichment_is_not_silently_deduplicated(self):
        from tests_news.test_pipeline import article
        from news_fusion.inputs import load_articles
        from news_fusion.io import write_records
        first=article(); first["events"]=[{"type":"earnings"}]
        second=dict(first,events=[{"type":"merger"}])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"articles.jsonl"
            write_records(path,[first,second])
            with self.assertRaisesRegex(ValueError,"Conflicting"):
                load_articles([path])


class CliAndCheckpointRegressionTests(unittest.TestCase):
    def test_predict_config_is_not_silently_ignored(self):
        with contextlib.redirect_stderr(io.StringIO()),patch.object(experiment,"predict") as predict:
            with self.assertRaises(SystemExit):
                experiment.main(["--config","wrong.yaml","predict","--checkpoint","a","--output","b"])
            predict.assert_not_called()

    def test_predict_dataset_is_forwarded_and_equals_baseline_syntax_is_supported(self):
        with contextlib.redirect_stdout(io.StringIO()),patch.object(experiment,"predict",return_value={}) as predict:
            experiment.main(["--dataset","SP500","predict","--checkpoint","a","--output","b"])
            self.assertEqual(predict.call_args.kwargs["dataset"],"SP500")
        with contextlib.redirect_stdout(io.StringIO()),patch.object(experiment,"configuration",configuration),patch.object(experiment,"train",return_value={}) as train:
            experiment.main(["train","--data","a","--output","b","--baseline=c"],required_mode="fusion")
            self.assertEqual(train.call_args.args[-2],Path("c"))

    def test_configured_panel_path_is_resolved_for_fusion(self):
        _,config=configuration(dataset="NYSE"); config["data"]["preprocessing"]="train_max_ratio"; config["training"]["device"]="cpu"; config["news"]["panel"]="panels/reviewed"
        config["workflow"]["mode"]="fusion"; config["workflow"]["baseline_checkpoint"]="base.pt"
        data=types.SimpleNamespace(axis=types.SimpleNamespace(dataset="NYSE"))
        with patch.object(experiment,"MarketData",return_value=data),patch.object(experiment,"load_panel",side_effect=ValueError("stop before training")) as loader:
            with self.assertRaisesRegex(ValueError,"stop before"):
                experiment.train(Path("D:/finance/model/StockMixer"),config,"unused","unused",baseline="base.pt")
            self.assertEqual(loader.call_args.args[0],Path("D:/finance/model/StockMixer/panels/reviewed").resolve())

    def test_atomic_checkpoint_failure_keeps_previous_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"last_model.pt"
            torch.save({"old":torch.ones(1)},path)
            before=path.read_bytes()
            def failed_save(value,stream):
                stream.write(b"partial"); raise OSError("simulated storage failure")
            with patch.object(torch,"save",side_effect=failed_save),self.assertRaises(OSError):
                experiment.atomic_checkpoint({"new":torch.zeros(1)},path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual([p.name for p in path.parent.iterdir()],["last_model.pt"])

    def test_approved_old_baseline_loads_but_arbitrary_code_or_fake_cutoff_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=make_market(Path(temporary)/"data"); data=MarketData(root)
            _,config=configuration(dataset="NYSE")
            checkpoint={"schema_version":"stockmixer_strict_v1","protocol":"strict","mode":"baseline","smoke_only":True,
                "dataset":"NYSE","backbone":backbone_contract(),"code_sha256":dict(experiment.AUDITED_PHASE1_BASELINE),
                "data_manifest_sha256":digest_file(root/"manifest.json"),"stock_axis_hash":data.axis.stock_hash,
                "date_axis_hash":data.axis.date_hash,"train_end_date":data.axis.dates[17],"selection_end_date":data.axis.dates[19],"config":config}
            self.assertEqual(experiment.validate_checkpoint(checkpoint,data),"audited_phase1_baseline")
            bad=copy.deepcopy(checkpoint); bad["selection_end_date"]="2010-01-01"
            with self.assertRaisesRegex(ValueError,"boundary"):
                experiment.validate_checkpoint(bad,data)
            bad=copy.deepcopy(checkpoint); bad["code_sha256"]["experiment.py"]="0"*64
            with self.assertRaisesRegex(ValueError,"code mismatch"):
                experiment.validate_checkpoint(bad,data)
            del data


if __name__=="__main__":
    unittest.main()
