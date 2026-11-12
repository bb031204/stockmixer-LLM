import json
import tempfile
import unittest
import types
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from news_fusion.encoder import LocalFinBert, encode_articles, validate_outputs
from news_fusion.inputs import load_articles, normalize_article
from news_fusion.io import digest, read_records, write_json, write_records
from news_fusion.panel import build_panel, compress, load_features, relationship_valid
from news_fusion.time_axis import Axis, Schedule
from news_fusion.dataset import NewsPanel

DATES=["2020-03-05","2020-03-06","2020-03-09","2020-03-10","2020-03-11","2020-03-12","2020-03-13","2020-03-16"]

def make_axis(root):
    raw={"dataset":"TEST","stock_uids":["TEST:A","TEST:B"],"trading_dates":DATES,
         "train_end_idx":4,"valid_end_idx":6,"source_sha256":{}}
    write_json(root/"axis.json",raw)
    axis=Axis.load(root/"axis.json")
    records=[{"date":d,"decision_cutoff_at":d+("T21:00:00Z" if i<2 else "T20:00:00Z")} for i,d in enumerate(DATES)]
    return axis,Schedule(axis,records)

def article(source="one",publication="2020-03-05T18:00:00Z",scopes=None):
    return normalize_article({"source_id":source,"title":"Earnings outlook", "summary":"Revenue improves for business "+source,
        "language":"en","market":"US","impact_scope":scopes or ["company"],"published_at":publication,
        "availability_basis":"observed","first_seen_at":publication,"fetched_at":publication,"feature_available_at":publication})

def company(source="one",**updates):
    value={"source_id":source,"channel":"company","dataset":"TEST","stock_uid":"TEST:A",
        "valid_from":"2010-01-01","valid_to":None,"known_from":"2010-01-01T00:00:00Z",
        "verification_status":"verified","mapping_source":"synthetic fixture"}
    value.update(updates)
    return value

class FakeBackend:
    """Deterministic synthetic features for contract tests; never represents real FinBERT."""
    def __init__(self):
        self.calls=0
        self.metadata={"model_id":"SYNTHETIC_TEST_ONLY","model_revision":"fixture-v1",
            "tokenizer_revision":"fixture-v1","model_available_at":"2010-01-01T00:00:00Z","max_length":512}
    def predict(self,articles):
        self.calls+=1
        vectors=np.stack([np.random.RandomState(int(a['input_text_hash'][:8],16)).normal(size=768) for a in articles]).astype(np.float32)
        return np.tile(np.array([[.6,.1,.3]],np.float32),(len(articles),1)),vectors,np.full(len(articles),20)

class InputEncoderTests(unittest.TestCase):
    def test_local_backend_cls_label_order_and_freezing_contract(self):
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter=torch.nn.Parameter(torch.ones(1))
                self.config=types.SimpleNamespace(hidden_size=768,num_labels=3,id2label={0:"neutral",1:"positive",2:"negative"})
            def forward(self,input_ids,attention_mask,**kwargs):
                n=input_ids.shape[0]
                hidden=torch.full((n,4,768),99.0)
                hidden[:,0]=2.0
                return types.SimpleNamespace(logits=torch.tensor([[1.,3.,0.]]).repeat(n,1),hidden_states=(hidden,))
        class FixtureTokenizer:
            def __call__(self,texts,**kwargs):
                return {"input_ids":torch.ones(len(texts),4,dtype=torch.long),
                        "attention_mask":torch.tensor([[1,1,1,0]]).repeat(len(texts),1)}
        fixture_module=types.SimpleNamespace(__version__="SYNTHETIC_TEST_ONLY",
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:FixtureTokenizer()),
            AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a,**k:FixtureModel()))
        with tempfile.TemporaryDirectory() as temp,patch.dict('sys.modules',transformers=fixture_module):
            directory=Path(temp)
            (directory/"model.safetensors").write_bytes(b'SYNTHETIC_WEIGHTS_NOT_A_MODEL')
            backend=LocalFinBert(directory)
            self.assertFalse(backend.model.training)
            self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
            probs,vectors,counts=backend.predict([article(),article("two")])
            self.assertEqual(probs.shape,(2,3))
            self.assertTrue(np.all(probs.argmax(1)==0))
            np.testing.assert_array_equal(vectors,np.full((2,768),2.0))
            np.testing.assert_array_equal(counts,[3,3])
            np.testing.assert_array_equal(probs,backend.predict([article(),article("two")])[0])

    def test_language_missing_summary_and_chinese(self):
        self.assertEqual(normalize_article({"source_id":"a","summary":"中文新闻内容","language":"en"})["input_status"],"unsupported_language")
        self.assertEqual(normalize_article({"source_id":"b","title":"English only","language":"en"})["input_status"],"empty_text")
        self.assertEqual(normalize_article({"source_id":"c","summary":"English article"})["input_status"],"unsupported_language")
        with self.assertRaises(ValueError):
            normalize_article({"summary":"No identity"})

    def test_title_dedup_and_label_order(self):
        a=normalize_article({"source_id":"a","title":"Revenue rises","summary":"Revenue rises","language":"en"})
        self.assertEqual(a["title"],"")
        self.assertEqual(LocalFinBert.label_order({0:"neutral",1:"negative",2:"positive"}),[2,1,0])
        with self.assertRaises(ValueError):
            LocalFinBert.label_order({0:"LABEL_0",1:"LABEL_1",2:"LABEL_2"})

    def test_bad_model_outputs(self):
        for probs in (np.array([[1,1,1]]),np.array([[np.nan,0,1]]),np.array([[1,-.1,.1]])):
            with self.assertRaises(ValueError):
                validate_outputs(probs,np.zeros((1,768)),1)

    def test_empty_and_conflicting_input_stop(self):
        with self.assertRaises(ValueError):
            load_articles([])
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"input.jsonl"
            a={"source_id":"a","summary":"Original English summary","language":"en"}
            write_records(p,[a,a])
            rows,stats=load_articles([p])
            self.assertEqual(len(rows),1)
            self.assertEqual(stats["duplicate_source_id"],1)
            write_records(p,[a,dict(a,summary="Changed English summary")])
            with self.assertRaises(ValueError):
                load_articles([p])

    def test_cache_and_no_text_persistence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); backend=FakeBackend(); a=article()
            encode_articles([a],backend,root/"one",root/"cache")
            encode_articles([a],backend,root/"two",root/"cache")
            self.assertEqual(backend.calls,1)
            first,vec,_=load_features(root/"one")
            second,_,_=load_features(root/"two")
            self.assertEqual(first[0]["finbert_completed_at"],second[0]["finbert_completed_at"])
            self.assertEqual(vec.shape,(1,768))
            self.assertEqual(vec.dtype,np.float16)
            del vec
            self.assertNotIn(a["summary"],(root/"one/article_finbert.jsonl").read_text())
            changed=dict(a,input_text_hash=digest("different summary"),summary="different summary")
            encode_articles([changed],backend,root/"three",root/"cache")
            self.assertEqual(backend.calls,2)
            backend.metadata["max_length"]=256
            encode_articles([a],backend,root/"four",root/"cache")
            self.assertEqual(backend.calls,3)
            with self.assertRaises(FileExistsError):
                encode_articles([a],backend,root/"one",root/"cache")

class TimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.axis,self.schedule=make_axis(Path(self.temp.name))

    def test_close_weekend_date_only_dst(self):
        for publication,expected in (("2020-03-05T21:01:00Z",1),("2020-03-06T22:00:00Z",2),
                                     ("2020-03-07",2),("2020-03-06",2),("2020-03-09T20:01:00Z",3)):
            self.assertEqual(self.schedule.first_session(article(publication=publication),"retrospective"),expected)

    def test_observed_uses_all_completion_times(self):
        a=article()
        a.update(finbert_completed_at="2020-03-05T19:00:00Z",news_feature_available_at="2020-03-05T19:00:00Z",
                 model_available_at="2010-01-01T00:00:00Z")
        self.assertEqual(self.schedule.first_session(a,"observed_pit"),0)
        for key in ("first_seen_at","fetched_at","feature_available_at","finbert_completed_at","news_feature_available_at","model_available_at"):
            delayed=dict(a,**{key:"2020-03-06T22:00:00Z"})
            self.assertEqual(self.schedule.first_session(delayed,"observed_pit"),2)
            missing=dict(a,**{key:None})
            with self.assertRaises(ValueError):
                self.schedule.first_session(missing,"observed_pit")
        with self.assertRaises(ValueError):
            self.schedule.first_session(dict(a,availability_basis="migration"),"observed_pit")

    def test_simulated_does_not_use_future_encoder_completion(self):
        a=article(publication="2020-03-06T20:59:00Z")
        a["finbert_completed_at"]="2026-01-01T00:00:00Z"
        self.assertEqual(self.schedule.first_session(a,"simulated_pit",300),2)
        with self.assertRaises(ValueError):
            self.schedule.first_session(article(publication="2020-03-06T10:00:00"),"retrospective")

    def test_early_close(self):
        dates=["2020-11-25","2020-11-27","2020-11-30","2020-12-01"]
        a=Axis("TEST",["a"],dates,2,3,{})
        schedule=Schedule(a,[{"date":d,"decision_cutoff_at":d+("T18:00:00Z" if i==1 else "T21:00:00Z")} for i,d in enumerate(dates)])
        self.assertEqual(schedule.first_session(article(publication="2020-11-27T19:00:00Z"),"retrospective"),2)

    def test_bitemporal_mapping(self):
        cutoff=self.schedule.cutoffs[0]
        base=company()
        self.assertTrue(relationship_valid(base,DATES[0],DATES[0],cutoff,"observed_pit"))
        for update in ({"known_from":"2020-03-06T00:00:00Z"},{"verification_status":"unverified"},
                       {"valid_from":"2020-03-06"},{"valid_to":"2020-03-05"},{"known_from":None}):
            self.assertFalse(relationship_valid(dict(base,**update),DATES[0],DATES[0],cutoff,"observed_pit"))

class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.axis,self.schedule=make_axis(self.root)

    def features(self,articles,name="features"):
        path=self.root/name
        encode_articles(articles,FakeBackend(),path,self.root/"cache")
        return path

    def test_end_to_end_causal_panel_and_decision_index(self):
        features=self.features([article(),article("market",scopes=["market"]),article("future","2020-03-12T22:00:00Z")])
        links=[company(),company("future",stock_uid="TEST:B")]
        output=self.root/"panel"
        manifest=build_panel(self.axis,self.schedule,features,links,[],output,projection="random_projection")
        panel=NewsPanel(output)
        self.assertEqual(panel.features["company"].shape,(2,8,40))
        self.assertEqual(panel.features["market"].shape,(8,40))
        self.assertEqual(panel.masks["company"][1,5],0)
        self.assertEqual(panel.masks["company"][1,6],1)
        self.assertTrue(np.all(panel.features["industry"]==0))
        sample=panel.window(np.int64(0),lookback=2)
        self.assertEqual((sample["decision_idx"],sample["target_idx"]),(1,2))
        np.testing.assert_array_equal(sample["company_news"],panel.features["company"][:,1])
        self.assertEqual(manifest["mode"],"retrospective")
        self.assertTrue(manifest["historical_model_replay"])
        self.assertEqual(manifest["transform"]["fit_mode"],"fixed_random_projection")
        panel.assert_compatible((2,8,5),self.axis.stock_uids,self.axis.dates)
        with self.assertRaises(ValueError):
            panel.assert_compatible((2,8,5),self.axis.stock_uids[::-1],self.axis.dates)
        with self.assertRaises(ValueError):
            panel.window(0,steps=3)

    def test_industry_only_verified_historical_members_receive(self):
        features=self.features([article()])
        link=company(channel="industry",industry_standard="GICS",industry_version="fixture",industry_code="10",evidence_type="inferred")
        member=company(); member.pop("source_id"); member.update(industry_standard="GICS",industry_version="fixture",industry_code="10")
        build_panel(self.axis,self.schedule,features,[link],[member],self.root/"panel",projection="random_projection")
        panel=NewsPanel(self.root/"panel")
        self.assertEqual(panel.masks["industry"][0,0],1)
        self.assertEqual(panel.masks["industry"][1,0],0)
        self.assertEqual(panel.features["industry"][0,0,37],1)
        self.assertTrue(np.all(panel.masks["company"]==0))

    def test_market_scope_is_not_inferred_from_us_directory(self):
        features=self.features([article()])
        with self.assertRaises(ValueError):
            build_panel(self.axis,self.schedule,features,[],[],self.root/"invalid",projection="random_projection")
        self.assertFalse((self.root/"invalid").exists())

    def test_training_only_pca_and_no_silent_fallback(self):
        train=np.random.RandomState(4).randn(40,768).astype(np.float32)
        vectors=np.concatenate([train,np.zeros((2,768),np.float32)])
        mean,matrix,meta=compress(vectors,list(range(40)))
        vectors[40:]=999
        mean2,matrix2,_=compress(vectors,list(range(40)))
        np.testing.assert_array_equal(mean,mean2)
        np.testing.assert_array_equal(matrix,matrix2)
        self.assertEqual(meta["fitted_articles"],40)
        with self.assertRaises(ValueError):
            compress(vectors,[0,1])

    def test_panel_pca_selection_excludes_validation_articles(self):
        articles=[article("training"+str(i)) for i in range(40)]
        articles += [article("heldout","2020-03-11T18:00:00Z")]
        links=[company(a["source_id"]) for a in articles]
        features=self.features(articles)
        m=build_panel(self.axis,self.schedule,features,links,[],self.root/"pca1")
        self.assertEqual(m["transform"]["fitted_articles"],40)
        with np.load(self.root/"pca1/embedding_transform.npz") as transform:
            first_mean=transform["mean"].copy()
            first_axes=transform["components"].copy()
        articles[-1]=dict(articles[-1],summary="Different heldout content",input_text_hash=digest("new heldout text"))
        features2=self.features(articles,"features2")
        build_panel(self.axis,self.schedule,features2,links,[],self.root/"pca2")
        with np.load(self.root/"pca2/embedding_transform.npz") as transform:
            np.testing.assert_array_equal(first_mean,transform["mean"])
            np.testing.assert_array_equal(first_axes,transform["components"])

    def test_duplicate_content_and_inferred_paths_not_double_counted(self):
        one=article(); two=dict(one,source_id="duplicate")
        features=self.features([one,two])
        build_panel(self.axis,self.schedule,features,[company(),company("duplicate")],[],self.root/"panel",projection="random_projection")
        panel=NewsPanel(self.root/"panel")
        self.assertAlmostEqual(float(panel.features["company"][0,0,35]),float(np.log(2)),places=6)

    def test_observed_rejects_today_generated_historical_features(self):
        features=self.features([article()])
        with self.assertRaises(ValueError):
            build_panel(self.axis,self.schedule,features,[company()],[],self.root/"panel",mode="observed_pit",projection="random_projection")

    def test_pre_dataset_articles_are_not_revived_as_fresh_news(self):
        features=self.features([article(publication="2010-01-01T00:00:00Z")])
        with self.assertRaises(ValueError):
            build_panel(self.axis,self.schedule,features,[company()],[],self.root/"panel",projection="random_projection")

    def test_simulated_encoding_delay_does_not_reset_article_age(self):
        features=self.features([article()])
        manifest=build_panel(self.axis,self.schedule,features,[company()],[],self.root/"panel",
            mode="simulated_pit",latency_seconds=24*3600,projection="random_projection")
        panel=NewsPanel(self.root/"panel")
        self.assertEqual(panel.masks["company"][0,0],0)
        self.assertEqual(panel.features["company"][0,1,36],1)
        self.assertEqual(panel.masks["company"][0,5],0)

    def test_artifact_tampering_is_rejected(self):
        features=self.features([article()])
        (features/"article_finbert.jsonl").write_text("[]\n")
        with self.assertRaises(ValueError):
            load_features(features)

if __name__=="__main__":
    unittest.main()
