"""Read precomputed panels at the input-window decision date, not the future target."""
import json
from pathlib import Path
from numbers import Integral
import numpy as np
from .io import SCHEMA, digest, verify_files
from .time_axis import Axis

class NewsPanel:
    def __init__(self,directory,verify_hashes=True):
        directory=Path(directory)
        self.directory=directory.resolve()
        self.manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        m=self.manifest
        if m.get("schema_version")!=SCHEMA or m.get("kind")!="daily_news_panel" or m.get("feature_dim")!=40:
            raise ValueError("Unsupported panel schema.")
        if verify_hashes:
            required=["axis.json","trading_dates.npy","embedding_transform.npz","decision_schedule.jsonl","provenance.jsonl"]
            required += [c+suffix for c in ("company","industry","market") for suffix in ("_news_daily.npy","_news_mask.npy")]
            verify_files(directory,m,required=required)
        self.axis=Axis.load(directory/"axis.json")
        if self.axis.stock_hash!=m["stock_axis_hash"] or self.axis.date_hash!=m["trading_dates_hash"]:
            raise ValueError("Manifest axis hash mismatch.")
        dates=np.load(directory/"trading_dates.npy",allow_pickle=False)
        if dates.tolist()!=self.axis.dates:
            raise ValueError("Saved trading dates do not match the axis.")
        n,t=len(self.axis.stock_uids),len(self.axis.dates)
        if (m["n_stocks"],m["n_dates"])!=(n,t):
            raise ValueError("Manifest axis size mismatch.")
        self.features,self.masks={},{}
        for channel in ("company","industry","market"):
            features=np.load(directory/(channel+"_news_daily.npy"),mmap_mode="r",allow_pickle=False)
            mask=np.load(directory/(channel+"_news_mask.npy"),mmap_mode="r",allow_pickle=False)
            expected=(t,40) if channel=="market" else (n,t,40)
            if features.shape!=expected or mask.shape!=expected[:-1]:
                raise ValueError("Panel shape mismatch: "+channel)
            # Bounded-memory checks instead of materializing N*T*40 multiple times.
            for index in range(features.shape[0]):
                part,valid=features[index],mask[index]
                if not np.isfinite(part).all() or not np.isin(valid,[0,1]).all():
                    raise ValueError("Invalid finite values/mask: "+channel)
                if np.any(part[np.asarray(valid)==0]!=0):
                    raise ValueError("Missing-news features must be exactly zero: "+channel)
            self.features[channel],self.masks[channel]=features,mask

    def assert_compatible(self,eod_shape,stock_uids,dates):
        if tuple(eod_shape[:2])!=(len(self.axis.stock_uids),len(self.axis.dates)):
            raise ValueError("Market array shape does not match news panels.")
        if digest(list(stock_uids))!=self.axis.stock_hash or digest(list(dates))!=self.axis.date_hash:
            raise ValueError("Market and news stock/date order mismatch; do not reorder silently.")

    def assert_experiment(self,axis,mode,lookback=16):
        self.assert_compatible((len(axis.stock_uids),len(axis.dates),5),axis.stock_uids,axis.dates)
        if self.axis.dataset!=axis.dataset or self.axis.train_end!=axis.train_end or self.axis.valid_end!=axis.valid_end:
            raise ValueError("News panel split differs from the training experiment; rebuild its PCA/panel.")
        transform=self.manifest.get("transform",{})
        if transform.get("training_decision_end_exclusive")!=axis.train_end-1:
            raise ValueError("News transform fitting boundary differs from training decisions.")
        if transform.get("training_decision_start_inclusive")!=lookback-1:
            raise ValueError("News transform fitting start differs from actual training decisions; rebuild panel.")
        if self.manifest.get("mode")!=mode:
            raise ValueError("News panel time mode mismatch.")
        if self.manifest.get("availability_comparison")!="strictly_before_cutoff":
            raise ValueError("Rebuild panel with strict cutoff comparison.")

    def window(self,offset,lookback=16,steps=1):
        if steps!=1:
            raise ValueError("This news feature version supports steps=1 only; no labels are silently changed.")
        if any(isinstance(v,bool) or not isinstance(v,Integral) for v in (offset,lookback)) or offset<0 or lookback<1:
            raise ValueError("Invalid input window.")
        offset,lookback=int(offset),int(lookback)
        decision=offset+lookback-1
        target=decision+steps
        if target>=len(self.axis.dates):
            raise IndexError("Target is outside the dataset.")
        result=self.decision(decision)
        result["target_idx"]=target
        return result

    def decision(self,decision):
        """Read features at an input date without requiring a future label."""
        if isinstance(decision,bool) or not isinstance(decision,Integral) or not 0<=decision<len(self.axis.dates):
            raise IndexError("Decision is outside the news date axis.")
        result={"decision_idx":int(decision)}
        for channel in ("company","industry","market"):
            if channel=="market":
                feature=self.features[channel][decision]
                mask=np.asarray(self.masks[channel][decision]).reshape(1,1)
            else:
                feature=self.features[channel][:,decision]
                mask=self.masks[channel][:,decision,None]
            # Writable copies prevent torch from writing into read-only memory maps.
            result[channel+"_news"]=np.array(feature,dtype=np.float32,copy=True)
            result[channel+"_mask"]=np.array(mask,dtype=np.float32,copy=True)
        return result
