"""Trading dates come from the dataset; decision cutoffs require an explicit calendar."""
from dataclasses import dataclass
from datetime import date as date_type
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import digest
from .validation import integer, finite_number

MODES=("observed_pit","simulated_pit","retrospective")

def timestamp(value):
    if not value:
        raise ValueError("missing_timestamp")
    try:
        ts=pd.Timestamp(value)
    except (ValueError,TypeError):
        raise ValueError("invalid_timestamp") from None
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError("timezone_required")
    return ts.tz_convert("UTC")

@dataclass
class Axis:
    dataset:str
    stock_uids:list
    dates:list
    train_end:int
    valid_end:int
    source:dict

    @classmethod
    def load(cls,path):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        axis=cls(raw["dataset"],raw["stock_uids"],raw["trading_dates"],
            raw["train_end_idx"],raw["valid_end_idx"],raw)
        if not isinstance(axis.stock_uids,list) or not axis.stock_uids or not all(isinstance(s,str) and s.strip() for s in axis.stock_uids) or len(set(axis.stock_uids))!=len(axis.stock_uids):
            raise ValueError("Stock identities must be nonempty and unique.")
        if not isinstance(axis.dates,list) or not axis.dates or not all(isinstance(d,str) for d in axis.dates) or axis.dates!=sorted(set(axis.dates)):
            raise ValueError("Dates must be strictly increasing and unique.")
        try:
            if any(date_type.fromisoformat(day).isoformat()!=day for day in axis.dates):
                raise ValueError("Noncanonical trading date.")
        except ValueError:
            raise ValueError("Trading dates must be valid ISO calendar dates.") from None
        integer(axis.train_end,"train_end",1)
        integer(axis.valid_end,"valid_end",1)
        if not 0<axis.train_end<axis.valid_end<len(axis.dates):
            raise ValueError("Invalid dataset split boundaries.")
        return axis

    @property
    def stock_hash(self):
        return digest(self.stock_uids)

    @property
    def date_hash(self):
        return digest(self.dates)

class Schedule:
    def __init__(self,axis,records):
        if [r["date"] for r in records]!=axis.dates:
            raise ValueError("Decision schedule must exactly match the dataset date axis.")
        self.axis=axis
        self.cutoffs=[timestamp(r["decision_cutoff_at"]) for r in records]
        if any(t.tz_convert("America/New_York").strftime("%Y-%m-%d")!=d for d,t in zip(axis.dates,self.cutoffs)):
            raise ValueError("Decision cutoff local date mismatch.")
        self.ns=np.array([t.value for t in self.cutoffs],dtype=np.int64)
        if np.any(np.diff(self.ns)<=0):
            raise ValueError("Decision cutoffs must be strictly increasing.")
        self.records=records

    def first_session(self,article,mode,latency_seconds=0):
        finite_number(latency_seconds,"simulated latency",minimum=0)
        if mode not in MODES:
            raise ValueError("Invalid time mode or simulated latency.")
        value=article.get("published_at")
        if not isinstance(value,str):
            raise ValueError("missing_publication")
        if len(value)==10 or article.get("publication_precision")=="date":
            date=pd.Timestamp(value[:10]).strftime("%Y-%m-%d")
            # Strictly later session, even when date is a weekend/holiday.
            threshold=(pd.Timestamp(date,tz="America/New_York")+pd.DateOffset(days=1)).tz_convert("UTC")
        else:
            threshold=timestamp(value)
        if mode=="simulated_pit":
            threshold+=pd.Timedelta(seconds=latency_seconds)
        if mode=="observed_pit":
            if article.get("availability_basis") not in ("observed","verified_audit"):
                raise ValueError("unobserved_availability")
            required=("first_seen_at","fetched_at","feature_available_at","finbert_completed_at",
                      "news_feature_available_at","model_available_at")
            threshold=max([threshold]+[timestamp(article.get(k)) for k in required])
            # Preserve and honor explicit upstream completion/commit evidence.
            optional=("llm_inferred_at","extracted_at","commit_verified_at","effective_feature_available_at")
            additional=[timestamp(article[k]) for k in optional if article.get(k) is not None]
            if article.get("llm_time_required") and not article.get("llm_inferred_at"):
                raise ValueError("llm_completion_unknown")
            if additional:
                threshold=max([threshold]+additional)
        # Equality at the cutoff is excluded in every mode, including retrospective replay.
        return int(np.searchsorted(self.ns,threshold.value,side="right"))

def export_schedule(axis,output):
    """Use exchange-calendars including early closes; no hardcoded 16:00 fallback."""
    import exchange_calendars as xcals
    from .io import write_records
    calendar=xcals.get_calendar("XNYS",start=axis.dates[0],end=axis.dates[-1])
    records=[]
    for day in axis.dates:
        if not calendar.is_session(day):
            raise ValueError("Dataset date missing in exchange calendar: "+day)
        close=calendar.session_close(day)
        records.append({"date":day,"decision_cutoff_at":close.isoformat(),
                        "calendar_source":"exchange_calendars:"+xcals.__version__+":XNYS"})
    Schedule(axis,records)
    if Path(output).exists():
        raise FileExistsError("Schedule already exists.")
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    write_records(output,records)
