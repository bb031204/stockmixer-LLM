"""Immutable market artifacts for the separate, target-split training protocol."""
import csv
import hashlib
import io
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json
from .time_axis import Axis
from .validation import integer, finite_number

SCHEMA = "stockmixer_strict_v1"
PROJECT = Path(__file__).resolve().parents[1]
PROTECTED = ("src/model.py", "src/load_data.py")  # Active dependencies; retired entry points are archived.
RELEASED_MODEL_SHA256 = "8e255da03d1efbbaead8a75f9561e6980c6b19a16f20fa2731f5ce99d4d2ec31"


def resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def configuration(path=None, dataset=None, project=None):
    root = Path(project or PROJECT).resolve()
    path = resolve(root, path or "configs/stockmixer.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config,dict):
        raise ValueError("Configuration must be a mapping.")
    if config.get("schema_version")=="stockmixer_config_v2":
        from .configuration import compile_config
        return root,compile_config(root,path,config,dataset)
    config["dataset"] = dataset or config.get("dataset")
    validate_config(config)
    return root, config


def validate_config(config):
    if config.get("schema_version") != "stockmixer_strict_config_v1":
        raise ValueError("Use configs/stockmixer.yaml for the single training entry.")
    sections={
        "data":{"dataset_root","axis_dir","identity_mapping","processed_csv_root","lookback","steps","preprocessing"},
        "model":{"backbone","scale_argument"},
        "training":{"epochs","learning_rate","alpha","numpy_seed","torch_seed","device","selection_metric","test_policy"},
        "fusion":{"enabled_channels","freeze_backbone","news_dim","hidden_dim","dropout","time_mode"},
        "news":{"source_root","adapted_snapshot","panel"}}
    if set(config)-{"workflow"}!={"schema_version","dataset",*sections}:
        raise ValueError("Missing or unknown top-level configuration key.")
    for section,fields in sections.items():
        if not isinstance(config[section],dict) or set(config[section])-({"metric_protocol"} if section=="training" else set())!=fields:
            raise ValueError("Missing or unknown configuration field in "+section)
    if config["dataset"] not in ("NASDAQ", "NYSE", "SP500"):
        raise ValueError("Unknown dataset.")
    if config["model"]["backbone"] != "released_legacy":
        raise ValueError("This entry point pins the released_legacy backbone; no silent model migration.")
    if integer(config["data"]["lookback"],"lookback",1) != 16 or integer(config["data"]["steps"],"steps",1) != 1:
        raise ValueError("The pinned released backbone/labels support lookback=16 and steps=1 only.")
    if config["data"]["preprocessing"] not in ("train_max_ratio","original"):
        raise ValueError("Use original or train_max_ratio preprocessing with a matching data artifact.")
    if config["training"]["selection_metric"] != "val_loss" or config["training"]["test_policy"] not in ("final_best_once","per_epoch"):
        raise ValueError("Use validation loss selection and test_policy=per_epoch or final_best_once.")
    if integer(config["model"]["scale_argument"],"scale_argument",1)!=3:
        raise ValueError("scale_argument is ineffective in this pinned model; keep the recorded value 3.")
    settings=config["training"]
    if settings.get("metric_protocol","valid_only") not in ("stockmixer_original","valid_only"):
        raise ValueError("metric_protocol must be stockmixer_original or valid_only.")
    integer(settings["epochs"],"epochs",1)
    finite_number(settings["learning_rate"],"learning_rate",strictly_positive=True)
    finite_number(settings["alpha"],"alpha",minimum=0)
    integer(settings["numpy_seed"],"numpy_seed",0,2**32-1)
    integer(settings["torch_seed"],"torch_seed",0,2**63-1)
    if not isinstance(settings["device"],str) or re.fullmatch(r"auto|cpu|cuda(?::\d+)?",settings["device"]) is None:
        raise ValueError("device must be auto, cpu, cuda or cuda:<index>.")
    fusion=config["fusion"]
    if fusion["freeze_backbone"] is not True:
        raise ValueError("This fusion protocol requires freeze_backbone=true.")
    integer(fusion['news_dim'],'news_dim',9,776)
    integer(fusion["hidden_dim"],"hidden_dim",1)
    if not 0<=finite_number(fusion["dropout"],"dropout",minimum=0)<1:
        raise ValueError("dropout must be below 1.")
    channels=fusion["enabled_channels"]
    if not isinstance(channels,list) or not all(isinstance(c,str) for c in channels) or len(set(channels))!=len(channels) or not set(channels)<={"company","industry","market"}:
        raise ValueError("enabled_channels must be a list of unique, supported channel names.")
    if fusion["time_mode"] not in ("retrospective","simulated_pit","observed_pit"):
        raise ValueError("Unsupported news time_mode.")
    for section,names in (("data",("dataset_root","axis_dir","identity_mapping","processed_csv_root")),("news",("source_root","adapted_snapshot"))):
        if any(not isinstance(config[section][name],str) or not config[section][name].strip() for name in names):
            raise ValueError("Configuration paths must be nonempty strings.")
    if config["news"]["panel"] is not None and (not isinstance(config["news"]["panel"],str) or not config["news"]["panel"].strip()):
        raise ValueError("news.panel must be null or a nonempty path.")


    if "workflow" in config:
        validate_workflow(config)


def validate_workflow(config):
    """Optional for historical checkpoints; mandatory for the single-entry workflow."""
    from datetime import date
    workflow=config["workflow"]
    fields={"mode","prepared_data","prepare_if_missing","output_directory","smoke","baseline_checkpoint","prediction"}
    if not isinstance(workflow,dict) or set(workflow)!=fields:
        raise ValueError("Missing or unknown workflow configuration field.")
    if workflow["mode"] not in ("baseline","fusion"):
        raise ValueError("workflow.mode must be baseline or fusion.")
    paths=workflow["prepared_data"]
    if not isinstance(paths,dict) or set(paths)!={"NASDAQ","NYSE","SP500"} or any(not isinstance(v,str) or not v.strip() for v in paths.values()):
        raise ValueError("workflow.prepared_data must provide paths for NASDAQ, NYSE and SP500.")
    for field in ("prepare_if_missing","smoke"):
        if not isinstance(workflow[field],bool):
            raise ValueError("workflow."+field+" must be a boolean.")
    for field in ("output_directory","baseline_checkpoint"):
        value=workflow[field]
        if value is not None and (not isinstance(value,str) or not value.strip()):
            raise ValueError("workflow."+field+" must be null or a nonempty path.")
    prediction=workflow["prediction"]
    if not isinstance(prediction,dict) or set(prediction)!={"enabled","as_of","allow_in_sample_replay"}:
        raise ValueError("Missing or unknown workflow.prediction configuration field.")
    if any(not isinstance(prediction[field],bool) for field in ("enabled","allow_in_sample_replay")):
        raise ValueError("Prediction enabled/replay flags must be booleans.")
    value=prediction["as_of"]
    if not isinstance(value,str):
        raise ValueError("Prediction as_of must be latest or a quoted YYYY-MM-DD string.")
    if value!="latest":
        try:
            if date.fromisoformat(value).isoformat()!=value:
                raise ValueError()
        except ValueError:
            raise ValueError("Prediction as_of must be latest or YYYY-MM-DD.") from None
    if workflow["mode"]=="baseline":
        if workflow["baseline_checkpoint"] is not None or config["news"]["panel"] is not None:
            raise ValueError("Baseline mode cannot silently consume a baseline checkpoint or news panel.")
    elif workflow["baseline_checkpoint"] is None or config["news"]["panel"] is None or not config["fusion"]["enabled_channels"]:
        raise ValueError("Fusion mode requires baseline_checkpoint, news.panel and enabled channels.")


@dataclass(frozen=True)
class TargetSplit:
    n_dates: int
    train_end: int
    valid_end: int
    lookback: int = 16
    steps: int = 1

    def __post_init__(self):
        for name in ("n_dates","train_end","valid_end","lookback","steps"):
            integer(getattr(self,name),name,1)
        if not (self.lookback >= 1 and self.steps >= 1 and
                self.lookback + self.steps - 1 < self.train_end < self.valid_end < self.n_dates):
            raise ValueError("Invalid target split/window boundaries.")

    def targets(self, phase):
        bounds = {"train": (self.lookback + self.steps - 1, self.train_end),
                  "validation": (self.train_end, self.valid_end), "test": (self.valid_end, self.n_dates)}
        return np.arange(*bounds[phase], dtype=np.int64)

    def offsets(self, phase):
        return self.targets(phase) - self.lookback - self.steps + 1

    def describe(self, dates):
        result = {}
        for phase in ("train", "validation", "test"):
            target = self.targets(phase)
            result[phase] = {"samples": len(target), "target_indices": [int(target[0]), int(target[-1])],
                             "target_dates": [dates[target[0]], dates[target[-1]]],
                             "offset_indices": [int(self.offsets(phase)[0]), int(self.offsets(phase)[-1])],
                             "decision_indices": [int(target[0]-self.steps), int(target[-1]-self.steps)]}
        return result


def positive_train_scale(values, missing, train_end, shared_close=False):
    """Cancel any previous constant divisor, retaining explicit per-feature missingness."""
    values = np.asarray(values, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if values.ndim != 3 or missing.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Invalid source feature tensor/missingness.")
    integer(train_end,"train_end",1,values.shape[1])
    fitting = np.where(missing[:, :train_end], -np.inf, values[:, :train_end])
    scale = fitting.max(axis=1)
    if shared_close:
        scale = np.repeat(scale[:, -1:], values.shape[-1], axis=1)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Every stock/feature needs a positive, valid training-period maximum.")
    normalized = values / scale[:, None, :]
    normalized[missing] = 1.1  # fixed original fill constant, applied AFTER scaling
    normalized = normalized.astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized features.")
    return normalized, scale


def verify_axis_sources(project, axis, dataset_root=None):
    for relative, expected in axis.source["source_sha256"].items():
        key=Path(relative)
        if key.is_absolute() or key.parts[0]!="dataset" or ".." in key.parts:
            raise ValueError("Invalid verified axis source path.")
        source=Path(dataset_root).joinpath(*key.parts[1:]) if dataset_root else project/key
        if digest_file(source) != expected:
            raise ValueError("Market source changed since stock/date-axis verification: " + relative)


def identity_rows(project, config, axis):
    path = resolve(project, config["data"]["identity_mapping"])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["dataset"] == axis.dataset]
    if [int(r["array_index"]) for r in rows] != list(range(len(axis.stock_uids))):
        raise ValueError("Identity mapping row order mismatch.")
    if [r["stock_uid"] for r in rows] != axis.stock_uids or any(not r["ticker"] for r in rows):
        raise ValueError("Identity mapping does not match the verified axis.")
    return rows, path


def read_market(project, config, axis, identities):
    """Reproduce and compare legacy arrays before deriving a new immutable artifact."""
    folder = resolve(project, config["data"]["dataset_root"]) / axis.dataset
    n, t = len(axis.stock_uids), len(axis.dates)
    source_hashes = {}
    if axis.dataset == "SP500":
        raw_path = folder / "baseline_data_sp500.npy"
        # This is a trusted, local legacy dataset containing pandas DataFrames.
        raw = np.load(raw_path, allow_pickle=True).item()
        if list(raw) != [r["ticker"] for r in identities]:
            raise ValueError("SP500 raw dictionary order differs from stock identities.")
        columns = ["open", "high", "low", "volume", "close"]
        full = np.stack([raw[r["ticker"]][columns].to_numpy(dtype=np.float64) for r in identities])
        old = np.load(folder / "SP500.npy", allow_pickle=False)
        if not np.array_equal(full / full.max(axis=1)[:, None, :], old):
            raise ValueError("SP500 raw-to-legacy feature correspondence failed.")
        values = full[:, 915:].copy()
        missing = np.zeros_like(values, dtype=bool)
        masks = np.ones((n, t), dtype=np.float32)
        source_hashes[str(raw_path)] = digest_file(raw_path)
        legacy_returns = None
        feature_names = columns
        provenance = "raw named columns; verified against all legacy SP500.npy elements; slice from index 915"
    else:
        loaded = {}
        for key in ("eod", "mask", "gt", "price"):
            path = folder / (key + "_data.pkl")
            with path.open("rb") as stream:
                loaded[key] = pickle.load(stream)
        values = np.empty((n, t, 5), dtype=np.float32)
        source_root = resolve(project, config["data"]["processed_csv_root"])
        for i, row in enumerate(identities):
            path = source_root / (axis.dataset + "_" + row["ticker"] + "_1.csv")
            content = path.read_bytes()
            source_hashes[str(path)] = hashlib.sha256(content).hexdigest()
            array = np.genfromtxt(io.BytesIO(content), delimiter=",", dtype=np.float32)
            if axis.dataset == "NASDAQ":
                array = array[:-1]
            if array.shape != (t, 6) or not np.array_equal(array[:, 0], np.arange(t)):
                raise ValueError("Processed CSV date/index shape mismatch: " + str(path))
            values[i] = array[:, 1:]
        missing = values == np.float32(-1234)
        old = np.where(missing, np.float32(1.1), values)
        masks = (~missing[:, :, -1]).astype(np.float32)
        if not np.array_equal(old, loaded["eod"]) or not np.array_equal(masks, loaded["mask"]):
            raise ValueError("Processed CSV features/masks do not exactly reproduce the dataset.")
        if not np.array_equal(loaded["price"], old[:, :, -1]):
            raise ValueError("Legacy price_data is not the final input feature.")
        legacy_returns = loaded["gt"]
        feature_names = ["close_ma5", "close_ma10", "close_ma20", "close_ma30", "close"]
        provenance = ("processed CSV: fixed positive full-period close divisor cancels algebraically when "
                      "rebased by training close maximum; original CSV rounding cannot be undone")
    if values.shape != (n, t, 5) or not np.isfinite(values).all():
        raise ValueError("Invalid market feature shape or values.")
    valid = masks.astype(bool)
    if (values[:, :, -1][valid] <= 0).any():
        raise ValueError("Nonpositive observed close price.")
    returns = np.zeros((n, t), dtype=np.float32)
    pair = valid[:, 1:] & valid[:, :-1]
    raw_close = values[:, :, -1].astype(np.float64)
    calculated = (raw_close[:, 1:] - raw_close[:, :-1]) / raw_close[:, :-1]
    returns[:, 1:][pair] = calculated[pair].astype(np.float32)
    if legacy_returns is not None:
        error = float(np.max(np.abs(returns[:, 1:][pair] - legacy_returns[:, 1:][pair])))
        if error > 2e-6:
            raise ValueError("Valid return labels differ from original data: " + str(error))
        # Keep original valid labels exactly; invalid labels are explicitly excluded by the window mask.
        returns[:, 1:][pair] = legacy_returns[:, 1:][pair]
    eod, scale = positive_train_scale(values, missing, axis.train_end, shared_close=axis.dataset != "SP500")
    return eod, masks, returns, scale, source_hashes, feature_names, provenance


def prepare_market(project, config, output):
    validate_config(config)
    if config["data"]["preprocessing"]=="original":
        from .original import prepare_original_market
        return prepare_original_market(project,config,output)
    axis_path = resolve(project, config["data"]["axis_dir"]) / (config["dataset"] + ".json")
    axis = Axis.load(axis_path)
    dataset_root=resolve(project,config["data"]["dataset_root"])
    verify_axis_sources(project, axis, dataset_root)
    identities, identity_path = identity_rows(project, config, axis)
    data, masks, returns, scale, sources, feature_names, provenance = read_market(project, config, axis, identities)
    split = TargetSplit(len(axis.dates), axis.train_end, axis.valid_end)
    # All observed returns retain their definition after scaling (the small tolerance covers float32).
    price = data[:, :, -1].copy()
    pair = (masks[:, 1:] * masks[:, :-1]).astype(bool)
    scaled_return = (price[:, 1:] - price[:, :-1]) / price[:, :-1]
    error = float(np.max(np.abs(scaled_return[pair] - returns[:, 1:][pair])))
    if error > 3e-6 or (price <= 0).any():
        raise ValueError("Price-space return invariance check failed.")
    verify_axis_sources(project, axis, dataset_root)
    for path, expected in sources.items():
        if digest_file(path) != expected:
            raise ValueError("Input changed while preparing market artifact: " + path)
    with new_artifact(output) as directory:
        for name, array in (("eod", data), ("mask", masks), ("returns", returns), ("price", price), ("scale", scale)):
            np.save(directory / (name + ".npy"), array, allow_pickle=False)
        write_json(directory / "axis.json", axis.source)
        write_json(directory / "source_hashes.json", sources)
        with (directory / "stock_identity.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(identities[0]))
            writer.writeheader(); writer.writerows(identities)
        manifest = {"schema_version": SCHEMA, "kind": "strict_market_data", "created_at": now(),
            "dataset": axis.dataset, "stock_axis_hash": axis.stock_hash, "date_axis_hash": axis.date_hash,
            "axis_file_sha256": digest_file(axis_path), "identity_mapping_sha256": digest_file(identity_path),
            "shape": list(data.shape), "feature_names": feature_names, "split": split.describe(axis.dates),
            "train_end_idx": axis.train_end, "valid_end_idx": axis.valid_end, "lookback": 16, "steps": 1,
            "source_dataset_root":str(dataset_root),
            "preprocessing": {"mode": "train_max_ratio", "fit_end_exclusive": axis.train_end,
                "scaling": "shared close maximum" if axis.dataset != "SP500" else "per-feature maximum",
                "provenance": provenance, "max_return_invariance_error": error, "missing_fill": 1.1},
            "backbone": backbone_contract(), "legacy_source_hashes": axis.source["source_sha256"],
            "protected_source_hashes": {name: digest_file(project / name) for name in PROTECTED},
            "limitations": ["Fixed historical stock pool; historical constituent membership not verified.",
                "Corporate-action/price revision point-in-time provenance is not verified.",
                "Train-only scaling and target separation do not establish observed-PIT news availability."],
            "files": file_manifest(directory)}
        write_json(directory / "manifest.json", manifest)
    return manifest


def backbone_contract():
    model_hash=digest_file(PROJECT / "src/model.py")
    if model_hash!=RELEASED_MODEL_SHA256:
        raise ValueError("This protocol pins the released model source. A different backbone needs an explicit new protocol.")
    return {"id": "released_legacy", "lookback": 16, "temporal_lengths": [16, 16, 8],
            "temporal_dim": 40, "scale_argument_effective": False,
            "model_sha256": model_hash}


class MarketData:
    def __init__(self, directory, project=None):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("schema_version") != SCHEMA or m.get("kind") != "strict_market_data":
            raise ValueError("A verified strict market artifact is required.")
        verify_files(self.directory, m, required=("eod.npy","mask.npy","returns.npy","price.npy","scale.npy","axis.json","stock_identity.csv","source_hashes.json"))
        if m["backbone"] != backbone_contract():
            raise ValueError("Backbone source/architecture mismatch; do not mix model directories.")
        self.axis = Axis.load(self.directory / "axis.json")
        if self.axis.stock_hash != m["stock_axis_hash"] or self.axis.date_hash != m["date_axis_hash"]:
            raise ValueError("Prepared market axis mismatch.")
        self.split = TargetSplit(len(self.axis.dates), self.axis.train_end, self.axis.valid_end, m["lookback"], m["steps"])
        if m["lookback"]!=16 or m["steps"]!=1 or m.get("dataset")!=self.axis.dataset:
            raise ValueError("Prepared market window/dataset contract mismatch.")
        if m.get("preprocessing",{}).get("mode")!="train_max_ratio" or m["preprocessing"].get("fit_end_exclusive")!=self.axis.train_end:
            raise ValueError("Prepared preprocessing fitting boundary mismatch.")
        if m["train_end_idx"] != self.axis.train_end or m["valid_end_idx"] != self.axis.valid_end:
            raise ValueError("Prepared market split mismatch.")
        self.arrays = {key: np.load(self.directory / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                       for key in ("eod", "mask", "returns", "price")}
        n, t = len(self.axis.stock_uids), len(self.axis.dates)
        for key, array in self.arrays.items():
            if array.shape != ((n, t, 5) if key == "eod" else (n, t)) or not np.isfinite(array).all():
                raise ValueError("Invalid prepared array: " + key)
        if not np.isin(self.arrays["mask"], [0, 1]).all() or (self.arrays["price"] <= 0).any():
            raise ValueError("Invalid price or missingness mask.")
        if not np.array_equal(self.arrays["price"],self.arrays["eod"][:,:,-1]):
            raise ValueError("base_price differs from the input close price.")
        if m.get("shape")!=[n,t,5] or m.get("split")!=self.split.describe(self.axis.dates):
            raise ValueError("Prepared market shape/target ledger mismatch.")
        scale=np.load(self.directory/"scale.npy",allow_pickle=False)
        if scale.shape!=(n,5) or not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError("Invalid train-only scale transform.")
        if self.axis.dataset!="SP500" and not np.array_equal(scale,np.repeat(scale[:,-1:],5,axis=1)):
            raise ValueError("Moving-average features require the same close-price scale.")
        valid=self.arrays["mask"].astype(bool)
        close=self.arrays["price"]
        maxima=np.where(valid[:,:self.axis.train_end],close[:,:self.axis.train_end],-np.inf).max(axis=1)
        if not np.allclose(maxima,1,rtol=0,atol=2e-6):
            raise ValueError("Training close maxima are not normalized to one.")
        pair=valid[:,1:] & valid[:,:-1]
        expected=(close[:,1:].astype(np.float64)-close[:,:-1])/close[:,:-1]
        if np.any(np.abs(expected[pair]-self.arrays["returns"][:,1:][pair])>3e-6):
            raise ValueError("Return labels differ from consecutive valid close prices.")
        with (self.directory/"stock_identity.csv").open(encoding="utf-8-sig",newline="") as stream:
            identities=list(csv.DictReader(stream))
        if [r.get("stock_uid") for r in identities]!=self.axis.stock_uids or [r.get("array_index") for r in identities]!=[str(i) for i in range(n)] or any(r.get("dataset")!=self.axis.dataset or not r.get("ticker") for r in identities):
            raise ValueError("Prepared prediction identities differ from the market stock axis.")
        if project:
            verify_axis_sources(Path(project), self.axis, m.get("source_dataset_root"))

    def window(self, target):
        integer(target,"target",0)
        decision = int(target) - 1
        x, price, mask = self.decision(decision)
        if target >= len(self.axis.dates):
            raise IndexError("Training/evaluation requires an in-axis target.")
        mask = np.minimum(mask, self.arrays["mask"][:, target, None])
        return x, price, np.array(self.arrays["returns"][:, target, None], copy=True), mask

    def decision(self, decision):
        integer(decision,"decision",0)
        if not self.split.lookback - 1 <= decision < len(self.axis.dates):
            raise IndexError("Decision window is outside the data axis.")
        begin = decision - self.split.lookback + 1
        return (np.array(self.arrays["eod"][:, begin:decision+1], copy=True),
                np.array(self.arrays["price"][:, decision, None], copy=True),
                np.array(self.arrays["mask"][:, begin:decision+1].min(axis=1, keepdims=True), copy=True))


def audit_news(project, config):
    """Compare current input IDs AND producer hashes, without inventing a new audit chain."""
    from .source_adapter import producer_hash
    root = resolve(project, config["news"]["source_root"])
    artifact = resolve(project, config["news"]["adapted_snapshot"])
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    verify_files(artifact, manifest)
    adapted = {r["source_id"]: r for r in read_records(artifact / "articles.jsonl")}
    current, physical, hashes = {}, {}, {}
    for scope in ("US", "GLOBAL"):
        folder = root / scope
        if not folder.is_dir():
            raise FileNotFoundError("Missing input directory: " + str(folder))
        paths = sorted(folder.glob("*.json")); physical[scope] = len(paths)
        for path in paths:
            payload = path.read_bytes()
            row = json.loads(payload.decode("utf-8-sig"))
            sid, hashed = row["source_id"], producer_hash(row)
            if sid in current and current[sid] != hashed:
                raise ValueError("Conflicting current article versions: " + sid)
            current[sid] = hashed
            hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    missing = sorted(set(adapted) - set(current))
    added = sorted(set(current) - set(adapted))
    changed = sorted(s for s in set(current) & set(adapted) if current[s] != adapted[s].get("source_record_hash"))
    if set(hashes)!={str(path) for scope in ("US","GLOBAL") for path in (root/scope).glob("*.json")}:
        raise ValueError("Current input membership changed during audit; retry with a frozen input directory.")
    for path, expected in hashes.items():
        if digest_file(path)!=expected:
            raise ValueError("Current news input changed during audit: " + path)
    # Return a concrete contract; disagreement never silently switches training inputs.
    return {"physical_files": physical, "current_unique_articles": len(current),
        "adapted_articles": len(adapted), "adapted_manifest_sha256": digest_file(artifact / "manifest.json"),
        "adapted_source_root": manifest["source_root"], "current_source_root": str(root),
        "missing_current_ids": missing, "new_current_ids": added, "changed_current_ids": changed,
        "exact_current_snapshot": not (missing or added or changed), "current_source_hashes": hashes,
        "decision": "Do not reuse as current snapshot until differences are explicitly resolved.",
        "model_inference_run": False, "historical_observed_pit_ready": False}



def load_market(directory,project=None):
    manifest=json.loads((Path(directory)/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind")=="original_market_data":
        from .original import OriginalMarketData
        return OriginalMarketData(directory,project)
    return MarketData(directory,project)
