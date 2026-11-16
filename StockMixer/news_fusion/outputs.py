"""StockMixer-style outputs; immutable flat artifacts remain available for exact replay."""
import csv
import json
import os
import re
import shutil
import sys
import uuid
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath

import yaml
from .io import digest_file, now, verify_files, write_json

PROJECT=Path(__file__).resolve().parents[1]
DIRECTORIES=("checkpoints","configs","figures","logs","predictions","tables")
SCHEMA="stockmixer_outputs_v1"
ANSI_RE=re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def timestamp(value=None):
    when=datetime.now(timezone.utc) if value is None else datetime.fromisoformat(value.replace("Z","+00:00"))
    if when.tzinfo is None:
        raise ValueError("Output timestamps require an explicit timezone.")
    return when.astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")


def destination(project, explicit=None, dataset=None, kind="run", category=None, suffix=""):
    """All CLI writes stay under the specified project's outputs, independent of cwd."""
    project=Path(project).resolve(); root=(project/"outputs").resolve()
    if explicit is None:
        label="_".join(str(x) for x in (dataset,kind) if x)
        label=re.sub(r"[^A-Za-z0-9_-]","_",label)
        base=(root/category if category else root)/(timestamp()+"_"+label+suffix)
        result=base; serial=1
        while result.exists():
            result=base.with_name(base.stem+f"_{serial:02d}"+base.suffix);serial+=1
        return result
    path=Path(explicit)
    if path.is_absolute():
        result=path.resolve()
    else:
        if ".." in path.parts:
            raise ValueError("Output path must stay inside project/outputs.")
        if path.parts and path.parts[0].lower() in ("outputs","runs"):
            if path.parts[0].lower()=="runs":
                warnings.warn("runs/ output is deprecated; writing to outputs/ instead.",FutureWarning)
            path=Path(*path.parts[1:])
        result=(root/path).resolve()
    try:
        relative=result.relative_to(root)
    except ValueError:
        raise ValueError("All CLI outputs must be under "+str(root)) from None
    if not relative.parts:
        raise ValueError("Choose a new output subdirectory; do not overwrite the outputs root.")
    return result


def run_destination(project, explicit=None, *, dataset, kind="E0", category=None):
    """Name a new experiment YYYYMMDD_HHMMSS_DATASET_details (Asia/Taipei).

    A custom leaf is a run label, while an already formatted name is retained.
    This does not affect stable prepared-data/cache paths or historical artifacts.
    """
    if dataset not in ("NASDAQ", "NYSE", "SP500"):
        raise ValueError("A verified dataset is required for experiment output names.")
    if explicit is None:
        return destination(project, dataset=dataset, kind=kind, category=category)
    path = destination(project, explicit)
    matched = re.fullmatch(r"(\d{8}_\d{6})_(NASDAQ|NYSE|SP500)_(.+)", path.name)
    if matched:
        datetime.strptime(matched[1], "%Y%m%d_%H%M%S")
        if matched[2] != dataset:
            raise ValueError("Output directory dataset differs from the selected dataset.")
        return path
    label = re.sub(r"[^\w-]+", "_", path.name).strip("_")
    if not label:
        raise ValueError("Output directory requires a nonempty run label.")
    return path.with_name(timestamp() + "_" + dataset + "_" + kind + "_" + label)


def initialize(root):
    for name in DIRECTORIES:
        (Path(root)/name).mkdir(parents=True,exist_ok=True)


def copy_atomic(source,target):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest_file(source)==digest_file(target):
        return
    temporary=target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
    try:
        shutil.copy2(source,temporary);os.replace(temporary,target)
    finally:
        if temporary.exists():
            temporary.unlink()


class Tee:
    def __init__(self,console,log):
        self.console,self.log=console,log
    def write(self,text):
        for stream in (self.console,self.log):
            output=text if getattr(stream,"isatty",lambda:False)() else ANSI_RE.sub("",text)
            stream.write(output)
        self.flush();return len(text)
    def flush(self):
        self.console.flush();self.log.flush()
    def isatty(self):
        return bool(getattr(self.console,"isatty",lambda:False)())
    @property
    def encoding(self):
        return getattr(self.console,"encoding",None) or "utf-8"


def refresh_training(archive,root):
    archive,root=Path(archive),Path(root)
    initialize(root)
    mapping={"config.json":"configs/config.json","model.txt":"configs/model_summary.txt",
        "history.json":"tables/history.json","sample_ledger.json":"tables/sample_ledger.json","epoch_samples.json":"tables/epoch_samples.json",
        "run_status.json":"logs/run_status.json","best_model.pt":"checkpoints/best_model.pt",
        "last_model.pt":"checkpoints/last_model.pt","test_predictions.npz":"predictions/test_predictions.npz",
        "test_metrics.json":"tables/test_metrics.json"}
    for name,relative in mapping.items():
        if (archive/name).exists():
            copy_atomic(archive/name,root/relative)
    if (archive/"config.json").exists():
        config=json.loads((archive/"config.json").read_text(encoding="utf-8"))
        (root/"configs/config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    if (archive/"history.json").exists():
        history=json.loads((archive/"history.json").read_text(encoding="utf-8"))
        with (root/"tables/history.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            if history:
                fields=list(dict.fromkeys(key for row in history for key in row))
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(history)


def plot_history(root,history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not history:
        return
    epochs=[row["epoch"] for row in history]
    figure,axis=plt.subplots(figsize=(9,5))
    for key,label in (("train_loss","Train"),("val_loss","Validation"),("test_loss","Test")):
        if any(key in row for row in history):
            axis.plot(epochs,[row.get(key) for row in history],label=label)
    axis.set(xlabel="Epoch",ylabel="Loss",title="StockMixer strict protocol: loss")
    axis.legend();axis.grid(alpha=.25);figure.tight_layout()
    figure.savefig(Path(root)/"figures/loss_curve.png",dpi=150);plt.close(figure)
    figure,axis=plt.subplots(figsize=(9,5))
    if any("val_ic" in row for row in history):
        series=(("val_ic","Validation IC"),("test_ic","Test IC"),("val_ric","Validation RIC"),("test_ric","Test RIC"))
    else:
        series=(("val_IC","Validation IC"),("val_RankIC","Validation RankIC"))
    for key,label in series:
        if any(key in row for row in history):
            axis.plot(epochs,[row.get(key) for row in history],label=label)
    axis.axhline(0,color="grey",linewidth=.8)
    axis.set(xlabel="Epoch",ylabel="Correlation / ratio",title="StockMixer: IC and RIC")
    axis.legend();axis.grid(alpha=.25);figure.tight_layout()
    figure.savefig(Path(root)/"figures/ic_curve.png",dpi=150);plt.close(figure)
    if any("val_sharpe5" in row for row in history):
        figure,axis=plt.subplots(figsize=(9,5))
        for key,label in (("val_sharpe5","Validation SR"),("test_sharpe5","Test SR")):
            if any(key in row for row in history):
                axis.plot(epochs,[row.get(key) for row in history],label=label)
        axis.set(xlabel="Epoch",ylabel="Sharpe ratio",title="StockMixer: SR")
        axis.legend();axis.grid(alpha=.25);figure.tight_layout()
        figure.savefig(Path(root)/"figures/sr_curve.png",dpi=150);plt.close(figure)
    (Path(root)/"figures/README.md").write_text(
        "Curves use only recorded evaluations. Test is diagnostic; checkpoint selection uses validation loss only.\n",encoding="utf-8")



def bundle_files(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():digest_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p!=root/"manifest.json" and (not {"_artifact","_prediction_artifact"}.intersection(p.relative_to(root).parts) or p in (root/"_artifact/manifest.json",root/"_prediction_artifact/manifest.json"))}


def verify_bundle(root):
    root=Path(root).resolve()
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=SCHEMA or manifest.get("kind") not in ("training_bundle","prediction_bundle"):
        raise ValueError("Unsupported outputs bundle.")
    files=manifest.get("files",{})
    required={"_artifact/manifest.json","configs/config.json","tables/metrics.json"}
    required|={"checkpoints/best_model.pt","checkpoints/last_model.pt","tables/history.csv"} if manifest["kind"]=="training_bundle" else {"predictions/predictions.csv"}
    if not required<=set(files):
        raise ValueError("Output bundle omits required files.")
    for name,expected in files.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(expected,str) or re.fullmatch(r"[a-f0-9]{64}",expected) is None:
            raise ValueError("Invalid output bundle path/hash.")
        path=(root/Path(*relative.parts)).resolve()
        path.relative_to(root)
        if digest_file(path)!=expected:
            raise ValueError("Output bundle file hash mismatch: "+name)
    archive=root/"_artifact"
    original=json.loads((archive/"manifest.json").read_text(encoding="utf-8"))
    verify_files(archive,original)
    if digest_file(archive/"manifest.json")!=manifest["artifact_manifest_sha256"]:
        raise ValueError("Immutable artifact manifest mismatch.")
    pairs=(("checkpoints/best_model.pt","best_model.pt"),("checkpoints/last_model.pt","last_model.pt")) if manifest["kind"]=="training_bundle" else (("predictions/predictions.csv","predictions.csv"),)
    for visible,immutable in pairs:
        if digest_file(root/visible)!=digest_file(archive/immutable):
            raise ValueError("Display output differs from its immutable original.")
    if "prediction_artifact_manifest_sha256" in manifest:
        expected_files={"_prediction_artifact/manifest.json","predictions/predictions.csv","tables/prediction_metrics.json","logs/predict.log"}
        if manifest["kind"]!="training_bundle" or not expected_files<=set(files):
            raise ValueError("Integrated prediction omits required files.")
        prediction_archive=root/"_prediction_artifact"
        prediction=json.loads((prediction_archive/"manifest.json").read_text(encoding="utf-8"))
        verify_files(prediction_archive,prediction,required=("predictions.csv",))
        if digest_file(prediction_archive/"manifest.json")!=manifest["prediction_artifact_manifest_sha256"]:
            raise ValueError("Integrated prediction manifest mismatch.")
        if prediction.get("dataset")!=manifest.get("dataset") or prediction.get("checkpoint_sha256")!=digest_file(root/"checkpoints/best_model.pt"):
            raise ValueError("Integrated prediction belongs to another checkpoint/dataset.")
        if digest_file(root/"predictions/predictions.csv")!=digest_file(prediction_archive/"predictions.csv"):
            raise ValueError("Integrated prediction differs from its immutable original.")
    return manifest



def publish_training(archive,root,data_contract=None,console_log=None):
    archive,root=Path(archive),Path(root)
    if archive.resolve()!= (root/"_artifact").resolve():
        raise ValueError("A training bundle must contain its own immutable _artifact directory.")
    manifest=json.loads((archive/"manifest.json").read_text(encoding="utf-8"))
    verify_files(archive,manifest)
    refresh_training(archive,root)
    if data_contract is not None:
        write_json(root/"configs/data_contract.json",data_contract)
    if console_log is not None:
        copy_atomic(console_log,root/"logs/train.log")
    elif not (root/"logs/train.log").exists():
        (root/"logs/train.log").write_text("Historical experiment: no console log was retained. See tables/history.csv.\n",encoding="utf-8")
    summary={"selection_metric":"val_loss","selection_mode":"min","best_epoch":manifest["best_epoch"],
        "best_validation_loss":manifest["validation"]["loss"],"best_valid_perf":manifest["validation"],
        "best_test_perf":manifest["test"],"test_metrics":manifest["test"],"final_test_evaluations":manifest["final_test_evaluations"],
        "test_policy":manifest.get("test_policy","final_best_once"),"metric_protocol":manifest.get("metric_protocol","valid_only"),
        "test_evaluations":manifest.get("test_evaluations",manifest["final_test_evaluations"]),
        "epoch_test_evaluations":manifest.get("epoch_test_evaluations",0),
        "protocol":manifest.get("protocol","strict"),"smoke_only":manifest["smoke_only"]}
    write_json(root/"tables/metrics.json",summary)
    with (root/"tables/metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["metric","best_validation","final_test"])
        for key in ("loss","mse_loss","rank_loss","mse","IC","RIC","prec_10","sharpe5") if manifest.get("metric_protocol")=="stockmixer_original" else ("loss","mse_loss","rank_loss","mse","rmse","IC","RankIC","ICIR","RIC","prec_10","sharpe5"):
            writer.writerow([key,manifest["validation"].get(key),manifest["test"].get(key)])
    plot_history(root,json.loads((archive/"history.json").read_text(encoding="utf-8")))
    sys.stdout.flush();sys.stderr.flush()
    bundle={"schema_version":SCHEMA,"kind":"training_bundle","created_at":now(),"dataset":manifest["dataset"],
        "protocol":manifest.get("protocol","strict"),"artifact_manifest_sha256":digest_file(archive/"manifest.json"),
        "best_epoch":manifest["best_epoch"],"smoke_only":manifest["smoke_only"],"files":bundle_files(root)}
    write_json(root/"manifest.json",bundle)
    verify_bundle(root)
    return bundle


def publish_prediction(archive,root):
    archive,root=Path(archive),Path(root);initialize(root)
    manifest=json.loads((archive/"manifest.json").read_text(encoding="utf-8"));verify_files(archive,manifest)
    copy_atomic(archive/"predictions.csv",root/"predictions/predictions.csv")
    config=manifest.get("config",{})
    write_json(root/"configs/config.json",config)
    (root/"configs/config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    write_json(root/"tables/metrics.json",{k:v for k,v in manifest.items() if k not in ("files","config")})
    (root/"logs/predict.log").write_text(json.dumps({k:manifest[k] for k in ("dataset","as_of_date","target_date","rows","valid_rows")},ensure_ascii=False,indent=2),encoding="utf-8")
    bundle={"schema_version":SCHEMA,"kind":"prediction_bundle","created_at":now(),"dataset":manifest["dataset"],
        "artifact_manifest_sha256":digest_file(archive/"manifest.json"),"files":bundle_files(root)}
    write_json(root/"manifest.json",bundle);verify_bundle(root);return bundle


def publish_training_prediction(archive,root):
    root=Path(root).resolve();archive=Path(archive).resolve()
    if archive!=root/"_prediction_artifact":
        raise ValueError("Integrated forecast must reside in its own training folder.")
    bundle=verify_bundle(root)
    if bundle["kind"]!="training_bundle" or "prediction_artifact_manifest_sha256" in bundle:
        raise ValueError("Expected a training bundle without an existing forecast.")
    prediction=json.loads((archive/"manifest.json").read_text(encoding="utf-8"))
    verify_files(archive,prediction,required=("predictions.csv",))
    training=json.loads((root/"_artifact/manifest.json").read_text(encoding="utf-8"))
    for key in ("dataset","protocol","data_manifest_sha256","stock_axis_hash","date_axis_hash"):
        if prediction.get(key)!=training.get(key):
            raise ValueError("Forecast metadata differs from training: "+key)
    if prediction.get("checkpoint_sha256")!=digest_file(root/"checkpoints/best_model.pt"):
        raise ValueError("Forecast must use this run's selected best checkpoint.")
    if (root/"predictions/predictions.csv").exists():
        raise FileExistsError("This training run already has a forecast.")
    copy_atomic(archive/"predictions.csv",root/"predictions/predictions.csv")
    write_json(root/"tables/prediction_metrics.json",{k:v for k,v in prediction.items() if k not in ("files","config")})
    write_json(root/"logs/predict.log",{k:prediction[k] for k in ("dataset","as_of_date","target_date","rows","valid_rows")})
    bundle["prediction_artifact_manifest_sha256"]=digest_file(archive/"manifest.json")
    bundle["files"]=bundle_files(root)
    temporary=root/(".manifest-"+uuid.uuid4().hex+".tmp")
    try:
        write_json(temporary,bundle)
        os.replace(temporary,root/"manifest.json")
    finally:
        if temporary.exists():
            temporary.unlink()
    verify_bundle(root)
    return bundle
