"""Offline CLI. This module intentionally provides no training/testing-loop entry point."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import yaml
from .io import digest_file, read_records
from .time_axis import Axis, Schedule, export_schedule

def main():
    parser=argparse.ArgumentParser(description="Standalone FinBERT/news panels; original StockMixer remains unchanged.")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config",type=Path,default=None)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("inspect")
    adapt=commands.add_parser("adapt",help="Read-only collector JSON/audit adaptation; no model inference.")
    adapt.add_argument("--source-root",type=Path,required=True)
    adapt.add_argument("--output",type=Path,required=True)
    adapt.add_argument("--markets",nargs="+",choices=("US","GLOBAL"),default=["GLOBAL","US"])
    adapted=commands.add_parser("validate-adapted",help="Verify adapted artifact and unchanged producer inputs.")
    adapted.add_argument("--artifact",type=Path,required=True)
    adapted.add_argument("--source-root",type=Path)
    encoder=commands.add_parser("encode")
    encoder.add_argument("--model-dir",type=Path,required=True)
    encoder.add_argument("--output",type=Path,required=True)
    encoder.add_argument("--cache",type=Path,required=True)
    encoder.add_argument("--input",type=Path,nargs="+")
    encoder.add_argument("--device",default="cpu")
    encoder.add_argument("--limit",type=int)
    encoder.add_argument("--format",choices=("jsonl","parquet"),default="jsonl")
    encoder.add_argument("--model-available-at",default=None)
    calendar=commands.add_parser("schedule")
    calendar.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
    calendar.add_argument("--output",type=Path,required=True)
    build=commands.add_parser("build")
    build.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
    build.add_argument("--features",type=Path,required=True)
    build.add_argument("--schedule",type=Path,required=True)
    build.add_argument("--links",type=Path)
    build.add_argument("--memberships",type=Path)
    build.add_argument("--output",type=Path,required=True)
    build.add_argument("--projection",choices=("pca","random_projection"),default=None)
    check=commands.add_parser("validate")
    check.add_argument("--panel",type=Path,required=True)
    args=parser.parse_args()
    project=args.project_root.resolve()
    config_path=args.config or project/"configs/news_pipeline.yaml"
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.command=="adapt":
        from .source_adapter import adapt_snapshot
        result=adapt_snapshot(args.source_root,args.output,project/"news_assets/axes",args.markets,
            progress=lambda current,total: print(f"[ADAPT] checked={current}/{total}",file=sys.stderr,flush=True))
        print(json.dumps({k:result[k] for k in ("unique_articles","ready_articles","review_required",
            "duplicate_views_removed","audit_family_counts","llm_time_known","datasets")},indent=2))
    elif args.command=="validate-adapted":
        from .source_adapter import validate_adapted
        print(json.dumps(validate_adapted(args.artifact,args.source_root),indent=2))
    elif args.command=="inspect":
        root=(project/config["news_root"]).resolve()
        news=list((root/"US").rglob("*.json"))+list((root/"GLOBAL").rglob("*.json"))
        news+=list((root/"US").rglob("*.jsonl"))+list((root/"GLOBAL").rglob("*.jsonl"))
        axes={}
        for name in ("NASDAQ","NYSE","SP500"):
            axis=Axis.load(project/"news_assets/axes"/(name+".json"))
            stale=[]
            for relative,expected in axis.source["source_sha256"].items():
                if digest_file(project/relative)!=expected:
                    stale.append(relative)
            axes[name]={"stocks":len(axis.stock_uids),"dates":len(axis.dates),
                "range":[axis.dates[0],axis.dates[-1]],"source_hash_mismatches":stale}
        print(json.dumps({"news_files":len(news),"axes":axes,"dependencies":{
            name:importlib.util.find_spec(name) is not None for name in ("torch","transformers","pyarrow","exchange_calendars")},
            "input_available":bool(news),"real_encoding_run":False,"baseline_modified":False},indent=2))
    elif args.command=="encode":
        from .inputs import load_articles
        from .encoder import LocalFinBert, encode_articles
        paths=args.input
        if paths is None:
            root=(project/config["news_root"]).resolve()
            paths=sorted(p for scope in ("US","GLOBAL") for p in (root/scope).rglob("*")
                         if p.suffix in (".json",".jsonl",".parquet"))
        articles,counts=load_articles(paths)
        if args.limit is not None:
            if args.limit<1:
                raise ValueError("limit must be positive.")
            articles=articles[:args.limit]
        backend=LocalFinBert(args.model_dir,args.device,config["finbert"]["max_length"],args.model_available_at)
        result=encode_articles(articles,backend,args.output,args.cache,
            batch_size=config["finbert"]["batch_size"],table_format=args.format)
        print(json.dumps({"input_counts":counts,"successful":result["successful"],"status_counts":result["status_counts"]},indent=2))
    elif args.command in ("schedule","build"):
        axis=Axis.load(project/"news_assets/axes"/(args.dataset+".json"))
        for relative,expected in axis.source["source_sha256"].items():
            if digest_file(project/relative)!=expected:
                raise ValueError("Market data changed since axis verification; regenerate/reverify axis first.")
        if args.command=="schedule":
            export_schedule(axis,args.output)
            print("Decision schedule exported; original data unchanged.")
        else:
            from .panel import build_panel
            schedule=Schedule(axis,read_records(args.schedule))
            settings=config["panel"]
            result=build_panel(axis,schedule,args.features,
                read_records(args.links) if args.links else [],read_records(args.memberships) if args.memberships else [],
                args.output,mode=settings["mode"],projection=args.projection or settings["projection"],
                lookback_sessions=settings["lookback_sessions"],tau=settings["decay_tau_sessions"],
                latency_seconds=settings["simulated_latency_seconds"],seed=settings["projection_seed"])
            print(json.dumps({"dataset":result["dataset"],"mode":result["mode"],"coverage":result["coverage"]},indent=2))
    else:
        from .dataset import NewsPanel
        panel=NewsPanel(args.panel)
        print(json.dumps({"dataset":panel.axis.dataset,"coverage":panel.manifest["coverage"],"valid":True},indent=2))

if __name__=="__main__":
    try:
        main()
    except (ValueError,OSError,ImportError) as error:
        print("News pipeline stopped: "+str(error),file=sys.stderr)
        sys.exit(2)
