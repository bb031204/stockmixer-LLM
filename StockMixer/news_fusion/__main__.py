"""Offline CLI. This module intentionally provides no training/testing-loop entry point."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import yaml
from .io import digest_file, read_records
from .time_axis import Axis, Schedule, export_schedule
from .outputs import destination
from .configuration import load_llm_config

def main(argv=None):
    parser=argparse.ArgumentParser(description="Standalone FinBERT/news panels; original StockMixer remains unchanged.")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config",type=Path,default=None)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("inspect")
    adapt=commands.add_parser("adapt",help="Read-only collector JSON/audit adaptation; no model inference.")
    adapt.add_argument("--source-root",type=Path,required=True)
    adapt.add_argument("--output",type=Path)
    adapt.add_argument("--markets",nargs="+",choices=("US","GLOBAL"),default=["GLOBAL","US"])
    adapted=commands.add_parser("validate-adapted",help="Verify adapted artifact and unchanged producer inputs.")
    adapted.add_argument("--artifact",type=Path,required=True)
    adapted.add_argument("--source-root",type=Path)
    encoder=commands.add_parser("encode")
    encoder.add_argument("--model-dir",type=Path)
    encoder.add_argument("--output",type=Path)
    encoder.add_argument("--cache",type=Path)
    encoder.add_argument("--input",type=Path,nargs="+")
    encoder.add_argument("--device",default=None)
    encoder.add_argument("--limit",type=int)
    encoder.add_argument("--format",choices=("jsonl","parquet"),default=None)
    encoder.add_argument("--model-available-at",default=None)
    encoder.add_argument("--check-input",action="store_true",help="Verify frozen inputs without loading FinBERT.")
    for command in ("routes-template","routes-publish"):
        route=commands.add_parser(command)
        route.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
        route.add_argument("--output",type=Path)
        if command=="routes-publish":
            route.add_argument("--review",type=Path,required=True)
            route.add_argument("--allow-partial",action="store_true")
    auto=commands.add_parser("routes-auto",help="Local Qwen review with evidence checks and versioned cache.")
    auto.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
    auto.add_argument("--output",type=Path)
    auto.add_argument("--limit",type=int)
    auto.add_argument("--source-id",action="append")
    auto.add_argument("--check-model",action="store_true")
    calendar=commands.add_parser("schedule")
    calendar.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
    calendar.add_argument("--output",type=Path)
    build=commands.add_parser("build")
    build.add_argument("--dataset",choices=("NASDAQ","NYSE","SP500"),required=True)
    build.add_argument("--features",type=Path,required=True)
    build.add_argument("--schedule",type=Path,required=True)
    build.add_argument("--links",type=Path)
    build.add_argument("--memberships",type=Path)
    build.add_argument("--market-routes",type=Path,help="Published evidence-bound market review artifact.")
    build.add_argument("--output",type=Path)
    build.add_argument("--projection",choices=("pca","random_projection"),default=None)
    check=commands.add_parser("validate")
    check.add_argument("--panel",type=Path,required=True)
    args=parser.parse_args(argv)
    project=args.project_root.resolve()
    if args.command in ("adapt","encode","schedule","build","routes-template","routes-publish","routes-auto"):
        category={"adapt":"artifacts/news/adapted","encode":"artifacts/news/article_features",
                  "schedule":"artifacts/schedules","build":"artifacts/news/panels",
                  "routes-template":"artifacts/news/routing_reviews","routes-publish":"artifacts/news/market_routes",
                  "routes-auto":"artifacts/news/routing_reviews"}[args.command]
        args.output=destination(project,args.output,dataset=getattr(args,"dataset",None),kind=args.command,
            category=category,suffix=".jsonl" if args.command=="schedule" else "")
    if args.command=="encode":
        args.cache=destination(project,args.cache or project/"outputs/artifacts/news/cache",kind="cache")
    config_path=args.config or project/"configs/llm.yaml"
    if not config_path.is_absolute():
        config_path=project/config_path
    config=load_llm_config(config_path)
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
    elif args.command=="routes-auto":
        from .auto_review import OllamaReviewer,auto_review
        from .offline_inputs import encoding_inputs
        settings=config["review"];backend=OllamaReviewer(settings)
        if args.check_model:
            print(json.dumps({"available":True,"model":backend.metadata,"model_inference_run":False},indent=2))
            return
        articles,_,provenance=encoding_inputs(project,config)
        cache=destination(project,settings["cache_dir"],kind="review-cache")
        result=auto_review(articles,args.dataset,args.output,cache,backend,settings,provenance,args.limit,args.source_id,
            progress=lambda n,total,stats:print(f"[REVIEW] {n}/{total} cache={stats.get('cache_hits',0)} calls={stats.get('model_calls',0)} failed={stats.get('failed_articles',0)}",file=sys.stderr,flush=True))
        print(json.dumps({"output":str(args.output),"review_file":str(args.output/"market_review.jsonl"),
            "decisions":result["decisions"],"stats":result["stats"],"human_sample_checked":False},indent=2))
    elif args.command in ("routes-template","routes-publish"):
        from .offline_inputs import encoding_inputs
        from .routing import review_template,publish_routes
        articles,_,provenance=encoding_inputs(project,config)
        if args.command=="routes-template":
            result=review_template(articles,args.dataset,args.output,provenance)
        else:
            review=args.review if args.review.is_absolute() else project/args.review
            result=publish_routes(articles,read_records(review),args.dataset,args.output,provenance,args.allow_partial)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=="encode":
        from .offline_inputs import encoding_inputs
        from .encoder import LocalFinBert, encode_articles
        articles,counts,provenance=encoding_inputs(project,config,args.input)
        if args.limit is not None:
            if args.limit<1:
                raise ValueError("limit must be positive.")
            articles=articles[:args.limit]
        if args.check_input:
            print(json.dumps({"input_counts":counts,"selected_articles":len(articles),"input_provenance":provenance,"model_inference_run":False},indent=2))
            return
        model_dir=args.model_dir or config["finbert"]["model_dir"]
        if not model_dir:
            raise ValueError("Configure finbert.model_dir in configs/llm.yaml before offline encoding.")
        model_dir=Path(model_dir)
        if not model_dir.is_absolute():
            model_dir=project/model_dir
        backend=LocalFinBert(model_dir,args.device or config["finbert"]["device"],config["finbert"]["max_length"],args.model_available_at)
        result=encode_articles(articles,backend,args.output,args.cache,
            batch_size=config["finbert"]["batch_size"],table_format=args.format or config["finbert"]["output_format"],input_provenance=provenance)
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
            market_routes=args.market_routes or config["market_routes"][args.dataset]
            if market_routes is not None:
                market_routes=Path(market_routes)
                if not market_routes.is_absolute():
                    market_routes=project/market_routes
            if "market" in config["fusion"]["enabled_channels"] and market_routes is None:
                raise ValueError("Configure market_routes."+args.dataset+" or pass --market-routes with a published review artifact.")
            result=build_panel(axis,schedule,args.features,
                read_records(args.links) if args.links else [],read_records(args.memberships) if args.memberships else [],
                args.output,mode=settings["mode"],projection=args.projection or settings["projection"],
                lookback_sessions=settings["lookback_sessions"],tau=settings["decay_tau_sessions"],
                latency_seconds=settings["simulated_latency_seconds"],seed=settings["projection_seed"],
                decision_lookback=settings.get("decision_lookback",16),market_routes=market_routes,components=settings['projection_components'])
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
