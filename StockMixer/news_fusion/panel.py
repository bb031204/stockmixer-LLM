"""Causal, independently masked company/industry/market panels; no training code."""
import json
from datetime import date as date_type, datetime
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .encoder import validate_outputs
from .io import SCHEMA, digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .time_axis import timestamp
from .validation import integer, finite_number

FEATURE_NAMES=["p_positive","p_negative","p_neutral"]+[f"embedding_{i}" for i in range(32)]+[
    "log1p_article_count","mean_age_sessions","inferred_share","mean_importance","importance_available_ratio"]
CHANNELS=("company","industry","market")

def load_features(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=SCHEMA or manifest.get("kind")!="article_features":
        raise ValueError("Unsupported article feature schema.")
    verify_files(directory,manifest,required=(manifest.get("record_table",""),"article_embeddings_fp16.npy"))
    records=read_records(directory/manifest["record_table"])
    if len({r['source_id'] for r in records})!=len(records):
        raise ValueError("Duplicate source_id in feature table.")
    vectors=np.load(directory/"article_embeddings_fp16.npy",mmap_mode="r",allow_pickle=False)
    success=[r for r in records if r["finbert_status"]=="success"]
    if manifest.get("articles")!=len(records) or manifest.get("successful")!=len(success):
        raise ValueError("Article feature manifest counts differ from its record table.")
    for row in success:
        value=row.get("embedding_row")
        if isinstance(value,bool) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or int(value)!=value:
            raise ValueError("Embedding row must be a finite integer index.")
        row["embedding_row"]=int(value)
    if vectors.shape!=(len(success),768) or sorted(r["embedding_row"] for r in success)!=list(range(len(success))):
        raise ValueError("Embedding row/index mismatch.")
    for r in success:
        if r["model_revision"]!=manifest["encoder"]["model_revision"] or r.get("model_id")!=manifest["encoder"].get("model_id"):
            raise ValueError("Mixed model revisions in one article feature artifact.")
        probs=np.array([[r["p_positive"],r["p_negative"],r["p_neutral"]]],dtype=np.float32)
        validate_outputs(probs,np.asarray(vectors[r["embedding_row"]])[None],1)
    return records,vectors,manifest

def relationship_valid(link,date,published_date,cutoff,mode):
    if link.get("verification_status")!="verified" or not link.get("mapping_source"):
        return False
    def day(value):
        if value is None or (isinstance(value,str) and not value):
            return None
        if isinstance(value,(date_type,datetime)):
            return value.isoformat()[:10]
        return value
    start,end=day(link.get("valid_from")),day(link.get("valid_to"))
    if not isinstance(start,str) or len(start)!=10:
        return False
    try:
        date_type.fromisoformat(start)
        if end is not None:
            if not isinstance(end,str) or len(end)!=10:
                return False
            date_type.fromisoformat(end)
    except ValueError:
        return False
    if not (start<=date and start<=published_date and (not end or (date<end and published_date<end))):
        return False
    if mode=="observed_pit":
        try:
            if timestamp(link.get("known_from"))>=cutoff:
                return False
            if link.get("known_to") and cutoff>=timestamp(link["known_to"]):
                return False
        except ValueError:
            return False
    return True

def compress(vectors,eligible,components=32,method="pca",seed=1234):
    if method=="random_projection":
        matrix=np.random.RandomState(seed).normal(size=(components,768)).astype(np.float32)/np.sqrt(components)
        return np.zeros(768,dtype=np.float32),matrix,{"fit_mode":"fixed_random_projection","seed":seed,"fitted_articles":0}
    if method!="pca":
        raise ValueError("Compression must be pca or explicit random_projection.")
    selected=np.asarray(vectors[eligible],dtype=np.float32)
    if len(selected)<=components:
        raise ValueError("Insufficient training-only eligible articles for PCA; provide more data or explicitly select random_projection.")
    mean=selected.mean(0)
    _,singular,axes=np.linalg.svd(selected-mean,full_matrices=False)
    if singular[components-1]<=max(singular[0]*1e-6,1e-8):
        raise ValueError("Training PCA rank is insufficient; no silent fallback.")
    # Fix signs for a stable serialized basis within this implementation/environment.
    matrix=axes[:components]
    signs=np.sign(matrix[np.arange(components),np.argmax(np.abs(matrix),axis=1)])
    matrix=matrix*signs[:,None]
    return mean,matrix,{"fit_mode":"train_only_pca","fitted_articles":len(selected)}

def build_panel(axis,schedule,feature_dir,links,memberships,output,mode="retrospective",
                projection="pca",lookback_sessions=5,tau=3.0,latency_seconds=0,seed=1234,decision_lookback=16):
    if schedule.axis.stock_hash!=axis.stock_hash or schedule.axis.date_hash!=axis.date_hash:
        raise ValueError("Schedule belongs to another axis.")
    integer(lookback_sessions,"lookback_sessions",1)
    integer(decision_lookback,"decision_lookback",1,axis.train_end-1)
    finite_number(tau,"decay tau",strictly_positive=True)
    finite_number(latency_seconds,"simulated latency",minimum=0)
    articles,vectors,feature_manifest=load_features(feature_dir)
    if not len(vectors):
        raise ValueError("No successfully encoded articles; a real panel cannot be built.")
    ids={a["source_id"] for a in articles}
    by_article=defaultdict(list)
    for link in links:
        if link.get("source_id") not in ids:
            raise ValueError("Relation source_id absent from the feature snapshot.")
        by_article[link["source_id"]].append(link)
    by_industry=defaultdict(list)
    for member in memberships:
        if member.get("dataset")==axis.dataset:
            key=tuple(member.get(k) for k in ("industry_standard","industry_version","industry_code"))
            if all(key):
                by_industry[key].append(member)
    stock_index={uid:i for i,uid in enumerate(axis.stock_uids)}
    groups=defaultdict(dict)
    eligible=set()
    rejections=Counter()
    accepted_ids=set()
    # Stable order ensures first-available copy wins content dedup at each decision, not globally in hindsight.
    ordered=[]
    for a in articles:
        if a["finbert_status"]!="success":
            rejections[a["finbert_status"]]+=1
            continue
        if a.get("market") not in ("US","GLOBAL"):
            rejections["unsupported_market"]+=1
            continue
        try:
            first=schedule.first_session(a,mode,latency_seconds)
            published_session=schedule.first_session(a,"retrospective")
        except ValueError as exc:
            rejections[str(exc)]+=1
            continue
        if first>=len(axis.dates):
            rejections["available_after_dataset"]+=1
            continue
        pub=a["published_at"]
        publication_date=pub[:10] if len(pub)==10 else timestamp(pub).tz_convert("America/New_York").strftime("%Y-%m-%d")
        if publication_date<axis.dates[0]:
            # The dataset has no prehistory sessions to measure age reliably; do not revive stale news at index zero.
            rejections["publication_before_dataset_start"]+=1
            continue
        if first>=published_session+lookback_sessions:
            rejections["stale_before_usable"]+=1
            continue
        ordered.append((first,a["source_id"],published_session,a))
    for first,_,published_session,a in sorted(ordered,key=lambda item:(item[0],item[1])):
        relations=[]
        for link in by_article.get(a["source_id"],[]):
            channel=link.get("channel")
            if channel=="company" and link.get("dataset")==axis.dataset and link.get("stock_uid") in stock_index:
                relations.append(("company",stock_index[link["stock_uid"]],link,link))
            elif channel=="industry" and link.get("dataset") in (None,axis.dataset) and link.get("verification_status")=="verified":
                key=tuple(link.get(k) for k in ("industry_standard","industry_version","industry_code"))
                if all(key) and link.get("evidence_type") in ("explicit","inferred"):
                    for member in by_industry.get(key,[]):
                        if member.get("stock_uid") in stock_index:
                            relations.append(("industry",stock_index[member["stock_uid"]],link,member))
        if "market" in a.get("impact_scope",[]):
            relations.append(("market",-1,None,None))
        if not relations:
            rejections["unmapped_article"]+=1
        for t in range(first,min(published_session+lookback_sessions,len(axis.dates))):
            for channel,stock,link,mapping in relations:
                date,cutoff=axis.dates[t],schedule.cutoffs[t]
                publication=a["published_at"]
                pub=publication[:10] if len(publication)==10 else timestamp(publication).tz_convert("America/New_York").strftime("%Y-%m-%d")
                if link is not None:
                    if not relationship_valid(link,date,pub,cutoff,mode):
                        rejections["invalid_or_unavailable_relation"]+=1
                        continue
                    if mapping is not link and not relationship_valid(mapping,date,pub,cutoff,mode):
                        rejections["invalid_or_unavailable_membership"]+=1
                        continue
                entry={"article":a,"age":t-published_session,"first_session":first,
                    "inferred":bool(link and link.get("evidence_type")=="inferred")}
                key=(channel,stock,t)
                content_hash=a["input_text_hash"]
                existing=groups[key].get(content_hash)
                if existing:
                    # Prefer explicit over inferred when the same article reaches a stock by multiple mappings.
                    existing["inferred"]=existing["inferred"] and entry["inferred"]
                    continue
                groups[key][content_hash]=entry
                accepted_ids.add(a["source_id"])
                # Last training label is train_end-1; its decision is train_end-2 for steps=1.
                if decision_lookback-1<=t<axis.train_end-1:
                    eligible.add(a["embedding_row"])
    if not groups:
        raise ValueError("No eligible article-channel pairs for this dataset/time mode; no empty real panel is published.")
    eligible=sorted(eligible)
    mean,matrix,transform_info=compress(vectors,eligible,method=projection,seed=seed)
    compact=(np.asarray(vectors,dtype=np.float32)-mean)@matrix.T
    if not np.isfinite(compact).all():
        raise ValueError("Nonfinite compressed embeddings.")
    n,total=len(axis.stock_uids),len(axis.dates)
    coverage={}
    with new_artifact(output) as directory:
        data,masks={},{}
        for channel in CHANNELS:
            shape=(total,40) if channel=="market" else (n,total,40)
            data[channel]=np.lib.format.open_memmap(directory/(channel+"_news_daily.npy"),mode="w+",dtype="float32",shape=shape)
            masks[channel]=np.lib.format.open_memmap(directory/(channel+"_news_mask.npy"),mode="w+",dtype="uint8",shape=shape[:-1])
            data[channel][:]=0
            masks[channel][:]=0
        with (directory/"provenance.jsonl").open("w",encoding="utf-8") as ledger:
            for (channel,stock,t),unique in sorted(groups.items()):
                entries=list(unique.values())
                weights=np.exp(-np.array([e["age"] for e in entries])/tau)
                weights/=weights.sum()
                arows=[e["article"] for e in entries]
                row=np.zeros(40,dtype=np.float32)
                row[:3]=weights@np.array([[a["p_positive"],a["p_negative"],a["p_neutral"]] for a in arows])
                row[3:35]=weights@compact[[a["embedding_row"] for a in arows]]
                row[35]=np.log1p(len(entries))
                row[36]=np.mean([e["age"] for e in entries])
                row[37]=np.mean([e["inferred"] for e in entries])
                importance=[float(a["importance"]) for a in arows if a.get("importance_verified") is True
                    and isinstance(a.get("importance"),(int,float)) and np.isfinite(a["importance"]) and 0<=a["importance"]<=1]
                row[38]=np.mean(importance) if importance else 0
                row[39]=len(importance)/len(entries)
                loc=t if channel=="market" else (stock,t)
                data[channel][loc]=row
                masks[channel][loc]=1
                for e,w in zip(entries,weights):
                    ledger.write(json.dumps({"source_id":e["article"]["source_id"],"channel":channel,
                        "stock_uid":None if stock<0 else axis.stock_uids[stock],"decision_idx":t,
                        "first_available_session":e["first_session"],"weight":float(w),"inferred":e["inferred"]})+"\n")
        for channel in CHANNELS:
            data[channel].flush()
            masks[channel].flush()
            coverage[channel]={"active_cells":int(masks[channel].sum()),"total_cells":int(masks[channel].size)}
        # Release Windows memory-map handles before publishing the directory.
        del data,masks
        np.savez(directory/"embedding_transform.npz",mean=mean,components=matrix)
        np.save(directory/"trading_dates.npy",np.asarray(axis.dates,dtype="U10"))
        write_json(directory/"axis.json",axis.source)
        write_records(directory/"decision_schedule.jsonl",schedule.records)
        transform_info.update(dataset=axis.dataset,training_decision_end_exclusive=axis.train_end-1,
            training_decision_start_inclusive=decision_lookback-1,
            eligible_training_articles=len(eligible),fit_source_ids_hash=digest(sorted(a["source_id"] for a in articles
                if a.get("embedding_row") in eligible)))
        manifest={"schema_version":SCHEMA,"kind":"daily_news_panel","created_at":now(),
            "dataset":axis.dataset,"stock_axis_hash":axis.stock_hash,"trading_dates_hash":axis.date_hash,
            "mode":mode,"historical_model_replay":mode!="observed_pit","feature_names":FEATURE_NAMES,
            "availability_comparison":"strictly_before_cutoff",
            "article_encoder":feature_manifest["encoder"],
            "market_routing_verified":False,
            "feature_dim":40,"n_stocks":n,"n_dates":total,"coverage":coverage,
            "lookback_sessions":lookback_sessions,"decay_tau_sessions":tau,"simulated_latency_seconds":latency_seconds,
            "age_basis":"first publication-eligible session, not delayed encoding completion",
            "prehistory_policy":"exclude publications before first dataset date; no age clipping to zero",
            "transform":transform_info,"article_artifact_manifest_sha256":digest_file(Path(feature_dir)/"manifest.json"),
            "relations_hash":digest(links),"memberships_hash":digest(memberships),
            "articles_accepted":len(accepted_ids),"rejections":dict(rejections),
            "baseline_training_modified":False,"news_sample_index":"decision_idx=offset+lookback-1; steps=1",
            "source_code_sha256":{p.name:digest_file(p) for p in Path(__file__).parent.glob("*.py")},
            "files":file_manifest(directory)}
        write_json(directory/"manifest.json",manifest)
    return manifest
