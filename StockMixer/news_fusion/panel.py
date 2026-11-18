"""Causal, independently masked company/industry/market panels; no training code."""
import json
import shutil
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

def relationship_valid(link,date,published_date,cutoff,mode,relation_time_basis="decision_and_event"):
    if link.get('verification_status') == 'reference_only':
        return (mode == 'retrospective' and link.get('membership_basis') == 'undated_dataset_reference'
                and link.get('historical_validity_verified') is False and bool(link.get('mapping_source')))
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
    if relation_time_basis not in ("decision_and_event","event_date"):
        raise ValueError("Unknown relation time basis")
    applies=start<=published_date and (not end or published_date<end)
    if relation_time_basis=="decision_and_event":
        applies=applies and start<=date and (not end or date<end)
    else:
        applies=applies and date>=published_date
    if not applies:
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
                projection="pca",lookback_sessions=5,tau=3.0,latency_seconds=0,seed=1234,decision_lookback=16,market_routes=None,relations_artifact=None,components=32,relation_time_basis="decision_and_event",market_channel=None):
    if schedule.axis.stock_hash!=axis.stock_hash or schedule.axis.date_hash!=axis.date_hash:
        raise ValueError("Schedule belongs to another axis.")
    if relation_time_basis not in ("decision_and_event","event_date"):
        raise ValueError("Unknown relation time basis")
    integer(lookback_sessions,"lookback_sessions",1)
    integer(decision_lookback,"decision_lookback",1,axis.train_end-1)
    finite_number(tau,"decay tau",strictly_positive=True)
    finite_number(latency_seconds,"simulated latency",minimum=0)
    integer(components,'projection_components',1,768)
    width=components+8
    stats_start=components+3
    feature_names=FEATURE_NAMES[:3]+['embedding_'+str(i) for i in range(components)]+FEATURE_NAMES[-5:]
    articles,vectors,feature_manifest=load_features(feature_dir)
    if not len(vectors):
        raise ValueError("No successfully encoded articles; a real panel cannot be built.")
    relations_manifest=None
    relations_manifest_sha=None
    if relations_artifact is not None:
        from .relations import load_relations
        relations_manifest_sha=digest_file(Path(relations_artifact)/'manifest.json')
        resolved_links,resolved_members,relations_manifest=load_relations(relations_artifact,axis)
        if digest(links)!=digest(resolved_links) or digest(memberships)!=digest(resolved_members):
            raise ValueError('Panel links/memberships differ from frozen historical relation proof')
        if relations_manifest.get('source_input',{}).get('snapshot_manifest_sha256')!=feature_manifest.get('input_provenance',{}).get('snapshot_manifest_sha256'):
            raise ValueError('Relation and FinBERT snapshots differ')
    reference_only = any(m.get('verification_status')=='reference_only' for m in memberships)
    if reference_only and mode!='retrospective':
        raise ValueError('Undated reference industry membership is retrospective-only')
    reviewed_market=None
    routing_manifest=None
    routing_source_hash=None
    if market_routes is not None:
        from .routing import feature_routes
        routing_source_hash=digest_file(Path(market_routes)/"manifest.json")
        reviewed_market,routing_manifest=feature_routes(market_routes,axis,articles,feature_manifest)
        if digest_file(Path(market_routes)/"manifest.json")!=routing_source_hash:
            raise ValueError("Market review changed while being loaded.")
    # 市场通道专项（M2/M4/M5）；market_channel=None 时完全保持旧行为。
    quality=None;official_routes=[];official_info=None;route_bases={};news_routes_total=None
    if market_channel is not None:
        from . import market_quality as mq
        mq.validate_settings(market_channel)
        if reviewed_market is None:
            raise ValueError('market_channel requires a published market-route review artifact')
        # FinBERT 特征表不含正文；证据来源按已绑定审核的标题/摘要判定。
        from .routing import load_routes
        # 审核绑定只保存标题/摘要；summary_origin 等来源声明取自同版本的 FinBERT 特征记录。
        feature_by_id={a['source_id']:a for a in articles}
        bound_text={b['source_id']:dict(feature_by_id.get(b['source_id'],{}),**b)
                    for b in load_routes(market_routes,axis.dataset)[1]}
        news_routes_total=len(reviewed_market)
        reviewed_market,route_bases=mq.filter_routes(reviewed_market,bound_text,set(market_channel['evidence_basis']))
        if not market_channel['news_routes']:
            reviewed_market=[]
        news_routes_kept=len(reviewed_market)
        if market_channel['official_events'] is not None:
            from .official_events import load_encoded
            o_records,o_vectors,official_routes,official_info=load_encoded(market_channel['official_events'],axis.dataset)
            encoder,other=feature_manifest['encoder'],official_info['encoder']
            if any(encoder.get(k)!=other.get(k) for k in ('model_id','model_revision','max_length','embedding_method','preprocessing_version')):
                raise ValueError('Official events were encoded with a different FinBERT configuration')
            if {a['source_id'] for a in articles}&{r['source_id'] for r in o_records}:
                raise ValueError('Official event source_id collides with a news article')
            offset=len(vectors)
            o_records=[dict(r,embedding_row=r['embedding_row']+offset) if r['finbert_status']=='success' else dict(r) for r in o_records]
            vectors=np.concatenate([np.asarray(vectors,dtype=np.float32),np.asarray(o_vectors,dtype=np.float32)])
            articles=articles+o_records
            reviewed_market=reviewed_market+official_routes
    market_per_stock=any(any(c['scope']!='US_EQUITY' for c in r.get('scope_claims',[])) for r in reviewed_market or [])
    market_securities=[]
    if market_per_stock:
        if relations_artifact is None:raise ValueError('Exchange scope requires frozen historical security proof')
        from .relations import validate_master
        market_securities,_,_=validate_master(axis,read_records(Path(relations_artifact)/'security_master.jsonl'),[],[])
    ids={a["source_id"] for a in articles}
    article_by_id={a['source_id']:a for a in articles}
    by_article=defaultdict(list)
    for link in links:
        if link.get("channel")=="market":
            raise ValueError("Market links must come from a published market-route review, not --links.")
        if link.get("source_id") not in ids:
            raise ValueError("Relation source_id absent from the feature snapshot.")
        article=article_by_id[link['source_id']]
        for key,article_key in (('article_record_version','record_version'),('article_input_text_hash','input_text_hash')):
            if key in link and link[key]!=article.get(article_key):
                raise ValueError('Relation is bound to a different article version')
        if link.get('channel')=='company' and link.get('dataset')==axis.dataset and 'array_index' in link:
            from .relations import axis_identity
            axis_identity(axis,link)
        by_article[link["source_id"]].append(link)
    if reviewed_market is not None:
        for route in reviewed_market:
            by_article[route["source_id"]].append(route)
    by_industry=defaultdict(list)
    for member in memberships:
        if member.get("dataset")==axis.dataset:
            if 'array_index' in member:
                from .relations import axis_identity
                axis_identity(axis,member)
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
    quarantined=set(((routing_manifest or {}).get("semantic_review") or {}).get("quarantined_source_ids",[]))
    for a in articles:
        if a["source_id"] in quarantined:
            rejections["semantic_quarantine"]+=1
            continue
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
            elif channel=="market" and reviewed_market is not None and link.get("dataset")==axis.dataset:
                if not market_per_stock:
                    relations.append(("market",-1,link,link))
                else:
                    from .upstream import market_owners
                    from .relations import publication_day
                    claims=link.get('scope_claims') or [{'scope':'US_EQUITY'}]
                    if any(c['scope']=='US_EQUITY' for c in claims):claims=[{'scope':'US_EQUITY'}]
                    mapped=set()
                    for claim in claims:
                        owners=market_owners(claim['scope'],market_securities,axis,publication_day(a))
                        if not owners:rejections['unmatched_historical_exchange_scope']+=1
                        for index,owner in owners:
                            if index not in mapped:
                                relations.append(('market',index,link,owner or link));mapped.add(index)
            elif channel=="industry" and link.get("dataset") in (None,axis.dataset) and link.get("verification_status")=="verified":
                key=tuple(link.get(k) for k in ("industry_standard","industry_version","industry_code"))
                if all(key) and link.get("evidence_type") in ("explicit","inferred"):
                    for member in by_industry.get(key,[]):
                        if member.get("stock_uid") in stock_index:
                            relations.append(("industry",stock_index[member["stock_uid"]],link,member))
        if reviewed_market is None and "market" in a.get("impact_scope",[]):
            relations.append(("market",-1,None,None))
        if not relations:
            rejections["unmapped_article"]+=1
        for t in range(first,min(published_session+lookback_sessions,len(axis.dates))):
            for channel,stock,link,mapping in relations:
                date,cutoff=axis.dates[t],schedule.cutoffs[t]
                publication=a["published_at"]
                pub=publication[:10] if len(publication)==10 else timestamp(publication).tz_convert("America/New_York").strftime("%Y-%m-%d")
                if link is not None:
                    if not relationship_valid(link,date,pub,cutoff,mode,relation_time_basis):
                        rejections["invalid_or_unavailable_relation"]+=1
                        continue
                    if mapping is not link and not relationship_valid(mapping,date,pub,cutoff,mode,relation_time_basis):
                        rejections["invalid_or_unavailable_membership"]+=1
                        continue
                exposure=finite_number(mapping.get('exposure_weight',1.0) if mapping else 1.0,'exposure_weight',strictly_positive=True)
                if exposure>1:raise ValueError('exposure_weight must be <= 1')
                entry={"article":a,"age":t-published_session,"first_session":first,'exposure':exposure,
                    "inferred":bool(link and link.get("evidence_type")=="inferred")}
                key=(channel,stock,t)
                content_hash=a["input_text_hash"]
                existing=groups[key].get(content_hash)
                if existing:
                    # Prefer explicit over inferred when the same article reaches a stock by multiple mappings.
                    existing["inferred"]=existing["inferred"] and entry["inferred"]
                    existing['exposure']=max(existing['exposure'],exposure)
                    continue
                groups[key][content_hash]=entry
                accepted_ids.add(a["source_id"])
                # Last training label is train_end-1; its decision is train_end-2 for steps=1.
                if decision_lookback-1<=t<axis.train_end-1:
                    eligible.add(a["embedding_row"])
    if not groups:
        raise ValueError("No eligible article-channel pairs for this dataset/time mode; no empty real panel is published.")
    event_of={}
    market_days=defaultdict(set)
    if market_channel is not None:
        from . import market_quality as mq
        market_entries={}
        for (channel,stock,t),unique in groups.items():
            if channel!='market':continue
            for e in unique.values():
                a=e['article'];market_days[t].add(a['source_id'])
                market_entries.setdefault(a['source_id'],{'source_id':a['source_id'],'session':t-e['age'],
                    'event_type':a.get('event_type'),'embedding_row':a['embedding_row']})
        dedup=market_channel['event_dedup']
        event_of=mq.cluster_events(list(market_entries.values()),vectors,dedup['window_sessions'],dedup['cosine_threshold'])
        event_types={sid:entry['event_type'] for sid,entry in market_entries.items()}
        speculative=sum(mq.speculative_summary(bound_text[sid]) for sid in market_entries if sid in bound_text)
        quality={'policy':{k:(str(v) if k=='official_events' and v is not None else v) for k,v in market_channel.items()},
            'news_routes_total':news_routes_total,'news_routes_kept':news_routes_kept,
            'news_route_evidence_basis':dict(Counter(route_bases.values())),'official':official_info,
            'market_articles':len(market_entries),'market_events':len(set(event_of.values())),
            'speculative_summary_articles':speculative,'count_feature_basis':'independent_events',
            'splits':mq.coverage(sorted((t,sorted(ids)) for t,ids in market_days.items()),event_of,event_types,axis,decision_lookback)}
    eligible=sorted(eligible)
    mean,matrix,transform_info=compress(vectors,eligible,components=components,method=projection,seed=seed)
    transform_info['components']=components
    compact=(np.asarray(vectors,dtype=np.float32)-mean)@matrix.T
    if not np.isfinite(compact).all():
        raise ValueError("Nonfinite compressed embeddings.")
    n,total=len(axis.stock_uids),len(axis.dates)
    coverage={}
    with new_artifact(output) as directory:
        data,masks={},{}
        for channel in CHANNELS:
            shape=(total,width) if channel=="market" and not market_per_stock else (n,total,width)
            data[channel]=np.lib.format.open_memmap(directory/(channel+"_news_daily.npy"),mode="w+",dtype="float32",shape=shape)
            masks[channel]=np.lib.format.open_memmap(directory/(channel+"_news_mask.npy"),mode="w+",dtype="uint8",shape=shape[:-1])
            data[channel][:]=0
            masks[channel][:]=0
        with (directory/"provenance.jsonl").open("w",encoding="utf-8") as ledger:
            for (channel,stock,t),unique in sorted(groups.items()):
                entries=list(unique.values())
                weights=np.exp(-np.array([e["age"] for e in entries])/tau)*np.array([e['exposure'] for e in entries])
                events=None
                if channel=="market" and event_of:
                    # 同一事件的多篇报道合计只占一个事件的权重，避免转载放大。
                    events=[event_of[e["article"]["source_id"]] for e in entries]
                    totals=defaultdict(float)
                    for name,w in zip(events,weights):totals[name]+=w
                    weights=np.array([w/totals[name] for name,w in zip(events,weights)])
                weights/=weights.sum()
                arows=[e["article"] for e in entries]
                row=np.zeros(width,dtype=np.float32)
                row[:3]=weights@np.array([[a["p_positive"],a["p_negative"],a["p_neutral"]] for a in arows])
                row[3:stats_start]=weights@compact[[a["embedding_row"] for a in arows]]
                row[stats_start]=np.log1p(len(set(events)) if events is not None else sum(e['exposure'] for e in entries))
                row[stats_start+1]=np.mean([e["age"] for e in entries])
                row[stats_start+2]=np.mean([e["inferred"] for e in entries])
                importance=[float(a["importance"]) for a in arows if a.get("importance_verified") is True
                    and isinstance(a.get("importance"),(int,float)) and np.isfinite(a["importance"]) and 0<=a["importance"]<=1]
                row[stats_start+3]=np.mean(importance) if importance else 0
                row[stats_start+4]=len(importance)/len(entries)
                loc=t if channel=="market" and not market_per_stock else (stock,t)
                data[channel][loc]=row
                masks[channel][loc]=1
                for e,w in zip(entries,weights):
                    record={"source_id":e["article"]["source_id"],"channel":channel,
                        "stock_uid":None if stock<0 else axis.stock_uids[stock],"decision_idx":t,
                        "first_available_session":e["first_session"],"weight":float(w),"exposure_weight":e['exposure'],"inferred":e["inferred"]}
                    if events is not None:record["market_event_id"]=event_of[e["article"]["source_id"]]
                    ledger.write(json.dumps(record)+"\n")
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
        write_records(directory/'article_relations.jsonl',links)
        write_records(directory/'industry_memberships.jsonl',memberships)
        if market_channel is not None:
            write_records(directory/'official_market_routes.jsonl',official_routes)
        if relations_manifest is not None:
            for name in relations_manifest['files']:
                shutil.copy2(Path(relations_artifact)/name,directory/('relation_'+name))
            shutil.copy2(Path(relations_artifact)/'manifest.json',directory/'relations_manifest.json')
            if digest_file(directory/'relations_manifest.json')!=relations_manifest_sha:
                raise ValueError('Historical relations changed during panel construction')
        transform_info.update(dataset=axis.dataset,training_decision_end_exclusive=axis.train_end-1,
            training_decision_start_inclusive=decision_lookback-1,
            eligible_training_articles=len(eligible),fit_source_ids_hash=digest(sorted(a["source_id"] for a in articles
                if a.get("embedding_row") in eligible)))
        routing_metadata=None
        if market_routes is not None:
            from .routing import FILES
            for name in routing_manifest["files"]:
                shutil.copy2(Path(market_routes)/name,directory/name)
            shutil.copy2(Path(market_routes)/"manifest.json",directory/"market_routing_manifest.json")
            routing_metadata={"manifest_sha256":routing_source_hash,
                "source_snapshot_manifest_sha256":routing_manifest["source_input"]["snapshot_manifest_sha256"],
                "accepted_routes":routing_manifest["accepted_routes"],"decisions":routing_manifest["decisions"]}
        manifest={"schema_version":SCHEMA,"kind":"daily_news_panel","created_at":now(),
            "dataset":axis.dataset,"stock_axis_hash":axis.stock_hash,"trading_dates_hash":axis.date_hash,
            "mode":mode,"historical_model_replay":mode!="observed_pit","feature_names":feature_names,
            "availability_comparison":"strictly_before_cutoff",
            "relation_time_basis":relation_time_basis,
            "article_encoder":feature_manifest["encoder"],
            "market_layout":"stock_date" if market_per_stock else "shared_date",
            "market_routing_verified":routing_manifest is not None,
            "market_routing":routing_metadata,
            "market_quality":quality,
            'historical_relations_verified':relations_manifest is not None,
            'industry_membership_policy':'undated_reference_exploration' if reference_only else 'verified_historical',
            'historical_industry_membership_verified':not reference_only,
            'exploratory_reference_stocks':len({m['stock_uid'] for m in memberships if m.get('verification_status')=='reference_only'}),
            'relations_manifest_sha256':relations_manifest_sha,
            "feature_dim":width,"n_stocks":n,"n_dates":total,"coverage":coverage,
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
        if routing_manifest is not None:
            from .routing import assert_panel_routes
            if digest_file(Path(market_routes)/"manifest.json")!=routing_metadata["manifest_sha256"]:
                raise ValueError("Market review changed while building the panel.")
            assert_panel_routes(directory,axis,manifest)
    return manifest
