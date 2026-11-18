"""Final input accounting after encoding, evidence/event filtering and market eligibility."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from .io import digest,digest_file,read_records
from .market_quality import gate_blockers


def valid_windows(data,lookback):
    mask=np.asarray(data.arrays['mask'])>0
    if data.split.lookback!=lookback or data.split.steps!=1:raise ValueError('Acceptance window must match model input')
    missing=np.pad(np.cumsum(~mask,axis=1,dtype=np.int32),((0,0),(1,0)))
    n,t=mask.shape;valid=np.zeros((n,t),dtype=bool)
    # column is decision index; window includes lookback input days AND the target label.
    targets=np.arange(lookback,t)
    valid[:,targets-1]=missing[:,targets+1]==missing[:,targets-lookback]
    return valid


def summarize_final(axis,masks,ledger,valid,config):
    n,t=valid.shape
    if (n,t)!=(len(axis.stock_uids),len(axis.dates)):raise ValueError('Eligibility axis mismatch')
    lookback=config['panel']['decision_lookback'];enabled=config['fusion']['enabled_channels']
    bounds={'train':(lookback-1,axis.train_end-1),'valid':(axis.train_end-1,axis.valid_end-1),
        'test':(axis.valid_end-1,t-1)}
    if config['pipeline']['data_protocol']=='original':bounds['original_train_candidates']=(lookback-1,axis.train_end+lookback-1)
    uid={s:i for i,s in enumerate(axis.stock_uids)}
    coverage={};eligible={};seen_masks={}
    for ch in ('company','industry','market'):
        mask=np.asarray(masks[ch],dtype=bool)
        if mask.shape not in ((t,),(n,t)):raise ValueError('Unexpected panel mask shape')
        coverage[ch]={};seen_masks[ch]=mask
        for split,(start,end) in bounds.items():
            raw=mask[start:end] if mask.ndim==1 else mask[:,start:end]
            use=(raw[None,:] if mask.ndim==1 else raw)&valid[:,start:end]
            coverage[ch][split]={'panel_active_mask_cells':int(raw.sum()),
                'panel_stock_decision_cells':int(raw.sum())*n if mask.ndim==1 else int(raw.sum()),
                'valid_price_label_cells':int(use.sum()),'active_days_after_price_filter':int(use.any(axis=0).sum()),
                'covered_stocks_after_price_filter':int(use.any(axis=1).sum()),
                'articles':0,'unique_texts':0,'events':0 if ch=='market' else None}
            eligible[(ch,split)]={'articles':set(),'texts':set(),'events':set(),'types':{}}
    for row in ledger:
        ch=row['channel'];d=row['decision_idx']
        if ch not in coverage or type(d) is not int or not 0<=d<t:raise ValueError('Invalid provenance cell')
        stock=row.get('stock_uid');mask=seen_masks[ch]
        if stock is None:
            if ch!='market' or mask.ndim!=1:raise ValueError('Only a shared market cell can broadcast')
            if not mask[d]:raise ValueError('Ledger references inactive shared market mask')
            usable=bool(valid[:,d].any())
        else:
            if stock not in uid or mask.ndim!=2 or not mask[uid[stock],d]:raise ValueError('Ledger stock/mask mismatch')
            usable=bool(valid[uid[stock],d])
        if not usable:continue
        for split,(start,end) in bounds.items():
            if not start<=d<end:continue
            x=eligible[(ch,split)];x['articles'].add(row['source_id'])
            x['texts'].add(row.get('input_text_hash',row['source_id']))
            if ch=='market':
                event=row.get('market_event_id')
                if not event:raise ValueError('Final market acceptance requires explicit independent event IDs')
                x['events'].add(event);x['types'].setdefault(event,row.get('event_type') or 'unknown')
    for (ch,split),values in eligible.items():
        stats=coverage[ch][split];stats.update(articles=len(values['articles']),unique_texts=len(values['texts']))
        if ch=='market':stats['events']=len(values['events'])
    splits={}
    from collections import Counter
    for split in ('train','valid','test'):
        c=coverage['market'][split];x=eligible[('market',split)]
        splits[split]={'articles':c['articles'],'events':c['events'],'active_days':c['active_days_after_price_filter'],
                       'event_types':dict(Counter(x['types'].values()))}
    blockers=[]
    for ch in enabled:
        actual=coverage[ch]['train']['valid_price_label_cells']
        if actual<config['pipeline']['minimum_training_cells']:
            blockers.append('Enabled channel '+ch+' has insufficient price-valid nominal training news: '+str(actual))
    if 'market' in enabled and config.get('market_channel') is not None:
        blockers+=gate_blockers({'splits':splits},config['market_channel']['gates'])
    return {'dataset':axis.dataset,'coverage':coverage,'market_quality_after_price_filter':{'splits':splits},
            'blockers':blockers,'ready':not blockers,'accounting_version':'encoded_event_evidence_price_v1',
            'split_basis':'target=decision+1; nominal splits; original training overlap reported separately'}


def audit_panel(panel,data,config,feature_dir):
    """Reconcile encoded features, review policy, event clustering, provenance and final masks."""
    panel.assert_experiment(data.axis,config['panel']['mode'],config['panel']['decision_lookback'])
    if feature_dir is None:raise ValueError('Final acceptance requires the pinned FinBERT feature artifact')
    from .panel import load_features
    articles,vectors,features=load_features(feature_dir)
    if digest_file(Path(feature_dir)/'manifest.json')!=panel.manifest['article_artifact_manifest_sha256']:
        raise ValueError('Final panel and FinBERT features belong to different artifacts')
    byid={a['source_id']:a for a in articles}
    if (config.get('market_channel') or {}).get('official_events'):
        from .official_events import load_encoded
        records,ov,_,_=load_encoded(config['market_channel']['official_events'],data.axis.dataset)
        offset=len(vectors);byid.update({a['source_id']:dict(a,embedding_row=a['embedding_row']+offset) for a in records})
        vectors=np.concatenate([np.asarray(vectors),np.asarray(ov)])
    rows=read_records(panel.directory/'provenance.jsonl');market_entries={}
    locations={ch:set() for ch in ('company','industry','market')};weights=defaultdict(float);texts=defaultdict(set)
    for row in rows:
        a=byid.get(row['source_id'])
        if a is None or a.get('finbert_status')!='success':raise ValueError('Panel uses a missing or failed FinBERT article')
        row.update(input_text_hash=a['input_text_hash'],event_type=a.get('event_type'))
        key=(row['channel'],row.get('stock_uid'),row['decision_idx'])
        if a['input_text_hash'] in texts[key]:raise ValueError('Duplicate article content in a final panel cell')
        texts[key].add(a['input_text_hash']);locations[row['channel']].add(key[1:])
        weight=row.get('weight')
        if not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<=0:raise ValueError('Invalid provenance weight')
        weights[key]+=weight
        if row['channel']=='market':
            from .time_axis import Schedule
            # Publication-session assignment must match the builder, including date-only conservative delay.
            market_entries.setdefault(a['source_id'],a)
    if any(abs(w-1)>1e-5 for w in weights.values()):raise ValueError('Panel cell article weights do not sum to one')
    if any(len(locations[ch])!=int(np.asarray(panel.masks[ch]).sum()) for ch in locations):
        raise ValueError('Panel active masks and provenance cell counts differ')
    if config.get('market_channel') is not None:
        from .routing import load_routes
        from .market_quality import filter_routes,cluster_events
        from .time_axis import Schedule
        bindings_path=panel.directory/'market_route_bindings.jsonl'
        bindings={a['source_id']:dict(byid.get(a['source_id'],{}),**a) for a in read_records(bindings_path)}
        routes,_,_=load_routes(panel.directory,panel.axis.dataset,'market_routing_manifest.json')
        kept,_=filter_routes(routes,bindings,set(config['market_channel']['evidence_basis']))
        allowed={r['source_id'] for r in kept} if config['market_channel']['news_routes'] else set()
        allowed.update(r['source_id'] for r in read_records(panel.directory/'official_market_routes.jsonl'))
        if set(market_entries)-allowed:raise ValueError('Panel contains market articles rejected by configured evidence policy')
        schedule=Schedule(panel.axis,read_records(panel.directory/'decision_schedule.jsonl'))
        entries=[{'source_id':sid,'session':schedule.first_session(a,'retrospective'),
                  'embedding_row':a['embedding_row'],'event_type':a.get('event_type')} for sid,a in market_entries.items()]
        policy=config['market_channel']['event_dedup']
        expected=cluster_events(entries,vectors,policy['window_sessions'],policy['cosine_threshold'])
        if any(r.get('market_event_id')!=expected[r['source_id']] for r in rows if r['channel']=='market'):
            raise ValueError('Market event IDs differ from independently replayed FinBERT clustering')
    else:
        for row in rows:
            if row['channel']=='market':row['market_event_id']='article_'+row['source_id']
    # Refit on exactly the original nominal-training article set; verify the stored PCA, not only its metadata.
    start=config['panel']['decision_lookback']-1;end=panel.axis.train_end-1
    eligible=sorted({byid[r['source_id']]['embedding_row'] for r in rows if start<=r['decision_idx']<end})
    from .panel import compress
    mean,matrix,info=compress(vectors,eligible,config['panel']['projection_components'],config['panel']['projection'],config['panel']['projection_seed'])
    saved=np.load(panel.directory/'embedding_transform.npz',allow_pickle=False)
    if not np.allclose(saved['mean'],mean,atol=1e-5,rtol=1e-4) or not np.allclose(saved['components'],matrix,atol=1e-5,rtol=1e-4):
        raise ValueError('Stored projection differs from independent training-only fit')
    if panel.manifest['transform'].get('eligible_training_articles')!=len(eligible):raise ValueError('PCA article count mismatch')
    expected_hash=digest(sorted(a['source_id'] for a in byid.values() if a.get('embedding_row') in eligible))
    if panel.manifest['transform'].get('fit_source_ids_hash')!=expected_hash:raise ValueError('PCA source identities mismatch')
    valid=valid_windows(data,config['panel']['decision_lookback'])
    result=summarize_final(panel.axis,panel.masks,rows,valid,config)
    result.update(feature_artifact_manifest_sha256=digest_file(Path(feature_dir)/'manifest.json'),
        panel_manifest_sha256=digest_file(panel.directory/'manifest.json'),
        encoded_articles=len(articles),finbert_successful=features['successful'],
        panel_materialized=True,projection=panel.manifest['transform'],
        evidence_and_event_replay_verified=True,training_only_projection_replayed=True)
    return result
