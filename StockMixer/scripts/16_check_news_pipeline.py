"""Read-only upstream audit and isolated downstream fusion probes (no formal training)."""
import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8-sig')
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def freeze_export(batch, root, require_complete=False):
    """Freeze only committed, exported articles; never synthesize an upstream publication."""
    from news_fusion.io import digest_file
    run = json.loads((batch/'run.json').read_text(encoding='utf-8'))
    progress = json.loads((batch/'progress.json').read_text(encoding='utf-8'))
    live = json.loads((batch/'live_progress.json').read_text(encoding='utf-8'))
    if run.get('classification_contract') != 'dictionary-scope-replacement-1':
        raise ValueError('Fresh replacement contract required')
    if require_complete:
        if progress['pending'] or live['pending']:
            raise ValueError('Upstream is incomplete; cannot produce a full-corpus report')
        acceptance = json.loads((batch/'acceptance.json').read_text(encoding='utf-8'))
        if acceptance.get('passed') is not True:
            raise ValueError('Full upstream acceptance required')
    source = batch/'adapted'
    target = root/'frozen_producer'
    pins = {}
    ids = sorted(p.stem for p in (source/'data/US').glob('*.json'))
    if not ids:
        raise ValueError('No exported records yet')
    # Each article is independently checked against its published audit and proof by adapt_snapshot.
    # Read twice: a concurrent re-export must fail rather than silently produce a mixed snapshot.
    for sid in ids:
        analysis_path = batch/'analysis'/(sid+'.json')
        analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
        if analysis.get('status') != 'completed' or analysis.get('fingerprint') != run['fingerprint']:
            raise ValueError('Uncommitted or wrong-run article: '+sid)
        for rel in ['data/US/'+sid+'.json', 'audit/'+sid+'.json', 'evidence/'+sid+'.json']:
            p = source/rel
            body = p.read_bytes()
            pins[str(p)] = hashlib.sha256(body).hexdigest()
            out = target/rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
        audit = json.loads((target/'audit'/(sid+'.json')).read_text(encoding='utf-8'))
        if audit.get('classification_contract') != 'dictionary-scope-replacement-1':
            raise ValueError('Old classification audit: '+sid)
        if audit.get('llm_inferred_at') != analysis['audit'].get('llm_inferred_at'):
            raise ValueError('Publication and analysis are not the same version: '+sid)
    if any(digest_file(p) != h for p, h in pins.items()):
        raise ValueError('An included export changed during freezing; rerun in a new directory')
    info = {'run': run, 'export_progress_at_start': progress, 'live_progress_at_start': live,
            'run_file_sha256':digest_file(batch/'run.json'),
            'frozen_exported_articles': len(ids), 'source_ids': ids,
            'partial': len(ids) != run['total'], 'diagnostic_only': True,
            'newer_unexported_analyses_excluded': True, 'source_files_sha256': pins}
    dump(root/'frozen_inputs.json', info)
    return target, info


def mapping_check(project, base, llm, root):
    from news_fusion.industry_dictionary import IndustryDictionary
    from news_fusion.industry_mapping import master_inputs, reference_inputs, verified_mapping
    from news_fusion.reference_membership import memberships
    from news_fusion.time_axis import Axis
    from news_fusion.io import read_records, digest_file
    resolve = lambda v: Path(v) if Path(v).is_absolute() else project/v
    dictionary = IndustryDictionary(resolve(llm['industry_dictionary']['directory']))
    axes = {d: Axis.load(resolve(base['data']['axis_dir'])/(d+'.json')) for d in ('NASDAQ','NYSE','SP500')}
    archive = resolve(llm['industry_dictionary']['reference_archive'])
    sp = resolve(base['data']['dataset_root'])/'SP500/sp500_ticker.csv'
    refs, names, refpins = reference_inputs(archive, sp, dictionary)
    tables, pins = master_inputs(project, llm)
    directory = resolve(llm['industry_dictionary']['output'])
    manifest = verified_mapping(directory, dictionary, axes, tables[0], tables[1], refs, names)
    reference_rows = read_records(directory/'reference_experiment_memberships.jsonl')
    counts, reference_by_dataset, unmatched = [], {}, []
    for dataset, axis in axes.items():
        stored = read_records(directory/(dataset+'_reference_experiment_memberships.jsonl'))
        if not stored:
            raise ValueError('Missing reference memberships: '+dataset)
        expected = memberships(axis, dictionary, archive, sp, stored[0]['known_from'])
        if stored != expected or [r for r in reference_rows if r['dataset']==dataset] != expected:
            raise ValueError('Reference members differ from the original released dataset: '+dataset)
        reference_by_dataset[dataset] = expected
        rows = read_records(directory/(dataset+'_stock_industry.jsonl'))
        unmatched.extend(r for r in rows if not r['reference_industry_key'])
        counts.append({'dataset': dataset, **{k:manifest['datasets'][dataset][k] for k in
            ('stocks','reference_matched_stocks','reference_unmatched_stocks','stocks_with_verified_dated_observations')},
            'reference_memberships_recomputed':len(expected), 'passed':True})
    if len(reference_rows) != sum(len(v) for v in reference_by_dataset.values()):
        raise ValueError('Undeclared reference memberships')
    table(root/'mapping_summary.csv', counts)
    table(root/'unmatched_stock_industry.csv', unmatched)
    result = {'passed':True, 'summary':counts, 'mapping_directory':str(directory),
              'dictionary_sha256':dictionary.sha256, 'reference_pins':refpins,
              'historical_input_pins':pins, 'mapping_manifest_sha256':digest_file(directory/'manifest.json'),
              'checks':['dictionary codes/names/version/hash','all three stock axes and array indices',
                        'CSV/JSONL/reverse-index consistency','dated SEC evidence and observation validity',
                        'reference memberships independently rebuilt from release archive and SP500 CSV'],
              'limitation':'Undated reference members are usable only in exploratory retrospective mode; they are not verified historical membership.'}
    dump(root/'mapping_check.json', result)
    return axes, dictionary, tables, reference_by_dataset, result


def phase_for_target(target, train_end, valid_end, n_dates, lookback=16):
    if not lookback <= target < n_dates:
        return None
    return 'train' if target < train_end else 'validation' if target < valid_end else 'test'


def enumerate_groups(axis, schedule, articles, links, members, securities, settings):
    """Pre-encoding expansion using the exact panel time/relationship/dedup rules."""
    from news_fusion.panel import relationship_valid
    from news_fusion.relations import publication_day
    from news_fusion.upstream import market_owners
    by_article, by_industry = defaultdict(list), defaultdict(list)
    for link in links:
        by_article[link['source_id']].append(link)
    for member in members:
        if member['dataset'] == axis.dataset:
            key = tuple(member.get(k) for k in ('industry_standard','industry_version','industry_code'))
            by_industry[key].append(member)
    indices = {uid:i for i,uid in enumerate(axis.stock_uids)}
    groups, rejected = defaultdict(dict), Counter()
    ordered = []
    mode, basis = settings['mode'], settings['relation_time_basis']
    for a in articles:
        try:
            first = schedule.first_session(a, mode, settings['simulated_latency_seconds'])
            published = schedule.first_session(a, 'retrospective')
            day = publication_day(a)
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        if first >= len(axis.dates):
            rejected['available_after_dataset'] += 1
        elif day < axis.dates[0]:
            rejected['publication_before_dataset_start'] += 1
        elif first >= published+settings['lookback_sessions']:
            rejected['stale_before_usable'] += 1
        else:
            ordered.append((first,a['source_id'],published,day,a))
    market_per_stock = any(l['channel']=='market' and any(c['scope']!='US_EQUITY' for c in l.get('scope_claims',[])) for l in links)
    for first, sid, published, day, a in sorted(ordered, key=lambda row:(row[0],row[1])):
        relations=[]
        for link in by_article[sid]:
            channel=link['channel']
            if channel=='company' and link.get('dataset')==axis.dataset and link.get('stock_uid') in indices:
                relations.append((channel,indices[link['stock_uid']],link,link))
            elif channel=='industry' and link.get('verification_status')=='verified':
                key=tuple(link.get(k) for k in ('industry_standard','industry_version','industry_code'))
                if all(key) and link.get('evidence_type') in ('explicit','inferred'):
                    for m in by_industry[key]:
                        relations.append((channel,indices[m['stock_uid']],link,m))
            elif channel=='market':
                if not market_per_stock:
                    relations.append((channel,-1,link,link))
                else:
                    claims=link.get('scope_claims') or [{'scope':'US_EQUITY'}]
                    if any(c['scope']=='US_EQUITY' for c in claims):
                        claims=[{'scope':'US_EQUITY'}]
                    seen=set()
                    for c in claims:
                        owners=market_owners(c['scope'],securities,axis,day)
                        if not owners:rejected['unmatched_historical_exchange_scope']+=1
                        for i,owner in owners:
                            if i not in seen:
                                relations.append((channel,i,link,owner or link)); seen.add(i)
        if not relations:rejected['unmapped_article']+=1
        for t in range(first,min(published+settings['lookback_sessions'],len(axis.dates))):
            for channel,stock,link,member in relations:
                if not relationship_valid(link,axis.dates[t],day,schedule.cutoffs[t],mode,basis):
                    rejected['invalid_or_unavailable_relation']+=1;continue
                if member is not link and not relationship_valid(member,axis.dates[t],day,schedule.cutoffs[t],mode,basis):
                    rejected['invalid_or_unavailable_membership']+=1;continue
                groups[(channel,stock,t)].setdefault(a['input_text_hash'],sid)
    return groups,dict(rejected)


def summarize_groups(axis, data, groups, enabled):
    import numpy as np
    rows, ledger=[] ,[]
    mask_cache={}
    for channel in ('company','industry','market'):
        for phase in ('train','validation','test','original_train_candidates'):
            ids,contents,days,stocks,cells,usable=set(),set(),set(),set(),set(),set()
            for (ch,stock,t),items in groups.items():
                target=t+1
                if ch!=channel:continue
                if phase=='original_train_candidates':
                    if not 16<=target<axis.train_end+16 or target>=len(axis.dates):continue
                elif phase_for_target(target,axis.train_end,axis.valid_end,len(axis.dates))!=phase:
                    continue
                owners=range(len(axis.stock_uids)) if stock==-1 else [stock]
                if target not in mask_cache:
                    mask_cache[target]=np.asarray(data.window(target)[3]).reshape(-1)>0
                for i in owners:
                    cells.add((i,t)); stocks.add(i)
                    if mask_cache[target][i]:usable.add((i,t))
                ids.update(items.values());contents.update(items);days.add(t)
            rows.append({'dataset':axis.dataset,'channel':channel,'split':phase,
                         'enabled_in_current_config':channel in enabled,
                         'mapped_unique_articles':len(ids),'unique_texts':len(contents),
                         'decision_days':len(days),'stocks':len(stocks),
                         'stock_decision_cells':len(cells),'valid_price_label_cells':len(usable),
                         'fresh_finbert_ready_articles':0,'fresh_training_panel_ready':False})
    for (channel,stock,t),items in groups.items():
        target=t+1
        phase=phase_for_target(target,axis.train_end,axis.valid_end,len(axis.dates))
        if phase:
            for content,sid in items.items():
                ledger.append({'dataset':axis.dataset,'channel':channel,'stock_uid':axis.stock_uids[stock] if stock>=0 else 'ALL_DATASET_STOCKS',
                               'decision_idx':t,'decision_date':axis.dates[t],'target_idx':target,
                               'target_date':axis.dates[target],'split':phase,'source_id':sid,'input_text_hash':content})
    return rows,ledger


def coverage_check(project, base, llm, root, axes, dictionary, tables, references, batch, require_complete):
    from news_fusion.source_adapter import adapt_snapshot
    from news_fusion.io import digest_file, read_records, write_records
    from news_fusion.inputs import normalize_article
    from news_fusion.retrace_adapter import automated_industry_reviews
    from news_fusion.industry_scope_review import apply_scope_reviews
    from news_fusion.relations import resolve_relations, validate_master
    from news_fusion.reference_membership import add_references
    from news_fusion.time_axis import Schedule, export_schedule
    from news_fusion.original import OriginalMarketData
    from news_fusion.upstream import make_reviews
    from news_fusion.routing import checked_reviews
    producer, frozen = freeze_export(batch,root,require_complete)
    adapt_snapshot(producer,root/'adapted',project/base['data']['axis_dir'],markets=('US',))
    articles=[normalize_article(a) for a in read_records(root/'adapted/articles.jsonl')]
    input_counts=dict(Counter(a['input_status'] for a in articles))
    articles=[a for a in articles if a['input_status']=='ready']
    for a in articles:
        if a['source_id'] in llm['pipeline']['industry_excluded_sources']:
            a['industry_excluded_proposals']=a.get('industry_impacts',[])
            a['industry_impacts']=[]
            a['industry_scope_review']={'decision':'exclude','review_method':'configured_exclusion',
                'rationale':'Preserve prior user exclusions; no peer propagation.'}
    automated=automated_industry_reviews(articles,llm['pipeline']['industry_min_confidence']) if llm['pipeline']['allow_automated_industry_scope'] else {}
    scope_path=llm['pipeline'].get('industry_scope_review')
    if scope_path and not Path(scope_path).is_absolute():scope_path=project/scope_path
    articles,scope=apply_scope_reviews(articles,directory=scope_path,required=llm['pipeline']['require_industry_scope_review'],
        snapshot_sha=digest_file(root/'adapted/manifest.json'),preserve_approval_time=True,automated=automated)
    all_rows,historical_rows,unresolved,stats,industry_audit=[],[],[],{},[]
    for dataset,axis in axes.items():
        print('[CHECK] coverage '+dataset,flush=True)
        data=OriginalMarketData(project/'outputs/artifacts/market_original'/dataset)
        if data.axis.stock_hash!=axis.stock_hash or data.axis.date_hash!=axis.date_hash:
            raise ValueError('Prepared market axes differ')
        export_schedule(axis,root/(dataset+'_schedule.jsonl'))
        schedule=Schedule(axis,read_records(root/(dataset+'_schedule.jsonl')))
        policy={'version':'evidence_mapping_v3','industry_min_confidence':llm['pipeline']['industry_min_confidence']}
        links,verified,missing,counts=resolve_relations(axis,articles,*tables,policy=policy,dictionary=dictionary)
        reviews=make_reviews(articles,dataset,llm['review']['min_confidence'])
        routes,routecounts=checked_reviews(articles,reviews,allow_partial=True,dataset=dataset)
        for route in routes:route.update(dataset=dataset,known_from=route['reviewed_at'])
        write_records(root/(dataset+'_market_reviews.jsonl'),reviews)
        links+=routes
        securities,_,_=validate_master(axis,*tables)
        current=add_references(verified,references[dataset]) if llm['pipeline']['industry_membership_mode']=='reference_retrospective' else verified
        for link in links:
            if link['channel']!='industry':continue
            key=tuple(link.get(k) for k in ('industry_standard','industry_version','industry_code'))
            selected=lambda rows:{m['stock_uid'] for m in rows if tuple(m.get(k) for k in ('industry_standard','industry_version','industry_code'))==key}
            industry_audit.append({'dataset':dataset,'source_id':link['source_id'],
                'industry_standard':key[0],'industry_version':key[1],'industry_code':key[2],
                'historical_candidate_stocks':len(selected(verified)),
                'reference_plus_historical_candidate_stocks':len(selected(current)),
                'note':'Candidates only; publication/decision validity is checked separately in coverage.'})
        groups,rejections=enumerate_groups(axis,schedule,articles,links,current,securities,llm['panel'])
        rows,ledger=summarize_groups(axis,data,groups,llm['fusion']['enabled_channels'])
        write_records(root/(dataset+'_mapped_news_ledger.jsonl'),ledger)
        all_rows+=rows
        hg,hr=enumerate_groups(axis,schedule,articles,links,verified,securities,llm['panel'])
        hrows,_=summarize_groups(axis,data,hg,llm['fusion']['enabled_channels']);historical_rows+=hrows
        unresolved.extend(dict(row,dataset=dataset) for row in missing)
        pca_ids={r['source_id'] for r in ledger if r['split']=='train'}
        stats[dataset]={'relations':counts,'market_review_decisions':routecounts,'time_mapping_rejections':rejections,
                        'historical_only_rejections':hr,'validity':'pre-encoding mapping; not a FinBERT panel',
                        'nominal_target_ranges':{'train':[axis.dates[16],axis.dates[axis.train_end-1]],
                         'validation':[axis.dates[axis.train_end],axis.dates[axis.valid_end-1]],
                         'test':[axis.dates[axis.valid_end],axis.dates[-1]]},
                        'original_train_validation_overlap_targets':16,
                        'pre_encoding_train_pca_candidate_articles':len(pca_ids),
                        'pca_components':llm['panel']['projection_components'],
                        'pca_sample_count_sufficient_upper_bound':len(pca_ids)>llm['panel']['projection_components'],
                        'fresh_finbert_not_run':True}
    table(root/'coverage.csv',all_rows)
    table(root/'coverage_verified_historical_only.csv',historical_rows)
    table(root/'industry_link_candidate_stocks.csv',industry_audit)
    write_records(root/'unresolved_relations.jsonl',unresolved)
    result={'frozen':{k:v for k,v in frozen.items() if k not in ('source_files_sha256','source_ids')},
            'input_status_counts':input_counts,'industry_scope_reviews':scope,'datasets':stats,
            'rows':all_rows,'historical_only_rows':historical_rows,
            'warning':'Snapshot is partial until all 1948 articles are committed/exported. Counts cannot be extrapolated to the entire corpus. Fresh FinBERT features/panels have not been generated.'}
    dump(root/'coverage_check.json',result)
    return result


def fusion_check(project,llm,root,steps=3):
    import numpy as np
    import torch
    from news_fusion.experiment import load_run_checkpoint, make_model, model_api, batch_forward
    from news_fusion.original import OriginalMarketData
    from news_fusion.fusion import StockMixerWithNews
    from news_fusion.dataset import NewsPanel
    from news_fusion.io import digest_file
    torch.set_num_threads(2)
    checkpoint_path=Path(llm['baseline_checkpoint'])
    if not checkpoint_path.is_absolute():checkpoint_path=project/checkpoint_path
    checkpoint=load_run_checkpoint(checkpoint_path)
    config=checkpoint['config'];dataset=config['dataset']
    data=OriginalMarketData(project/'outputs/artifacts/market_original'/dataset)
    backbone=make_model(data,config,torch.device('cpu'),fusion=False)
    # Checkpoint state key is verified by load_run_checkpoint before loading weights.
    state=checkpoint.get('model_state',checkpoint.get('model_state_dict'))
    if state is None:raise ValueError('Unsupported checkpoint state key: '+','.join(checkpoint))
    backbone.load_state_dict(state)
    torch.manual_seed(config['training']['torch_seed'])
    model=StockMixerWithNews(backbone,**{k:llm['fusion'][k] for k in ('news_dim','hidden_dim','dropout','freeze_backbone','enabled_channels')})
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=llm['training']['learning_rate'])
    # Functional probe: real stock window/labels, synthetic features, all masks active.
    # Separate real FinBERT panel probe below prevents confusing synthetic and real inputs.
    target=int(data.split.targets('train')[0])
    x,price,truth,mask=[torch.tensor(np.array(a),dtype=torch.float32) for a in data.window(target)]
    n=len(data.axis.stock_uids);dim=llm['fusion']['news_dim']
    news={}
    for ch in ('company','industry','market'):
        news[ch+'_news']=torch.randn(n,dim)
        news[ch+'_mask']=torch.ones(n,1)
    _,get_loss=model_api()
    traces=[]
    for step in range(steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        prediction=model(x,price,**news)
        loss=get_loss(prediction,truth,price,mask,n,config['training']['alpha'])[0]
        loss.backward()
        for ch in model.branches:
            for component in ('adapter','gate','head'):
                ps=list(getattr(model.branches[ch],component).parameters())
                traces.append({'step':step+1,'channel':ch,'component':component,
                    'gradient_l2':sum(float(p.grad.square().sum()) for p in ps if p.grad is not None)**.5,
                    'has_gradient':any(p.grad is not None for p in ps),'loss':float(loss)})
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction,details=model(x,price,**news,return_details=True)
        repeated=model(x,price,**news)
        silent={}
        for ch in model.branches:
            silent[ch+'_news']=torch.full((n,dim),float('nan'))
            silent[ch+'_mask']=torch.zeros(n,1)
        no_news=model(x,price,**silent)
        branch_stats={}
        for ch,branch in model.branches.items():
            h=branch.adapter(news[ch+'_news'])
            gate=torch.sigmoid(branch.gate(torch.cat([h,details['baseline_return']],dim=-1)))
            branch_stats[ch]={'enabled':ch in model.enabled_channels,'gate_mean':float(gate.mean()),
                'delta_return_mean_abs':float(details['deltas'][ch].abs().mean()),
                'max_parameter_update':max(float((v-before['branches.'+ch+'.'+k]).abs().max()) for k,v in branch.state_dict().items())}
    frozen=all(torch.equal(v,before['stock_mixer.'+k]) for k,v in backbone.state_dict().items())
    checks={'backbone_unchanged':frozen,'no_news_exact_baseline_even_with_inactive_nan':torch.equal(no_news,details['baseline_price']),
            'eval_repeat_exact':torch.equal(prediction,repeated),
            'enabled_branches_updated':all(branch_stats[c]['max_parameter_update']>0 for c in model.enabled_channels),
            'disabled_branches_unchanged':all(v['max_parameter_update']==0 for c,v in branch_stats.items() if c not in model.enabled_channels),
            'enabled_gate_and_adapter_gradients_after_head_initialization':all(any(r['channel']==c and r['component']==comp and r['step']>1 and r['gradient_l2']>0 for r in traces) for c in model.enabled_channels for comp in ('gate','adapter'))}
    # Validate and inspect the exact legacy panel used by the static-metric run, read-only.
    legacy=project/'outputs/20261118_103625_NASDAQ_fusion/_artifact'
    real={'available':False,'reason':'No legacy run config found'}
    if (legacy/'config.json').exists():
        old=json.loads((legacy/'config.json').read_text(encoding='utf-8'))
        path=Path(old['news']['panel'])
        if not path.is_absolute():path=project/path
        panel=NewsPanel(path)
        counts=[]
        for ch in ('company','industry','market'):
            for phase in ('train','validation','test'):
                targets=data.split.targets('validation' if phase=='validation' else phase)
                active=panel.masks[ch][targets-1] if panel.masks[ch].ndim==1 else panel.masks[ch][:,targets-1]
                counts.append({'channel':ch,'split':phase,'active_mask_cells':int(active.sum()),
                               'total_mask_cells':int(active.size),'active_fraction':float(active.mean())})
        active_targets=[int(t) for t in data.split.targets('train') if any(np.asarray(panel.decision(int(t)-1)[c+'_mask']).any() for c in old['fusion']['enabled_channels'])]
        real={'available':True,'panel':str(path),'panel_manifest_sha256':digest_file(path/'manifest.json'),
              'legacy_enabled_channels':old['fusion']['enabled_channels'],'coverage':counts,
              'note':'Legacy real FinBERT features, not the fresh retrace snapshot. Training candidates include original validation overlap.'}
        if active_targets:
            torch.manual_seed(config['training']['torch_seed'])
            real_backbone=make_model(data,config,torch.device('cpu'),fusion=False)
            real_backbone.load_state_dict(state)
            actual=StockMixerWithNews(real_backbone,**{k:old['fusion'][k] for k in ('news_dim','hidden_dim','dropout','freeze_backbone','enabled_channels')})
            opt=torch.optim.Adam([p for p in actual.parameters() if p.requires_grad],lr=old['training']['learning_rate'])
            real_trace=[];real_target=active_targets[0]
            actual.eval()
            with torch.no_grad():pred_before=batch_forward(actual,data,real_target,torch.device('cpu'),panel)[0].clone()
            for step in range(steps):
                actual.train(); opt.zero_grad(set_to_none=True)
                pred,tr,pr,ma=batch_forward(actual,data,real_target,torch.device('cpu'),panel)
                loss=get_loss(pred,tr,pr,ma,n,old['training']['alpha'])[0]
                loss.backward()
                real_trace.append({'step':step+1,'loss':float(loss),'gradient_l2':sum(float(p.grad.square().sum()) for p in actual.parameters() if p.grad is not None)**.5})
                opt.step()
            actual.eval()
            with torch.no_grad():
                pred_after=batch_forward(actual,data,real_target,torch.device('cpu'),panel)[0]
                base_prices=torch.tensor(np.array(data.window(real_target)[1]),dtype=torch.float32)
                delta_returns=(pred_after-pred_before)/base_prices
            real.update(target_idx=real_target,target_date=data.axis.dates[real_target],steps=real_trace,
                        prediction_max_abs_change=float((pred_after-pred_before).abs().max()),
                        prediction_mean_abs_change=float((pred_after-pred_before).abs().mean()),
                        return_delta_mean=float(delta_returns.mean()),return_delta_cross_stock_std=float(delta_returns.std()))
            checks['real_finbert_probe_updates_prediction']=real['prediction_max_abs_change']>0
        # An interrupted checkpoint is evidence only, never promoted into load_run_checkpoint or training.
        archived_path=legacy/'last_model.pt'
        if archived_path.exists() and old['fusion']['enabled_channels']==['market']:
            archived=torch.load(archived_path,map_location='cpu',weights_only=True)
            if archived.get('config')!=old or archived.get('dataset')!=dataset:
                raise ValueError('Archived diagnostic checkpoint metadata differs')
            archived_model=StockMixerWithNews(make_model(data,config,torch.device('cpu'),fusion=False),
                **{k:old['fusion'][k] for k in ('news_dim','hidden_dim','dropout','freeze_backbone','enabled_channels')})
            archived_model.load_state_dict(archived['model_state_dict']);archived_model.eval()
            slope_bounds=[]
            if panel.manifest.get('market_layout','shared_date')=='shared_date':
                with torch.no_grad():
                    for decision in range(len(data.axis.dates)):
                        inp=panel.decision(decision)
                        if not np.asarray(inp['market_mask']).any():continue
                        features=torch.tensor(inp['market_news'],dtype=torch.float32).reshape(1,-1)
                        branch=archived_model.branches['market']
                        amplitude=float(branch.head(branch.adapter(features)).item())
                        slope_bounds.append(1-abs(amplitude*float(branch.gate.weight[0,-1]))/4)
            history=json.loads((legacy/'history.json').read_text(encoding='utf-8'))
            ranges={k:{'first':history[0][k],'last':history[-1][k],
                       'min':min(r[k] for r in history),'max':max(r[k] for r in history)}
                    for k in ('val_ic','val_ric','test_ic','test_ric','val_prec_10','test_prec_10')}
            real['archived_interrupted_run_evidence']={'checkpoint':str(archived_path),
                'checkpoint_sha256':digest_file(archived_path),'unsealed_run_not_trainable_input':True,
                'recorded_epochs':len(history),'history_metric_ranges':ranges,
                'shared_market_minimum_monotonic_derivative_bound':min(slope_bounds) if slope_bounds else None,
                'all_shared_market_days_strictly_monotonic_in_base_return':bool(slope_bounds) and min(slope_bounds)>0,
                'frozen_backbone_matches_baseline':all(torch.equal(v,state[k]) for k,v in archived_model.stock_mixer.state_dict().items()),
                'explanation':'For shared market features, r_new = r_base + A_day * sigmoid(B_day + w*r_base). Its derivative is >= 1-|A_day*w|/4. A positive bound preserves exact-arithmetic stock rankings; finite-precision ties and evaluator masking may still affect numerical metrics. RIC in the original evaluator is mean(daily IC)/std(daily IC), not RankIC.'}
    result={'checks':checks,'passed':all(checks.values()),'synthetic_feature_probe':{'steps':steps,'dataset':dataset,'branches':branch_stats},
            'legacy_real_finbert_probe':real,'registered_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'effective_enabled_parameters':sum(p.numel() for c in model.enabled_channels for p in model.branches[c].parameters()),
            'baseline_checkpoint':str(checkpoint_path),'baseline_checkpoint_sha256':digest_file(checkpoint_path),
            'no_formal_training_or_checkpoint_writes':True,
            'interpretation':'Zero-initialized heads make adapter/gate gradients zero on the first step. They must become nonzero after head updates. Sparse masks, frozen backbone and small residuals can keep rounded IC/RIC unchanged; this does not establish predictive benefit.'}
    table(root/'fusion_gradients.csv',traces)
    dump(root/'fusion_check.json',result)
    return result


def write_report(root,mapping,coverage,fusion):
    lines=['# 新闻接入检查：覆盖、映射、融合','',
        '本报告为隔离诊断；未调用 Qwen、未启动正式训练、未修改上游重处理任务或正式配置。','']
    if mapping:
        lines+=['## 1. 行业到股票映射','',
                '|数据集|股票数|参考行业匹配|未匹配|有日期核验的股票|','|---|---:|---:|---:|---:|']
        for r in mapping['summary']:lines.append('|{dataset}|{stocks}|{reference_matched_stocks}|{reference_unmatched_stocks}|{stocks_with_verified_dated_observations}|'.format(**r))
        lines+=['','所有股票轴、数组下标、字典代码/版本、正反向表及来源验证通过；参考成员表已从原始来源独立重建核对。',
                '参考行业无历史有效期，只能用于探索性回溯实验；不能当作真实历史行业成员。未匹配明细见 `unmatched_stock_industry.csv`。','']
    if coverage:
        f=coverage['frozen'];count=f['frozen_exported_articles'];total=f['run']['total']
        lines+=['## 2. 三通道覆盖','',f'本次冻结已导出的 {count}/{total} 篇新闻。上游采样时已处理 {f["live_progress_at_start"]["processed_total"]} 篇；尚未导出的结果未混入本统计。',
                '这些是编码前的映射覆盖，不是已经生成的 FinBERT 训练特征。新快照的 FinBERT 和正式面板尚未重建。',
                '按预测目标日期划分；一条新闻跨多个决策日仍按篇数去重。有效格子还要求完整价格窗口及标签有效。','',
                '|数据集|通道|训练篇数|验证篇数|测试篇数|训练有效股票日格子|','|---|---|---:|---:|---:|---:|']
        for ds in ('NASDAQ','NYSE','SP500'):
            for ch in ('company','industry','market'):
                rows={r['split']:r for r in coverage['rows'] if r['dataset']==ds and r['channel']==ch}
                lines.append(f'|{ds}|{ch}|{rows["train"]["mapped_unique_articles"]}|{rows["validation"]["mapped_unique_articles"]}|{rows["test"]["mapped_unique_articles"]}|{rows["train"]["valid_price_label_cells"]}|')
        lines+=['','`coverage.csv` 另列原始代码实际训练候选范围，包含前16个验证目标；`coverage_verified_historical_only.csv` 排除无日期参考成员。',
                '日期/证据/身份拒绝原因见 `coverage_check.json` 和 `unresolved_relations.jsonl`；逐文章分配见各数据集 `*_mapped_news_ledger.jsonl`。',
                '**部分批次无法代表1948篇全库覆盖。全部重处理完成后必须重新检查；即使映射有覆盖，也仍须完成真实 FinBERT 编码及训练期 PCA 秩验证。**','']
        for ds,stats in coverage['datasets'].items():
            lines.append(f'- {ds}：本快照训练期 PCA 候选新闻 {stats["pre_encoding_train_pca_candidate_articles"]} 篇；当前 PCA 维度 {stats["pca_components"]}；样本数上界是否足够：{stats["pca_sample_count_sufficient_upper_bound"]}。尚未做真实编码及秩验证。')
        lines.append('')
    if fusion:
        lines+=['## 3. 融合功能检查','',f'功能检查通过：{fusion["passed"]}。',
                f'登记的可训练参数：{fusion["registered_trainable_parameters"]}；当前启用通道实际参与参数：{fusion["effective_enabled_parameters"]}。','']
        for k,v in fusion['checks'].items():lines.append(f'- {k}: {v}')
        lines+=['','真实行情窗口和临时随机新闻特征用于检查梯度路径；旧版真实 FinBERT 面板另做隔离诊断，两者分开记录。',
                'Head 零初始化导致第一步 Adapter/Gate 梯度为0属于正常现象，后续梯度与参数更新见 `fusion_gradients.csv`。',
                '骨干冻结、新闻覆盖稀疏、修正量较小与终端保留位数都会使 IC/RIC 看起来不变。梯度检查通过不能证明预测效果更好。',
                '旧实验只启用 market，当前配置启用 market/industry；旧面板不代表新重处理结果。']
        archived=fusion['legacy_real_finbert_probe'].get('archived_interrupted_run_evidence')
        if archived:
            lines+=['',f'旧实验保留了 {archived["recorded_epochs"]} 轮记录。共享市场分支对基础收益率的最小导数下界为 {archived["shared_market_minimum_monotonic_derivative_bound"]}。',
                    '该下界大于0时，市场分支在数学上仍保持原股票排序；这解释了 loss 变化而选股指标几乎不变。公司/行业通道可提供不同股票的不同信息，但仍需充分覆盖与真实对照实验。',
                    '**原始评估器 RIC = 日度IC均值 / 日度IC标准差，并非 RankIC。** 旧实验未封存的权重仅用于只读诊断，未作为可复用正式训练输入。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--step',choices=['all','mapping','coverage','fusion'],default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    parser.add_argument('--probe-steps',type=int,default=3)
    args=parser.parse_args(argv)
    if not 3<=args.probe_steps<=10:parser.error('--probe-steps must be 3..10 (isolated diagnosis only)')
    project=args.project.resolve();sys.path.insert(0,str(project))
    from news_fusion.configuration import read_yaml,load_llm_config
    from news_fusion.io import digest_file, file_manifest
    base=read_yaml(project/'configs/stockmixer.yaml');llm=load_llm_config(project/'configs/llm.yaml')
    config_pins={str(p):digest_file(p) for p in [project/'configs/stockmixer.yaml',project/'configs/llm.yaml']}
    if llm['pipeline']['data_protocol']!='original':
        parser.error('This checker currently audits original_reproduction; strict protocol requires its prepared arrays and split ledger')
    root=args.output or project/'outputs/artifacts/news/diagnostics'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_NASDAQ_NYSE_SP500_checks')
    root=root.resolve();root.relative_to(project/'outputs')
    root.mkdir(parents=True,exist_ok=False)
    dump(root/'run_status.json',{'status':'running','step':args.step,'diagnostic_only':True})
    mapping=coverage=fusion=None
    try:
        if args.step in ('all','mapping','coverage'):
            print('[CHECK] mapping',flush=True)
            axes,dictionary,tables,references,mapping=mapping_check(project,base,llm,root)
        if args.step in ('all','coverage'):
            batch=Path(llm['pipeline']['retrace_artifact'])
            if not batch.is_absolute():batch=project/batch
            coverage=coverage_check(project,base,llm,root,axes,dictionary,tables,references,batch,args.require_complete)
        if args.step in ('all','fusion'):
            print('[CHECK] fusion (CPU, temporary weights)',flush=True)
            fusion=fusion_check(project,llm,root,args.probe_steps)
        if any(digest_file(p)!=h for p,h in config_pins.items()):
            raise ValueError('Configuration changed during diagnostics; rerun in a new output directory')
        write_report(root,mapping,coverage,fusion)
        passed=fusion is None or fusion['passed']
        dump(root/'run_status.json',{'status':'completed','functional_checks_passed':passed,'diagnostic_only':True,
             'partial_news_snapshot':coverage['frozen']['partial'] if coverage else None})
        dump(root/'manifest.json',{'kind':'isolated_news_pipeline_diagnostics','created_at':datetime.now().astimezone().isoformat(),
             'code_sha256':digest_file(__file__),'configuration_sha256':config_pins,
             'files':file_manifest(root),'diagnostic_only':True})
        print('[DONE] '+str(root/'REPORT.md'),flush=True)
        return 0 if passed else 1
    except Exception as exc:
        dump(root/'run_status.json',{'status':'failed','error':repr(exc),'diagnostic_only':True})
        raise


if __name__=='__main__':
    raise SystemExit(main())
