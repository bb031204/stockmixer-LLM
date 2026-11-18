"""Encode and audit final news inputs without activating snapshots or training StockMixer."""
import argparse,copy,importlib.util,json,sys
from datetime import datetime
from pathlib import Path


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work-dir',type=Path,help='Resume an isolated directory with frozen_inputs.json/adapted; never a formal training output')
    parser.add_argument('--allow-partial',action='store_true',help='Explicit incomplete-corpus diagnosis; cannot be marked full training ready')
    parser.add_argument('--datasets',nargs='+',choices=['NASDAQ','NYSE','SP500'])
    args=parser.parse_args(argv);project=args.project.resolve();sys.path.insert(0,str(project))
    from news_fusion.configuration import read_yaml,load_llm_config
    from news_fusion.io import digest_file,read_records,write_records,write_json,file_manifest,now
    spec=importlib.util.spec_from_file_location('precheck',project/'scripts/16_check_news_pipeline.py')
    pre=importlib.util.module_from_spec(spec);spec.loader.exec_module(pre)
    base=read_yaml(project/'configs/stockmixer.yaml');config=load_llm_config(project/'configs/llm.yaml')
    pins={str(f):digest_file(f) for f in [project/'configs/stockmixer.yaml',project/'configs/llm.yaml']}
    resolve=lambda v:Path(v) if Path(v).is_absolute() else project/v
    root=(args.work_dir or project/'outputs/artifacts/news/acceptance'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_news_acceptance')).resolve()
    root.relative_to(project/'outputs')
    global ACTIVE_ROOT
    ACTIVE_ROOT=root
    if args.work_dir:
        if not (root/'frozen_inputs.json').is_file():raise ValueError('Resume only an isolated frozen-news acceptance directory')
        if (root/'acceptance.json').exists():raise ValueError('This report is already sealed; choose a new directory')
    else:root.mkdir(parents=True,exist_ok=False)
    from news_fusion.source_adapter import adapt_snapshot,validate_adapted
    if not (root/'frozen_inputs.json').exists():
        producer,_=pre.freeze_export(resolve(config['pipeline']['retrace_artifact']),root,not args.allow_partial)
        adapt_snapshot(producer,root/'adapted',resolve(base['data']['axis_dir']),markets=('US',))
    frozen=json.loads((root/'frozen_inputs.json').read_text(encoding='utf-8'))
    if frozen['partial'] and not args.allow_partial:raise ValueError('Frozen snapshot is partial; explicit --allow-partial required')
    validate_adapted(root/'adapted')
    config=copy.deepcopy(config);config['adapted_snapshot']=str(root/'adapted')
    from news_fusion.offline_inputs import encoding_inputs
    articles,counts,provenance=encoding_inputs(project,config)
    initial={a['source_id']:a for a in articles}
    write_json(root/'run_status.json',{'status':'running','diagnostic_only':frozen['partial']})
    from news_fusion.panel import load_features,build_panel,compress
    features=root/'article_features'
    if not features.exists():
        import torch
        from news_fusion.encoder import LocalFinBert,encode_articles
        torch.set_num_threads(config['pipeline']['finbert_threads'])
        model=resolve(config['finbert']['model_dir']);installed=json.loads((model/'download_manifest.json').read_text(encoding='utf-8'))
        if any(digest_file(model/n)!=h for n,h in installed['files'].items()):raise ValueError('Local FinBERT changed')
        backend=LocalFinBert(model,config['finbert']['device'],config['finbert']['max_length'],installed['model_available_at'])
        encode_articles(articles,backend,features,resolve(config['pipeline']['cache']),config['finbert']['batch_size'],config['finbert']['output_format'],provenance,
            progress=lambda n,t,c:print('[ENCODE]',n,t,c,flush=True))
    encoded,vectors,fm=load_features(features)
    if fm['input_provenance']['snapshot_manifest_sha256']!=provenance['snapshot_manifest_sha256']:raise ValueError('Features use a different snapshot')
    from news_fusion.pipeline import validate_feature_settings
    model=resolve(config['finbert']['model_dir']);installed=json.loads((model/'download_manifest.json').read_text(encoding='utf-8'))
    validate_feature_settings(fm,config,installed)
    eb={a['source_id']:a for a in encoded}
    if set(eb)!=set(initial):raise ValueError('Feature articles differ from frozen snapshot')
    if any(eb[s]['input_text_hash']!=a['input_text_hash'] for s,a in initial.items()):raise ValueError('Feature text hashes differ')
    from news_fusion.sample_review import apply_review,market_proof
    path=config['pipeline'].get('sample_review')
    articles,sample_summary=apply_review(articles,resolve(path) if path else None)
    for a in articles:
        if a['source_id'] in config['pipeline']['industry_excluded_sources']:a['industry_impacts']=[]
    from news_fusion.retrace_adapter import automated_industry_reviews
    from news_fusion.industry_scope_review import apply_scope_reviews
    scope_path=config['pipeline'].get('industry_scope_review')
    articles,scope_summary=apply_scope_reviews(articles,resolve(scope_path) if scope_path else None,
        config['pipeline']['require_industry_scope_review'],provenance['snapshot_manifest_sha256'],True,
        automated_industry_reviews(articles,config['pipeline']['industry_min_confidence']) if config['pipeline']['allow_automated_industry_scope'] else None)
    axes,dictionary,tables,refs,mapping=pre.mapping_check(project,base,config,root)
    datasets=args.datasets or config['pipeline']['datasets'];results={};flat=[]
    from news_fusion.routing import publish_routes,load_routes
    from news_fusion.semantic_review import apply_sample
    from news_fusion.upstream import make_reviews
    from news_fusion.relations import build_relations,load_relations,validate_master
    from news_fusion.time_axis import export_schedule,Schedule
    from news_fusion.protocol import load_market
    from news_fusion.market_quality import filter_routes,cluster_events
    from news_fusion.final_acceptance import summarize_final,valid_windows,audit_panel
    from news_fusion.dataset import NewsPanel
    for dataset in datasets:
        print('[ACCEPT]',dataset,flush=True);axis=axes[dataset]
        reviews=make_reviews(articles,dataset,config['review']['min_confidence'])
        approvals_before_sample=sum(r['decision']=='include' for r in reviews)
        proof=market_proof(articles,reviews,manifest_sha=sample_summary.get('review_manifest_sha256'))
        if proof:reviews,_=apply_sample(articles,reviews,proof['rows'],proof['confirmation'])
        route_dir=root/(dataset+'_routes');publish_routes(articles,reviews,dataset,route_dir,provenance,True,semantic_audit=proof)
        routes,_,route_manifest=load_routes(route_dir,dataset)
        paths=[resolve(config['pipeline'][k]) if config['pipeline'][k] else None for k in ('security_master','industry_memberships','industry_aliases')]
        relations=root/(dataset+'_relations')
        reference_sources=(resolve(config['industry_dictionary']['reference_archive']),resolve(base['data']['dataset_root'])/'SP500/sp500_ticker.csv') if config['pipeline']['industry_membership_mode']=='reference_retrospective' else None
        build_relations(axis,articles,paths,relations,provenance,policy={'version':'evidence_mapping_v3','industry_min_confidence':config['pipeline']['industry_min_confidence']},dictionary=dictionary,reference_sources=reference_sources)
        links,members,rm=load_relations(relations,axis)
        export_schedule(axis,root/(dataset+'_schedule.jsonl'));schedule=Schedule(axis,read_records(root/(dataset+'_schedule.jsonl')))
        # Match the public configuration's prepared-data path without invoking fusion preparation recursively.
        from news_fusion.configuration import compile_config
        baseline=copy.deepcopy(base);baseline['llm']['enabled']=False;baseline['market_name']=dataset;baseline['data']['protocol']=config['pipeline']['data_protocol']
        compiled=compile_config(project,project/'configs/stockmixer.yaml',baseline,dataset)
        data=load_market(resolve(compiled['workflow']['prepared_data'][dataset]))
        success=[dict(eb[a['source_id']],**a) for a in articles if eb[a['source_id']]['finbert_status']=='success']
        quality=config.get('market_channel');allowed=routes;bases={}
        if quality:
            if quality['official_events'] is not None:raise ValueError('This standalone snapshot audit requires news-only inputs; official events are audited by main.py validation')
            allowed,bases=filter_routes(routes,{a['source_id']:a for a in success},set(quality['evidence_basis']))
            if not quality['news_routes']:allowed=[]
        securities,_,_=validate_master(axis,*tables)
        groups,rejections=pre.enumerate_groups(axis,schedule,success,links+allowed,members,securities,config['panel'])
        entries={}
        for (ch,stock,t),items in groups.items():
            if ch=='market':
                for sid in items.values():
                    a=eb[sid];entries.setdefault(sid,{'source_id':sid,'session':schedule.first_session(a,'retrospective'),
                        'embedding_row':a['embedding_row'],'event_type':a.get('event_type')})
        events=cluster_events(list(entries.values()),vectors,quality['event_dedup']['window_sessions'],quality['event_dedup']['cosine_threshold']) if quality else {sid:'article_'+sid for sid in entries}
        import numpy as np
        masks={ch:np.zeros((len(axis.stock_uids),len(axis.dates)),dtype=bool) for ch in ('company','industry','market')}
        ledger=[];eligible=set()
        for (ch,stock,t),items in groups.items():
            masks[ch][slice(None) if stock<0 else stock,t]=True
            for text,sid in items.items():
                owners=range(len(axis.stock_uids)) if stock<0 else [stock]
                for i in owners:ledger.append({'source_id':sid,'channel':ch,'stock_uid':axis.stock_uids[i],'decision_idx':t,
                    'input_text_hash':text,'event_type':eb[sid].get('event_type'),'market_event_id':events.get(sid)})
                if config['panel']['decision_lookback']-1<=t<axis.train_end-1:eligible.add(eb[sid]['embedding_row'])
        final=summarize_final(axis,masks,ledger,valid_windows(data,config['panel']['decision_lookback']),config)
        final.update(panel_materialized=False,encoded_articles=len(encoded),finbert_successful=fm['successful'],
            market_routes_before_sample_restrictions=approvals_before_sample,
            market_routes_before_evidence_filter=len(routes),market_routes_after_evidence_filter=len(allowed),
            market_route_evidence_basis=bases,pca_train_articles=len(eligible),time_mapping_rejections=rejections)
        panel_path=root/(dataset+'_panel')
        try:
            compress(vectors,sorted(eligible),config['panel']['projection_components'],config['panel']['projection'],config['panel']['projection_seed'])
        except ValueError as e:
            final['blockers'].append(str(e));final['projection_passed']=False
        else:
            pc=config['panel']
            build_panel(axis,schedule,features,links,members,panel_path,pc['mode'],pc['projection'],pc['lookback_sessions'],pc['decay_tau_sessions'],
                pc['simulated_latency_seconds'],pc['projection_seed'],pc['decision_lookback'],route_dir,relations,pc['projection_components'],pc['relation_time_basis'],quality)
            audited=audit_panel(NewsPanel(panel_path),data,config,features)
            final.update(audited);final['projection_passed']=True
        final['ready']=not final['blockers'] and final['panel_materialized'] and not frozen['partial']
        if frozen['partial']:final['blockers'].append('Incomplete source corpus; diagnostic snapshot cannot activate formal training')
        for ch,splits in final['coverage'].items():
            for phase,stat in splits.items():flat.append({'dataset':dataset,'channel':ch,'split':phase,**stat})
        results[dataset]=final;write_json(root/(dataset+'_final_inputs.json'),final)
    pre.table(root/'final_input_coverage.csv',flat)
    report={'kind':'final_news_input_acceptance_v1','created_at':now(),'partial':frozen['partial'],
        'frozen_exported_articles':frozen['frozen_exported_articles'],'source_total':frozen['run']['total'],
        'adapter':json.loads((root/'adapted/manifest.json').read_text(encoding='utf-8')),
        'finbert':{k:fm.get(k) for k in ('articles','successful','counts','status_counts')},
        'sample_review':sample_summary,'industry_scope_review':scope_summary,'datasets':results,
        'ready':all(r['ready'] for r in results.values()),'formal_training_started':False}
    write_json(root/'acceptance.json',report)
    lines=['# 最终新闻输入验收','',f'冻结 {frozen["frozen_exported_articles"]}/{frozen["run"]["total"]} 篇；FinBERT成功 {fm["successful"]}/{fm["articles"]} 篇；正式训练就绪：{report["ready"]}。',
      '统计已扣除编码失败、证据筛选、身份/行业/时间不匹配、重复文本和无效完整行情窗口；市场事件按真实FinBERT余弦聚类计数，重复报道仍保留但按同一事件分配权重。',
      'PCA或覆盖门槛未通过时，不发布空面板，不降低门槛，不启动训练。','',
      '|数据集|通道|训练文章|验证文章|测试文章|训练有效股票日|','|---|---|---:|---:|---:|---:|']
    for ds,f in results.items():
        for ch,c in f['coverage'].items():lines.append(f'|{ds}|{ch}|{c["train"]["articles"]}|{c["valid"]["articles"]}|{c["test"]["articles"]}|{c["train"]["valid_price_label_cells"]}|')
    for ds,f in results.items():
        lines+=['',f'## {ds} 阻断项']+['- '+b for b in f['blockers']]
        lines+=['',f'训练期PCA候选文章：{f.get("pca_train_articles")}；面板已生成：{f["panel_materialized"]}。']
    lines+=['','完整事件数、有效天数及门槛见各市场 `*_final_inputs.json`；逐通道分划分见 `final_input_coverage.csv`。',
        '当前实验口径仍为 original_reproduction + retrospective，原始训练候选含16个验证目标；实际训练候选覆盖另列，不把该实验称为无泄露PIT。']
    (root/'FINAL_ACCEPTANCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if any(digest_file(p)!=h for p,h in pins.items()):raise ValueError('Configuration changed during acceptance')
    write_json(root/'run_status.json',{'status':'completed','ready':report['ready'],'partial':frozen['partial'],'formal_training_started':False})
    write_json(root/'manifest.json',{'kind':'isolated_final_news_acceptance','created_at':now(),
        'configuration_sha256':pins,'script_sha256':digest_file(__file__),'files':file_manifest(root)})
    print('[DONE]',root/'FINAL_ACCEPTANCE_REPORT.md',flush=True)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        root=globals().get('ACTIVE_ROOT')
        # Do not mutate a sealed report when the caller accidentally reuses its directory.
        if root and root.exists() and not (root/'acceptance.json').exists():
            (root/'run_status.json').write_text(json.dumps({'status':'failed','error':repr(exc),'formal_training_started':False},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
