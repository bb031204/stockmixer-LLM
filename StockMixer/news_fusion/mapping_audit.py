"""Read-only real-data mapping audit; candidate outputs never imply training readiness."""
from collections import Counter
from pathlib import Path
from .io import digest_file, file_manifest, new_artifact, now, read_records, write_json, write_records
from .inputs import normalize_article
from .offline_inputs import encoding_inputs
from .relations import resolve_relations, validate_master
from .source_adapter import Snapshot, adapt_one, producer_hash
from .time_axis import Axis
from .upstream import make_reviews


def audit_mapping(project, config, axis_directory, output):
    project = Path(project).resolve()
    def path(value):
        p=Path(value)
        return p.resolve() if p.is_absolute() else (project/p).resolve()
    active, _, provenance = encoding_inputs(project,config)
    settings=config['pipeline']
    names=settings['datasets']
    master_paths=[path(settings[k]) if settings[k] else None for k in
                  ('security_master','industry_memberships','industry_aliases')]
    master_hashes={str(p):digest_file(p) for p in master_paths if p}
    for parent in {p.parent for p in master_paths if p}:
        import json
        manifest=parent/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('kind')=='sec_sic_dated_historical_observations':
            from .master_data import verify_observations
            verify_observations(parent)
    tables=[read_records(p) if p else [] for p in master_paths]
    source_root=settings.get('upstream_source_root')
    candidates=[]; proofs=[]; checks=[]; source=None; live_rows={}; audits=[]
    if source_root:
        source=Snapshot(path(source_root),('US','GLOBAL'),allow_append_only=True)
        views={}
        for market,files in source.membership.items():
            for relative in files:
                row=source.json(relative);sid=row['source_id']
                if sid in live_rows and producer_hash(live_rows[sid])!=producer_hash(row):
                    raise ValueError('Conflicting upstream copies for '+sid)
                live_rows[sid]=row;views.setdefault(sid,set()).add(market)
        for sid,row in sorted(live_rows.items()):
            adapted,proof,check=adapt_one(source,row,views[sid])
            checks.append(dict(check,adapter_status=adapted['adapter_status']))
            proofs.append(proof)
            descriptor=proof['selected_audit']
            if descriptor:
                audits.append({'source_id':sid,'path':descriptor['path'],'sha256':descriptor['sha256'],
                               'payload':source.json(descriptor['path'])})
            if adapted['adapter_status']=='ready':candidates.append(normalize_article(adapted))
        source.verify()
    dictionary=None
    if config.get('industry_dictionary'):
        from .industry_dictionary import IndustryDictionary
        dictionary=IndustryDictionary(path(config['industry_dictionary']['directory']))
    axes={name:Axis.load(Path(axis_directory)/(name+'.json')) for name in names}
    report={'kind':'news_mapping_gap_audit','created_at':now(),'model_calls':0,'training_run':False,
            'source_read_only':True,'active_snapshot':provenance,'active_articles':len(active),
            'upstream_unique_articles':len(live_rows),'upstream_ready_candidates':len(candidates),
            'upstream_not_ready':len(live_rows)-len(candidates),
            'upstream_industry_analysis_articles':sum(bool(a['payload'].get('industry_impacts')) for a in audits),
            'upstream_typed_scope_articles':sum(bool(a.get('market_scope') or a.get('affected_markets')) for a in candidates),
            'master_input_sha256':master_hashes,'datasets':{},'versions':{}}
    with new_artifact(output) as directory:
        write_records(directory/'upstream_candidate_articles.jsonl',candidates)
        write_records(directory/'upstream_provenance.jsonl',proofs)
        write_records(directory/'upstream_audits.jsonl',audits)
        write_records(directory/'upstream_time_checks.jsonl',checks)
        if source:
            write_json(directory/'upstream_source_snapshot.json',{'source_root':str(source.root),'market_files':source.membership,
                       'files':source.hashes,'snapshot_started_at':source.started_at,'later_additions_excluded':source.new_files})
        coverage=[]
        for name,axis in axes.items():
            securities,memberships,aliases=validate_master(axis,*tables)
            identities={s['stock_uid'] for s in securities}
            report['datasets'][name]={'stocks':len(axis.stock_uids),'verified_stock_observations':len(identities),
                'unverified_stocks':len(axis.stock_uids)-len(identities),'security_rows':len(securities),
                'industry_membership_rows':len(memberships),'industry_alias_rows':len(aliases),
                'continuous_history_complete':False}
            for i,uid in enumerate(axis.stock_uids):
                observations=[s for s in securities if s['stock_uid']==uid]
                coverage.append({'dataset':name,'stock_uid':uid,'array_index':i,
                    'dataset_ticker':axis.source.get('tickers',[v.split(':',1)[1] for v in axis.stock_uids])[i],
                    'status':'partial_verified_observations' if observations else 'no_verified_historical_observation',
                    'observations':[{k:s.get(k) for k in ('security_id','company_name','exchange','valid_from','valid_to','known_from','mapping_source')} for s in observations]})
            for label,articles in [('active',active),('upstream',candidates)]:
                links,_,pending,counts=resolve_relations(axis,articles,*tables,policy={'version':'evidence_mapping_v2',
                    'industry_min_confidence':settings['industry_min_confidence']},dictionary=dictionary)
                market_reviews=make_reviews(articles,name,config['review']['min_confidence'])
                write_records(directory/(name+'_'+label+'_links.jsonl'),links)
                write_records(directory/(name+'_'+label+'_unmatched.jsonl'),pending)
                write_records(directory/(name+'_'+label+'_market_review.jsonl'),market_reviews)
                report['versions'].setdefault(label,{})[name]={'articles':len(articles),'relation_counts':counts,
                    'unmatched_reasons':dict(Counter(r['reason'] for r in pending)),
                    'market_decisions':dict(Counter(r['decision'] for r in market_reviews)),
                    'market_pending_reasons':dict(Counter(r['upstream_mapping_reason'] for r in market_reviews if r['decision']=='pending'))}
        write_records(directory/'stock_mapping_coverage.jsonl',coverage)
        report['limitations']=['Deterministic verified matches are not calibrated probabilities; mapping_confidence remains null.',
            'SEC-SIC evidence covers isolated filing dates, never interpolated continuous listing/industry history.',
            'Missing upstream confidence, literal evidence or typed market scope is pending; no downstream Qwen rerun.',
            'Upstream candidate artifacts are diagnostics, not a published training snapshot.',
            'Original records and inference/availability timestamps are unchanged.']
        if source:source.verify()
        if any(digest_file(p)!=h for p,h in master_hashes.items()):raise ValueError('Master inputs changed during audit')
        report['source_code_sha256']={p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}
        report['files']=file_manifest(directory)
        write_json(directory/'manifest.json',report)
    return report
