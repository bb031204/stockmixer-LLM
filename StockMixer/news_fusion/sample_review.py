"""Source-bound assistant sample findings. Restrictions only; never human approval."""
import copy
import json
from pathlib import Path
from .io import digest,digest_file,read_records,verify_files
from .time_axis import timestamp


def claim_pin(a):
    return digest({k:a.get(k) for k in ('source_id','source_record_hash','input_text_hash',
        'impact_scope','industry_impacts','market_scope','scope_evidence')})


def apply_review(articles,directory):
    if directory is None:return articles,{'reviewed_samples_present':0,'applied_restrictions':0,'human_confirmed_samples':0}
    directory=Path(directory)
    m=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if m.get('kind')!='source_bound_semantic_sample_v1':raise ValueError('Wrong sample review kind')
    verify_files(directory,m,required=('reviews.jsonl',))
    rows=read_records(directory/'reviews.jsonl')
    if m.get('sample_count')!=len(rows):raise ValueError('Sample review count mismatch')
    byid={a['source_id']:a for a in articles};seen=set();blocked={};present=0
    from datetime import datetime,timezone
    for row in rows:
        sid=row['source_id']
        if sid in seen:raise ValueError('Duplicate sample review')
        seen.add(sid)
        if row.get('review_method')!='assistant_secondary_review' or row.get('reviewed_by')!='Codex':
            raise ValueError('This artifact cannot claim human approval')
        if timestamp(row.get('reviewed_at')).to_pydatetime()>datetime.now(timezone.utc):raise ValueError('Future sample review')
        if len(row.get('rationale',''))<8:raise ValueError('Substantive review rationale required')
        channels=row.get('exclude_channels')
        if not isinstance(channels,list) or len(set(channels))!=len(channels) or not set(channels)<={'company','industry','market'}:
            raise ValueError('Sample restrictions must be explicit channel exclusions')
        if row.get('verdict') not in ('pass_with_limits','restrict','needs_review'):raise ValueError('Unknown sample verdict')
        keys=row.get('exclude_industry_keys',[])
        if not isinstance(keys,list) or not all(isinstance(k,str) for k in keys) or len(set(keys))!=len(keys):raise ValueError('Invalid industry key restrictions')
        if (channels or keys) and row['verdict']!='restrict':raise ValueError('Only restrictive findings may change routing')
        if sid not in byid:continue
        a=byid[sid];present+=1
        if row.get('claim_pin')!=claim_pin(a):raise ValueError('Sample review is stale for article '+sid)
        if not set(keys)<={i.get('industry_key') for i in a.get('industry_impacts',[])}:raise ValueError('Restricted industry key is absent from this article')
        quote=row.get('evidence_quote')
        if not isinstance(quote,str) or len(quote)<8:raise ValueError('Sample evidence quote required')
        source=(a.get('title') or '')+' '+(a.get('summary') or '')
        from .source_proof import bound_quote
        if quote not in source and not any(bound_quote(a,f,quote) for f in ('industry_impacts','scope_evidence','impact_scope_evidence')):
            raise ValueError('Sample evidence is not bound to article '+sid)
        blocked[sid]=row
    output=[]
    for original in articles:
        a=copy.deepcopy(original);row=blocked.get(a['source_id'])
        if row:
            a['semantic_sample_review']={k:row.get(k) for k in ('verdict','exclude_channels','exclude_industry_keys','rationale','review_method','reviewed_by','reviewed_at','claim_pin')}
            if 'industry' in row['exclude_channels']:
                a['sample_excluded_industry_impacts']=a.get('industry_impacts',[])
                a['industry_impacts']=[]
            elif row.get('exclude_industry_keys'):
                a['sample_excluded_industry_impacts']=[i for i in a.get('industry_impacts',[]) if i.get('industry_key') in row['exclude_industry_keys']]
                a['industry_impacts']=[i for i in a.get('industry_impacts',[]) if i.get('industry_key') not in row['exclude_industry_keys']]
            # Company routing is checked by the deterministic resolver; drop its suggestions only.
            if 'company' in row['exclude_channels']:a['entities']=[]
            # Market exclusions are enforced in the review stage without changing article_pin.
        output.append(a)
    return output,{'sample_count':len(rows),'reviewed_samples_present':present,
        'applied_restrictions':sum(bool(r['exclude_channels'] or r.get('exclude_industry_keys')) for sid,r in blocked.items()),
        'human_confirmed_samples':0,'review_manifest_sha256':digest_file(directory/'manifest.json'),
        'review_scope':'risk-targeted sample, not population accuracy or full-source factual verification'}


def market_proof(articles,reviews,existing=None,manifest_sha=None):
    """Use the existing semantic overlay/replay contract for market exclusions."""
    byid={a['source_id']:a for a in articles}
    from .semantic_review import decision_pin
    from .routing import article_pin
    rows={r['source_id']:copy.deepcopy(r) for r in (existing or {}).get('rows',[])}
    added=False
    for row in reviews:
        audit=byid[row['source_id']].get('semantic_sample_review') or {}
        if 'market' not in audit.get('exclude_channels',[]):continue
        previous=rows.get(row['source_id'])
        if previous and previous.get('review_method')=='human_confirmation':
            if previous['decision']=='include':raise ValueError('Sample restriction conflicts with explicit human market approval')
            continue
        rows[row['source_id']]={'source_id':row['source_id'],'article_pin':article_pin(byid[row['source_id']]),
            'original_decision_pin':decision_pin(row),'decision':'exclude','review_method':'assistant_secondary_review',
            'reviewed_by':audit['reviewed_by'],'reviewed_at':audit['reviewed_at'],'evidence_quote':None,
            'rationale':audit['rationale'],'apply_decision':True,'quarantine':False,
            'quality_flags':['source_bound_sample_market_exclusion']}
        added=True
    if not added:return existing
    return {'rows':list(rows.values()),'confirmation':(existing or {}).get('confirmation',{}),
        'original_reviews':reviews,'manifest_sha256':digest([manifest_sha,(existing or {}).get('manifest_sha256')])}
