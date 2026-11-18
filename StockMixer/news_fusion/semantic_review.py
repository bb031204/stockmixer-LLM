"""Replayable semantic sample review; assistant judgments never count as human signatures."""
import copy
import json
from pathlib import Path
from .io import digest, digest_file, read_records, verify_files
from .routing import article_pin, keyed
from .time_axis import timestamp


def decision_pin(row):
    return digest({k:row.get(k) for k in ('source_id','article_pin','decision','evidence_quote','rationale',
                                         'reviewed_by','reviewed_at')})


def load_sample(directory, articles, reviews=None,snapshot_manifest_sha256=None):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='semantic_sample_review':
        raise ValueError('Not a semantic sample review artifact')
    if snapshot_manifest_sha256 is not None and manifest.get('snapshot_manifest_sha256')!=snapshot_manifest_sha256:
        raise ValueError('Semantic sample belongs to another frozen snapshot')
    verify_files(directory,manifest,required=('semantic_reviews.jsonl','user_confirmation.json','original_market_review.jsonl'))
    if reviews is None:
        reviews=read_records(directory/'original_market_review.jsonl')
    rows=read_records(directory/'semantic_reviews.jsonl')
    confirmation=json.loads((directory/'user_confirmation.json').read_text(encoding='utf-8'))
    from .routing import checked_reviews
    checked_reviews(articles,reviews,True,reviews[0]['dataset'] if reviews else None)
    _,summary=apply_sample(articles,reviews,rows,confirmation)
    if manifest.get('summary')!=summary:raise ValueError('Semantic manifest summary differs from its actual sample records')
    return {'rows':rows,'confirmation':confirmation,'manifest_sha256':digest_file(directory/'manifest.json'),
            'source':str(directory),'original_reviews':reviews}


def apply_sample(articles,reviews,rows,confirmation):
    from datetime import datetime, timezone
    by_id=keyed(articles); original=keyed(reviews); selected=keyed(rows)
    result=copy.deepcopy(reviews); targets=keyed(result)
    humans=0; assistants=0; applied=0; quarantined=[]
    for sid,row in selected.items():
        if sid not in by_id or sid not in original or row.get('article_pin')!=article_pin(by_id[sid]):
            raise ValueError('Semantic sample does not match the frozen article')
        if row.get('original_decision_pin')!=decision_pin(original[sid]):
            raise ValueError('Semantic sample belongs to a different original review decision')
        method=row.get('review_method')
        if method not in ('assistant_secondary_review','human_confirmation'):
            raise ValueError('Unknown semantic reviewer method')
        if not isinstance(row.get('reviewed_by'),str) or not row['reviewed_by'].strip():
            raise ValueError('Semantic reviewer identity is required')
        if timestamp(row.get('reviewed_at')).to_pydatetime()>datetime.now(timezone.utc):
            raise ValueError('Future semantic review time')
        if not isinstance(row.get('rationale'),str) or len(row['rationale'].strip())<8:
            raise ValueError('Substantive semantic rationale is required')
        if type(row.get('apply_decision')) is not bool or type(row.get('quarantine')) is not bool:
            raise ValueError('Semantic apply/quarantine flags must be booleans')
        if row.get('decision') not in ('include','exclude','pending'):
            raise ValueError('Invalid semantic decision')
        if method=='human_confirmation':
            check=confirmation.get('samples',{}).get(sid,{})
            if confirmation.get('response')!='认可' or confirmation.get('reviewed_by')!=row['reviewed_by'] or check.get('article_pin')!=row['article_pin'] or check.get('decision')!=row['decision']:
                raise ValueError('Human label lacks a matching actual user confirmation')
            humans+=1
        else:
            assistants+=1
            if row['apply_decision'] and row['decision']=='include' and original[sid]['decision']!='include':
                raise ValueError('Assistant sample cannot silently promote an unapproved article to the market channel')
        if row['decision']=='include':
            quote=row.get('evidence_quote')
            text=(by_id[sid].get('title') or '')+' '+(by_id[sid].get('summary') or '')
            if not isinstance(quote,str) or len(quote)<8 or quote not in text:
                raise ValueError('Included semantic review lacks a verbatim article quote')
        if row['quarantine']:
            if row['decision']=='include':raise ValueError('A quarantined article cannot be included')
            quarantined.append(sid)
        if row['apply_decision']:
            targets[sid].update({k:row.get(k) for k in ('decision','evidence_quote','rationale','reviewed_by','reviewed_at','review_method')})
            targets[sid]['semantic_review']={'original_decision_pin':row['original_decision_pin'],
                                            'quarantine':row['quarantine'],'quality_flags':row.get('quality_flags',[])}
            applied+=1
    summary={'sample_count':len(rows),'human_confirmed_samples':humans,'assistant_reviewed_samples':assistants,
             'applied_samples':applied,'human_sample_checked':humans>0,
             'quarantined_source_ids':sorted(quarantined),'scope':'sampled frozen title/summary only; not all articles or source-document factual verification'}
    return result,summary


def annotate_articles(articles,proof):
    _,summary=apply_sample(articles,proof['original_reviews'],proof['rows'],proof['confirmation'])
    blocked=set(summary['quarantined_source_ids'])
    return [dict(a,semantic_quarantine=True) if a['source_id'] in blocked else copy.deepcopy(a) for a in articles]
