"""Scope review for proposed industry links, independent of company/market routing."""
import copy
import json
from datetime import datetime,timezone
from pathlib import Path
from .io import read_records,verify_files,digest_file,digest
from .routing import article_pin,keyed
from .time_axis import timestamp


def apply_scope_reviews(articles,directory=None,required=False,snapshot_sha=None,preserve_approval_time=False,automated=None):
    by_id=keyed(articles);reviews={};confirmation={};metadata={}
    if directory is not None:
        directory=Path(directory)
        manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('kind')!='industry_scope_review' or (snapshot_sha is not None and manifest.get('snapshot_manifest_sha256')!=snapshot_sha):
            raise ValueError('Industry scope review snapshot/kind mismatch')
        verify_files(directory,manifest,required=('industry_reviews.jsonl','user_confirmation.json'))
        reviews=keyed(read_records(directory/'industry_reviews.jsonl'))
        confirmation=json.loads((directory/'user_confirmation.json').read_text(encoding='utf-8'))
        metadata={'manifest_sha256':digest_file(directory/'manifest.json'),'directory':str(directory)}
    human=0;assistant=0
    for sid,row in reviews.items():
        if sid not in by_id or row.get('article_pin')!=article_pin(by_id[sid]):raise ValueError('Industry scope review article pin mismatch')
        if row.get('decision') not in ('include','exclude'):raise ValueError('Invalid industry scope decision')
        if not isinstance(row.get('reviewed_by'),str) or not row['reviewed_by'].strip() or len(row.get('rationale',''))<8:
            raise ValueError('Industry scope reviewer and rationale are required')
        if timestamp(row.get('reviewed_at')).to_pydatetime()>datetime.now(timezone.utc):raise ValueError('Future industry scope review')
        if row.get('review_method')=='human_confirmation':
            proof=confirmation.get('samples',{}).get(sid,{})
            response=confirmation.get('response')
            acknowledged=response=='认可' or (response=='是公司' and row['decision']=='exclude')
            if not acknowledged or confirmation.get('reviewed_by')!=row['reviewed_by'] or proof.get('article_pin')!=row['article_pin'] or proof.get('decision')!=row['decision']:
                raise ValueError('Industry review lacks a matching actual user confirmation')
            human+=1
        elif row.get('review_method')=='assistant_secondary_review':
            if row['decision']!='exclude':raise ValueError('Assistant scope review may restrict but cannot approve new industry propagation')
            assistant+=1
        else:raise ValueError('Unknown industry scope review method')
        if row['decision']=='include':
            if row.get('approved_impacts_hash')!=digest(by_id[sid].get('industry_impacts',[])):
                raise ValueError('Industry approval is not bound to these specific model proposals')
            text=(by_id[sid].get('title') or '')+' '+(by_id[sid].get('summary') or '')
            quote=row.get('evidence_quote')
            if not isinstance(quote,str) or len(quote)<8 or quote not in text:raise ValueError('Approved industry scope requires verbatim evidence')
    automatic=0
    for sid,row in (automated or {}).items():
        if sid in reviews:continue  # actual user exclusions always take precedence
        if sid not in by_id or row.get('review_method')!='automated_dictionary_summary_evidence_v1' or row.get('reviewed_by')!='upstream_evidence_gate_not_human':
            raise ValueError('Invalid verified upstream automatic scope review')
        if row.get('decision')!='include' or row.get('approved_impacts_hash')!=digest(by_id[sid].get('industry_impacts',[])):
            raise ValueError('Automatic scope approval differs from verified proposals')
        if timestamp(row.get('reviewed_at')).to_pydatetime()>datetime.now(timezone.utc):raise ValueError('Future automatic review')
        reviews[sid]=row;automatic+=1
    output=[];removed=0;pending=0;retained=0;pending_ids=[]
    for original in articles:
        a=copy.deepcopy(original);proposals=a.get('industry_impacts',[]);review=reviews.get(a['source_id'])
        if proposals and ((review and review['decision']=='exclude') or (required and not review)):
            if review:removed+=len(proposals)
            else:pending+=len(proposals);pending_ids.append(a['source_id'])
            a['industry_impacts']=[]
            a['industry_scope_review']={'decision':review['decision'] if review else 'pending',
                'reviewed_by':review.get('reviewed_by') if review else None,
                'rationale':review.get('rationale') if review else 'Industry propagation awaits semantic scope review.',
                'proposed_impacts':proposals,**metadata}
        else:
            retained+=len(proposals)
            if preserve_approval_time and proposals and review and review['decision']=='include':
                a['industry_scope_review']={k:review.get(k) for k in ('decision','review_method','reviewed_by','reviewed_at','rationale','approved_impacts_hash')}
                a['industry_scope_review'].update(metadata)
        output.append(a)
    return output,dict(metadata,human_confirmed_samples=human,assistant_reviewed_samples=assistant,automated_evidence_gated_samples=automatic,
        removed_impacts=removed,pending_impacts=pending,retained_impacts=retained,
        pending_source_ids=pending_ids,require_review=required)
