"""Replayable import of upstream industry-only batches; no model calls or stock guesses."""
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile
from .industry_dictionary import IndustryDictionary
from .io import digest,digest_file,file_manifest,new_artifact,now,read_records,verify_files,write_json,write_records
from .source_adapter import producer_hash
from .snapshot import validate_snapshot
from .inputs import normalize_article,text

_VALIDATED={}

KIND='frozen_industry_enriched_selection_v1'
BATCH_KIND='legacy-dictionary-industry-v1'
EXCLUDED={'1b02e559d81c568ffa4b6ff8','62f0ea7d749999a245b9525b','777b97f602ed63a72ee4f488'}
QUARANTINE={'00cf47b880e8fc0684b11eaa','01d91eb93988b6150f2cc241'}
FIELDS=('industry_key','quote','scope','scope_quote','mechanism','direction','confidence')

def parse(raw):
    def reject(value):raise ValueError('Nonfinite batch JSON')
    return json.loads(raw,parse_constant=reject)

def archive_bytes(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        result={}
        for name in names:
            if '\\' in name or name.startswith('/') or any(p in ('','..','.') for p in name.split('/')) or ':' in name:
                raise ValueError('Unsafe artifact archive member')
            if archive.getinfo(name).file_size>100_000_000:raise ValueError('Oversized archive member')
            result[name]=archive.read(name)
    return result

def verify_payload(payload):
    manifest=parse(payload['manifest.json'])
    if set(payload)!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Archived artifact has extra/missing files')
    for name,expected in manifest['files'].items():
        if hashlib.sha256(payload[name]).hexdigest()!=expected:raise ValueError('Archived evidence SHA mismatch')
    return manifest

def batch_rows(payload,dictionary):
    manifest=verify_payload(payload);run=parse(payload['run.json'])
    if manifest['kind']!=BATCH_KIND or run['kind']!=BATCH_KIND or manifest['pending']!=0:
        raise ValueError('Industry batch must finish inspecting the complete frozen corpus')
    settings=run['settings']
    if settings['dictionary_sha256']!=dictionary.sha256 or run['fingerprint']!=producer_hash(settings) or manifest['fingerprint']!=run['fingerprint']:
        raise ValueError('Industry batch configuration/dictionary fingerprint mismatch')
    if producer_hash(payload['industry_prompt.txt'].decode('utf-8').replace('\r\n','\n'))!=settings['prompt_sha256']:
        raise ValueError('Industry batch prompt changed')
    index=[parse(line) for line in payload['input_index.jsonl'].splitlines() if line.strip()]
    frozen=archive_bytes(io.BytesIO(payload['raw_news.zip']));raw={}
    for entry in index:
        body=frozen[entry['archive_path']]
        record=parse(body.decode('utf-8-sig'))
        if hashlib.sha256(body).hexdigest()!=entry['sha256'] or producer_hash(record)!=entry['record_hash'] or record['source_id']!=entry['source_id']:
            raise ValueError('Industry batch raw record binding mismatch')
        sid=record['source_id']
        if sid in raw and producer_hash(raw[sid])!=producer_hash(record):raise ValueError('Conflicting batch copies')
        raw[sid]=record
    values=[parse(line) for line in payload['industry_classifications.jsonl'].splitlines() if line.strip()]
    result={r['source_id']:r for r in values}
    if len(result)!=len(values) or set(result)!=set(raw) or len(raw)!=run['unique_articles']:
        raise ValueError('Industry batch identity/coverage mismatch')
    parent_rows={}
    if 'parent_batch.zip' in payload:
        parent_payload=archive_bytes(io.BytesIO(payload['parent_batch.zip']))
        parent_rows,parent_raw,_=batch_rows(parent_payload,dictionary)
        parent_sha=hashlib.sha256(parent_payload['manifest.json']).hexdigest()
        if run.get('parent_manifest_sha256')!=parent_sha or settings.get('parent_manifest_sha256')!=parent_sha or manifest.get('parent_manifest_sha256')!=parent_sha:
            raise ValueError('Recovery parent manifest binding differs')
        if raw!=parent_raw or sorted(run['recovery_target_ids'])!=sorted(s for s,r in parent_rows.items() if r['status']!='completed'):
            raise ValueError('Recovery source corpus or failed-record selection changed')
    for sid,item in result.items():
        inherited=sid in parent_rows and parent_rows[sid]['status']=='completed'
        if inherited and item!=parent_rows[sid]:raise ValueError('A completed parent analysis was rewritten or re-inferred')
        valid_fingerprint=(inherited or item['fingerprint']==run['fingerprint'])
        if sid in parent_rows and not inherited:
            if item.get('recovery_parent_analysis_sha256')!=producer_hash(parent_rows[sid]) or item.get('prompt_sha256')!=settings['prompt_sha256'] or not item.get('recovery_attempted_at'):
                raise ValueError('Recovered record lacks the failed-parent and actual recovery provenance')
        if parse(payload['analysis/'+sid+'.json'])!=item or not valid_fingerprint or item['record_hash']!=producer_hash(raw[sid]):
            raise ValueError('Industry sidecar identity/fingerprint mismatch')
        summary=raw[sid]['主要内容']
        if item['summary_sha256']!=producer_hash(summary) or item['text_basis']!='stored_summary_only' or item['historical_observed_pit_ready'] is not False:
            raise ValueError('Industry summary evidence/provenance mismatch')
        for field in ('feature_available_at','llm_inferred_at'):
            if item.get(field):
                stamp=datetime.fromisoformat(item[field].replace('Z','+00:00'))
                if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):raise ValueError('Invalid batch availability time')
        if item['status'] not in ('completed','needs_review'):raise ValueError('Unknown batch status')
        if item['status']!='completed' and (item.get('industries') or item.get('industry_impacts')):
            raise ValueError('Failed classification cannot publish industry features')
        candidates=dictionary.candidates(summary,settings['candidate_limit'],settings['active_standards'],headline=raw[sid].get('title') or '')
        if item['candidate_keys']!=[c['key'] for c in candidates]:raise ValueError('Dictionary retrieval cannot be replayed')
        selected=[]
        for topic in item.get('industries',[]):
            key=topic['industry_key'];row=dictionary.by_key.get(key);quote=topic['quote'];scope=topic['scope_quote'];score=topic['confidence']
            if topic.get('scope') not in ('company_only','industry_wide') or topic.get('direction') not in ('positive','negative','mixed','unclear') or isinstance(score,bool) or not isinstance(score,(int,float)) or not 0<=score<=1:
                raise ValueError('Invalid classified industry scope/direction/confidence')
            if row is None or key not in item['candidate_keys'] or not isinstance(quote,str) or quote not in summary or not dictionary.evidence_matches(key,quote):
                raise ValueError('Unverified industry key/summary quote')
            for field,name in (('industry_standard','industry_standard'),('industry_version','industry_version'),('industry_code','industry_code'),('industry_name_standard','name')):
                if topic[field]!=row[name]:raise ValueError('Industry canonical dictionary field differs')
            if topic['dictionary_sha256']!=dictionary.sha256 or topic['dictionary_version']!=dictionary.manifest['version'] or topic['original_evidence_verified'] is not False or topic['evidence_basis']!='stored_summary_only':
                raise ValueError('Summary annotation cannot claim original-document verification')
            broad=r'\b(industry|industries|sector|manufacturers|producers|lenders|banks|retailers|automakers|utilities|airlines|carriers|insurers|companies|firms|competitors|suppliers)\b'
            effect=r'\b(demand|supply|shortage|shortages|regulat\w*|tariffs?|tax\w*|competition|competitive|prices?|pricing|costs?|production|sales|grow\w*|declin\w*|reduc\w*|increas\w*|decreas\w*|requir\w*|restrict\w*|disrupt\w*|capacity|outlook|adopt\w*|shift\w*|rate\w*)\b'
            approved=(topic['scope']=='industry_wide' and isinstance(scope,str) and len(scope)>=8 and scope in summary
                and bool(re.search(broad,scope,re.I)) and bool(re.search(effect,scope,re.I))
                and len(topic['mechanism'])>=12 and not isinstance(score,bool) and isinstance(score,(int,float))
                and settings['min_confidence']<=score<=1 and sid not in EXCLUDED|QUARANTINE)
            if topic['propagation_approved'] is not approved:raise ValueError('Industry publication gate cannot be replayed')
            if approved:
                selected.append(dict(topic,industry_name_original=topic['industry_name_standard'],industry_name=topic['industry_name_standard'],
                    quote=quote,evidence=quote,industry_scope_quote=scope,evidence_type='inferred',impact_role='affected',
                    horizon='unspecified',llm_inferred_at=item['llm_inferred_at']))
        if selected!=item.get('industry_impacts',[]):raise ValueError('Industry impacts differ from evidence-gated classifications')
    return result,raw,manifest

def context_scope_evidence(topic,summary,minimum=.85,maximum_chars=600):
    """Recover a bounded verbatim context for short scope fragments; no new semantic/model labels.

    Only already industry-wide Qwen proposals about explicit US regulatory actions or
    official sector statistics are eligible. A company financing its own operations is not.
    """
    quote=topic.get('scope_quote');score=topic.get('confidence')
    if topic.get('scope')!='industry_wide' or not isinstance(quote,str) or len(quote)<8 or quote not in summary:
        return None
    if isinstance(score,bool) or not isinstance(score,(float,int)) or not minimum<=score<=1 or len(topic.get('mechanism',''))<12:return None
    position=summary.find(quote);end=position+len(quote)
    boundaries=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z])',summary)]+[len(summary)]
    left=max(p for p in boundaries if p<=position);right=min(p for p in boundaries if p>=end)
    # Include one neighboring sentence only when the exact scope fragment is retained.
    start_index=boundaries.index(left);end_index=boundaries.index(right)
    options=[summary[left:right].strip()]
    if start_index>0:options.append(summary[boundaries[start_index-1]:right].strip())
    if end_index+1<len(boundaries):options.append(summary[left:boundaries[end_index+1]].strip())
    broad=r'\b(industry|industries|sector|manufacturers|producers|lenders|banks|retailers|automakers|utilities|airlines|carriers|insurers|companies|firms|suppliers)\b'
    regulator=r'\b(Federal Reserve|Federal Deposit Insurance Corporation|FDIC|Office of the Comptroller|OCC|Securities and Exchange Commission|SEC|Food and Drug Administration|FDA|Environmental Protection Agency|EPA)\b'
    action=r'\b(issued|published|proposed|adopted|released|announced|finalized|requires?|requiring|amended|established)\b'
    rule=r'\b(guidance|regulations?|rules?|requirements?|standards?|restrictions?)\b'
    statistics=r'\b(Bureau of Labor Statistics|Labor Department|Department of Labor|Census Bureau|Department of Commerce)\b'
    metric=r'\b(employment|jobs|payrolls|output|production|sales|prices|orders)\b'
    for context in options:
        if len(context)>maximum_chars or quote not in context or not re.search(broad,context,re.I):continue
        regulation=all(re.search(p,context,re.I) for p in (regulator,action,rule))
        sector_statistics=bool(re.search(statistics,context,re.I) and re.search(metric,context,re.I) and re.search(r'\d',context))
        if regulation or sector_statistics:return context
    return None

def merged_articles(base,analyses,batch_manifest_sha,legacy=None,raw=None,scope_reviewed_at=None,minimum=.85):
    merged=[]
    for original in base:
        previous=(legacy or {}).get(original['source_id'])
        inheritance='new_source' if previous is None else 'not_reused_source_or_text_version_changed'
        if previous is not None:
            if previous['source_record_hash']==original['source_record_hash'] and normalize_article(previous)['input_text_hash']==normalize_article(original)['input_text_hash']:
                original=previous;inheritance='exact_source_and_text_reused'
        a=copy.deepcopy(original);sid=a['source_id'];result=analyses[sid]
        if a['source_record_hash']!=result['record_hash']:raise ValueError('Frozen news differs from batch source record')
        selected=[];regional_pending=[];classification_pending=[];context_validated=[]
        candidates=copy.deepcopy(result['industry_impacts'])
        source_summary=(raw or {}).get(sid,{}).get('主要内容',original.get('summary',''))
        if sid not in EXCLUDED|QUARANTINE:
            existing={v['industry_key'] for v in candidates}
            for topic in result['industries']:
                if topic['industry_key'] in existing:continue
                context=context_scope_evidence(topic,source_summary,minimum=minimum)
                if context is None:continue
                impact=dict(topic,industry_name_original=topic['industry_name_standard'],industry_name=topic['industry_name_standard'],
                    evidence=topic['quote'],industry_scope_quote=context,source_scope_quote=topic['scope_quote'],
                    evidence_type='inferred',impact_role='affected',horizon='unspecified',llm_inferred_at=result['llm_inferred_at'],
                    review_method='deterministic_verbatim_context_validation_v2',industry_scope_status='automated_context_evidence_gate_passed',propagation_approved=True)
                candidates.append(impact);existing.add(topic['industry_key']);context_validated.append(topic['industry_key'])
        for impact in candidates:
            quote=impact['industry_scope_quote']
            target=impact.get('industry_name_standard','').upper()
            evidence=impact.get('quote','')+' '+quote
            retail=target.startswith('RETAIL-') and not re.search(r'\b(retail\w*|grocer\w*|supermarket\w*|stores?|shops?|shopping)\b',evidence,re.I)
            wholesale=target.startswith('WHOLESALE-') and not re.search(r'\b(wholesale\w*|distribut\w*)\b',evidence,re.I)
            if retail or wholesale:
                classification_pending.append(impact);continue
            foreign=re.search(r'\b(euro\s*zone|european|europe|china|chinese|japan|japanese|british|united kingdom|india|indian|canadian|canada)\b',quote,re.I)
            us=re.search(r'\b(united states|u\.?s\.?|american)\b',quote,re.I)
            if foreign and not us:regional_pending.append(impact)
            else:selected.append(impact)
        a['industry_impacts']=copy.deepcopy(selected)
        for impact in a['industry_impacts']:
            impact['summary_original_quote']=impact['quote']
            impact['quote']=impact['evidence']=text(impact['quote'])
            impact['industry_scope_quote']=text(impact['industry_scope_quote'])
        # Keep original company/market binding fields intact; new industry evidence has its own timing.
        a['industry_enrichment']={'kind':BATCH_KIND,'analysis_sha256':producer_hash(result),
            'legacy_review_inheritance':inheritance,
            'batch_manifest_sha256':batch_manifest_sha,'dictionary':result['dictionary'],
            'model':result.get('model'),'model_digest':result.get('model_digest'),'fingerprint':result.get('fingerprint'),
            'classified_industries':result['industries'],'status':result['status'],
            'feature_available_at':result['feature_available_at'],'llm_inferred_at':result['llm_inferred_at'],
            'regional_scope_pending_impacts':regional_pending,
            'classification_qualification_pending_impacts':classification_pending,
            'context_validated_scope_keys':context_validated,
            'scope_reviewed_at':scope_reviewed_at,
            'scope_review_method':'automated_dictionary_summary_evidence_v1','original_document_revalidated':False}
        from .time_axis import timestamp
        a['effective_feature_available_at']=max(timestamp(v) for v in (a.get('effective_feature_available_at'),result['feature_available_at'],scope_reviewed_at) if v).isoformat()
        a['record_version']=digest([original['record_version'],producer_hash(result),a['industry_impacts'],scope_reviewed_at,'context_evidence_validation_v2'])
        merged.append(a)
    return merged

def checked_exclusions(base_payload,originals,raw):
    # English FinBERT cannot encode an unsupported-language source. Keep it in the batch,
    # but require the original adapter's exact version and explicit rejection reason.
    parent=archive_bytes(io.BytesIO(base_payload['parent_artifact.zip']))
    review={r['source_id']:r for r in (parse(line) for line in parent['review_records.jsonl'].splitlines() if line.strip())}
    missing=set(raw)-{r['source_id'] for r in originals};excluded=[]
    for sid in sorted(missing):
        r=review.get(sid)
        if r is None or r['source_record_hash']!=producer_hash(raw[sid]) or r.get('adapter_errors')!=['encoder_input:unsupported_language']:
            raise ValueError('An encoding-ready article was omitted from the imported batch')
        excluded.append({'source_id':sid,'source_record_hash':producer_hash(raw[sid]),'reason':'unsupported_finbert_language','article':r})
    if not {r['source_id'] for r in originals}<=set(raw):raise ValueError('Base contains news outside batch')
    return excluded

def import_batch(base,batch,dictionary_directory,output,legacy_snapshot=None):
    base,batch,output=map(Path,(base,batch,output));validate_snapshot(base)
    dictionary=IndustryDictionary(dictionary_directory)
    payload={name:(batch/name).read_bytes() for name in json.loads((batch/'manifest.json').read_text(encoding='utf-8'))['files']}
    payload['manifest.json']=(batch/'manifest.json').read_bytes();analyses,raw,metadata=batch_rows(payload,dictionary)
    originals=read_records(base/'articles.jsonl')
    base_payload={name:(base/name).read_bytes() for name in json.loads((base/'manifest.json').read_text(encoding='utf-8'))['files']}
    excluded=checked_exclusions(base_payload,originals,raw)
    legacy={}
    if legacy_snapshot:
        legacy_snapshot=Path(legacy_snapshot);validate_snapshot(legacy_snapshot)
        legacy={r['source_id']:r for r in read_records(legacy_snapshot/'articles.jsonl')}
    reviewed_at=now();minimum=parse(payload['run.json'])['settings']['min_confidence']
    with new_artifact(output) as directory:
        base_manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
        for filename,source,pieces in (('base_snapshot.zip',base,None),('upstream_industry_batch.zip',None,payload)):
            with zipfile.ZipFile(directory/filename,'w',zipfile.ZIP_DEFLATED) as archive:
                if pieces:
                    for name,body in pieces.items():archive.writestr(name,body)
                else:
                    for name in list(base_manifest['files'])+['manifest.json']:archive.writestr(name,(source/name).read_bytes())
        if legacy_snapshot:
            legacy_manifest=json.loads((legacy_snapshot/'manifest.json').read_text(encoding='utf-8'))
            with zipfile.ZipFile(directory/'legacy_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in list(legacy_manifest['files'])+['manifest.json']:archive.writestr(name,(legacy_snapshot/name).read_bytes())
        # Keep the unchanged raw archive accessible to the existing SEC evidence builder.
        (directory/'raw_news.zip').write_bytes(payload['raw_news.zip'])
        for name in dictionary.manifest['files']:(directory/('dictionary_'+name)).write_bytes((dictionary.directory/name).read_bytes())
        (directory/'dictionary_manifest.json').write_bytes((dictionary.directory/'manifest.json').read_bytes())
        values=merged_articles(originals,analyses,hashlib.sha256(payload['manifest.json']).hexdigest(),legacy,raw,reviewed_at,minimum)
        write_records(directory/'articles.jsonl',values)
        write_records(directory/'excluded_articles.jsonl',excluded)
        write_json(directory/'manifest.json',{'kind':KIND,'schema_version':KIND,'created_at':now(),'articles':len(values),
            'ready_articles':len(values),'time_mode':'retrospective','historical_observed_pit_ready':False,
            'enriched_industry_articles':sum(bool(a['industry_impacts']) for a in values),
            'context_reviewed_articles':sum(bool(a['industry_enrichment']['context_validated_scope_keys']) for a in values),
            'regional_scope_pending_articles':sum(bool(a['industry_enrichment']['regional_scope_pending_impacts']) for a in values),
            'upstream_batch_manifest_sha256':hashlib.sha256(payload['manifest.json']).hexdigest(),
            'legacy_metadata_policy':'Exact raw-record and text hash only; preserve pinned company/market metadata, independently version new industry evidence.',
            'scope_reviewed_at':reviewed_at,'enrichment_code_sha256':digest_file(__file__),
            'batch_summary':{k:metadata[k] for k in ('total','processed','status_counts','articles_with_dictionary_mentions','articles_with_industry_impacts','industry_impacts')},
            'files':file_manifest(directory)})
        validate_enriched(directory)
    return json.loads((output/'manifest.json').read_text(encoding='utf-8'))

def validate_enriched(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!=KIND or manifest.get('time_mode')!='retrospective' or manifest.get('historical_observed_pit_ready') is not False:
        raise ValueError('Invalid enriched snapshot kind/time policy')
    verify_files(directory,manifest)
    key=(str(directory.resolve()),digest_file(directory/'manifest.json'))
    if (directory/'legacy_snapshot.zip').is_file()!=('legacy_snapshot.zip' in manifest['files']):raise ValueError('Unmanifested legacy source archive')
    if key in _VALIDATED:return copy.deepcopy(_VALIDATED[key])
    base=archive_bytes(directory/'base_snapshot.zip');verify_payload(base)
    with tempfile.TemporaryDirectory(prefix='stockmixer-batch-verify-') as temp:
        for name,body in base.items():
            if Path(name).name!=name:raise ValueError('Base snapshot must be flat')
            (Path(temp)/name).write_bytes(body)
        validate_snapshot(temp)
    dictionary=IndustryDictionary(directory,manifest_name='dictionary_manifest.json',dictionary_name='dictionary_industry_dictionary.jsonl',verify_sources=True)
    payload=archive_bytes(directory/'upstream_industry_batch.zip');analyses,raw,metadata=batch_rows(payload,dictionary)
    if hashlib.sha256(payload['manifest.json']).hexdigest()!=manifest['upstream_batch_manifest_sha256']:
        raise ValueError('Enriched snapshot batch manifest binding mismatch')
    if (directory/'raw_news.zip').read_bytes()!=payload['raw_news.zip']:raise ValueError('Raw archive differs from upstream batch')
    originals=[parse(line) for line in base['articles.jsonl'].splitlines() if line.strip()]
    legacy={}
    if (directory/'legacy_snapshot.zip').is_file():
        inherited=archive_bytes(directory/'legacy_snapshot.zip');verify_payload(inherited)
        with tempfile.TemporaryDirectory(prefix='stockmixer-legacy-verify-') as temp:
            for name,body in inherited.items():
                if Path(name).name!=name:raise ValueError('Legacy snapshot must be flat')
                (Path(temp)/name).write_bytes(body)
            validate_snapshot(temp)
        legacy={r['source_id']:r for r in (parse(line) for line in inherited['articles.jsonl'].splitlines() if line.strip())}
    expected=merged_articles(originals,analyses,manifest['upstream_batch_manifest_sha256'],legacy,raw,manifest['scope_reviewed_at'],parse(payload['run.json'])['settings']['min_confidence'])
    actual=read_records(directory/'articles.jsonl')
    excluded=checked_exclusions(base,originals,raw)
    if actual!=expected or len(actual)!=manifest['articles'] or read_records(directory/'excluded_articles.jsonl')!=excluded:
        raise ValueError('Enriched snapshot articles differ from independently replayed batch')
    if manifest['enriched_industry_articles']!=sum(bool(a['industry_impacts']) for a in actual) or manifest['context_reviewed_articles']!=sum(bool(a['industry_enrichment']['context_validated_scope_keys']) for a in actual):
        raise ValueError('Enriched industry summary counts differ from their evidence')
    result={'valid':True,'articles':len(actual),'summary_only':True,'historical_observed_pit_ready':False}
    _VALIDATED[key]=result
    return copy.deepcopy(result)

def automated_reviews(directory,articles):
    validate_enriched(directory)
    expected={r['source_id']:normalize_article(r) for r in read_records(Path(directory)/'articles.jsonl')}
    result={}
    for article in articles:
        a=expected[article['source_id']]
        if article.get('industry_impacts')!=a.get('industry_impacts') or article.get('industry_enrichment')!=a.get('industry_enrichment'):
            raise ValueError('Industry review proposals differ from the verified upstream batch')
        if a.get('industry_impacts'):
            result[a['source_id']]={'decision':'include','review_method':'automated_dictionary_summary_evidence_v1',
              'reviewed_by':'upstream_evidence_gate_not_human','reviewed_at':a['industry_enrichment']['scope_reviewed_at'],
              'rationale':'Dictionary choice, verbatim summary scope and economic mechanism independently replayed; not a human review.',
              'approved_impacts_hash':digest(a['industry_impacts']),'batch_manifest_sha256':a['industry_enrichment']['batch_manifest_sha256']}
    return result

REVIEW_KIND='subset_extended_market_review_v1'

def extended_rows(articles,old_articles,old_reviews,dataset):
    from .routing import article_pin
    previous={a['source_id']:normalize_article(a) for a in old_articles};old={r['source_id']:r for r in old_reviews};result=[]
    for a in articles:
        sid=a['source_id']
        if normalize_article(a)['input_text_hash']!=a.get('input_text_hash'):
            raise ValueError('Review text differs from its pinned content hash')
        if sid in old and sid in previous and article_pin(a)==article_pin(previous[sid]):
            result.append(dict(old[sid],dataset=dataset))
        else:
            result.append({'source_id':sid,'dataset':dataset,'article_pin':article_pin(a),
              'title':a['title'],'summary':a['summary'],'published_at':a['published_at'],
              'candidate_market_scope':'market' in a.get('impact_scope',[]),'decision':'pending',
              'evidence_quote':None,'rationale':'New source outside the previously reviewed snapshot; no automatic market promotion.',
              'reviewed_by':None,'reviewed_at':None,'review_method':'unreviewed_changed_source' if sid in old else 'unreviewed_new_source'})
    return result

def extend_market_review(source,legacy_snapshot,articles,dataset,output,provenance,settings):
    from .review_reuse import validate_review_artifact
    from .routing import checked_reviews
    source,legacy_snapshot,output=map(Path,(source,legacy_snapshot,output))
    old_articles=[normalize_article(a) for a in read_records(legacy_snapshot/'articles.jsonl')]
    original=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if original['source_input']['snapshot_manifest_sha256']!=digest_file(legacy_snapshot/'manifest.json'):
        raise ValueError('Original review differs from archived legacy snapshot')
    validate_review_artifact(source,old_articles,original['dataset'],original['source_input'],settings)
    values=extended_rows(articles,old_articles,read_records(source/'market_review.jsonl'),dataset)
    _,counts=checked_reviews(articles,values,True,dataset)
    with new_artifact(output) as directory:
        for filename,root in (('source_review.zip',source),('source_snapshot.zip',legacy_snapshot)):
            m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
            with zipfile.ZipFile(directory/filename,'w',zipfile.ZIP_DEFLATED) as archive:
                for name in list(m['files'])+['manifest.json']:archive.writestr(name,(root/name).read_bytes())
        write_records(directory/'market_review.jsonl',values)
        write_json(directory/'run_status.json',{'status':'completed','selected_articles':len(articles),
            'model_calls':0,'inherited_decisions':sum(r.get('review_method') not in ('unreviewed_new_source','unreviewed_changed_source') for r in values),
            'changed_sources_pending':sum(r.get('review_method')=='unreviewed_changed_source' for r in values),
            'new_sources_status':'pending; not promoted to market','finished_at':now()})
        write_json(directory/'manifest.json',{'kind':REVIEW_KIND,'created_at':now(),'dataset':dataset,'target_market':'US',
            'source_input':provenance,'selected_articles':len(articles),'decisions':counts,'settings':settings,
            'files':file_manifest(directory)})
        validate_extended_review(directory,articles,dataset,provenance,settings)
    return output

def validate_extended_review(directory,articles,dataset,provenance,settings):
    from .review_reuse import validate_review_artifact
    from .routing import checked_reviews
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory,manifest)
    if manifest.get('kind')!=REVIEW_KIND or manifest.get('dataset')!=dataset or manifest.get('settings')!=settings or manifest['source_input'].get('snapshot_manifest_sha256')!=provenance.get('snapshot_manifest_sha256'):
        raise ValueError('Extended review settings/dataset/snapshot mismatch')
    snapshot=archive_bytes(directory/'source_snapshot.zip');verify_payload(snapshot)
    original=archive_bytes(directory/'source_review.zip');old_manifest=verify_payload(original)
    if old_manifest['source_input']['snapshot_manifest_sha256']!=hashlib.sha256(snapshot['manifest.json']).hexdigest():
        raise ValueError('Extended review source snapshot differs from original decision provenance')
    old_articles=[normalize_article(parse(line)) for line in snapshot['articles.jsonl'].splitlines() if line.strip()]
    with tempfile.TemporaryDirectory(prefix='stockmixer-review-verify-') as temp:
        for name,body in original.items():
            if Path(name).name!=name:raise ValueError('Original market review must be flat')
            (Path(temp)/name).write_bytes(body)
        validate_review_artifact(temp,old_articles,old_manifest['dataset'],old_manifest['source_input'],settings)
    old_reviews=[parse(line) for line in original['market_review.jsonl'].splitlines() if line.strip()]
    expected=extended_rows(articles,old_articles,old_reviews,dataset)
    if read_records(directory/'market_review.jsonl')!=expected:raise ValueError('Extended market decisions changed from inherited bindings')
    _,counts=checked_reviews(articles,expected,True,dataset)
    status=json.loads((directory/'run_status.json').read_text(encoding='utf-8'))
    if status.get('status')!='completed' or status.get('model_calls')!=0 or manifest['decisions']!=counts or manifest['selected_articles']!=len(articles):
        raise ValueError('Extended market review status/counts mismatch')
    return manifest
