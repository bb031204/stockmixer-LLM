"""Reuse a common US-market review across stock universes without rerunning pending decisions."""
import json
import shutil
from pathlib import Path
from .auto_review import PROMPT_VERSION,SYSTEM_PROMPT,RESPONSE_SCHEMA
from .io import digest,digest_file,file_manifest,new_artifact,now,read_records,verify_files,write_json,write_records
from .routing import checked_reviews


def validate_review_artifact(directory,articles,dataset,provenance,settings):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')=='completed_market_review_v1':
        from .market_review_completion import validate_completed_review
        return validate_completed_review(directory,articles,dataset,provenance,settings)
    if manifest.get('kind')=='subset_extended_market_review_v1':
        from .legacy_industry import validate_extended_review
        return validate_extended_review(directory,articles,dataset,provenance,settings)
    if manifest.get('kind')=='upstream_market_review':
        from .upstream import validate_review
        return validate_review(directory,articles,dataset,provenance,settings['min_confidence'])
    verify_files(directory,manifest,required=('market_review.jsonl','run_status.json','model.json','settings.json','prompt.txt','response_schema.json'))
    status=json.loads((directory/'run_status.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='automated_market_review' or manifest.get('target_market')!='US' or manifest.get('dataset')!=dataset or status.get('status')!='completed' or manifest.get('selected_articles')!=len(articles):
        raise ValueError('Only a completed full-snapshot common US market review can be reused')
    if manifest.get('source_input',{}).get('snapshot_manifest_sha256')!=provenance.get('snapshot_manifest_sha256'):
        raise ValueError('Reused review belongs to another snapshot')
    if manifest.get('settings')!=settings or manifest.get('prompt_version')!=PROMPT_VERSION or manifest.get('prompt_sha256')!=digest(SYSTEM_PROMPT) or manifest.get('response_schema_sha256')!=digest(RESPONSE_SCHEMA):
        raise ValueError('Reused review has different inference settings or prompt; select a matching review or regenerate using the cache')
    model=manifest.get('model',{})
    if model.get('model')!=settings['model']:
        raise ValueError('Review model does not match configured model')
    expected=settings.get('expected_model_digest')
    if expected and str(model.get('model_digest','')).removeprefix('sha256:')!=expected.removeprefix('sha256:'):
        raise ValueError('Review model digest differs from configured expected_model_digest')
    if json.loads((directory/'model.json').read_text(encoding='utf-8'))!=model or json.loads((directory/'settings.json').read_text(encoding='utf-8'))!=settings:
        raise ValueError('Review model/settings files contradict the manifest')
    _,counts=checked_reviews(articles,read_records(directory/'market_review.jsonl'),True,dataset)
    if counts!=manifest.get('decisions'):
        raise ValueError('Review decision counts differ from its records')
    validate_rebound(directory,articles,provenance)
    return manifest


def rebind_review(source,articles,dataset,output,provenance,settings):
    source=Path(source)
    original_sha=digest_file(source/'manifest.json')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    validate_review_artifact(source,articles,manifest['dataset'],provenance,settings)
    verify_files(source,manifest,required=('market_review.jsonl','run_status.json','model.json','settings.json','prompt.txt','response_schema.json'))
    status=json.loads((source/'run_status.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='automated_market_review' or manifest.get('target_market')!='US' or status.get('status')!='completed' or manifest.get('selected_articles')!=len(articles):
        raise ValueError('Only a completed full-snapshot common US market review can be reused')
    if manifest.get('source_input',{}).get('snapshot_manifest_sha256')!=provenance.get('snapshot_manifest_sha256'):
        raise ValueError('Reused review belongs to another snapshot')
    if manifest.get('settings')!=settings or manifest.get('prompt_version')!=PROMPT_VERSION or manifest.get('prompt_sha256')!=digest(SYSTEM_PROMPT) or manifest.get('response_schema_sha256')!=digest(RESPONSE_SCHEMA):
        raise ValueError('Reused review has different inference settings or prompt')
    original=read_records(source/'market_review.jsonl')
    checked_reviews(articles,original,True,manifest['dataset'])
    rows=[dict(row,dataset=dataset) for row in original]
    checked_reviews(articles,rows,True,dataset)
    created=now()
    with new_artifact(output) as directory:
        for name in manifest['files']:
            shutil.copy2(source/name,directory/('source_'+name))
        shutil.copy2(source/'manifest.json',directory/'source_review_manifest.json')
        for name in ('model.json','settings.json','prompt.txt','response_schema.json'):
            shutil.copy2(source/name,directory/name)
        write_records(directory/'market_review.jsonl',rows)
        sample=[]
        for decision in ('include','exclude','pending'):
            sample.extend([row for row in rows if row['decision']==decision][:10])
        write_records(directory/'quality_sample.jsonl',sample)
        write_json(directory/'run_status.json',{'status':'completed','started_at':created,'finished_at':created,
            'selected_articles':len(rows),'stats':{'processed':len(rows),'model_calls':0,'reused_decisions':len(rows)}})
        clone=dict(manifest,dataset=dataset,created_at=created,model_inference_run=False,
            stats={'processed':len(rows),'model_calls':0,'reused_decisions':len(rows)},
            source_statistics=manifest['stats'],rebound_from={'dataset':manifest['dataset'],'manifest_sha256':original_sha,
                'common_target_market':'US','changes':['dataset field only; review completion timestamps and decisions preserved']},
            files=file_manifest(directory))
        clone['reuse_code_sha256']=digest_file(Path(__file__))
        write_json(directory/'manifest.json',clone)
        if digest_file(source/'manifest.json')!=original_sha:
            raise ValueError('Review source changed while being reused')
        validate_rebound(directory,articles,provenance)
    return clone


def validate_rebound(directory,articles,provenance):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory,manifest)
    proof=manifest.get('rebound_from')
    if not proof:return
    if digest_file(directory/'source_review_manifest.json')!=proof['manifest_sha256']:
        raise ValueError('Rebound source manifest differs')
    original=json.loads((directory/'source_review_manifest.json').read_text(encoding='utf-8'))
    verify_files(directory,{'files':{'source_'+k:v for k,v in original['files'].items()}})
    if original.get('source_input',{}).get('snapshot_manifest_sha256')!=provenance.get('snapshot_manifest_sha256'):
        raise ValueError('Rebound review snapshot differs')
    rows=read_records(directory/'source_market_review.jsonl')
    expected=[dict(row,dataset=manifest['dataset']) for row in rows]
    if digest(expected)!=digest(read_records(directory/'market_review.jsonl')):
        raise ValueError('Rebound changed a decision, evidence or review timestamp')
    checked_reviews(articles,rows,True,original['dataset'])
    checked_reviews(articles,expected,True,manifest['dataset'])
