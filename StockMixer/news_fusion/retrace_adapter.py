"""Import completed retrace exports. Never launches Qwen, downloads, or training."""
import json
from pathlib import Path
from .io import digest, now


def import_retrace(batch, output, axes_dir=None, allow_partial=False):
    batch=Path(batch).resolve()
    progress=json.loads((batch/'progress.json').read_text(encoding='utf-8'))
    if progress['pending'] and not allow_partial:
        raise ValueError('Retrace still has pending articles; wait for completion or use explicit --allow-partial for diagnostics')
    if not progress['pending']:
        acceptance=json.loads((batch/'acceptance.json').read_text(encoding='utf-8'))
        if acceptance.get('passed') is not True:raise ValueError('Retrace acceptance has not passed')
        if not allow_partial:
            manifest=json.loads((batch/'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('classification_contract')!='dictionary-scope-replacement-1':
                raise ValueError('Fresh dictionary/scope replacement required; legacy classification unions cannot activate training')
            from .io import digest_file
            for name,expected in manifest['files'].items():
                path=(batch/name).resolve()
                try:path.relative_to(batch)
                except ValueError:raise ValueError('Retrace manifest escaped its batch') from None
                if not path.is_file() or digest_file(path)!=expected:
                    raise ValueError('Retrace files changed since acceptance: '+name)
    from .source_adapter import adapt_snapshot
    return adapt_snapshot(batch/'adapted',output,axes_dir,markets=('US',))


def automated_industry_reviews(articles, minimum):
    """Already-grounded scope evidence, with explicit nonhuman review provenance."""
    from .source_proof import bound_quote
    from .upstream import confidence
    result={}
    for a in articles:
        impacts=a.get('industry_impacts') or []
        if not impacts:continue
        source=(a.get('title') or '')+' '+(a.get('summary') or '')
        good=True
        for item in impacts:
            quote=item.get('industry_scope_quote') or ''
            score=confidence(item.get('confidence'))
            if (item.get('propagation_approved') is not True or score is None or score<minimum
                    or len(quote)<8 or (quote not in source and not bound_quote(a,'industry_impacts',quote))):
                good=False;break
        if good:
            result[a['source_id']]={'decision':'include','review_method':'automated_dictionary_summary_evidence_v1',
                'reviewed_by':'upstream_evidence_gate_not_human','reviewed_at':a['llm_inferred_at'],
                'approved_impacts_hash':digest(impacts),
                'rationale':'Bound upstream industry scope and literal source evidence passed; not human approval.'}
    return result
