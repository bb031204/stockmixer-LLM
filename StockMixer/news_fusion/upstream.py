"""Deterministic reuse of collector analysis. No model calls or identity guesses."""
import json
import re
from collections import Counter
from pathlib import Path
from .io import digest, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .inputs import text
from .time_axis import timestamp

SCOPES = {'US_EQUITY': None, 'NASDAQ_LISTED': 'NASDAQ', 'NYSE_LISTED': 'NYSE'}
POLICY = 'upstream_evidence_mapping_v1'


def industry_impact(value):
    """Keep the producer's originals; adapt aliases without inventing evidence/confidence."""
    if not isinstance(value, dict):
        raise ValueError('Industry impact must be an object')
    result = dict(value)
    for canonical, aliases in {'industry_name': ('industry_name_original', 'industry', '行业'),
                               'evidence': ('quote', '证据')}.items():
        values = [value[k] for k in (canonical, *aliases) if value.get(k) is not None]
        if values and any(v != values[0] for v in values):
            raise ValueError('Conflicting industry field aliases: ' + canonical)
        if values:
            result[canonical] = values[0]
        if result.get(canonical) is not None and not isinstance(result[canonical],str):
            raise ValueError('Invalid industry field type: '+canonical)
    if result.get('mechanism') is not None and not isinstance(result['mechanism'],str):
        raise ValueError('Industry mechanism must be text')
    return result


def confidence(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
        return None
    return float(value)


def scope_claims(article, minimum=.85):
    """Typed scope needs pinned upstream audit, literal evidence and completed inference."""
    raw = article.get('market_scope')
    alternate = article.get('affected_markets')
    raw = [raw] if isinstance(raw, str) else raw
    alternate = [alternate] if isinstance(alternate, str) else alternate
    if raw and alternate and raw != alternate:
        return [], 'conflicting_market_scope_fields'
    scopes = raw or alternate
    if not scopes:
        return [], 'missing_typed_market_scope'
    if not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes) or len(set(scopes)) != len(scopes) or not set(scopes) <= set(SCOPES):
        return [], 'unsupported_market_scope'
    proof = article.get('upstream_analysis') or {}
    if not isinstance(proof,dict):return [], 'invalid_upstream_proof_type'
    audit = proof.get('audit') or {}
    if not isinstance(audit,dict):return [], 'invalid_upstream_audit_type'
    if proof.get('record_hash') != article.get('source_record_hash') or audit.get('record_hash_matches') is not True or not re.fullmatch('[a-f0-9]{64}', str(audit.get('sha256'))):
        return [], 'unbound_upstream_audit'
    if not all(article.get(k) for k in ('qwen_model_revision', 'prompt_version', 'llm_inferred_at')):
        return [], 'missing_upstream_model_provenance'
    try:
        if timestamp(article['llm_inferred_at']) > timestamp(now()):
            return [], 'future_upstream_inference'
    except (ValueError, TypeError):
        return [], 'invalid_upstream_inference_time'
    evidence = article.get('scope_evidence')
    evidence = [evidence] if isinstance(evidence, dict) else evidence
    if not isinstance(evidence, list):
        return [], 'missing_market_scope_evidence'
    source = text(article.get('title')) + ' ' + text(article.get('summary'))
    claims = []
    for scope in scopes:
        rows = [r for r in evidence if isinstance(r, dict) and r.get('scope') == scope]
        if len(rows) != 1:
            return [], 'ambiguous_or_missing_scope_evidence'
        row = rows[0]
        quote = text(row.get('evidence_quote', row.get('quote')))
        rationale = text(row.get('rationale', row.get('mechanism')))
        score = confidence(row.get('confidence'))
        from .source_proof import bound_quote
        if len(quote) < 8 or (quote.casefold() not in source.casefold() and not bound_quote(article, 'scope_evidence', quote)):
            return [], 'market_evidence_not_in_pinned_text'
        if len(rationale) < 8:
            return [], 'missing_market_scope_rationale'
        if score is None or score < minimum:
            return [], 'missing_or_low_scope_source_confidence'
        claims.append({'scope': scope, 'evidence_quote': quote, 'rationale': rationale,
                       'source_confidence': score, 'mapping_confidence': None,
                       'mapping_method': 'exact_scope_enum_and_historical_exchange',
                       'confidence_interpretation': 'uncalibrated_model_self_assessment'})
    return claims, None


def make_reviews(articles, dataset, minimum=.85):
    from .routing import article_pin
    rows = []
    for article in articles:
        claims, reason = scope_claims(article, minimum)
        row = {'source_id': article['source_id'], 'dataset': dataset, 'article_pin': article_pin(article),
               'decision': 'pending', 'reviewed_by': None, 'reviewed_at': None,
               'evidence_quote': None, 'rationale': None, 'review_method': POLICY,
               'upstream_mapping_reason': reason, 'upstream_min_confidence': minimum}
        if claims:
            row.update(decision='include', reviewed_by='collector_analysis_reuse_not_human',
                       reviewed_at=article['llm_inferred_at'], evidence_quote=claims[0]['evidence_quote'],
                       rationale=claims[0]['rationale'], scope_claims=claims)
        elif not article.get('market_scope') and not article.get('affected_markets') and article.get('impact_scope') and 'market' not in article['impact_scope']:
            # Deterministic exclusion never elevates an unverified scope to an accepted route.
            row.update(decision='exclude', reviewed_by='deterministic_scope_filter_not_human',
                       reviewed_at=article.get('llm_inferred_at') or article.get('feature_available_at') or '1970-01-01T00:00:00Z',
                       rationale='Upstream impact_scope has no market channel; do not broaden company/industry scope.')
        rows.append(row)
    return rows


def build_review(articles, dataset, output, provenance, minimum=.85):
    rows = make_reviews(articles, dataset, minimum)
    with new_artifact(output) as directory:
        write_records(directory/'market_review.jsonl', rows)
        write_records(directory/'article_bindings.jsonl', articles)
        write_json(directory/'run_status.json', {'status': 'completed', 'model_calls': 0})
        write_json(directory/'manifest.json', {'kind': 'upstream_market_review', 'policy': POLICY,
            'dataset': dataset, 'target_market': 'US', 'created_at': now(), 'source_input': provenance,
            'minimum_confidence': minimum, 'selected_articles': len(articles), 'model_calls': 0,
            'decisions': dict(Counter(r['decision'] for r in rows)), 'files': file_manifest(directory)})
    return validate_review(output, articles, dataset, provenance, minimum)


def validate_review(directory, articles, dataset, provenance, minimum):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory, manifest, required=('market_review.jsonl', 'article_bindings.jsonl', 'run_status.json'))
    expected = make_reviews(articles, dataset, minimum)
    if manifest.get('kind') != 'upstream_market_review' or manifest.get('policy') != POLICY or manifest.get('dataset') != dataset or manifest.get('minimum_confidence') != minimum or manifest.get('model_calls') != 0:
        raise ValueError('Upstream review policy/dataset/settings mismatch')
    if provenance.get('mode') != 'verified_snapshot' or manifest.get('source_input') != provenance:
        raise ValueError('Upstream review must bind the exact verified snapshot')
    if digest(read_records(directory/'article_bindings.jsonl')) != digest(articles) or read_records(directory/'market_review.jsonl') != expected or manifest.get('selected_articles') != len(articles) or manifest.get('decisions') != dict(Counter(r['decision'] for r in expected)):
        raise ValueError('Upstream review cannot replay from its pinned source')
    if json.loads((directory/'run_status.json').read_text(encoding='utf-8')) != {'status': 'completed', 'model_calls': 0}:
        raise ValueError('Upstream review run status mismatch')
    return manifest


def market_owners(scope, securities, axis, day):
    """Listing venue is historical master evidence, never the dataset's name."""
    from .relations import valid_on
    if scope not in SCOPES:
        raise ValueError('Unsupported market scope')
    if scope == 'US_EQUITY':
        return [(i, None) for i in range(len(axis.stock_uids))]
    return [(s['array_index'], s) for s in securities if s['exchange'] == SCOPES[scope] and valid_on(s, day)]
