"""Replay bounded collector evidence without fetching text or trusting a model flag."""
import hashlib
import json
import re


def bound_quote(article, field, quote):
    if not isinstance(quote, str) or len(quote.strip()) < 8:
        return False
    proof = article.get('upstream_analysis') or {}
    bundle = proof.get('bound_source_evidence') or {}
    try:
        raw_audit, raw_pack = bundle['audit_json'], bundle['pack_json']
        if hashlib.sha256(raw_audit.encode('utf-8')).hexdigest() != proof['audit']['sha256']:
            return False
        audit, pack = json.loads(raw_audit), json.loads(raw_pack)
        pin = audit['evidence_pack']
        if hashlib.sha256(raw_pack.encode('utf-8')).hexdigest() != pin['sha256']:
            return False
        from .inputs import text
        record = article['source_record_hash']
        if (proof['record_hash'] != record or audit['record_hash'] != record or pack['record_hash'] != record
                or audit['source_id'] != article['source_id'] or pack['source_id'] != article['source_id']
                or proof['audit']['record_hash_matches'] is not True
                or pack['schema_version'] != 'market-proof-1'
                or not re.fullmatch('[a-f0-9]{64}', str(audit.get('source_body_hash')))
                or pack['source_body_hash'] != audit['source_body_hash']):
            return False
        for index, item in enumerate(audit.get(field, [])):
            for key in ('quote', 'industry_scope_quote', 'evidence_quote'):
                if text(item.get(key)) != text(quote):
                    continue
                if item.get('original_evidence_verified') is not True or item.get('evidence_basis') != 'original_article_quote_verified_before_raw_cleanup':
                    continue
                if any(x.get('field') == field and x.get('index') == index and x.get('key') == key
                       and x.get('quote') == item[key] and x.get('quote_sha256') == hashlib.sha256(item[key].encode('utf-8')).hexdigest()
                       for x in pack['evidence']):
                    return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return False
