"""Explicit undated dataset references for exploratory retrospective experiments only."""
from pathlib import Path
from .industry_mapping import reference_inputs


def memberships(axis, dictionary, archive, sp500, known_from):
    references, _, pins = reference_inputs(archive, sp500, dictionary)
    result = []
    for i, uid in enumerate(axis.stock_uids):
        ticker = axis.source.get('tickers', [s.split(':', 1)[1] for s in axis.stock_uids])[i]
        keys = sorted(set(references[axis.dataset].get(ticker, [])))
        if len(keys) > 1:
            raise ValueError('Conflicting reference industry: ' + uid)
        for key in keys:
            entry = dictionary.by_key[key]
            result.append(dict(dataset=axis.dataset, stock_uid=uid, array_index=i, dataset_ticker=ticker,
                industry_key=key, industry_standard=entry['industry_standard'], industry_version=entry['industry_version'],
                industry_code=entry['industry_code'], industry_name=entry['name'], exposure_weight=1.0,
                verification_status='reference_only', membership_basis='undated_dataset_reference',
                mapping_source=entry['source'], evidence=entry['name'], known_from=known_from,
                valid_from=None, valid_to=None, historical_validity_verified=False,
                reference_source_sha256=entry['source_sha256']))
    return result


def add_references(verified, references):
    # Different classifications remain distinct; no guessed SIC-to-TRSR/GICS crosswalk.
    historical = {(m['stock_uid'], m['industry_standard'], m['industry_version']) for m in verified}
    return verified + [r for r in references if (r['stock_uid'], r['industry_standard'], r['industry_version']) not in historical]
