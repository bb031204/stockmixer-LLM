"""M1：只为从未审核的市场记录补跑本地 Qwen 审核，其余决定逐字继承。

- 选择条件：来源审核中 decision=pending 且 review_method 以 unreviewed_ 开头。
- Prompt、模型 digest、置信阈值沿用 configs/llm.yaml 的 review 段，与旧审核完全一致。
- 来源审核和本次 auto_review 运行目录均打包进新产物，validate 时逐条重放比较。
- reviewed_at 为本次真实审核完成时间，不沿用旧审核时间。
"""
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records

KIND = 'completed_market_review_v1'
UNREVIEWED = ('unreviewed_new_source', 'unreviewed_changed_source')


def unreviewed_ids(rows):
    return [r['source_id'] for r in rows if r.get('decision') == 'pending' and r.get('review_method') in UNREVIEWED]


def _archive(directory, target):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory, manifest)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(manifest['files']) + ['manifest.json']:
            archive.writestr(name, (directory/name).read_bytes())
    return digest_file(directory/'manifest.json')


def _extract(archive_path, target):
    from .legacy_industry import archive_bytes, verify_payload
    payload = archive_bytes(archive_path)
    verify_payload(payload)
    for name, body in payload.items():
        if Path(name).name != name:
            raise ValueError('Archived review must be flat')
        (Path(target)/name).write_bytes(body)
    return json.loads(payload['manifest.json'])


def merge(source_rows, auto_rows, selected, dataset):
    auto = {r['source_id']: r for r in auto_rows}
    result = []
    for row in source_rows:
        sid = row['source_id']
        result.append(dict(auto[sid], dataset=dataset) if sid in selected else dict(row, dataset=dataset))
    return result


def complete_review(source, articles, dataset, output, cache, backend, settings, provenance, limit=None, progress=None):
    from .auto_review import auto_review
    from .review_reuse import validate_review_artifact
    from .routing import checked_reviews
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_review_artifact(source, articles, dataset, provenance, settings)
    source_rows = read_records(source/'market_review.jsonl')
    selected = unreviewed_ids(source_rows)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError('Source review has no unreviewed records; nothing to complete')
    with tempfile.TemporaryDirectory(prefix='stockmixer-m1-', dir=output.parent if output.parent.exists() else None) as temp:
        run = Path(temp)/'auto_review'
        auto_review(articles, dataset, run, cache, backend, settings, provenance, source_ids=selected, progress=progress)
        status = json.loads((run/'run_status.json').read_text(encoding='utf-8'))
        if status.get('status') != 'completed':
            raise ValueError('Completion review did not finish; partial results stay in the cache only')
        rows = merge(source_rows, read_records(run/'market_review.jsonl'), set(selected), dataset)
        _, counts = checked_reviews(articles, rows, True, dataset)
        with new_artifact(output) as directory:
            source_sha = _archive(source, directory/'source_review.zip')
            _archive(run, directory/'completion_run.zip')
            write_records(directory/'market_review.jsonl', rows)
            write_json(directory/'selected_source_ids.json', selected)
            newly = Counter(r['decision'] for r in rows if r['source_id'] in set(selected))
            write_json(directory/'run_status.json', {'status': 'completed', 'selected_articles': len(rows),
                'completed_source_ids': len(selected), 'model_calls': status.get('stats', {}).get('model_calls', 0),
                'finished_at': now()})
            write_json(directory/'manifest.json', {'kind': KIND, 'created_at': now(), 'dataset': dataset,
                'target_market': 'US', 'source_input': provenance, 'selected_articles': len(rows),
                'source_review_manifest_sha256': source_sha, 'completed_source_ids': len(selected),
                'completion_limited': limit is not None, 'decisions': counts, 'new_decisions': dict(newly),
                'settings': settings, 'human_sample_checked': False,
                'confidence_basis': 'model_self_report_not_calibrated', 'files': file_manifest(directory)})
            validate_completed_review(directory, articles, dataset, provenance, settings)
    return output


def validate_completed_review(directory, articles, dataset, provenance, settings):
    from .review_reuse import validate_review_artifact
    from .routing import checked_reviews
    from .auto_review import PROMPT_VERSION, SYSTEM_PROMPT, RESPONSE_SCHEMA
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory, manifest, required=('source_review.zip', 'completion_run.zip', 'market_review.jsonl',
                                                'selected_source_ids.json', 'run_status.json'))
    if manifest.get('kind') != KIND or manifest.get('dataset') != dataset or manifest.get('settings') != settings:
        raise ValueError('Completed review kind/dataset/settings mismatch')
    if manifest['source_input'].get('snapshot_manifest_sha256') != provenance.get('snapshot_manifest_sha256'):
        raise ValueError('Completed review belongs to another snapshot')
    selected = json.loads((directory/'selected_source_ids.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='stockmixer-m1-verify-') as temp:
        source_dir, run_dir = Path(temp)/'source', Path(temp)/'run'
        source_dir.mkdir(); run_dir.mkdir()
        source_manifest = _extract(directory/'source_review.zip', source_dir)
        if digest_file(source_dir/'manifest.json') != manifest['source_review_manifest_sha256']:
            raise ValueError('Archived source review manifest differs')
        validate_review_artifact(source_dir, articles, source_manifest['dataset'], provenance, settings)
        source_rows = read_records(source_dir/'market_review.jsonl')
        allowed = set(unreviewed_ids(source_rows))
        if len(set(selected)) != len(selected) or not set(selected) <= allowed:
            raise ValueError('Completion selected a record that was already reviewed')
        if not manifest.get('completion_limited') and set(selected) != allowed:
            raise ValueError('Full completion must cover every unreviewed record')
        run = _extract(directory/'completion_run.zip', run_dir)
        if run.get('kind') != 'automated_market_review' or run.get('settings') != settings or run.get('dataset') != dataset:
            raise ValueError('Completion run kind/settings mismatch')
        if run.get('prompt_version') != PROMPT_VERSION or run.get('prompt_sha256') != digest(SYSTEM_PROMPT) or run.get('response_schema_sha256') != digest(RESPONSE_SCHEMA):
            raise ValueError('Completion run used a different prompt; it is not comparable to inherited decisions')
        if run.get('selected_articles') != len(selected) or json.loads((run_dir/'run_status.json').read_text(encoding='utf-8')).get('status') != 'completed':
            raise ValueError('Completion run is incomplete')
        expected = merge(source_rows, read_records(run_dir/'market_review.jsonl'), set(selected), dataset)
    rows = read_records(directory/'market_review.jsonl')
    if rows != expected:
        raise ValueError('Completed decisions differ from the archived source and completion run')
    _, counts = checked_reviews(articles, rows, True, dataset)
    if manifest.get('decisions') != counts or manifest.get('selected_articles') != len(articles):
        raise ValueError('Completed review counts mismatch')
    return manifest
