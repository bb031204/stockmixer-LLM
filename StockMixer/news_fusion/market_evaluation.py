"""M3：市场审核人工评估集。抽样与评分均不调用模型；人工标注必须由真实审核者填写。"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .market_quality import evidence_basis, speculative_summary, text_origin
from .routing import article_pin
from .time_axis import timestamp

KIND = 'market_review_evaluation_sample_v1'
LABELS = ('market', 'not_market', 'uncertain')
GUIDELINE_VERSION = 'us_market_label_v1'
GUIDELINE = """标注问题：仅根据给出的标题和摘要，这条信息是否属于“美国股票市场整体层面”的信息？
market：美国货币政策、全国性经济数据、系统性金融或流动性风险，或文本明确说明对美国股市整体有影响的全球冲击。
not_market：单一公司的财报、申报、并购、产品、人事、品牌；单一行业或地方新闻，且文本没有说明会传导到整个市场。
uncertain：文本不足以判断，或摘要明显是对原文的推测性转述（例如“likely related to”）。
evidence_quote：从标题或摘要中逐字复制支持判断的一句话；判为 uncertain 时可以留空。
不要参考事后股价或其他资料。"""


def stratum(row, article):
    decision = row['decision']
    candidate = 'market' in article.get('impact_scope', [])
    if decision == 'include':
        return 'include'
    if decision == 'exclude':
        return 'exclude_candidate' if candidate else 'exclude_noncandidate'
    return 'pending_unreviewed' if str(row.get('review_method', '')).startswith('unreviewed_') else 'pending_uncertain'


def build_sample(articles, reviews, output, per_stratum=40, seed=1234, provenance=None):
    by_id = {a['source_id']: a for a in articles}
    groups = defaultdict(list)
    for row in sorted(reviews, key=lambda r: r['source_id']):
        groups[stratum(row, by_id[row['source_id']])].append(row)
    rng = random.Random(seed)
    rows = []
    for name in sorted(groups):
        chosen = rng.sample(groups[name], min(per_stratum, len(groups[name])))
        for row in sorted(chosen, key=lambda r: r['source_id']):
            a = by_id[row['source_id']]
            rows.append({'source_id': row['source_id'], 'article_pin': article_pin(a), 'stratum': name,
                'stratum_population': len(groups[name]), 'title': a['title'], 'summary': a['summary'],
                'published_at': a.get('published_at'), 'text_origin': text_origin(a),
                'speculative_summary': speculative_summary(a),
                # 模型决定只供评分，标注界面应隐藏，避免锚定。
                'model_decision': row['decision'], 'model_confidence': row.get('model_confidence'),
                'model_evidence_basis': evidence_basis(a, row.get('evidence_quote')) if row['decision'] == 'include' else None,
                'human_label': None, 'evidence_quote': None, 'event_type': None, 'notes': None,
                'reviewed_by': None, 'reviewed_at': None})
    with new_artifact(output) as directory:
        write_records(directory/'sample.jsonl', rows)
        (directory/'guideline.md').write_text(GUIDELINE, encoding='utf-8')
        write_json(directory/'manifest.json', {'kind': KIND, 'created_at': now(), 'seed': seed,
            'per_stratum': per_stratum, 'guideline_version': GUIDELINE_VERSION,
            'strata': {k: len(v) for k, v in sorted(groups.items())}, 'sampled': len(rows),
            'source_input': provenance, 'review_rows_sha256': digest(reviews),
            'human_labels_required': True, 'files': file_manifest(directory)})
    return output


def check_label(row):
    if row.get('human_label') is None:
        return False
    if row['human_label'] not in LABELS:
        raise ValueError('Invalid human_label: ' + row['source_id'])
    reviewer = row.get('reviewed_by')
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.startswith(('ollama:', 'collector_', 'deterministic_', 'assistant', 'upstream_')):
        raise ValueError('Human label needs a real reviewer identity: ' + row['source_id'])
    timestamp(row.get('reviewed_at'))
    if row['human_label'] != 'uncertain':
        quote = row.get('evidence_quote') or ''
        if len(quote) < 8 or quote.casefold() not in ((row['title'] or '') + ' ' + row['summary']).casefold():
            raise ValueError('Human evidence must be a verbatim excerpt: ' + row['source_id'])
    return True


def evaluate(sample_dir, labels_path, output):
    """labels_path：标注者填写后的 sample.jsonl 副本。只读取人工字段，其余字段必须与样本一致。"""
    sample_dir = Path(sample_dir)
    manifest = json.loads((sample_dir/'manifest.json').read_text(encoding='utf-8'))
    verify_files(sample_dir, manifest, required=('sample.jsonl', 'guideline.md'))
    if manifest.get('kind') != KIND:
        raise ValueError('Not a market evaluation sample')
    sample = {r['source_id']: r for r in read_records(sample_dir/'sample.jsonl')}
    labelled = read_records(labels_path)
    human = ('human_label', 'evidence_quote', 'event_type', 'notes', 'reviewed_by', 'reviewed_at')
    rows = []
    for row in labelled:
        base = sample.get(row.get('source_id'))
        if base is None or {k: v for k, v in row.items() if k not in human} != {k: v for k, v in base.items() if k not in human}:
            raise ValueError('Labelled row differs from the frozen sample: ' + str(row.get('source_id')))
        if check_label(row):
            rows.append(row)
    if not rows:
        raise ValueError('No human labels yet; fill sample.jsonl copies before evaluation')
    by_stratum = defaultdict(Counter)
    for row in rows:
        by_stratum[row['stratum']][row['human_label']] += 1
    population = {r['stratum']: r['stratum_population'] for r in sample.values()}

    def rate(stratum_name, label):
        counts = by_stratum.get(stratum_name, Counter())
        decided = counts['market'] + counts['not_market']
        return (counts[label] / decided if decided else None), decided

    precision, n_include = rate('include', 'market')
    # 候选召回：以分层抽样比例外推各层的 market 数量，再算 include 占比。
    estimated = {}
    for name, pop in population.items():
        share, decided = rate(name, 'market')
        estimated[name] = None if share is None else share * pop
    known = [v for k, v in estimated.items() if k != 'exclude_noncandidate' and v is not None]
    recall = (estimated['include'] / sum(known)) if estimated.get('include') and known and sum(known) else None
    bins = defaultdict(Counter)
    for row in rows:
        if row['model_decision'] == 'include' and row['human_label'] != 'uncertain' and isinstance(row.get('model_confidence'), (int, float)):
            bins['%.2f' % (int(row['model_confidence'] * 20) / 20)][row['human_label']] += 1
    origin = defaultdict(Counter)
    for row in rows:
        origin[row['text_origin']][row['human_label']] += 1
    report = {'kind': 'market_review_evaluation_v1', 'created_at': now(), 'guideline_version': manifest['guideline_version'],
        'sample_manifest_sha256': digest_file(sample_dir/'manifest.json'), 'labelled': len(rows),
        'reviewers': sorted({r['reviewed_by'] for r in rows}),
        'include_precision': precision, 'include_precision_n': n_include,
        'candidate_recall_estimate': recall, 'recall_basis': 'stratified extrapolation excluding non-candidates; approximate',
        'labels_by_stratum': {k: dict(v) for k, v in sorted(by_stratum.items())},
        'labels_by_text_origin': {k: dict(v) for k, v in sorted(origin.items())},
        'include_confidence_bins': {k: dict(v) for k, v in sorted(bins.items())},
        'speculative_summary_uncertain': sum(1 for r in rows if r['speculative_summary'] and r['human_label'] == 'uncertain'),
        'speculative_summary_labelled': sum(1 for r in rows if r['speculative_summary'])}
    with new_artifact(output) as directory:
        write_records(directory/'labels.jsonl', rows)
        write_json(directory/'evaluation.json', report)
        write_json(directory/'manifest.json', {'kind': 'market_review_evaluation_v1', 'created_at': report['created_at'],
            'sample_directory': str(sample_dir.resolve()), 'files': file_manifest(directory)})
    return report
