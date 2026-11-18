"""市场通道质量：证据来源（M2）、事件去重（M5）、实验门槛（M6）。不调用任何模型。"""
import copy
import re
from collections import Counter, defaultdict
import numpy as np
from .inputs import text
from .io import digest
from .validation import integer, finite_number

# 证据来源分级：
#   source_text        引文出现在上游明确声明为原文（标题或原文摘录）的文本中；
#   title_only         引文出现在非空标题中，但上游未声明标题来源；
#   unverified_summary 引文只能在摘要中找到，摘要来源未声明或为模型生成。
BASES = ('source_text', 'title_only', 'unverified_summary')
ORIGINS = ('source_excerpt', 'source_title', 'model_summary', 'unknown')
SPECULATIVE = re.compile(r"^the (document|press release|filing|article|report|text)\b|\b(likely|appears to|may be related)\b", re.I)
SPLITS = ('train', 'valid', 'test')

DEFAULTS = {
    'news_routes': True,
    'evidence_basis': ['source_text'],
    'official_events': None,
    'event_dedup': {'window_sessions': 1, 'cosine_threshold': 0.92},
    'gates': {'min_events': {'train': None, 'valid': None, 'test': None},
              'min_active_days': {'train': None, 'valid': None, 'test': None},
              'max_event_type_share': 0.5},
    'evaluation': None,
}


def validate_settings(settings):
    from .configuration import mapping, nonempty, boolean
    mapping(settings, DEFAULTS, 'market_channel')
    boolean(settings['news_routes'], 'market_channel.news_routes')
    if not settings['news_routes'] and settings['official_events'] is None:
        raise ValueError('market_channel must enable news routes or official events')
    basis = settings['evidence_basis']
    if not isinstance(basis, list) or not basis or len(set(basis)) != len(basis) or not set(basis) <= set(BASES):
        raise ValueError('market_channel.evidence_basis must be a nonempty unique subset of ' + str(BASES))
    nonempty(settings['official_events'], 'market_channel.official_events', True)
    nonempty(settings['evaluation'], 'market_channel.evaluation', True)
    dedup = settings['event_dedup']
    mapping(dedup, DEFAULTS['event_dedup'], 'market_channel.event_dedup')
    integer(dedup['window_sessions'], 'event_dedup.window_sessions', 0, 10)
    threshold = finite_number(dedup['cosine_threshold'], 'event_dedup.cosine_threshold')
    if not 0 < threshold <= 1:
        raise ValueError('event_dedup.cosine_threshold must be in (0, 1]')
    gates = settings['gates']
    mapping(gates, DEFAULTS['gates'], 'market_channel.gates')
    for key in ('min_events', 'min_active_days'):
        mapping(gates[key], SPLITS, 'market_channel.gates.' + key)
        for split, value in gates[key].items():
            if value is not None:
                integer(value, 'gates.' + key + '.' + split, 1)
    share = gates['max_event_type_share']
    if share is not None and not 0 < finite_number(share, 'gates.max_event_type_share') <= 1:
        raise ValueError('gates.max_event_type_share must be in (0, 1] or null')
    return settings


def defaults():
    return copy.deepcopy(DEFAULTS)


def text_origin(article):
    """只读取上游显式声明的来源；未声明一律为 unknown，不按文风推断。"""
    value = article.get('summary_origin')
    if value is None:
        return 'unknown'
    if value not in ORIGINS:
        raise ValueError('Unsupported summary_origin: ' + str(value))
    return value


def evidence_basis(article, quote):
    q = text(quote).casefold()
    if len(q) < 8:
        return 'unverified_summary'
    from .source_proof import bound_quote
    if bound_quote(article,'scope_evidence',text(quote)):
        return 'source_text'
    title, summary = text(article.get('title')).casefold(), text(article.get('summary')).casefold()
    origin = text_origin(article)
    if origin == 'source_excerpt' and (q in summary or q in title):
        return 'source_text'
    if title and q in title:
        return 'source_text' if origin == 'source_title' else 'title_only'
    return 'unverified_summary'


def speculative_summary(article):
    """报告项：摘要中带推测措辞。只统计，不作为自动排除规则。"""
    return bool(SPECULATIVE.search(text(article.get('summary'))))


def filter_routes(routes, articles_by_id, allowed):
    """按证据来源过滤已发布的市场路由；返回保留路由与逐条来源。"""
    kept, bases = [], {}
    for route in routes:
        article = articles_by_id[route['source_id']]
        basis = evidence_basis(article, route.get('evidence_quote'))
        bases[route['source_id']] = basis
        if basis in allowed:
            kept.append(route)
    return kept, bases


def split_of(index, axis, lookback):
    if lookback - 1 <= index < axis.train_end - 1:
        return 'train'
    if axis.train_end - 1 <= index < axis.valid_end - 1:
        return 'valid'
    if axis.valid_end - 1 <= index < len(axis.dates) - 1:
        return 'test'
    return None


def cluster_events(entries, vectors, window_sessions, threshold):
    """确定性贪心聚类。entries: [{source_id, session, event_type, embedding_row}]。

    同一事件要求：发布交易日相差不超过 window_sessions；与簇首文章的 FinBERT 余弦相似度
    不低于 threshold；双方 event_type 都已知时必须相同。
    """
    ordered = sorted(entries, key=lambda e: (e['session'], e['source_id']))
    clusters, result = [], {}
    for entry in ordered:
        vector = np.asarray(vectors[entry['embedding_row']], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        chosen = None
        for cluster in clusters:
            if entry['session'] - cluster['session'] > window_sessions:
                continue
            if entry['event_type'] and cluster['event_type'] and entry['event_type'] != cluster['event_type']:
                continue
            denominator = norm * cluster['norm']
            similarity = float(vector @ cluster['vector'] / denominator) if denominator > 0 else 0.0
            if similarity >= threshold:
                chosen = cluster
                break
        if chosen is None:
            chosen = {'id': 'mev_' + digest([entry['source_id']])[:16], 'session': entry['session'],
                      'event_type': entry['event_type'], 'vector': vector, 'norm': norm}
            clusters.append(chosen)
        result[entry['source_id']] = chosen['id']
    return result


def coverage(market_cells, event_of, event_type_of, axis, lookback):
    """market_cells: [(decision_idx, [source_id,...])]。按切分统计文章、事件、有效日期和事件类型。"""
    stats = {split: {'articles': set(), 'events': set(), 'active_days': set()} for split in SPLITS}
    for index, source_ids in market_cells:
        split = split_of(index, axis, lookback)
        if split is None:
            continue
        stats[split]['active_days'].add(index)
        for sid in source_ids:
            stats[split]['articles'].add(sid)
            stats[split]['events'].add(event_of[sid])
    result = {}
    for split, values in stats.items():
        types = Counter()
        first = {}
        for sid in sorted(values['articles']):
            first.setdefault(event_of[sid], event_type_of.get(sid) or 'unknown')
        for event in values['events']:
            types[first[event]] += 1
        result[split] = {'articles': len(values['articles']), 'events': len(values['events']),
                         'active_days': len(values['active_days']), 'event_types': dict(sorted(types.items()))}
    return result


def gate_blockers(quality, gates):
    """M6：门槛必须事先写入配置；未设置的门槛本身就是阻断原因。"""
    if not isinstance(quality, dict) or 'splits' not in quality:
        return ['Market panel lacks market_quality; rebuild the panel with the current market_channel policy']
    blockers = []
    for key, field in (('min_events', 'events'), ('min_active_days', 'active_days')):
        for split in SPLITS:
            threshold = gates[key][split]
            actual = quality['splits'][split][field]
            if threshold is None:
                blockers.append('market_channel.gates.%s.%s is not set; fix it before the first E3 run' % (key, split))
            elif actual < threshold:
                blockers.append('Market %s %s=%d below gate %d' % (split, field, actual, threshold))
    share = gates['max_event_type_share']
    if share is not None:
        for split in SPLITS:
            values = quality['splits'][split]['event_types']
            total = sum(values.values())
            if total and max(values.values()) / total > share:
                top = max(values, key=values.get)
                blockers.append('Market %s event type %s share %.2f exceeds %.2f' % (split, top, values[top] / total, share))
    return blockers


def funnel(articles, reviews, axis, lookback=16):
    """审核漏斗报告：按发布日期近似映射到切分；正式数字以面板 market_quality 为准。"""
    by_id = {a['source_id']: a for a in articles}
    dates = axis.dates
    result = {split: Counter() for split in SPLITS + ('outside_axis',)}
    sessions = defaultdict(set)
    for row in reviews:
        article = by_id[row['source_id']]
        published = article.get('published_at') or ''
        day = published[:10]
        # 只有日期的记录保守顺延到严格晚于该日期的第一个交易日。
        position = int(np.searchsorted(dates, day, side='right' if len(published) == 10 else 'left'))
        split = split_of(position, axis, lookback) if position < len(dates) else None
        bucket = result[split or 'outside_axis']
        bucket['articles'] += 1
        bucket['decision_' + str(row['decision'])] += 1
        bucket['method_' + str(row.get('review_method'))] += 1
        if 'market' in article.get('impact_scope', []):
            bucket['upstream_market_candidate'] += 1
            bucket['candidate_' + str(row['decision'])] += 1
        if speculative_summary(article):
            bucket['speculative_summary'] += 1
        bucket['origin_' + text_origin(article)] += 1
        if row['decision'] == 'include':
            bucket['include_basis_' + evidence_basis(article, row.get('evidence_quote'))] += 1
            if split:
                sessions[split].add(position)
    report = {split: dict(sorted(values.items())) for split, values in result.items()}
    for split in SPLITS:
        report[split]['include_first_sessions'] = len(sessions[split])
    return report
