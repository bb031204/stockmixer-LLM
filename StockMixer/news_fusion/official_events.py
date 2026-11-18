"""M4：官方宏观发布（FOMC / BLS / BEA）作为结构化市场信息来源。

本模块不联网、不调用 Qwen。输入是上游 U 采集后保存的 JSONL（每行一条首次发布记录），
本模块负责校验、生成可编码的文章表、FinBERT 离线编码，以及按固定规则生成市场路由。
官方文本是原文摘录，路由按来源类型规则生成，规则版本写入 manifest。
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from .inputs import normalize_article, text
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .time_axis import timestamp

KIND = 'official_macro_events_v1'
SOURCE_TYPE = 'official_macro_release'
ROUTE_RULE = 'official_release_rule_v1'
# 事件类型 → 允许的官方域名。只接受这些机构的首次发布文本。
EVENT_HOSTS = {
    'fomc_statement': ('federalreserve.gov',),
    'fomc_minutes': ('federalreserve.gov',),
    'cpi': ('bls.gov',),
    'employment_situation': ('bls.gov',),
    'gdp': ('bea.gov',),
    'personal_income_outlays': ('bea.gov',),
}
REQUIRED = ('source_id', 'source_type', 'event_type', 'title', 'text', 'source_url', 'published_at',
            'fetched_at', 'content_sha256', 'first_release', 'publication_time_basis')


def content_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def check_record(row):
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise ValueError('Official event missing fields: ' + ','.join(missing))
    sid = row['source_id']
    if not isinstance(sid, str) or not sid.startswith('official:'):
        raise ValueError('Official event source_id must start with official:')
    if row['source_type'] != SOURCE_TYPE:
        raise ValueError('Unsupported official source_type: ' + sid)
    if row['event_type'] not in EVENT_HOSTS:
        raise ValueError('Unsupported official event_type: ' + str(row['event_type']))
    url = urlsplit(row['source_url'])
    host = (url.hostname or '').lower()
    if url.scheme != 'https' or not any(host == h or host.endswith('.' + h) for h in EVENT_HOSTS[row['event_type']]):
        raise ValueError('Official event URL is not on the agency domain for its event type: ' + sid)
    if row['first_release'] is not True:
        raise ValueError('Only first-release text is allowed; revised values would leak later information: ' + sid)
    if row['publication_time_basis'] != 'official_page_timestamp':
        raise ValueError('Publication time must come from the official page: ' + sid)
    body, title = text(row['text']), text(row['title'])
    if len(body) < 40 or len(title) < 8:
        raise ValueError('Official event text/title too short: ' + sid)
    if content_hash(row['text']) != row['content_sha256']:
        raise ValueError('Official event text hash mismatch: ' + sid)
    published, fetched = timestamp(row['published_at']), timestamp(row['fetched_at'])
    if len(row['published_at']) <= 10:
        raise ValueError('Official events require an exact release time with timezone: ' + sid)
    if fetched < published:
        raise ValueError('Official event fetched before publication: ' + sid)
    return row


def to_article(row):
    raw = {'source_id': row['source_id'], 'title': row['title'], 'summary': row['text'], 'language': 'en',
           'market': 'US', 'impact_scope': ['market'], 'published_at': row['published_at'],
           'publication_precision': 'datetime', 'first_seen_at': row['fetched_at'], 'fetched_at': row['fetched_at'],
           'feature_available_at': None, 'availability_basis': 'retrospective_official_release',
           'summary_origin': 'source_excerpt', 'source_type': SOURCE_TYPE, 'source_url': row['source_url'],
           'event_type': row['event_type']}
    article = normalize_article(raw)
    if article['input_status'] != 'ready':
        raise ValueError('Official event is not encoding-ready: ' + row['source_id'])
    return article


def build_events(source, output):
    """source：上游保存的官方发布 JSONL。输出独立版本目录，不覆盖已有产物。"""
    source = Path(source).resolve()
    rows = read_records(source)
    if not rows:
        raise ValueError('No official events supplied')
    seen = set()
    for row in rows:
        check_record(row)
        if row['source_id'] in seen:
            raise ValueError('Duplicate official source_id: ' + row['source_id'])
        seen.add(row['source_id'])
    articles = [to_article(r) for r in sorted(rows, key=lambda r: (r['published_at'], r['source_id']))]
    with new_artifact(output) as directory:
        write_records(directory/'source_records.jsonl', rows)
        write_records(directory/'articles.jsonl', articles)
        write_json(directory/'manifest.json', {'kind': KIND, 'created_at': now(), 'source_file': str(source),
            'source_file_sha256': digest_file(source), 'articles': len(articles), 'route_rule': ROUTE_RULE,
            'event_types': dict(Counter(r['event_type'] for r in rows)),
            'publication_range': [articles[0]['published_at'], articles[-1]['published_at']],
            'time_mode': 'retrospective', 'historical_observed_pit_ready': False,
            'files': file_manifest(directory)})
    return validate_events(output)


def validate_events(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory, manifest, required=('source_records.jsonl', 'articles.jsonl'))
    if manifest.get('kind') != KIND or manifest.get('route_rule') != ROUTE_RULE:
        raise ValueError('Unsupported official event artifact')
    rows = read_records(directory/'source_records.jsonl')
    for row in rows:
        check_record(row)
    expected = [to_article(r) for r in sorted(rows, key=lambda r: (r['published_at'], r['source_id']))]
    if read_records(directory/'articles.jsonl') != expected or manifest['articles'] != len(expected):
        raise ValueError('Official event articles differ from their source records')
    return manifest


def encode_events(directory, backend, output, cache, batch_size=16):
    from .encoder import encode_articles
    manifest = validate_events(directory)
    articles = read_records(Path(directory)/'articles.jsonl')
    provenance = {'mode': 'official_events', 'official_directory': str(Path(directory).resolve()),
                  'official_manifest_sha256': digest_file(Path(directory)/'manifest.json'), 'articles': len(articles)}
    result = encode_articles(articles, backend, output, cache, batch_size, 'jsonl', provenance)
    if result['successful'] != manifest['articles']:
        raise ValueError('Official event encoding incomplete; inspect the feature manifest')
    return result


def route(article, dataset, known_from):
    published = article['published_at']
    day = timestamp(published).tz_convert('America/New_York').strftime('%Y-%m-%d')
    return {'source_id': article['source_id'], 'article_pin': digest([article['source_id'], article['input_text_hash'], published]),
            'channel': 'market', 'target_market': 'US', 'dataset': dataset, 'verification_status': 'verified',
            'mapping_source': ROUTE_RULE, 'valid_from': day, 'valid_to': None, 'known_from': known_from,
            'evidence_quote': article['title'], 'evidence_basis': 'source_text', 'event_type': article['event_type'],
            'reviewed_by': 'official_source_rule_not_human', 'reviewed_at': known_from,
            'rationale': 'Official first-release macro publication; routed by source-type rule, not by model judgment.'}


def load_encoded(features_dir, dataset):
    """返回官方事件特征记录、向量、市场路由及元数据，供 panel 合并。"""
    from .panel import load_features
    features_dir = Path(features_dir).resolve()
    records, vectors, feature_manifest = load_features(features_dir)
    provenance = feature_manifest.get('input_provenance') or {}
    if provenance.get('mode') != 'official_events':
        raise ValueError('market_channel.official_events must point to encoded official event features')
    events_dir = Path(provenance['official_directory'])
    if digest_file(events_dir/'manifest.json') != provenance['official_manifest_sha256']:
        raise ValueError('Official event artifact changed after encoding')
    events = validate_events(events_dir)
    articles = {a['source_id']: a for a in read_records(events_dir/'articles.jsonl')}
    if {r['source_id'] for r in records} != set(articles):
        raise ValueError('Official features and event articles differ')
    routes = []
    for record in records:
        article = articles[record['source_id']]
        if record['input_text_hash'] != article['input_text_hash'] or record.get('event_type') != article['event_type']:
            raise ValueError('Official feature row differs from its event article')
        if record['finbert_status'] == 'success':
            routes.append(route(article, dataset, events['created_at']))
    info = {'features_manifest_sha256': digest_file(features_dir/'manifest.json'), 'encoder': feature_manifest['encoder'],
            'events_manifest_sha256': provenance['official_manifest_sha256'], 'route_rule': ROUTE_RULE,
            'routes': len(routes), 'routes_sha256': digest(routes), 'event_types': events['event_types']}
    return records, vectors, routes, info
