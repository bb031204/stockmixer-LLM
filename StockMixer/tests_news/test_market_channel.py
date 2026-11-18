"""市场通道专项（M1–M6）的合成夹具契约测试；不代表真实新闻、真实模型或真实审核。"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from news_fusion import market_quality as mq
from news_fusion.configuration import REVIEW_DEFAULTS
from news_fusion.io import read_records, write_records, write_json, file_manifest
from news_fusion.inputs import normalize_article
from news_fusion.market_review_completion import complete_review, validate_completed_review, unreviewed_ids
from news_fusion.official_events import build_events, check_record, validate_events
from news_fusion.routing import article_pin
from tests_news.test_auto_review import FixtureBackend, response
from tests_news.test_pipeline import article


def source_review(root, articles, settings, provenance):
    """合成 subset_extended 替身：直接写 automated_market_review，第二篇保持 unreviewed。"""
    from news_fusion.auto_review import auto_review
    auto_review(articles, 'NASDAQ', root/'base', root/'cache', FixtureBackend([response()]), settings, provenance, source_ids=[articles[0]['source_id']])
    rows = read_records(root/'base/market_review.jsonl')
    rows[1].update(review_method='unreviewed_new_source', rationale='New source outside the previously reviewed snapshot.')
    write_records(root/'base/market_review.jsonl', rows)
    manifest = json.loads((root/'base/manifest.json').read_text(encoding='utf-8'))
    manifest['selected_articles'] = len(articles)
    manifest['files'] = file_manifest(root/'base')
    write_json(root/'base/manifest.json', manifest)
    return root/'base'


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.articles = [article('one'), article('two')]
        self.settings = copy.deepcopy(REVIEW_DEFAULTS)
        self.provenance = {'mode': 'verified_snapshot', 'snapshot_manifest_sha256': 'b'*64, 'synthetic_fixture': True}
        self.source = source_review(self.root, self.articles, self.settings, self.provenance)

    def tearDown(self): self.temp.cleanup()

    def validate(self, path):
        # 合成源审核为 automated_market_review；绕过全量快照条件，直接校验合并逻辑。
        from unittest.mock import patch
        with patch('news_fusion.review_reuse.validate_review_artifact'):
            return validate_completed_review(path, self.articles, 'NASDAQ', self.provenance, self.settings)

    def complete(self, backend):
        from unittest.mock import patch
        with patch('news_fusion.review_reuse.validate_review_artifact'):
            return complete_review(self.source, self.articles, 'NASDAQ', self.root/'done', self.root/'cache2', backend,
                                   self.settings, self.provenance)

    def test_only_unreviewed_records_are_sent_and_inherited_rows_are_unchanged(self):
        self.assertEqual(unreviewed_ids(read_records(self.source/'market_review.jsonl')), ['two'])
        backend = FixtureBackend([response()])
        self.complete(backend)
        self.assertEqual(backend.calls, 1)
        before = read_records(self.source/'market_review.jsonl'); after = read_records(self.root/'done/market_review.jsonl')
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[1]['decision'], 'include'); self.assertEqual(after[1]['review_method'], 'local_llm')
        self.assertNotEqual(after[1]['reviewed_at'], before[0]['reviewed_at'])
        manifest = self.validate(self.root/'done')
        self.assertEqual(manifest['new_decisions'], {'include': 1})

    def test_edited_inherited_decision_fails_replay(self):
        self.complete(FixtureBackend([response()]))
        path = self.root/'done/market_review.jsonl'
        rows = read_records(path); rows[0]['decision'] = 'exclude'; write_records(path, rows)
        manifest = json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        manifest['files'] = file_manifest(path.parent); write_json(path.parent/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'differ'):
            self.validate(path.parent)

    def test_nothing_to_complete_is_an_error(self):
        rows = read_records(self.source/'market_review.jsonl'); rows[1]['review_method'] = 'local_llm'
        write_records(self.source/'market_review.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'no unreviewed'):
            self.complete(FixtureBackend())


class EvidenceBasisTests(unittest.TestCase):
    def test_summary_quote_is_unverified_unless_origin_is_declared(self):
        a = normalize_article({'source_id': 's', 'title': '', 'summary': 'The Federal Reserve raised its policy rate by 25 basis points.',
                               'language': 'en', 'market': 'US', 'published_at': '2016-12-14'})
        self.assertEqual(mq.text_origin(a), 'unknown')
        self.assertEqual(mq.evidence_basis(a, 'The Federal Reserve raised its policy rate'), 'unverified_summary')
        a['summary_origin'] = 'source_excerpt'
        self.assertEqual(mq.evidence_basis(a, 'The Federal Reserve raised its policy rate'), 'source_text')
        a['summary_origin'] = 'model_summary'
        self.assertEqual(mq.evidence_basis(a, 'The Federal Reserve raised its policy rate'), 'unverified_summary')

    def test_title_quote_without_declared_origin_is_title_only(self):
        a = normalize_article({'source_id': 's', 'title': 'Fed raises rates for second time in a decade',
                               'summary': 'Officials cited labor markets.', 'language': 'en', 'market': 'US'})
        self.assertEqual(mq.evidence_basis(a, 'Fed raises rates'), 'title_only')

    def test_filter_routes_and_speculative_flag(self):
        a = article('x'); b = dict(article('y'), summary_origin='source_excerpt'); c = article('z')
        routes = [{'source_id': 'x', 'evidence_quote': 'Revenue improves for business'},
                  {'source_id': 'y', 'evidence_quote': 'Revenue improves for business'},
                  {'source_id': 'z', 'evidence_quote': 'Earnings outlook'}]
        kept, bases = mq.filter_routes(routes, {'x': a, 'y': b, 'z': c}, {'source_text'})
        self.assertEqual([r['source_id'] for r in kept], ['y'])
        self.assertEqual(bases, {'x': 'unverified_summary', 'y': 'source_text', 'z': 'title_only'})
        self.assertTrue(mq.speculative_summary({'summary': 'The press release is likely related to Apple.'}))
        self.assertFalse(mq.speculative_summary({'summary': 'Payrolls rose by 200,000 in May.'}))

    def test_unknown_declared_origin_is_rejected(self):
        with self.assertRaises(ValueError):
            mq.text_origin({'summary_origin': 'guessed'})


class EventDedupTests(unittest.TestCase):
    def test_rewrites_cluster_and_distinct_types_do_not(self):
        base = np.random.RandomState(0).normal(size=768).astype(np.float32)
        other = np.random.RandomState(1).normal(size=768).astype(np.float32)
        vectors = np.stack([base, base + 0.01, other, base])
        entries = [{'source_id': 'a', 'session': 10, 'event_type': None, 'embedding_row': 0},
                   {'source_id': 'b', 'session': 11, 'event_type': None, 'embedding_row': 1},
                   {'source_id': 'c', 'session': 10, 'event_type': None, 'embedding_row': 2},
                   {'source_id': 'd', 'session': 10, 'event_type': 'cpi', 'embedding_row': 3}]
        result = mq.cluster_events(entries, vectors, 1, 0.92)
        self.assertEqual(result['a'], result['b'])
        self.assertNotEqual(result['a'], result['c'])
        self.assertEqual(result['a'], result['d'])  # 簇首类型未知，不阻止合并
        far = mq.cluster_events([dict(entries[0]), dict(entries[1], session=13)], vectors, 1, 0.92)
        self.assertNotEqual(far['a'], far['b'])
        typed = mq.cluster_events([dict(entries[0], event_type='gdp'), dict(entries[3])], vectors, 1, 0.92)
        self.assertNotEqual(typed['a'], typed['d'])


class GateTests(unittest.TestCase):
    def quality(self, events=10, days=10, types=None):
        split = {'articles': events, 'events': events, 'active_days': days, 'event_types': types or {'a': 5, 'b': 5}}
        return {'splits': {s: dict(split) for s in mq.SPLITS}}

    def gates(self, value=5):
        return {'min_events': {s: value for s in mq.SPLITS}, 'min_active_days': {s: value for s in mq.SPLITS},
                'max_event_type_share': 0.6}

    def test_unset_gates_block(self):
        gates = mq.defaults()['gates']
        self.assertTrue(any('not set' in b for b in mq.gate_blockers(self.quality(), gates)))

    def test_pass_low_coverage_and_concentration(self):
        self.assertEqual(mq.gate_blockers(self.quality(), self.gates()), [])
        self.assertTrue(mq.gate_blockers(self.quality(events=3), self.gates()))
        self.assertTrue(any('share' in b for b in mq.gate_blockers(self.quality(types={'a': 9, 'b': 1}), self.gates())))
        self.assertTrue(mq.gate_blockers(None, self.gates()))

    def test_settings_validation(self):
        settings = mq.defaults()
        mq.validate_settings(settings)
        for key, value in (('evidence_basis', []), ('evidence_basis', ['guess']), ('news_routes', 'yes')):
            bad = mq.defaults(); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                mq.validate_settings(bad)
        bad = mq.defaults(); bad['news_routes'] = False
        with self.assertRaises(ValueError):
            mq.validate_settings(bad)


def official(sid='official:fomc:2016-12-14', **updates):
    body = 'Information received since the Federal Open Market Committee met in November indicates that the labor market has continued to strengthen.'
    row = {'source_id': sid, 'source_type': 'official_macro_release', 'event_type': 'fomc_statement',
           'title': 'Federal Reserve issues FOMC statement', 'text': body,
           'source_url': 'https://www.federalreserve.gov/newsevents/pressreleases/monetary20161214a.htm',
           'published_at': '2016-12-14T14:00:00-05:00', 'fetched_at': '2026-11-18T00:00:00+00:00',
           'content_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(), 'first_release': True,
           'publication_time_basis': 'official_page_timestamp'}
    row.update(updates)
    return row


class OfficialEventTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self): self.temp.cleanup()

    def test_valid_events_build_and_replay(self):
        write_records(self.root/'events.jsonl', [official(), official('official:fomc:2017-03-15', published_at='2017-03-15T14:00:00-04:00')])
        manifest = build_events(self.root/'events.jsonl', self.root/'out')
        self.assertEqual(manifest['articles'], 2)
        articles = read_records(self.root/'out/articles.jsonl')
        self.assertTrue(all(a['summary_origin'] == 'source_excerpt' and a['impact_scope'] == ['market'] for a in articles))
        self.assertEqual(mq.evidence_basis(articles[0], articles[0]['title']), 'source_text')
        validate_events(self.root/'out')

    def test_rejections(self):
        cases = [dict(first_release=False), dict(source_url='https://example.com/fomc'),
                 dict(source_url='http://www.federalreserve.gov/x'), dict(event_type='cpi'),
                 dict(published_at='2016-12-14'), dict(content_sha256='0'*64),
                 dict(publication_time_basis='estimated'), dict(fetched_at='2016-12-13T00:00:00+00:00'),
                 dict(source_id='fomc-2016')]
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ValueError):
                check_record(official(**update))

    def test_tampered_artifact_fails(self):
        write_records(self.root/'events.jsonl', [official()])
        build_events(self.root/'events.jsonl', self.root/'out')
        path = self.root/'out/articles.jsonl'
        rows = read_records(path); rows[0]['published_at'] = '2016-12-13T14:00:00-05:00'; write_records(path, rows)
        manifest = json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        manifest['files'] = file_manifest(path.parent); write_json(path.parent/'manifest.json', manifest)
        with self.assertRaises(ValueError):
            validate_events(self.root/'out')


class PanelIntegrationTests(unittest.TestCase):
    def setUp(self):
        from tests_news.test_pipeline import make_axis
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.axis, self.schedule = make_axis(self.root)
        self.provenance = {'mode': 'verified_snapshot', 'snapshot_manifest_sha256': 'a'*64, 'synthetic_unit_fixture': True}

    def tearDown(self): self.temp.cleanup()

    def routes(self, articles, quote='Revenue improves for business'):
        from news_fusion.routing import review_template, publish_routes
        review_template(articles, 'TEST', self.root/'template', self.provenance)
        rows = read_records(self.root/'template/market_review.jsonl')
        for row in rows:
            row.update(decision='include', reviewed_at='2020-01-01T00:00:00Z', reviewed_by='SYNTHETIC UNIT REVIEWER',
                       rationale='Synthetic fixture only: market impact.', evidence_quote=quote)
        publish_routes(articles, rows, 'TEST', self.root/'routes', self.provenance)
        return self.root/'routes'

    def official_features(self, published):
        from news_fusion.encoder import encode_articles
        from news_fusion.official_events import validate_events
        from news_fusion.io import digest_file
        from tests_news.test_pipeline import FakeBackend
        write_records(self.root/'events.jsonl', [official('official:fomc:x', published_at=published, fetched_at=published)])
        build_events(self.root/'events.jsonl', self.root/'events')
        articles = read_records(self.root/'events/articles.jsonl')
        provenance = {'mode': 'official_events', 'official_directory': str(self.root/'events'),
                      'official_manifest_sha256': digest_file(self.root/'events/manifest.json'), 'articles': 1}
        encode_articles(articles, FakeBackend(), self.root/'official_features', self.root/'cache', input_provenance=provenance)
        return self.root/'official_features'

    def test_evidence_filter_official_merge_and_event_counts(self):
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        from news_fusion.dataset import NewsPanel
        from tests_news.test_pipeline import FakeBackend
        day = self.axis.dates[2]
        news = [article('n1', publication=day+'T15:00:00Z', scopes=['market']),
                article('n2', publication=day+'T15:30:00Z', scopes=['market'])]
        routes = self.routes(news, quote='Earnings outlook')
        encode_articles(news, FakeBackend(), self.root/'features', self.root/'cache', input_provenance=self.provenance)
        channel = mq.defaults()
        channel['evidence_basis'] = ['source_text', 'title_only']
        channel['official_events'] = str(self.official_features(day+'T14:00:00-05:00'))
        channel['event_dedup']['cosine_threshold'] = 0.99
        manifest = build_panel(self.axis, self.schedule, self.root/'features', [], [], self.root/'panel', market_routes=routes,
                               projection='random_projection', decision_lookback=2, market_channel=channel)
        quality = manifest['market_quality']
        self.assertEqual(quality['news_routes_total'], 2); self.assertEqual(quality['news_routes_kept'], 2)
        self.assertEqual(quality['news_route_evidence_basis'], {'title_only': 2})
        self.assertEqual(quality['official']['routes'], 1)
        self.assertEqual(quality['market_articles'], 3)
        panel = NewsPanel(self.root/'panel')
        ledger = read_records(self.root/'panel/provenance.jsonl')
        self.assertIn('official:fomc:x', {r['source_id'] for r in ledger})
        self.assertTrue(all('market_event_id' in r for r in ledger if r['channel'] == 'market'))
        self.assertGreater(panel.masks['market'].sum(), 0)

    def test_default_basis_drops_unverified_news_routes(self):
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        from tests_news.test_pipeline import FakeBackend
        day = self.axis.dates[2]
        news = [article('n1', publication=day+'T15:00:00Z', scopes=['market'])]
        routes = self.routes(news)
        encode_articles(news, FakeBackend(), self.root/'features', self.root/'cache', input_provenance=self.provenance)
        channel = mq.defaults()
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            build_panel(self.axis, self.schedule, self.root/'features', [], [], self.root/'panel', market_routes=routes,
                        projection='random_projection', decision_lookback=2, market_channel=channel)


class FunnelTests(unittest.TestCase):
    def test_funnel_counts_by_split(self):
        from tests_news.test_pipeline import make_axis
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        axis, _ = make_axis(Path(temp.name))
        a = article('m', publication=axis.dates[2]+'T15:00:00Z', scopes=['market'])
        rows = [{'source_id': 'm', 'article_pin': article_pin(a), 'decision': 'include', 'evidence_quote': 'Revenue improves for business',
                 'review_method': 'local_llm'}]
        report = mq.funnel([a], rows, axis, lookback=2)
        self.assertEqual(report['train']['decision_include'], 1)
        self.assertEqual(report['train']['upstream_market_candidate'], 1)
        self.assertEqual(report['train']['include_basis_unverified_summary'], 1)


if __name__ == '__main__':
    unittest.main()
