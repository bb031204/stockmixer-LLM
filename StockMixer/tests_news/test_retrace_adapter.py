import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from news_fusion.source_adapter import adapt_snapshot,producer_hash,validate_adapted
from news_fusion.source_proof import bound_quote
from news_fusion.upstream import scope_claims,make_reviews
from news_fusion.routing import publish_routes,load_routes
from news_fusion.market_quality import evidence_basis
from news_fusion.io import write_json,read_records,digest
from news_fusion.reference_membership import memberships
from news_fusion.panel import relationship_valid
from news_fusion.retrace_adapter import import_retrace
from tests_news.test_source_adapter import SourceAdapterTests

class RetraceAdapterTests(unittest.TestCase):
    def fixture(self,root):
        for name in ('evidence','data/US','audit'):(root/name).mkdir(parents=True,exist_ok=True)
        helper=SourceAdapterTests();row=helper.row()
        row['时间信息']['availability_basis']='retrospective_retrace_current_inference_not_historical_PIT'
        quote='US equity markets face a change in financing conditions.'
        item=dict(scope='US_EQUITY',evidence_quote=quote,rationale='Financing conditions influence broad US risk assets.',confidence=.8,
            evidence_basis='original_article_quote_verified_before_raw_cleanup',original_evidence_verified=True)
        audit=dict(source_id=row['source_id'],record_hash=producer_hash(row),source_body_hash='c'*64,
            language='en',canonical_market='US',model='qwen3:8b',model_digest='fixture',prompt_version='fixture',
            llm_inferred_at='2020-03-05T18:30:00Z',market_scope_version='market-proof-1',market_scope=['US_EQUITY'],scope_evidence=[item])
        pack=dict(schema_version='market-proof-1',source_id=row['source_id'],record_hash=producer_hash(row),
            source_body_hash='c'*64,evidence=[dict(field='scope_evidence',index=0,key='evidence_quote',quote=quote,
            quote_sha256=hashlib.sha256(quote.encode()).hexdigest())])
        write_json(root/'evidence/article.json',pack)
        audit['evidence_pack']=dict(path='adapted/evidence/article.json',sha256=hashlib.sha256((root/'evidence/article.json').read_bytes()).hexdigest())
        write_json(root/'data/US/article.json',row);write_json(root/'audit/article.json',audit)
        return quote

    def test_original_only_proof_reaches_route_and_quality_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'producer';quote=self.fixture(source)
            adapt_snapshot(source,root/'snapshot',markets=('US',))
            a=read_records(root/'snapshot/articles.jsonl')[0]
            self.assertNotIn(quote,a['summary'])
            self.assertTrue(bound_quote(a,'scope_evidence',quote))
            self.assertFalse(a['upstream_observed_evidence_complete'])
            self.assertEqual(len(scope_claims(a,.75)[0]),1)
            self.assertFalse(scope_claims(a,.85)[0])
            publish_routes([a],make_reviews([a],'NASDAQ',.75),'NASDAQ',root/'routes',
                           {'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64},True)
            routes,bindings,_=load_routes(root/'routes','NASDAQ')
            self.assertEqual(len(routes),1)
            self.assertEqual(evidence_basis(bindings[0],quote),'source_text')
            self.assertTrue(validate_adapted(root/'snapshot')['valid'])
            for change in ('quote','audit','pack','record'):
                b=copy.deepcopy(a)
                if change=='quote':self.assertFalse(bound_quote(b,'scope_evidence','Invented broad market claim.'));continue
                if change=='record':b['source_record_hash']='d'*64
                else:b['upstream_analysis']['bound_source_evidence'][change+'_json']+=' '
                self.assertFalse(bound_quote(b,'scope_evidence',quote))

    def test_reference_is_axis_aligned_and_never_valid_in_pit(self):
        axis=SimpleNamespace(dataset='NASDAQ',stock_uids=['NASDAQ:B','NASDAQ:A'],source={'tickers':['B','A']})
        dictionary=SimpleNamespace(by_key={'TRSR:bank':dict(industry_standard='TRSR-INDUSTRY',industry_version='undated',industry_code='bank',name='Banks',source='fixture archive',source_sha256='f'*64)})
        with patch('news_fusion.reference_membership.reference_inputs',return_value=({'NASDAQ':{'A':['TRSR:bank']}},{},{})):
            rows=memberships(axis,dictionary,None,None,'2026-11-18T00:00:00Z')
        self.assertEqual((rows[0]['stock_uid'],rows[0]['array_index']),('NASDAQ:A',1))
        self.assertIsNone(rows[0]['valid_from'])
        self.assertTrue(relationship_valid(rows[0],'2015-01-05','2015-01-02',None,'retrospective'))
        self.assertFalse(relationship_valid(rows[0],'2015-01-05','2015-01-02',None,'observed_pit'))

    def test_incomplete_batch_blocks_formal_import(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_json(root/'progress.json',{'pending':1})
            with self.assertRaisesRegex(ValueError,'pending'):import_retrace(root,root/'out')

    def test_legacy_union_batch_cannot_be_formally_imported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_json(root/'progress.json',{'pending':0})
            write_json(root/'acceptance.json',{'passed':True})
            write_json(root/'manifest.json',{'files':{}})
            with self.assertRaisesRegex(ValueError,'Fresh dictionary/scope replacement'):import_retrace(root,root/'out')

    def test_modified_accepted_batch_cannot_be_formally_imported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_json(root/'progress.json',{'pending':0})
            write_json(root/'acceptance.json',{'passed':True})
            write_json(root/'manifest.json',{'classification_contract':'dictionary-scope-replacement-1',
                'files':{'adapted/data/US/a.json':'0'*64}})
            with self.assertRaisesRegex(ValueError,'changed since acceptance'):import_retrace(root,root/'out')

    def test_industry_reference_reaches_only_its_stock_and_masks_others(self):
        from tests_news.test_pipeline import make_axis,FakeBackend,article
        from news_fusion.encoder import encode_articles
        from news_fusion.panel import build_panel
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);axis,schedule=make_axis(root);a=article(scopes=['industry'])
            provenance={'mode':'verified_snapshot','snapshot_manifest_sha256':'d'*64}
            encode_articles([a],FakeBackend(),root/'features',root/'cache',input_provenance=provenance)
            link=dict(source_id=a['source_id'],channel='industry',dataset='TEST',verification_status='verified',
                mapping_source='synthetic scoped-news fixture',valid_from='2020-03-05',valid_to=None,
                known_from='2020-03-05T00:00:00Z',industry_standard='TRSR-INDUSTRY',industry_version='undated',industry_code='bank',evidence_type='inferred')
            member=dict(dataset='TEST',stock_uid='TEST:B',array_index=1,dataset_ticker='B',
                industry_standard='TRSR-INDUSTRY',industry_version='undated',industry_code='bank',
                verification_status='reference_only',membership_basis='undated_dataset_reference',
                historical_validity_verified=False,mapping_source='synthetic dataset fixture',exposure_weight=1.0)
            build_panel(axis,schedule,root/'features',[link],[member],root/'panel',projection='random_projection',
                components=2,decision_lookback=2)
            mask=np.load(root/'panel/industry_news_mask.npy')
            self.assertFalse(mask[0].any());self.assertTrue(mask[1].any())
            manifest=json.loads((root/'panel/manifest.json').read_text())
            self.assertFalse(manifest['historical_industry_membership_verified'])
            self.assertEqual(manifest['industry_membership_policy'],'undated_reference_exploration')
            with self.assertRaisesRegex(ValueError,'retrospective-only'):
                build_panel(axis,schedule,root/'features',[link],[member],root/'bad',mode='observed_pit',
                    projection='random_projection',components=2,decision_lookback=2)
