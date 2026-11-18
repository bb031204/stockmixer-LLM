import copy,hashlib,io,json,unittest,zipfile
from unittest.mock import patch
from news_fusion import legacy_industry as m
from news_fusion.industry_scope_review import apply_scope_reviews
from news_fusion.inputs import normalize_article
from news_fusion.routing import article_pin
from news_fusion.io import digest

class Tests(unittest.TestCase):
    def article(self):
        return dict(source_id='a',source_record_hash='r',record_version='old',title='Supply',summary='Semiconductor manufacturers face supply shortages.',
          published_at='2014-01-02',language='en',market='US',impact_scope=['company'],industry_impacts=[],
          feature_available_at='2026-01-01T00:00:00Z',effective_feature_available_at='2026-01-01T00:00:00Z')
    def analysis(self):
        return dict(record_hash='r',industry_impacts=[],dictionary={'sha256':'a'*64},industries=[],status='completed',
          feature_available_at='2026-02-01T00:00:00+00:00',llm_inferred_at='2026-02-01T00:00:00+00:00')
    def test_market_binding_unchanged_and_industry_availability_advances(self):
        a=self.article();r=m.merged_articles([a],{'a':self.analysis()},'b'*64)[0]
        self.assertEqual(article_pin(normalize_article(a)),article_pin(normalize_article(r)))
        self.assertEqual(r['effective_feature_available_at'],'2026-02-01T00:00:00+00:00')
        self.assertEqual(a['industry_impacts'],[])
    def test_changed_source_cannot_inherit_review(self):
        a=self.article();old=dict(a,source_record_hash='changed')
        result=m.merged_articles([a],{'a':self.analysis()},'b'*64,{'a':old})[0]
        self.assertEqual(result['source_record_hash'],'r');self.assertEqual(result['industry_enrichment']['legacy_review_inheritance'],'not_reused_source_or_text_version_changed')
    def test_changed_text_cannot_inherit_review(self):
        a=self.article();old=dict(a,summary='New statement.')
        result=m.merged_articles([a],{'a':self.analysis()},'b'*64,{'a':old})[0]
        self.assertEqual(result['summary'],a['summary'])
    def test_industry_metadata_survives_encoding_normalization(self):
        r=m.merged_articles([self.article()],{'a':self.analysis()},'b'*64)[0]
        self.assertEqual(normalize_article(r)['industry_enrichment'],r['industry_enrichment'])
    def test_archive_path_traversal_rejected(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('../bad','x')
        data.seek(0)
        with self.assertRaises(ValueError):m.archive_bytes(data)
    def test_archive_duplicate_rejected(self):
        import warnings
        data=io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(data,'w') as z:z.writestr('a','x');z.writestr('a','y')
        data.seek(0)
        with self.assertRaises(ValueError):m.archive_bytes(data)
    def test_self_declared_unverified_proposals_cannot_be_approved(self):
        with patch.object(m,'validate_enriched',side_effect=ValueError('invalid snapshot')):
            with self.assertRaises(ValueError):m.automated_reviews('fake',[self.article()])
    def test_automatic_gate_does_not_count_as_human(self):
        a=self.article();a['industry_impacts']=[{'industry_key':'SEC-SIC:3674','evidence':'Semiconductor manufacturers face supply shortages.'}]
        row=dict(decision='include',review_method='automated_dictionary_summary_evidence_v1',reviewed_by='upstream_evidence_gate_not_human',
          reviewed_at='2026-01-01T00:00:00Z',approved_impacts_hash=digest(a['industry_impacts']),rationale='Explicit sector-wide shortage evidence.')
        result,summary=apply_scope_reviews([a],required=True,preserve_approval_time=True,automated={'a':row})
        self.assertEqual(summary['human_confirmed_samples'],0);self.assertEqual(summary['automated_evidence_gated_samples'],1)
        self.assertEqual(result[0]['industry_impacts'],a['industry_impacts'])
    def test_modified_proposal_rejected(self):
        row=dict(decision='include',review_method='automated_dictionary_summary_evidence_v1',reviewed_by='upstream_evidence_gate_not_human',
          reviewed_at='2026-01-01T00:00:00Z',approved_impacts_hash='wrong')
        with self.assertRaises(ValueError):apply_scope_reviews([self.article()],required=True,automated={'a':row})
    def test_new_sources_not_silently_added_to_market(self):
        from news_fusion.routing import article_pin
        a=normalize_article(self.article());new=dict(a,source_id='b')
        old={'source_id':'a','dataset':'NASDAQ','article_pin':article_pin(a),'decision':'exclude'}
        values=m.extended_rows([a,new],[a],[old],'NYSE')
        self.assertEqual(values[0],dict(old,dataset='NYSE'));self.assertEqual(values[1]['decision'],'pending')
    def test_changed_binding_rejects_old_market_approval(self):
        a=normalize_article(self.article());new=dict(a,summary='Changed content')
        from news_fusion.routing import article_pin
        old={'source_id':'a','dataset':'NASDAQ','article_pin':article_pin(a),'decision':'include'}
        with self.assertRaises(ValueError):m.extended_rows([new],[a],[old],'NYSE')
    def test_foreign_region_not_automatically_broadcast_to_us(self):
        a=self.analysis();a['industry_impacts']=[dict(industry_key='SEC-SIC:6029',quote='Euro zone banks lower rates.',evidence='Euro zone banks lower rates.',industry_scope_quote='Euro zone banks lower rates.')]
        r=m.merged_articles([self.article()],{'a':a},'b'*64)[0]
        self.assertEqual(r['industry_impacts'],[]);self.assertEqual(len(r['industry_enrichment']['regional_scope_pending_impacts']),1)
    def test_short_scope_can_recover_explicit_regulatory_context(self):
        summary='The Federal Reserve issued guidance for state member banks involved in energy lending.'
        t=dict(scope='industry_wide',scope_quote='state member banks',confidence=.9,mechanism='Guidance changes sector-wide lending requirements.')
        self.assertEqual(m.context_scope_evidence(t,summary),summary)
    def test_multi_company_financing_is_not_regulatory_sector_evidence(self):
        summary='Major life insurance companies provided financing to a single property company.'
        t=dict(scope='industry_wide',scope_quote='life insurance companies',confidence=.9,mechanism='A financing transaction between several counterparties.')
        self.assertIsNone(m.context_scope_evidence(t,summary))
    def test_context_recovery_requires_model_proposal_and_verbatim_scope(self):
        summary='The Federal Reserve issued guidance for state member banks.'
        t=dict(scope='company_only',scope_quote='state member banks',confidence=.9,mechanism='Lending guidance changes.')
        self.assertIsNone(m.context_scope_evidence(t,summary));t['scope']='industry_wide';t['scope_quote']='all foreign banks'
        self.assertIsNone(m.context_scope_evidence(t,summary))
    def test_generic_food_industry_does_not_prove_food_retail_target(self):
        a=self.analysis();a['industry_impacts']=[dict(industry_key='SEC-SIC:5400',industry_name_standard='RETAIL-FOOD STORES',
            quote='The food industry faces rising costs.',industry_scope_quote='The food industry faces rising costs.')]
        r=m.merged_articles([self.article()],{'a':a},'b'*64)[0]
        self.assertEqual(r['industry_impacts'],[]);self.assertEqual(len(r['industry_enrichment']['classification_qualification_pending_impacts']),1)

if __name__=='__main__':unittest.main()
