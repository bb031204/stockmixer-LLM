import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from news_fusion.io import digest, write_json
from news_fusion.panel import relationship_valid
from news_fusion.routing import article_pin,publish_routes,load_routes
from news_fusion.semantic_review import apply_sample,decision_pin

from news_fusion import master_data as builder


class DatedEvidenceTests(unittest.TestCase):
    def test_official_filing_url_only(self):
        self.assertEqual(builder.canonical_index('https://www.sec.gov/Archives/edgar/data/1000045/0001193125-13-029629-index.html'),
            'https://www.sec.gov/Archives/edgar/data/1000045/000119312513029629/0001193125-13-029629-index.htm')
        for value in ['https://evil.test/Archives/edgar/data/1/0001193125-13-029629-index.html',
                      'http://www.sec.gov/Archives/edgar/data/1/0001193125-13-029629-index.html']:
            with self.assertRaises(ValueError):builder.canonical_index(value)

    def test_dated_filing_not_current_ticker_list(self):
        source='''<div>Filing Date 2013-01-30</div><div>NICHOLAS FINANCIAL INC (Filer) CIK: 0001000045</div>
         <div>SIC: 6153 Short-Term Business Credit Institutions</div>
         <table><tr><td>1</td><td>RELEASE</td><td><a href="/Archives/a.htm">a.htm</a></td><td>EX-99.1</td></tr></table>'''
        day,code,name,docs=builder.index_fields(source,'1000045')
        self.assertEqual((day,code,name),('2013-01-30','6153','Short-Term Business Credit Institutions'))
        self.assertEqual(docs,[('/Archives/a.htm','EX-99.1')])
        with self.assertRaises(ValueError):builder.index_fields(source,'12345')

    def test_exchange_ticker_boundaries(self):
        self.assertIsNotNone(builder.trading_quote('Nicholas Financial, Inc. (NASDAQ: NICK) common shares','NICK','NASDAQ'))
        self.assertIsNone(builder.trading_quote('Nicholas Financial, Inc. (NASDAQ: NICKX)','NICK','NASDAQ'))
        self.assertIsNone(builder.trading_quote('Nicholas Financial, Inc. (NYSE: NICK)','NICK','NASDAQ'))

    def test_snapshot_day_can_attribute_later_event_impact_only_explicitly(self):
        row={'valid_from':'2013-01-30','valid_to':'2013-01-31','verification_status':'verified','mapping_source':'SEC'}
        self.assertFalse(relationship_valid(row,'2013-01-31','2013-01-30',None,'retrospective'))
        self.assertTrue(relationship_valid(row,'2013-01-31','2013-01-30',None,'retrospective','event_date'))
        self.assertFalse(relationship_valid(row,'2013-02-01','2013-01-31',None,'retrospective','event_date'))
        self.assertFalse(relationship_valid(row,'2013-01-29','2013-01-30',None,'retrospective','event_date'))

    def test_event_date_does_not_bypass_observed_availability(self):
        import pandas as pd
        row={'valid_from':'2013-01-30','valid_to':'2013-01-31','verification_status':'verified','mapping_source':'SEC',
             'known_from':'2026-11-17T00:00:00+00:00'}
        self.assertFalse(relationship_valid(row,'2013-01-31','2013-01-30',pd.Timestamp('2013-01-31T21:00:00Z'),'observed_pit','event_date'))


class SemanticReviewTests(unittest.TestCase):
    def setUp(self):
        self.article={'source_id':'a','title':'','summary':'A single company changes its product mix.',
            'input_text_hash':digest(['','A single company changes its product mix.']),
            'published_at':'2013-01-30','market':'US','impact_scope':['company']}
        self.review={'source_id':'a','article_pin':article_pin(self.article),'dataset':'NASDAQ',
            'decision':'include','evidence_quote':self.article['summary'],'rationale':'Original model incorrectly calls this broad market.',
            'reviewed_by':'local_llm','reviewed_at':'2026-11-16T00:00:00Z'}
        self.sample={'source_id':'a','article_pin':article_pin(self.article),'original_decision_pin':decision_pin(self.review),
            'decision':'exclude','evidence_quote':None,'rationale':'Only a company event; no broad market mechanism.',
            'review_method':'assistant_secondary_review','reviewed_by':'Codex assistant','reviewed_at':'2026-11-17T00:00:00Z',
            'apply_decision':True,'quarantine':False}

    def test_assistant_is_not_a_human_signature(self):
        _,report=apply_sample([self.article],[self.review],[self.sample],{})
        self.assertFalse(report['human_sample_checked']);self.assertEqual(report['human_confirmed_samples'],0)
        invalid=dict(self.sample,review_method='human_confirmation')
        with self.assertRaisesRegex(ValueError,'actual user'):apply_sample([self.article],[self.review],[invalid],{})

    def test_human_confirmation_is_limited_to_matching_sample(self):
        row=dict(self.sample,review_method='human_confirmation',reviewed_by='current_session_user')
        confirmation={'response':'认可','reviewed_by':'current_session_user',
            'samples':{'a':{'article_pin':row['article_pin'],'decision':'exclude'}}}
        _,report=apply_sample([self.article],[self.review],[row],confirmation)
        self.assertEqual(report['human_confirmed_samples'],1)
        confirmation['samples']['a']['decision']='include'
        with self.assertRaises(ValueError):apply_sample([self.article],[self.review],[row],confirmation)

    def test_original_decision_and_source_mutation_rejected(self):
        invalid=dict(self.review,rationale='A changed original rationale.')
        with self.assertRaisesRegex(ValueError,'different original'):apply_sample([self.article],[invalid],[self.sample],{})
        invalid=dict(self.sample,article_pin='0'*64)
        with self.assertRaisesRegex(ValueError,'frozen article'):apply_sample([self.article],[self.review],[invalid],{})

    def test_assistant_cannot_promote_pending_market_news(self):
        old=dict(self.review,decision='pending');row=dict(self.sample,original_decision_pin=decision_pin(old),decision='include',evidence_quote=self.article['summary'])
        with self.assertRaisesRegex(ValueError,'silently promote'):apply_sample([self.article],[old],[row],{})

    def test_published_overlay_replays_after_reload(self):
        updated,summary=apply_sample([self.article],[self.review],[self.sample],{})
        proof={'rows':[self.sample],'confirmation':{},'original_reviews':[self.review],'manifest_sha256':'a'*64}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'routes'
            publish_routes([self.article],updated,'NASDAQ',out,{'mode':'verified_snapshot','snapshot_manifest_sha256':'b'*64},True,semantic_audit=proof)
            routes,_,manifest=load_routes(out,'NASDAQ')
            self.assertEqual(routes,[]);self.assertEqual(manifest['semantic_review']['human_confirmed_samples'],0)
            data=json.loads((out/'semantic_user_confirmation.json').read_text(encoding='utf-8'));data['response']='fabricated'
            write_json(out/'semantic_user_confirmation.json',data)
            with self.assertRaises(ValueError):load_routes(out,'NASDAQ')

    def test_quarantined_include_is_rejected(self):
        row=dict(self.sample,decision='include',quarantine=True,evidence_quote=self.article['summary'])
        with self.assertRaisesRegex(ValueError,'quarantined'):apply_sample([self.article],[self.review],[row],{})

    def test_explicit_industry_is_not_invented_from_company_earnings(self):
        from news_fusion.industry_review import parse_impacts
        impact={'industry_name':'Short-Term Business Credit Institutions','evidence_type':'explicit',
            'evidence':'A single company changes its product mix.','mechanism':'The company needs more credit.',
            'direction':'positive_conditional','confidence':.9,'event_type':'earnings',
            'impact_role':'potential_beneficiary','importance':.7,'horizon':'short'}
        response={'done':True,'message':{'content':json.dumps({'impacts':[impact]})}}
        with self.assertRaisesRegex(ValueError,'not stated'):
            parse_impacts(response,self.article,[{'alias':'Short-Term Business Credit Institutions'}],1)


if __name__=='__main__':unittest.main()
