"""Boundary tests for coverage accounting; no production news or model files written."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('pipeline_diagnostics',PROJECT/'scripts/16_check_news_pipeline.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
from news_fusion.time_axis import Axis,Schedule
from news_fusion.panel import relationship_valid


class CoverageBoundaries(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        days=pd.bdate_range('2018-01-02',periods=40).strftime('%Y-%m-%d').tolist()
        self.axis=Axis('NASDAQ',['NASDAQ:A','NASDAQ:B'],days,25,32,{})
        self.schedule=Schedule(self.axis,[{'date':d,'decision_cutoff_at':d+'T21:00:00Z'} for d in days])
        self.settings={'mode':'retrospective','relation_time_basis':'event_date','simulated_latency_seconds':300,'lookback_sessions':5}
        self.a={'source_id':'a','input_text_hash':'same','published_at':days[23], 'publication_precision':'date'}
        self.link={'source_id':'a','dataset':'NASDAQ','channel':'industry','verification_status':'verified',
                   'mapping_source':'evidence','valid_from':days[0],'valid_to':None,
                   'industry_standard':'TEST','industry_version':'v1','industry_code':'1','evidence_type':'explicit'}
        self.member={'dataset':'NASDAQ','stock_uid':'NASDAQ:A','array_index':0,
                     'verification_status':'reference_only','membership_basis':'undated_dataset_reference',
                     'historical_validity_verified':False,'mapping_source':'released-reference',
                     'industry_standard':'TEST','industry_version':'v1','industry_code':'1'}

    def expand(self,articles=None,links=None,members=None):
        return audit.enumerate_groups(self.axis,self.schedule,articles or [self.a],links or [self.link],members or [self.member],[],self.settings)[0]

    def test_date_only_no_same_day_and_exactly_five_sessions(self):
        groups=self.expand()
        self.assertEqual({k[2] for k in groups},set(range(24,29)))
        self.assertEqual({k[1] for k in groups},{0})

    def test_industry_versions_never_cross_map(self):
        member=dict(self.member,industry_version='v2')
        self.assertEqual(self.expand(members=[member]),{})

    def test_duplicate_text_keeps_first_available_source_per_stock_day(self):
        groups=self.expand(articles=[dict(self.a,source_id='b'),self.a],links=[dict(self.link,source_id='b'),self.link])
        self.assertTrue(all(list(items.values())==['a'] for items in groups.values()))

    def test_reference_membership_cannot_enter_observed_pit(self):
        self.assertFalse(relationship_valid(self.member,self.axis.dates[24],self.axis.dates[23],self.schedule.cutoffs[24],'observed_pit'))

    def test_expired_membership_is_excluded(self):
        member=dict(self.member,verification_status='verified',valid_from=self.axis.dates[0],valid_to=self.axis.dates[23])
        self.assertEqual(self.expand(members=[member]),{})

    def test_target_date_split_differs_from_publication_date(self):
        self.assertEqual(audit.phase_for_target(24,25,32,40),'train')
        self.assertEqual(audit.phase_for_target(25,25,32,40),'validation')
        self.assertIsNone(audit.phase_for_target(15,25,32,40))

    def test_shared_market_counts_stocks_but_excludes_invalid_labels(self):
        data=SimpleNamespace(window=lambda t:(None,None,None,np.array([[1],[0]])))
        groups={('market',-1,24):{'same':'a'}}
        rows,_=audit.summarize_groups(self.axis,data,groups,['market'])
        row=next(r for r in rows if r['channel']=='market' and r['split']=='validation')
        self.assertEqual((row['mapped_unique_articles'],row['stock_decision_cells'],row['valid_price_label_cells']),(1,2,1))


if __name__=='__main__':unittest.main()
