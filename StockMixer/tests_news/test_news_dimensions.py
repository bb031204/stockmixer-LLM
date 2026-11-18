import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from news_fusion.encoder import encode_articles
from news_fusion.panel import build_panel
from news_fusion.dataset import NewsPanel
from news_fusion.fusion import StockMixerWithNews
from tests_news.test_pipeline import make_axis,article,company,FakeBackend
from tests_news.test_fusion import baseline


class NewsDimensionTests(unittest.TestCase):
    def test_configurable_projection_changes_only_news_shape_and_keeps_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);axis,schedule=make_axis(root)
            encode_articles([article()],FakeBackend(),root/'features',root/'cache')
            manifest=build_panel(axis,schedule,root/'features',[company()],[],root/'panel',
                                 projection='random_projection',decision_lookback=2,components=16)
            self.assertEqual(manifest['feature_dim'],24)
            self.assertEqual(manifest['transform']['components'],16)
            panel=NewsPanel(root/'panel')
            row=panel.features['company'][0,1]
            self.assertEqual(row.shape,(24,))
            np.testing.assert_allclose(row[:3],[.6,.1,.3])
            self.assertAlmostEqual(float(row[19]),float(np.log(2)),places=5)
            self.assertEqual(panel.decision(1)['company_news'].shape,(2,24))
            del row,panel  # Release Windows memory maps before removing the fixture directory.

    def test_smaller_pca_still_fits_training_articles_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);axis,schedule=make_axis(root)
            rows=[article('a'),article('b'),article('future',publication='2020-03-16T18:00:00Z')]
            encode_articles(rows,FakeBackend(),root/'features',root/'cache')
            manifest=build_panel(axis,schedule,root/'features',[company('a'),company('b'),company('future')],[],
                                 root/'panel',decision_lookback=2,components=1)
            self.assertEqual(manifest['feature_dim'],9)
            self.assertEqual(manifest['transform']['eligible_training_articles'],2)
            self.assertEqual(manifest['transform']['fitted_articles'],2)
            self.assertEqual(NewsPanel(root/'panel').features['company'].shape,(2,8,9))

    def test_alternate_news_width_keeps_original_stockmixer_bitwise_at_initialization(self):
        torch.manual_seed(123);torch.set_num_threads(1)
        original=baseline.StockMixer(12,16,5,4,3)
        wrapper=StockMixerWithNews(original,news_dim=24,hidden_dim=32)
        x=torch.rand(12,16,5);price=torch.rand(12,1)+.5
        supplied={key:torch.randn(12,24) for key in ('company_news','industry_news','market_news')}
        masks={key:torch.ones(12,1) for key in ('company_mask','industry_mask','market_mask')}
        self.assertTrue(torch.equal(original(x),wrapper(x,price,**supplied,**masks)))
        self.assertEqual(wrapper.stock_mixer.state_dict().keys(),original.state_dict().keys())
