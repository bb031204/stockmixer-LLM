import copy
import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from news_fusion.fusion import StockMixerWithNews

BASELINE_SRC=Path(os.environ.get("NEWS_BASELINE_SRC",str(Path(__file__).resolve().parents[1]/"src")))
spec=importlib.util.spec_from_file_location("untouched_stockmixer_baseline",BASELINE_SRC/"model.py")
baseline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)
        self.base=baseline.StockMixer(12,16,5,4,3)
        self.wrapper=StockMixerWithNews(self.base)
        self.x=torch.rand(12,16,5)
        self.price=torch.rand(12,1)+.5
        self.news={channel+"_news":torch.randn(12,40) for channel in ("company","industry","market")}
        self.masks={channel+"_mask":torch.ones(12,1) for channel in ("company","industry","market")}

    def activate_heads(self):
        with torch.no_grad():
            for branch in self.wrapper.branches.values():
                branch.head.weight.fill_(.01)
                branch.head.bias.fill_(.02)

    def test_zero_initialized_heads_exactly_match_backbone(self):
        self.assertTrue(torch.equal(self.base(self.x),self.wrapper(self.x,self.price,**self.news,**self.masks)))

    def test_no_news_after_head_training_and_inactive_nans(self):
        self.activate_heads()
        features={k:torch.full_like(v,float('nan')) for k,v in self.news.items()}
        masks={k:torch.zeros_like(v) for k,v in self.masks.items()}
        expected=self.base(self.x)
        self.assertTrue(torch.equal(expected,self.wrapper(self.x,self.price,**features,**masks)))
        self.assertTrue(torch.equal(expected,self.wrapper(self.x,self.price)))

    def test_independent_channel_gates(self):
        self.activate_heads()
        for active in ("company","industry","market"):
            masks={k:torch.zeros_like(v) for k,v in self.masks.items()}
            masks[active+"_mask"]=torch.ones(12,1)
            _,info=self.wrapper(self.x,self.price,**self.news,**masks,return_details=True)
            for channel,delta in info["deltas"].items():
                if channel!=active:
                    self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
                else:
                    self.assertGreater(delta.abs().sum().item(),0)

    def test_market_broadcast_and_stock_conditioning(self):
        branch=self.wrapper.branches["market"]
        with torch.no_grad():
            branch.head.bias.fill_(.1)
            branch.gate.weight.zero_()
            branch.gate.weight[0,-1]=1
        result,details=self.wrapper(self.x,self.price,market_news=torch.randn(40),market_mask=torch.ones(1,1),return_details=True)
        self.assertEqual(result.shape,(12,1))
        self.assertGreater(details["deltas"]["market"].std().item(),0)

    def test_bad_shapes_prices_masks_and_active_nan(self):
        cases=[dict(base_price=torch.zeros_like(self.price)),dict(company_news=torch.zeros(12,39),company_mask=torch.ones(12,1)),
            dict(company_news=self.news["company_news"]),dict(company_mask=torch.ones(12,1)),
            dict(company_news=torch.full((12,40),float('nan')),company_mask=torch.ones(12,1)),
            dict(company_news=self.news["company_news"],company_mask=torch.full((12,1),.5))]
        for options in cases:
            inputs=dict(stock_x=self.x,base_price=self.price)
            inputs.update(options)
            with self.subTest(options=list(options)),self.assertRaises(ValueError):
                self.wrapper(**inputs)

    def test_loss_backward_updates_news_only_and_backbone_eval(self):
        self.wrapper.train()
        self.assertFalse(self.wrapper.stock_mixer.training)
        before=copy.deepcopy(self.base.state_dict())
        optimizer=torch.optim.Adam([p for p in self.wrapper.parameters() if p.requires_grad],lr=.001)
        # Two updates allow the Adapter/Gate to receive gradients after a zero Head starts learning.
        for _ in range(2):
            optimizer.zero_grad()
            prediction=self.wrapper(self.x,self.price,**self.news,**self.masks)
            loss,*_=baseline.get_loss(prediction,torch.randn(12,1)*.01,self.price,torch.ones(12,1),12,.1)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.wrapper.branches.parameters() if p.grad is not None))
            optimizer.step()
        self.assertTrue(all(torch.equal(value,self.base.state_dict()[name]) for name,value in before.items()))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))
        for branch in self.wrapper.branches.values():
            self.assertGreater(branch.head.weight.abs().sum().item(),0)
            self.assertGreater(branch.gate.weight.grad.abs().sum().item(),0)

    def test_disabled_channels_and_checkpoint_roundtrip(self):
        wrapper=StockMixerWithNews(copy.deepcopy(self.base),enabled_channels=("company",))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fusion.pt"
            torch.save(wrapper.state_dict(),path)
            restored=StockMixerWithNews(copy.deepcopy(self.base),enabled_channels=("company",))
            restored.load_state_dict(torch.load(path,weights_only=True))
            a,info=restored(self.x,self.price,**self.news,**self.masks,return_details=True)
            self.assertTrue(torch.equal(a,wrapper(self.x,self.price,**self.news,**self.masks)))
            self.assertTrue(torch.equal(info["deltas"]["industry"],torch.zeros(12,1)))

    def test_real_market_arrays_with_unchanged_baselines(self):
        project=BASELINE_SRC.parent
        for dataset,count,market in (("NASDAQ",1026,20),("NYSE",1737,25),("SP500",474,8)):
            with self.subTest(dataset=dataset):
                if dataset=="SP500":
                    eod=np.load(project/"dataset/SP500/SP500.npy",mmap_mode='r')[:,915:]
                    prices=eod[:,:,-1]
                    mask=np.ones((count,1),np.float32)
                    gt=(prices[:,16]/prices[:,15]-1)[:,None]
                else:
                    arrays=[]
                    for key in ("eod_data","price_data","mask_data","gt_data"):
                        with (project/"dataset"/dataset/(key+".pkl")).open('rb') as f:
                            arrays.append(pickle.load(f))
                    eod,prices,all_masks,all_gt=arrays
                    mask=all_masks[:,:17].min(1)[:,None]
                    gt=all_gt[:,16,None]
                x=torch.tensor(np.array(eod[:,:16]),dtype=torch.float32)
                price=torch.tensor(prices[:,15,None],dtype=torch.float32)
                base=baseline.StockMixer(count,16,5,market,3)
                wrapped=StockMixerWithNews(base)
                prediction=wrapped(x,price)
                self.assertTrue(torch.equal(prediction,base(x)))
                loss,*_=baseline.get_loss(prediction,torch.tensor(gt,dtype=torch.float32),price,
                    torch.tensor(mask,dtype=torch.float32),count,.1)
                self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_fusion_forward_backward(self):
        wrapped=self.wrapper.cuda()
        prediction=wrapped(self.x.cuda(),self.price.cuda(),
            **{k:v.cuda() for k,v in self.news.items()},**{k:v.cuda() for k,v in self.masks.items()})
        loss,*_=baseline.get_loss(prediction,torch.randn(12,1,device='cuda')*.01,
            self.price.cuda(),torch.ones(12,1,device='cuda'),12,.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in wrapped.branches.parameters() if p.grad is not None))

if __name__=="__main__":
    unittest.main()
