"""Optional price-compatible wrapper; neither imports FinBERT nor alters StockMixer."""
import torch
from torch import nn

CHANNELS=("company","industry","market")

class NewsBranch(nn.Module):
    def __init__(self,news_dim,hidden_dim,dropout):
        super().__init__()
        self.adapter=nn.Sequential(nn.LayerNorm(news_dim),nn.Linear(news_dim,hidden_dim),
            nn.GELU(),nn.Dropout(dropout))
        self.gate=nn.Linear(hidden_dim+1,1)
        self.head=nn.Linear(hidden_dim,1)
        nn.init.constant_(self.gate.bias,-2.0)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self,features,mask,base_return):
        # Sanitize inactive slots before the Adapter: multiplying NaN by zero is insufficient.
        clean=torch.where(mask,features,torch.zeros_like(features))
        h=self.adapter(clean)
        gate=torch.sigmoid(self.gate(torch.cat([h,base_return],dim=-1)))
        return torch.where(mask,gate*self.head(h),torch.zeros_like(base_return))

class StockMixerWithNews(nn.Module):
    def __init__(self,stock_mixer,news_dim=40,hidden_dim=32,dropout=0.0,
                 freeze_backbone=True,enabled_channels=CHANNELS):
        super().__init__()
        if not set(enabled_channels)<=set(CHANNELS):
            raise ValueError("Unknown news channel.")
        if news_dim<1 or hidden_dim<1 or not 0<=dropout<1:
            raise ValueError("Invalid adapter dimensions/dropout.")
        self.stock_mixer=stock_mixer
        self.news_dim=news_dim
        self.enabled_channels=frozenset(enabled_channels)
        self.branches=nn.ModuleDict({k:NewsBranch(news_dim,hidden_dim,dropout) for k in CHANNELS})
        self.freeze_backbone=freeze_backbone
        if freeze_backbone:
            self.stock_mixer.requires_grad_(False)
            self.stock_mixer.eval()

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.stock_mixer.eval()
        return self

    def _prepare(self,features,mask,baseline,channel):
        n=baseline.shape[0]
        if mask is None:
            if features is not None:
                raise ValueError(channel+"_mask is required when features are supplied.")
            return torch.zeros(n,self.news_dim,device=baseline.device,dtype=baseline.dtype),torch.zeros_like(baseline,dtype=torch.bool)
        if not torch.is_tensor(mask) or mask.device!=baseline.device:
            raise ValueError(channel+" mask must be a tensor on the model device.")
        if mask.ndim==0:
            mask=mask.reshape(1,1)
        elif mask.ndim==1:
            mask=mask[:,None]
        mask_shapes=((n,1),(1,1)) if channel=="market" else ((n,1),)
        if mask.shape not in mask_shapes or not torch.isfinite(mask).all() or not ((mask==0)|(mask==1)).all():
            raise ValueError(channel+" mask must be binary [N,1] or [1,1].")
        mask=mask.bool().expand(n,1)
        if features is None:
            if mask.any():
                raise ValueError(channel+" features are missing for an active mask.")
            features=baseline.new_zeros(n,self.news_dim)
        if not torch.is_tensor(features) or features.device!=baseline.device:
            raise ValueError(channel+" features must be a tensor on the model device.")
        if features.ndim==1 and channel=="market":
            features=features[None]
        feature_shapes=((n,self.news_dim),(1,self.news_dim)) if channel=="market" else ((n,self.news_dim),)
        if features.shape not in feature_shapes:
            raise ValueError(channel+" features must have shape [N,news_dim] (market may broadcast).")
        features=features.to(dtype=baseline.dtype).expand(n,self.news_dim)
        if not torch.isfinite(features[mask[:,0]]).all():
            raise ValueError(channel+" active features must be finite.")
        return features,mask

    def forward(self,stock_x,base_price,company_news=None,industry_news=None,market_news=None,
                company_mask=None,industry_mask=None,market_mask=None,return_details=False):
        if base_price.shape!=(stock_x.shape[0],1) or base_price.device!=stock_x.device:
            raise ValueError("base_price must have shape [N,1] on the stock input device.")
        if not torch.isfinite(base_price).all() or (base_price<=0).any():
            raise ValueError("base_price must be finite and strictly positive.")
        base=self.stock_mixer(stock_x)
        if base.shape!=base_price.shape or not torch.isfinite(base).all():
            raise ValueError("Backbone must return finite prices of shape [N,1].")
        base_return=(base-base_price)/base_price
        deltas={}
        supplied={"company":(company_news,company_mask),"industry":(industry_news,industry_mask),
                  "market":(market_news,market_mask)}
        for channel in CHANNELS:
            if channel not in self.enabled_channels:
                deltas[channel]=torch.zeros_like(base)
                continue
            features,mask=self._prepare(*supplied[channel],base,channel)
            deltas[channel]=self.branches[channel](features,mask,base_return)
        total=sum(deltas.values())
        # Algebraically equivalent to P_t*(1+r_base+delta), without an unnecessary roundtrip.
        prediction=base+base_price*total
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite fusion output.")
        if return_details:
            return prediction,{"baseline_price":base,"baseline_return":base_return,"deltas":deltas}
        return prediction
