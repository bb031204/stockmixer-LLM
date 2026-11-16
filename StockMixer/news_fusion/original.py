"""Official StockMixer arrays, get_batch and persistent candidate-window sampling."""
import csv
import json
import pickle
import random
from pathlib import Path
import numpy as np
from .io import digest_file,file_manifest,new_artifact,verify_files,write_json,now
from .protocol import SCHEMA,PROTECTED,MarketData,TargetSplit,backbone_contract,resolve,verify_axis_sources,identity_rows
from .time_axis import Axis
from .validation import integer


class OriginalSplit(TargetSplit):
    @property
    def samples_per_epoch(self):
        return self.train_end-self.lookback-self.steps+1

    def targets(self,phase):
        if phase=='train':
            return np.arange(self.lookback+self.steps-1,self.train_end+self.lookback+self.steps-1,dtype=np.int64)
        return super().targets(phase)

    def describe(self,dates):
        result=super().describe(dates)
        result['train'].update(candidate_offsets=[0,self.train_end-1],candidate_count=self.train_end,
            samples_per_epoch=self.samples_per_epoch,persistent_candidate_shuffle=True,
            validation_label_overlap_indices=[self.train_end,self.train_end+self.lookback+self.steps-2])
        return result


def prepare_original_market(project,config,output):
    axis_path=resolve(project,config['data']['axis_dir'])/(config['dataset']+'.json')
    axis=Axis.load(axis_path)
    root=resolve(project,config['data']['dataset_root'])
    verify_axis_sources(project,axis,root)
    identities,identity_path=identity_rows(project,config,axis)
    folder=root/axis.dataset
    if axis.dataset=='SP500':
        path=folder/'SP500.npy'
        data=np.load(path,allow_pickle=False)[:,915:,:]
        arrays={'eod':data,'price':data[:,:,-1], 'mask':np.ones(data.shape[:2]),'returns':np.zeros(data.shape[:2])}
        # Same scalar expression/order as the official per-stock/per-day loop.
        arrays['returns'][:,1:]=(data[:,1:,-1]-data[:,:-1,-1])/data[:,:-1,-1]
        sources={str(path):digest_file(path)}
        features=['open','high','low','volume','close']
    else:
        arrays={};sources={}
        for name,stem in (('eod','eod'),('price','price'),('mask','mask'),('returns','gt')):
            path=folder/(stem+'_data.pkl')
            with path.open('rb') as stream:
                arrays[name]=pickle.load(stream)
            sources[str(path)]=digest_file(path)
        features=['close_ma5','close_ma10','close_ma20','close_ma30','close']
    n,t=len(axis.stock_uids),len(axis.dates)
    if arrays['eod'].shape!=(n,t,5) or any(arrays[k].shape!=(n,t) for k in ('price','mask','returns')):
        raise ValueError('Original dataset dimensions differ from the verified axis')
    split=OriginalSplit(t,axis.train_end,axis.valid_end)
    if split.targets('train')[-1]>=axis.valid_end:
        raise ValueError('Original candidate windows would enter test or exceed data; unsupported split')
    verify_axis_sources(project,axis,root)
    for source,expected in sources.items():
        if digest_file(source)!=expected:
            raise ValueError('Input changed during original data preparation: '+source)
    with new_artifact(output) as directory:
        for name,array in arrays.items():
            if not isinstance(array,np.ndarray) or not np.isfinite(array).all():
                raise ValueError('Original arrays must be finite numpy arrays')
            np.save(directory/(name+'.npy'),array,allow_pickle=False)
        write_json(directory/'axis.json',axis.source)
        write_json(directory/'source_hashes.json',sources)
        with (directory/'stock_identity.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(identities[0]));writer.writeheader();writer.writerows(identities)
        manifest={'schema_version':SCHEMA,'kind':'original_market_data','created_at':now(),'dataset':axis.dataset,
            'stock_axis_hash':axis.stock_hash,'date_axis_hash':axis.date_hash,'shape':list(arrays['eod'].shape),
            'feature_names':features,'split':split.describe(axis.dates),'train_end_idx':axis.train_end,'valid_end_idx':axis.valid_end,
            'lookback':16,'steps':1,'source_dataset_root':str(root),'backbone':backbone_contract(),
            'axis_file_sha256':digest_file(axis_path),'identity_mapping_sha256':digest_file(identity_path),
            'preprocessing':{'mode':'original','normalization':'dataset as supplied; no extra train-max rebasing',
                'dtypes':{name:str(value.dtype) for name,value in arrays.items()}},
            'sampling':'shuffle persistent arange(valid_index), then take first valid_index-lookback-steps+1',
            'limitations':['Official candidate windows include validation labels; this is original reproduction, not leakage-free evaluation.',
                'Original normalization is retained and can use full-period information.'],
            'protected_source_hashes':{name:digest_file(Path(project)/name) for name in PROTECTED},'files':file_manifest(directory)}
        write_json(directory/'manifest.json',manifest)
    return manifest


class OriginalMarketData(MarketData):
    protocol='original_reproduction'

    def __init__(self,directory,project=None):
        self.directory=Path(directory).resolve()
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        m=self.manifest
        if m.get('schema_version')!=SCHEMA or m.get('kind')!='original_market_data' or m.get('preprocessing',{}).get('mode')!='original':
            raise ValueError('Expected original reproduction market artifact')
        verify_files(self.directory,m,required=('eod.npy','price.npy','returns.npy','mask.npy','axis.json','stock_identity.csv','source_hashes.json'))
        if m['backbone']!=backbone_contract():
            raise ValueError('Original artifact backbone mismatch')
        self.axis=Axis.load(self.directory/'axis.json')
        if self.axis.stock_hash!=m['stock_axis_hash'] or self.axis.date_hash!=m['date_axis_hash'] or self.axis.dataset!=m['dataset']:
            raise ValueError('Original market axes differ from manifest')
        self.split=OriginalSplit(len(self.axis.dates),self.axis.train_end,self.axis.valid_end,m['lookback'],m['steps'])
        if m['lookback']!=16 or m['steps']!=1 or m['split']!=self.split.describe(self.axis.dates) or self.split.targets('train')[-1]>=self.axis.valid_end:
            raise ValueError('Original sampling/window contract mismatch')
        self.arrays={name:np.load(self.directory/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('eod','price','mask','returns')}
        n,t=len(self.axis.stock_uids),len(self.axis.dates)
        for name,array in self.arrays.items():
            if array.shape!=((n,t,5) if name=='eod' else (n,t)) or not np.isfinite(array).all():
                raise ValueError('Invalid original array: '+name)
        if list(self.arrays['eod'].shape)!=m['shape'] or not np.isin(self.arrays['mask'],[0,1]).all() or (self.arrays['price']<=0).any():
            raise ValueError('Invalid original dimensions/masks/prices')
        if not np.array_equal(self.arrays['price'],self.arrays['eod'][:,:,-1]):
            raise ValueError('Original price is not the last feature')
        valid=self.arrays['mask'].astype(bool)
        pair=valid[:,1:] & valid[:,:-1]
        price=self.arrays['price'].astype(np.float64)
        expected=(price[:,1:]-price[:,:-1])/price[:,:-1]
        if np.any(np.abs(expected[pair]-self.arrays['returns'][:,1:][pair])>3e-6):
            raise ValueError('Original labels differ from valid close-to-close returns')
        with (self.directory/'stock_identity.csv').open(encoding='utf-8-sig',newline='') as stream:
            identities=list(csv.DictReader(stream))
        if [r['stock_uid'] for r in identities]!=self.axis.stock_uids or [r['array_index'] for r in identities]!=[str(i) for i in range(n)] or any(r['dataset']!=self.axis.dataset or not r['ticker'] for r in identities):
            raise ValueError('Original identity order mismatch')
        if project:
            verify_axis_sources(project,self.axis,m['source_dataset_root'])

    def get_batch(self,offset=None):
        if offset is None:
            offset=random.randrange(0,self.axis.train_end)
        integer(offset,'offset',0)
        seq_len=self.split.lookback;steps=self.split.steps
        if offset+seq_len+steps>len(self.axis.dates):
            raise IndexError('Original batch outside the data axis')
        mask_batch=np.min(self.arrays['mask'][:,offset:offset+seq_len+steps],axis=1)
        return (self.arrays['eod'][:,offset:offset+seq_len,:],np.expand_dims(mask_batch,axis=1),
            np.expand_dims(self.arrays['price'][:,offset+seq_len-1],axis=1),
            np.expand_dims(self.arrays['returns'][:,offset+seq_len+steps-1],axis=1))

    def window(self,target):
        integer(target,'target',self.split.lookback+self.split.steps-1)
        x,mask,price,truth=self.get_batch(int(target)-self.split.lookback-self.split.steps+1)
        return x,price,truth,mask
