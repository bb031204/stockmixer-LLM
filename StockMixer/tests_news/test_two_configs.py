import ast
import contextlib
import copy
import csv
import importlib.util
import io
import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from news_fusion import experiment as e
from news_fusion.configuration import configuration_sources,load_llm_config
from news_fusion.original import OriginalMarketData,OriginalSplit
from news_fusion.protocol import PROJECT,SCHEMA,backbone_contract,configuration,load_market
from news_fusion.io import write_json,file_manifest
from tests_news.test_workflow import temporary_project


def original_fixture(root):
    root.mkdir(parents=True)
    n,t,train,valid=12,50,20,38
    dates=pd.bdate_range('2020-01-02',periods=t).strftime('%Y-%m-%d').tolist()
    axis={'dataset':'NYSE','stock_uids':['NYSE:S'+str(i) for i in range(n)],'trading_dates':dates,
        'train_end_idx':train,'valid_end_idx':valid,'market_num':4,'source_sha256':{}}
    write_json(root/'axis.json',axis)
    price=.5+np.arange(n)[:,None]*.01+np.arange(t)[None,:]*.001
    arrays={'eod':np.repeat(price[:,:,None],5,axis=2),'price':price,'mask':np.ones((n,t)), 'returns':np.zeros((n,t))}
    arrays['returns'][:,1:]=(price[:,1:]-price[:,:-1])/price[:,:-1]
    for name,value in arrays.items():
        np.save(root/(name+'.npy'),value,allow_pickle=False)
    write_json(root/'source_hashes.json',{})
    with (root/'stock_identity.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','array_index','stock_uid','ticker']);writer.writeheader()
        writer.writerows({'dataset':'NYSE','array_index':i,'stock_uid':uid,'ticker':'S'+str(i)} for i,uid in enumerate(axis['stock_uids']))
    split=OriginalSplit(t,train,valid)
    write_json(root/'manifest.json',{'schema_version':SCHEMA,'kind':'original_market_data','preprocessing':{'mode':'original'},
        'dataset':'NYSE','source_dataset_root':str(root.parent/'dataset'),'stock_axis_hash':__import__('news_fusion.time_axis',fromlist=['Axis']).Axis.load(root/'axis.json').stock_hash,
        'date_axis_hash':__import__('news_fusion.time_axis',fromlist=['Axis']).Axis.load(root/'axis.json').date_hash,
        'backbone':backbone_contract(),'lookback':16,'steps':1,'split':split.describe(dates),'shape':[n,t,5],
        'train_end_idx':train,'valid_end_idx':valid,'files':file_manifest(root)})
    return root


def public_config():
    config=yaml.safe_load((PROJECT/'configs/stockmixer.yaml').read_text(encoding='utf-8-sig'))
    config['market_name']='NYSE'
    config['llm']['enabled']=False
    return config


def baseline_configuration(dataset=None):
    # Baseline tests must not depend on the user's current fusion switch.
    from news_fusion.configuration import compile_config
    return PROJECT,compile_config(PROJECT,PROJECT/'configs/stockmixer.yaml',public_config(),dataset)


class TwoConfigTests(unittest.TestCase):
    def test_disabled_switch_never_reads_missing_llm_file(self):
        with temporary_project() as tmp:
            path=Path(tmp)/'stockmixer.yaml';c=public_config();c['llm']['config']='absent.yaml'
            path.write_text(yaml.safe_dump(c),encoding='utf-8')
            _,resolved=configuration(path,project=PROJECT)
            self.assertEqual(resolved['workflow']['mode'],'baseline')
            self.assertEqual(resolved['data']['preprocessing'],'original')
            self.assertEqual(configuration_sources(path),[path])

    def test_enabled_switch_uses_only_llm_module_parameters_and_records_both_sources(self):
        with temporary_project() as tmp:
            path=Path(tmp)/'stockmixer.yaml';c=public_config();c['llm']['enabled']=True
            llm=yaml.safe_load((PROJECT/'configs/llm.yaml').read_text(encoding='utf-8-sig'))
            llm['baseline_checkpoint']='outputs/baseline/checkpoints/best_model.pt';llm['panels']['NYSE']='outputs/panel'
            llm['training']={'epochs':3,'learning_rate':.0003};llm['fusion']['hidden_dim']=64
            (path.parent/'llm.yaml').write_text(yaml.safe_dump(llm),encoding='utf-8')
            path.write_text(yaml.safe_dump(c),encoding='utf-8')
            _,resolved=configuration(path,project=PROJECT)
            self.assertEqual(resolved['workflow']['mode'],'fusion')
            self.assertEqual(resolved['training']['epochs'],3)
            self.assertEqual(resolved['fusion']['hidden_dim'],64)
            self.assertEqual(resolved['news']['panel'],'outputs/panel')
            self.assertEqual(len(configuration_sources(path)),2)

    def test_enabled_switch_rejects_unprepared_real_inputs(self):
        with temporary_project() as tmp:
            path=Path(tmp)/'stockmixer.yaml';c=public_config();c['llm']['enabled']=True
            (path.parent/'llm.yaml').write_bytes((PROJECT/'configs/llm.yaml').read_bytes())
            path.write_text(yaml.safe_dump(c),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'baseline_checkpoint and panels.NYSE'):
                configuration(path,project=PROJECT)

    def test_public_typo_or_wrong_market_shape_is_rejected(self):
        for field in ('typo','shape'):
            with self.subTest(field=field),temporary_project() as tmp:
                c=public_config()
                if field=='typo':c['alpah']=.1
                else:c['datasets']['NYSE']['stock_num']=1
                path=Path(tmp)/'stockmixer.yaml';path.write_text(yaml.safe_dump(c),encoding='utf-8')
                with self.assertRaises(ValueError):configuration(path,project=PROJECT)

    def test_llm_parameters_are_validated_even_in_offline_commands(self):
        for value in (["market","market"],[],['unknown']):
            with temporary_project() as tmp:
                llm=yaml.safe_load((PROJECT/'configs/llm.yaml').read_text(encoding='utf-8-sig'))
                llm['fusion']['enabled_channels']=value
                path=Path(tmp)/'llm.yaml';path.write_text(yaml.safe_dump(llm),encoding='utf-8')
                with self.assertRaises(ValueError):load_llm_config(path)

    def test_dataset_switch_uses_corresponding_original_artifact(self):
        for name in ('NASDAQ','NYSE','SP500'):
            _,c=baseline_configuration(dataset=name)
            self.assertEqual(c['dataset'],name)
            self.assertEqual(Path(c['workflow']['prepared_data'][name]).name,name)
            self.assertIn('market_original',c['workflow']['prepared_data'][name])


class OriginalFunctionTests(unittest.TestCase):
    def test_original_get_batch_matches_unmodified_official_function(self):
        source=ast.parse(Path('D:/finance/baseline/StockMixer-master/src/train.py').read_text(encoding='utf-8-sig'))
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='get_batch')
        with temporary_project() as tmp:
            data=OriginalMarketData(original_fixture(Path(tmp)/'data'))
            namespace={'np':np,'random':random,'valid_index':data.axis.train_end,'lookback_length':16,'steps':1,
                'eod_data':np.array(data.arrays['eod']),'mask_data':np.array(data.arrays['mask']),'price_data':np.array(data.arrays['price']),'gt_data':np.array(data.arrays['returns'])}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'official_get_batch','exec'),namespace)
            for offset in (0,1,19):
                expected=namespace['get_batch'](offset);actual=data.get_batch(offset)
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(actual,expected)))
            self.assertLess(data.arrays['price'][:,:data.axis.train_end].max(),1.)
            del expected,actual,namespace,data

    def test_real_original_training_loop_matches_weights_best_metrics_and_sampling(self):
        source=ast.parse(Path('D:/finance/baseline/StockMixer-master/src/train.py').read_text(encoding='utf-8-sig'))
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('get_batch','validate')]
        loop=next(n for n in source.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
        with temporary_project() as tmp:
            root=Path(tmp);data_dir=original_fixture(root/'data');data=OriginalMarketData(data_dir)
            _,config=baseline_configuration(dataset="NYSE");config['training']['device']='cpu';config['training']['epochs']=2
            torch.manual_seed(config['training']['torch_seed']);np.random.seed(config['training']['numpy_seed'])
            model=e.make_model(data,config,torch.device('cpu'));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            _,get_loss=e.model_api()
            from news_fusion.original_evaluator import evaluate
            namespace={'random':random,'np':np,'torch':torch,'device':torch.device('cpu'),'stock_num':12,'lookback_length':16,'steps':1,
                'valid_index':20,'test_index':38,'trade_dates':50,'epochs':2,'alpha':.1,'model':model,'optimizer':optimizer,
                'get_loss':get_loss,'evaluate':evaluate,'best_valid_loss':np.inf,'best_valid_perf':None,'best_test_perf':None,
                'batch_offsets':np.arange(20,dtype=int),'eod_data':np.array(data.arrays['eod']),'mask_data':np.array(data.arrays['mask']),
                'price_data':np.array(data.arrays['price']),'gt_data':np.array(data.arrays['returns'])}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=functions+[loop],type_ignores=[]),'official_training_reference','exec'),namespace)
                result=e.train(root,config,data_dir,root/'run',epochs=2,smoke=True)
            checkpoint=e.load_run_checkpoint(root/'run/checkpoints/last_model.pt')
            self.assertTrue(all(torch.equal(value,checkpoint['model_state_dict'][key]) for key,value in model.state_dict().items()))
            for key,value in namespace['best_valid_perf'].items():
                self.assertEqual(result['validation'][key],value)
            for key,value in namespace['best_test_perf'].items():
                self.assertEqual(result['test'][key],value)
            records=json.loads((root/'run/tables/epoch_samples.json').read_text())
            rng=np.random.RandomState(config['training']['numpy_seed']);offsets=np.arange(20,dtype=int)
            for row in records:
                rng.shuffle(offsets)
                self.assertEqual(row['offsets'],offsets[:4].tolist())
            self.assertEqual(result['protocol'],'original_reproduction')
            del checkpoint,data,model,optimizer,namespace

    def test_original_and_strict_data_cannot_be_silently_mixed(self):
        with temporary_project() as tmp:
            root=Path(tmp);original_fixture(root/'data');_,config=baseline_configuration(dataset="NYSE")
            config['data']['preprocessing']='train_max_ratio'
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                e.train(root,config,root/'data',root/'bad',epochs=1,smoke=True)


if __name__=='__main__':
    unittest.main()
