"""Two public configuration files; compile to the pinned runtime/checkpoint schema."""
import copy
import json
import re
from pathlib import Path
import yaml
from .validation import integer,finite_number


def mapping(value,fields,label):
    if not isinstance(value,dict) or set(value)!=set(fields):
        raise ValueError('Missing or unknown field in '+label)


def nonempty(value,label,nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value,str) or not value.strip():
        raise ValueError(label+' must be a nonempty string'+(' or null' if nullable else ''))


def boolean(value,label):
    if not isinstance(value,bool):
        raise ValueError(label+' must be boolean')


def read_yaml(path):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value,dict):
        raise ValueError('Configuration must be a mapping: '+str(path))
    return value


REVIEW_DEFAULTS={'provider':'ollama','endpoint':'http://127.0.0.1:11434','model':'qwen3:8b','expected_model_digest':None,
    'temperature':0.0,'seed':1234,'think':False,'num_ctx':8192,'num_predict':1024,'timeout_seconds':120,
    'max_attempts':2,'min_confidence':0.85,'max_input_chars':20000,'checkpoint_every':25,
    'max_consecutive_failures':3,'cache_dir':'outputs/artifacts/news/review_cache'}


def validate_review_settings(settings):
    mapping(settings,REVIEW_DEFAULTS,'review')
    if settings['provider']!='ollama':raise ValueError('review.provider must be ollama')
    for key in ('endpoint','model','cache_dir'):nonempty(settings[key],'review.'+key)
    value=settings['expected_model_digest']
    if value is not None and (not isinstance(value,str) or re.fullmatch(r'(?:sha256:)?[a-f0-9]{64}',value) is None):
        raise ValueError('review.expected_model_digest must be null or an exact SHA256')
    boolean(settings['think'],'review.think')
    for key,lo,hi in (('seed',0,2**32-1),('num_ctx',1024,65536),('num_predict',128,4096),('max_attempts',1,5),
                      ('max_input_chars',128,100000),('checkpoint_every',1,1000),('max_consecutive_failures',1,50)):
        integer(settings[key],'review.'+key,lo,hi)
    for key,lo,hi in (('temperature',0,2),('min_confidence',0,1),('timeout_seconds',1,600)):
        value=finite_number(settings[key],'review.'+key)
        if not lo<=value<=hi:raise ValueError('review.'+key+' is outside its supported range')


def load_llm_config(path):
    c=read_yaml(path)
    from .pipeline import DEFAULTS, validate_settings
    c.setdefault('pipeline',copy.deepcopy(DEFAULTS))
    c.setdefault('market_routes',{name:None for name in ('NASDAQ','NYSE','SP500')})
    c.setdefault('review',copy.deepcopy(REVIEW_DEFAULTS))
    c.setdefault('historical_master',None)
    c.setdefault('industry_dictionary',None)
    if c['industry_dictionary'] is not None:
        mapping(c['industry_dictionary'],('directory','reference_archive','output'),'industry_dictionary')
        for k,v in c['industry_dictionary'].items():nonempty(v,'industry_dictionary.'+k)
    if c['historical_master'] is not None:
        from .master_data import validate_settings as validate_historical_settings
        validate_historical_settings(c['historical_master'])
    c['pipeline'].setdefault('industry_excluded_sources',[])
    c['pipeline'].setdefault('retrace_artifact',None)
    c['pipeline'].setdefault('industry_membership_mode','verified_historical')
    c['pipeline'].setdefault('analysis_source','upstream_only')
    c['pipeline'].setdefault('upstream_source_root','D:/financial_data_project2')
    c['pipeline'].setdefault('semantic_review',None)
    c['pipeline'].setdefault('industry_keep_alive','10m')
    c['pipeline'].setdefault('industry_scope_review',None)
    c['pipeline'].setdefault('require_industry_scope_review',False)
    c['pipeline'].setdefault('industry_inference_artifact',None)
    for key in ('industry_batch_artifact','industry_batch_base_snapshot','industry_batch_snapshot_output','industry_batch_legacy_snapshot'):
        c['pipeline'].setdefault(key,None)
    c['pipeline'].setdefault('allow_automated_industry_scope',False)
    # market_channel 为可选段：缺省时不加入配置，避免改变已有流水线状态的配置指纹。
    mapping({k:v for k,v in c.items() if k!='market_channel'},('industry_dictionary','historical_master','schema_version','news_root','adapted_snapshot','baseline_checkpoint','panels','market_routes','review','finbert','panel','fusion','training','pipeline'),'llm config')
    if c.get('market_channel') is not None:
        from .market_quality import validate_settings as validate_market_channel
        validate_market_channel(c['market_channel'])
    validate_settings(c['pipeline'])
    validate_review_settings(c['review'])
    if c['schema_version']!='stockmixer_llm_config_v1':
        raise ValueError('Use llm.yaml schema stockmixer_llm_config_v1')
    for key in ('news_root','adapted_snapshot'):
        nonempty(c[key],key)
    nonempty(c['baseline_checkpoint'],'baseline_checkpoint',True)
    mapping(c['panels'],('NASDAQ','NYSE','SP500'),'panels')
    for key,value in c['panels'].items():
        nonempty(value,'panels.'+key,True)
    mapping(c['market_routes'],('NASDAQ','NYSE','SP500'),'market_routes')
    for name,value in c['market_routes'].items():
        nonempty(value,'market_routes.'+name,True)
    f=c['finbert']
    mapping(f,('model_id','model_dir','local_files_only','max_length','batch_size','pooling','supported_languages','device','output_format'),'finbert')
    if f['model_id']!='ProsusAI/finbert' or f['local_files_only'] is not True or f['pooling']!='cls_last_hidden':
        raise ValueError('FinBERT must use the pinned local-only model and cls_last_hidden pooling')
    nonempty(f['model_dir'],'finbert.model_dir',True)
    integer(f['max_length'],'max_length',8,512);integer(f['batch_size'],'batch_size',1)
    if f['supported_languages']!=['en'] or f['output_format'] not in ('jsonl','parquet'):
        raise ValueError('Unsupported FinBERT language/output format')
    nonempty(f['device'],'finbert.device')
    if re.fullmatch(r'cpu|cuda(?::\d+)?',f['device']) is None:
        raise ValueError('finbert.device must be cpu, cuda or cuda:<index>')
    p=c['panel']
    p.setdefault('projection_components',32)
    p.setdefault('relation_time_basis','decision_and_event')
    if p['relation_time_basis'] not in ('decision_and_event','event_date'):
        raise ValueError('Unknown panel relation_time_basis')
    mapping(p,('relation_time_basis','decision_lookback','mode','projection','projection_seed','projection_components','lookback_sessions','decay_tau_sessions','simulated_latency_seconds'),'panel')
    components=integer(p['projection_components'],'projection_components',1,768)
    if p['decision_lookback']!=16 or isinstance(p['decision_lookback'],bool) or p['mode'] not in ('retrospective','simulated_pit','observed_pit') or p['projection'] not in ('pca','random_projection'):
        raise ValueError('Unsupported panel window/time/projection')
    integer(p['projection_seed'],'projection_seed',0,2**32-1)
    integer(p['lookback_sessions'],'lookback_sessions',1)
    finite_number(p['decay_tau_sessions'],'decay_tau_sessions',strictly_positive=True)
    finite_number(p['simulated_latency_seconds'],'simulated_latency_seconds',minimum=0)
    mapping(c['fusion'],('enabled_channels','news_dim','hidden_dim','dropout','freeze_backbone'),'fusion')
    fusion=c['fusion']
    channels=fusion['enabled_channels']
    if not isinstance(channels,list) or not channels or not all(isinstance(v,str) for v in channels) or len(set(channels))!=len(channels) or not set(channels)<={'company','industry','market'}:
        raise ValueError('fusion.enabled_channels must be nonempty, unique known channels')
    if fusion['freeze_backbone'] is not True or integer(fusion['news_dim'],'news_dim',9,776)!=components+8:
        raise ValueError('Fusion requires a frozen backbone and news_dim=projection_components+8')
    integer(fusion['hidden_dim'],'hidden_dim',1)
    if not 0<=finite_number(fusion['dropout'],'dropout',minimum=0)<1:
        raise ValueError('Fusion dropout must be below 1')
    mapping(c['training'],('epochs','learning_rate'),'llm training')
    integer(c['training']['epochs'],'llm epochs',1)
    finite_number(c['training']['learning_rate'],'llm learning_rate',strictly_positive=True)
    return c


def configuration_sources(path):
    path=Path(path).resolve()
    c=read_yaml(path)
    result=[path]
    if c.get('schema_version')=='stockmixer_config_v2' and c['llm']['enabled'] is True:
        llm=Path(c['llm']['config'])
        result.append(llm.resolve() if llm.is_absolute() else (path.parent/llm).resolve())
    return result


def compile_config(project,path,raw,dataset=None):
    from .protocol import validate_config
    root=Path(project).resolve();path=Path(path).resolve()
    c=copy.deepcopy(raw)
    mapping(c,('schema_version','market_name','llm','lookback_length','steps','fea_num','scale_factor','activation',
               'datasets','data','training','output','prediction'),'stockmixer config')
    if c['schema_version']!='stockmixer_config_v2':
        raise ValueError('Unknown StockMixer public config schema')
    market=dataset or c['market_name']
    if market not in ('NASDAQ','NYSE','SP500'):
        raise ValueError('Unknown market_name')
    mapping(c['llm'],('enabled','config'),'llm switch')
    boolean(c['llm']['enabled'],'llm.enabled');nonempty(c['llm']['config'],'llm.config')
    if integer(c['lookback_length'],'lookback_length',1)!=16 or integer(c['steps'],'steps',1)!=1 or integer(c['fea_num'],'fea_num',1)!=5 or c['activation']!='GELU':
        raise ValueError('Pinned original StockMixer supports lookback=16, steps=1, fea_num=5 and recorded activation=GELU')
    integer(c['scale_factor'],'scale_factor',1)
    mapping(c['datasets'],('NASDAQ','NYSE','SP500'),'datasets')
    for name,profile in c['datasets'].items():
        mapping(profile,('stock_num','market_num','valid_index','test_index'),'datasets.'+name)
        for key,value in profile.items():
            integer(value,key,1)
    d=c['data'];mapping(d,('protocol','dataset_root','axis_dir','identity_mapping','processed_csv_root','prepared_root','prepare_if_missing'),'data')
    if d['protocol'] not in ('original','strict'):
        raise ValueError('data.protocol must be original or strict')
    for key in ('dataset_root','axis_dir','identity_mapping','processed_csv_root'):
        nonempty(d[key],'data.'+key)
    nonempty(d['prepared_root'],'data.prepared_root',True)
    boolean(d['prepare_if_missing'],'prepare_if_missing')
    axis_path=Path(d['axis_dir'])
    axis_path=(axis_path if axis_path.is_absolute() else root/axis_path)/(market+'.json')
    axis=json.loads(axis_path.read_text(encoding='utf-8'))
    profile=c['datasets'][market]
    expected={'stock_num':len(axis['stock_uids']),'market_num':axis['market_num'],'valid_index':axis['train_end_idx'],'test_index':axis['valid_end_idx']}
    if profile!=expected:
        raise ValueError('Market parameters differ from verified stock/date axis: '+str(expected))
    t=c['training'];mapping(t,('epochs','learning_rate','alpha','numpy_seed','torch_seed','device'),'training')
    mapping(c['output'],('directory','smoke'),'output')
    nonempty(c['output']['directory'],'output.directory',True);boolean(c['output']['smoke'],'output.smoke')
    mapping(c['prediction'],('enabled','as_of','allow_in_sample_replay'),'prediction')
    enabled=c['llm']['enabled']
    # Baseline is independent of the LLM file and of optional NLP dependencies.
    llm=None
    if enabled:
        llm_path=Path(c['llm']['config'])
        llm_path=llm_path if llm_path.is_absolute() else path.parent/llm_path
        llm=load_llm_config(llm_path)
        if llm['baseline_checkpoint'] is None or llm['panels'][market] is None:
            from .pipeline import prepared_inputs,checked_state,baseline_signature
            try:
                artifacts=prepared_inputs(root,llm,market)
                if checked_state(root,llm).get('baseline_parameters_sha256')!=baseline_signature(c):
                    raise ValueError('StockMixer parameters differ from prepared strict E0; rebuild preparation/baseline')
            except (ValueError,OSError) as error:
                raise ValueError('LLM fusion is enabled: configure baseline_checkpoint and panels.'+market+
                    ' or complete main.py --step prepare. '+str(error)) from error
            if llm['baseline_checkpoint'] is None:
                llm['baseline_checkpoint']=artifacts['baseline_checkpoint']
            if llm['panels'][market] is None:
                llm['panels'][market]=artifacts['panel']
        # Original reproduction remains intact. News E0/fusion use the same isolated strict protocol.
        d=copy.deepcopy(d)
        d['protocol']=llm['pipeline']['data_protocol']
        d['prepared_root']=None
        panel_path=Path(llm['panels'][market])
        panel_path=panel_path if panel_path.is_absolute() else root/panel_path
        if panel_path.exists():
            from .dataset import NewsPanel
            from .time_axis import Axis
            from .pipeline import validate_panel_settings
            panel=NewsPanel(panel_path)
            validate_panel_settings(panel,llm,Axis.load(axis_path))
            del panel
    fusion=copy.deepcopy(llm['fusion']) if llm else {'enabled_channels':['market'],'news_dim':40,'hidden_dim':32,'dropout':0.,'freeze_backbone':True}
    fusion['time_mode']=llm['panel']['mode'] if llm else 'retrospective'
    training=dict(t,selection_metric='val_loss',test_policy='per_epoch',metric_protocol='stockmixer_original')
    if llm:
        training.update(llm['training'])
    prepared=d['prepared_root'] or ('outputs/artifacts/market_original' if d['protocol']=='original' else 'outputs/artifacts/market')
    result={'schema_version':'stockmixer_strict_config_v1','dataset':market,
        'data':{'dataset_root':d['dataset_root'],'axis_dir':d['axis_dir'],'identity_mapping':d['identity_mapping'],
                'processed_csv_root':d['processed_csv_root'],'lookback':c['lookback_length'],'steps':c['steps'],
                'preprocessing':'original' if d['protocol']=='original' else 'train_max_ratio'},
        'model':{'backbone':'released_legacy','scale_argument':c['scale_factor']},'training':training,'fusion':fusion,
        'news':{'source_root':llm['news_root'] if llm else '../data_llm',
                'adapted_snapshot':llm['adapted_snapshot'] if llm else 'outputs/artifacts/news/adapted/20261113_current_v1',
                'panel':llm['panels'][market] if llm else None},
        'workflow':{'mode':'fusion' if enabled else 'baseline','prepared_data':{name:str(Path(prepared)/name) for name in c['datasets']},
            'prepare_if_missing':d['prepare_if_missing'],'output_directory':c['output']['directory'],'smoke':c['output']['smoke'],
            'baseline_checkpoint':llm['baseline_checkpoint'] if llm else None,'prediction':copy.deepcopy(c['prediction'])}}
    validate_config(result)
    return result
