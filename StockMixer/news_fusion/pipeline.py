"""Resumable offline news preparation and config-driven StockMixer entry point."""
import argparse
import copy
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
from .io import digest, digest_file, now, read_records, verify_files, write_json
from .outputs import destination
from .time_axis import Axis, Schedule, export_schedule

DEFAULTS = {
    'industry_excluded_sources': [],
    'retrace_artifact': None,
    'industry_membership_mode': 'verified_historical',
    'industry_batch_artifact': None, 'industry_batch_base_snapshot': None,
    'industry_batch_legacy_snapshot': None,
    'industry_batch_snapshot_output': None, 'allow_automated_industry_scope': False,
    'python': 'C:/Users/marti/anaconda3/envs/stockmixer/python.exe',
    'model_directory': 'C:/Users/marti/anaconda3/envs/stockmixer/models/ProsusAI-finbert/0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031',
    'model_revision': '0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031',
    'datasets': ['NASDAQ', 'NYSE', 'SP500'],
    'state': 'outputs/artifacts/news/pipeline_state.json',
    'data_protocol': 'strict', 'allow_partial_routes': True,
    'review_limit': None, 'encoding_limit': None,
    'cache': 'outputs/artifacts/news/cache', 'minimum_training_cells': 1,
    'security_master': None, 'industry_memberships': None, 'industry_aliases': None,
    'review_artifacts': {'NASDAQ': None, 'NYSE': None, 'SP500': None},
    'features': None, 'semantic_review': None,
    'finbert_threads': 4,
    'analysis_source': 'upstream_only',
    'upstream_source_root': 'D:/financial_data_project2',
    'infer_industries': False, 'industry_min_confidence': 0.85, 'industry_max_impacts': 3,
    'industry_cache': 'outputs/artifacts/news/industry_cache',
    'industry_keep_alive': '10m',
    'industry_scope_review': None, 'require_industry_scope_review': False,
    'industry_inference_artifact': None,
}
STEPS = ('setup', 'review', 'publish', 'encode', 'industry_mapping', 'relations', 'schedule', 'panels', 'validate', 'baseline', 'prepare', 'train')


@contextmanager
def state_lock(path):
    path=Path(path).with_suffix('.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell()==0:stream.write(b'1');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Another news pipeline stage is running; finish it before starting the next step') from error
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def validate_settings(settings):
    from .configuration import mapping, nonempty, boolean
    from .validation import integer
    optional={'sample_review','auto_import_retrace','launch_state','baseline_checkpoints'}
    mapping({k:v for k,v in settings.items() if k not in optional}, DEFAULTS, 'pipeline')
    nonempty(settings.get('sample_review'),'pipeline.sample_review',True)
    boolean(settings.get('auto_import_retrace',True),'pipeline.auto_import_retrace')
    nonempty(settings.get('launch_state','outputs/artifacts/news/launch_state.json'),'pipeline.launch_state')
    if 'baseline_checkpoints' in settings:
        mapping(settings['baseline_checkpoints'],('NASDAQ','NYSE','SP500'),'pipeline.baseline_checkpoints')
        for value in settings['baseline_checkpoints'].values():
            nonempty(value,'pipeline.baseline_checkpoints',True)
    for key in ('python', 'model_directory', 'model_revision', 'state', 'cache','industry_cache'):
        nonempty(settings[key], 'pipeline.' + key)
    for key in ('security_master', 'industry_memberships', 'industry_aliases', 'features','semantic_review','industry_scope_review','industry_inference_artifact','upstream_source_root'):
        nonempty(settings[key], 'pipeline.' + key, True)
    if settings['data_protocol'] not in ('strict', 'original'):
        raise ValueError('pipeline.data_protocol must be strict or original; original is a label-overlap reproduction experiment')
    names = settings['datasets']
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or not set(names) <= {'NASDAQ', 'NYSE', 'SP500'}:
        raise ValueError('pipeline.datasets must be unique supported datasets')
    mapping(settings['review_artifacts'], ('NASDAQ', 'NYSE', 'SP500'), 'pipeline.review_artifacts')
    for value in settings['review_artifacts'].values():
        nonempty(value, 'review_artifacts', True)
    boolean(settings['allow_partial_routes'], 'allow_partial_routes')
    boolean(settings['infer_industries'],'infer_industries')
    if settings['analysis_source'] not in ('upstream_only', 'downstream_legacy'):
        raise ValueError('Unknown analysis_source')
    if settings['analysis_source']=='upstream_only' and settings['infer_industries']:
        raise ValueError('upstream_only forbids downstream Qwen industry inference')
    boolean(settings['require_industry_scope_review'],'require_industry_scope_review')
    if settings.get('industry_membership_mode','verified_historical') not in ('verified_historical','reference_retrospective'):
        raise ValueError('Unknown industry_membership_mode')
    ids=settings.get('industry_excluded_sources',[])
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or any(not isinstance(s,str) or not s for s in ids):
        raise ValueError('industry_excluded_sources must contain unique source IDs')
    nonempty(settings.get('retrace_artifact'),'retrace_artifact',True)
    boolean(settings['allow_automated_industry_scope'],'allow_automated_industry_scope')
    for key in ('industry_batch_artifact','industry_batch_base_snapshot','industry_batch_snapshot_output','industry_batch_legacy_snapshot'):
        nonempty(settings[key],key,True)
    import re
    if not isinstance(settings['industry_keep_alive'],str) or re.fullmatch(r'[1-9]\d*[smh]',settings['industry_keep_alive']) is None:
        raise ValueError('industry_keep_alive must be a positive duration such as 10m')
    from .validation import finite_number
    confidence=finite_number(settings['industry_min_confidence'],'industry_min_confidence',minimum=0)
    if confidence>1:raise ValueError('industry_min_confidence must be <= 1')
    integer(settings['industry_max_impacts'],'industry_max_impacts',1,10)
    integer(settings['minimum_training_cells'], 'minimum_training_cells', 1)
    integer(settings['finbert_threads'], 'finbert_threads', 1, 64)
    for key in ('review_limit', 'encoding_limit'):
        if settings[key] is not None:
            integer(settings[key], key, 1)


def settings_fingerprint(config):
    return digest(config)


def baseline_signature(base):
    fields=('lookback_length','steps','fea_num','scale_factor','activation','datasets','training')
    values={key:base[key] for key in fields}
    values['data']={k:v for k,v in base['data'].items() if k not in ('protocol','prepared_root','prepare_if_missing')}
    return digest(values)


def validate_news_protocol(config):
    # Original candidate sampling intentionally overlaps validation labels. It is
    # supported only as explicitly labelled retrospective reproduction.
    if config.get('pipeline', {}).get('data_protocol', 'strict') == 'original' and config['panel']['mode'] != 'retrospective':
        raise ValueError('Original reproduction requires panel.mode: retrospective; it is not a leakage-free PIT experiment')


def validate_baseline_source(parent, data, expected):
    from .experiment import validate_checkpoint
    validate_checkpoint(parent, data)
    if parent['mode'] != 'baseline' or parent['smoke_only']:
        raise ValueError('News fusion requires a completed non-smoke E0')
    if parent['config']['data']['preprocessing'] != expected['data']['preprocessing']:
        raise ValueError('E0 preprocessing must match the configured original/strict protocol')
    for key in ('epochs', 'learning_rate', 'alpha', 'numpy_seed', 'torch_seed', 'selection_metric', 'test_policy', 'metric_protocol'):
        if parent['config']['training'].get(key) != expected['training'].get(key):
            raise ValueError('E0 training parameter differs from current config: ' + key)


def checked_state(project, config):
    path = destination(project, config['pipeline']['state'])
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('settings_sha256') != settings_fingerprint(config):
        raise ValueError('News config changed after preparation. Run main.py with --new, or restore the prepared config.')
    snapshot = Path(config['adapted_snapshot'])
    snapshot = snapshot if snapshot.is_absolute() else Path(project) / snapshot
    if digest_file(snapshot / 'manifest.json') != state.get('snapshot_manifest_sha256'):
        raise ValueError('Prepared news snapshot has changed')
    for path, expected in state.get('mapping_input_sha256', {}).items():
        if digest_file(path) != expected:
            raise ValueError('Historical mapping changed; rebuild relations/panels')
    for path,expected in state.get('accepted_manifests',{}).items():
        if digest_file(Path(path)/'manifest.json')!=expected:
            raise ValueError('Accepted artifact manifest changed: '+path+'; rebuild a new preparation version')
        artifact=json.loads((Path(path)/'manifest.json').read_text(encoding='utf-8'))
        if artifact.get('kind')=='sec_sic_dated_historical_observations':
            from .master_data import verify_observations
            verify_observations(path)
    return state


def prepared_inputs(project, config, dataset):
    """Resolve only fully validated artifacts; used by the public training config."""
    state = checked_state(project, config)
    if state.get('steps', {}).get('prepare') != 'completed':
        raise ValueError('Run main.py --step prepare after completing offline preprocessing and the strict baseline')
    if dataset not in state.get('artifacts',{}):
        raise ValueError('Dataset was not prepared: '+dataset+'; include it in pipeline.datasets and rebuild')
    artifacts = state['artifacts'][dataset]
    for key in ('panel', 'baseline_checkpoint'):
        if not artifacts.get(key):
            raise ValueError('Missing prepared ' + key + ' for ' + dataset)
    verify_artifact(Path(artifacts['panel']))
    from .dataset import NewsPanel
    panel=NewsPanel(artifacts['panel'])
    from .experiment import load_run_checkpoint
    from .protocol import load_market
    parent=load_run_checkpoint(artifacts['baseline_checkpoint'])
    data=load_market(parent['data_directory'],project)
    report=readiness(panel,config,data,state.get('features'));del panel,data
    if not report['ready']:raise ValueError('Prepared panel no longer passes enabled-channel readiness')
    return artifacts


def verify_artifact(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    verify_files(path, manifest)
    return manifest


def validate_feature_settings(manifest,config,installed):
    encoder=manifest.get('encoder',{})
    expected={'model_id':config['finbert']['model_id'],'max_length':config['finbert']['max_length'],
              'embedding_method':config['finbert']['pooling']}
    if any(encoder.get(k)!=v for k,v in expected.items()):
        raise ValueError('Cached FinBERT features differ from configured model/length/pooling; re-encode a new artifact')
    if installed.get('revision')!=config['pipeline']['model_revision']:
        raise ValueError('Installed model revision differs from pipeline.model_revision')
    hashes=encoder.get('local_file_sha256',{})
    for name in ('config.json','model.safetensors','tokenizer_config.json','special_tokens_map.json','vocab.txt'):
        if not installed.get('files',{}).get(name) or hashes.get(name)!=installed['files'][name]:
            raise ValueError('Cached FinBERT weight/tokenizer differs from the configured pinned installation: '+name)


def readiness(panel, config, data=None, feature_dir=None):
    axis = panel.axis
    panel.assert_experiment(axis, config['panel']['mode'], config['panel']['decision_lookback'])
    validate_panel_settings(panel,config)
    splits = {'train': (15, axis.train_end - 1), 'valid': (axis.train_end - 1, axis.valid_end - 1),
              'test': (axis.valid_end - 1, len(axis.dates) - 1)}
    report = {'dataset': axis.dataset, 'mode': panel.manifest['mode'],
              'historical_model_replay': panel.manifest['historical_model_replay'],
              'projection': panel.manifest['transform'], 'coverage': {}, 'blockers': [],
              'industry_membership_policy':panel.manifest.get('industry_membership_policy','verified_historical'),
              'historical_industry_membership_verified':panel.manifest.get('historical_industry_membership_verified',True)}
    for channel in ('company', 'industry', 'market'):
        report['coverage'][channel] = {}
        for split, (start, end) in splits.items():
            masks = panel.masks[channel][start:end] if panel.masks[channel].ndim == 1 else panel.masks[channel][:, start:end]
            report['coverage'][channel][split] = {'active_cells': int(masks.sum()), 'total_cells': int(masks.size),
                'active_days':int(np.count_nonzero(masks if masks.ndim==1 else np.any(masks,axis=0))),
                'decision_days':end-start}
        if channel in config['fusion']['enabled_channels'] and report['coverage'][channel]['train']['active_cells'] < config['pipeline']['minimum_training_cells']:
            report['blockers'].append('Enabled channel has insufficient verified training news: ' + channel)
    if 'market' in config['fusion']['enabled_channels']:
        panel.assert_market_routing()
        if config.get('market_channel') is not None:
            # M6：门槛在配置中事先写定；未设置或未达标都阻断正式融合。
            from .market_quality import gate_blockers
            quality=panel.manifest.get('market_quality')
            report['market_quality']=quality
            report['blockers'].extend(gate_blockers(quality,config['market_channel']['gates']))
    report['data_protocol'] = config['pipeline']['data_protocol']
    report['validation_label_overlap_possible'] = config['pipeline']['data_protocol'] == 'original'
    report['final_inputs_verified']=False
    if data is not None:
        from .final_acceptance import audit_panel
        final=audit_panel(panel,data,config,feature_dir)
        report['final_inputs']=final
        report['final_inputs_verified']=True
        report['blockers'].extend(final['blockers'])
    report['ready'] = not report['blockers']
    return report


def validate_panel_settings(panel,config,axis=None):
    validate_news_protocol(config)
    if axis is not None:
        panel.assert_experiment(axis,config['panel']['mode'],config['panel']['decision_lookback'])
    p=config['panel'];m=panel.manifest
    if m['feature_dim']!=config['fusion']['news_dim'] or m['feature_dim']!=p['projection_components']+8:
        raise ValueError('Panel news dimension differs from configured projection/fusion')
    for field in ('lookback_sessions','simulated_latency_seconds'):
        if m.get(field)!=p[field]:raise ValueError('Panel differs from configured '+field)
    if m.get('decay_tau_sessions')!=p['decay_tau_sessions']:
        raise ValueError('Panel differs from configured decay_tau_sessions')
    if m.get('relation_time_basis','decision_and_event')!=p.get('relation_time_basis','decision_and_event'):
        raise ValueError('Panel relation time basis differs from configured policy')
    transform=m['transform'];expected='train_only_pca' if p['projection']=='pca' else 'fixed_random_projection'
    if transform.get('fit_mode')!=expected or transform.get('components',m['feature_dim']-8)!=p['projection_components']:
        raise ValueError('Panel projection differs from current config')
    if expected=='fixed_random_projection' and transform.get('seed')!=p['projection_seed']:
        raise ValueError('Panel random projection seed differs from current config')
    expected_reference=config.get('pipeline',{}).get('industry_membership_mode','verified_historical')=='reference_retrospective'
    if m.get('historical_industry_membership_verified') is False and not expected_reference:
        raise ValueError('Panel contains exploratory undated industry references; config requires verified history')
    channel=config.get('market_channel')
    quality=m.get('market_quality')
    if (channel is None)!=(quality is None):
        raise ValueError('Panel market_channel policy differs from current config; rebuild panels')
    if channel is not None:
        policy={k:v for k,v in channel.items() if k not in ('gates','official_events')}
        if {k:v for k,v in quality['policy'].items() if k not in ('gates','official_events')}!=policy or (channel['official_events'] is None)!=(quality['official'] is None):
            raise ValueError('Panel market_channel evidence/dedup/source policy differs from current config; rebuild panels')
    encoder=m.get('article_encoder',{})
    if encoder.get('model_id')!=config['finbert']['model_id'] or encoder.get('max_length')!=config['finbert']['max_length'] or encoder.get('embedding_method')!=config['finbert']['pooling']:
        raise ValueError('Panel FinBERT encoder differs from current config')
    if p.get('mode',m.get('mode'))=='observed_pit' and 'industry' in config['fusion']['enabled_channels']:
        from .relations import require_industry_timing
        require_industry_timing(panel.assert_historical_relations(),p['mode'],config['fusion']['enabled_channels'])


class Pipeline:
    def __init__(self, project, config_path=None, fresh=False):
        from .configuration import load_llm_config, read_yaml
        self.project = Path(project).resolve()
        self.base_path = self.resolve(config_path or 'configs/stockmixer.yaml')
        self.base = read_yaml(self.base_path)
        llm_path = Path(self.base['llm']['config'])
        self.llm_path = llm_path if llm_path.is_absolute() else self.base_path.parent / llm_path
        self.config = load_llm_config(self.llm_path)
        validate_news_protocol(self.config)
        self.settings = self.config['pipeline']
        if Path(sys.executable).resolve() != Path(self.settings['python']).resolve():
            raise ValueError('Use configured interpreter: ' + self.settings['python'])
        self.state_path = destination(self.project, self.settings['state'])
        if self.state_path.exists() and not fresh:
            self.state = checked_state(self.project, self.config)
            if baseline_signature(self.base)!=self.state.get('baseline_parameters_sha256'):
                raise ValueError('StockMixer parameters changed after preparation; use --new for a matching baseline')
        else:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = '_'.join(self.settings['datasets'])
            root = destination(self.project, 'outputs/artifacts/news/pipelines/' + stamp + '_' + name + '_preparation')
            root.mkdir(parents=True, exist_ok=False)
            self.state = {'kind': 'offline_news_pipeline', 'created_at': now(), 'root': str(root),
                          'settings_sha256': settings_fingerprint(self.config),
                          'snapshot_manifest_sha256': digest_file(self.resolve(self.config['adapted_snapshot']) / 'manifest.json'),
                          'base_config_sha256': digest(self.base), 'steps': {},
                          'baseline_parameters_sha256':baseline_signature(self.base),
                          'artifacts': {name: {} for name in self.settings['datasets']}}
            write_json(root / 'stockmixer.source.json', self.base)
            write_json(root / 'llm.source.json', self.config)
            write_json(root / 'source_code_sha256.json',{p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')})
            self.save()
        self.root = Path(self.state['root'])
        self.axes = {name: Axis.load(self.resolve(self.base['data']['axis_dir']) / (name + '.json')) for name in self.settings['datasets']}

    @contextmanager
    def exclusive(self):
        """OS releases the lock after interruption; parallel steps cannot overwrite shared state."""
        with state_lock(self.state_path):
            self.state=checked_state(self.project,self.config)
            self.root=Path(self.state['root'])
            yield

    def resolve(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.project / path).resolve()

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state['updated_at'] = now()
        temporary = self.state_path.with_suffix('.tmp')
        write_json(temporary, self.state)
        os.replace(temporary, self.state_path)
        write_json(Path(self.state['root']) / 'state.json', self.state)

    def articles(self):
        from .offline_inputs import encoding_inputs
        articles,counts,provenance=encoding_inputs(self.project, self.config)
        path=self.settings.get('sample_review')
        if path:
            from .sample_review import apply_review
            directory=self.resolve(path)
            articles,summary=apply_review(articles,directory)
            self.pin_artifact(directory)
            self.state['semantic_sample_review']=summary
            provenance['source_bound_sample_review_manifest_sha256']=summary['review_manifest_sha256']
        return articles,counts,provenance

    def require(self, step):
        if self.state['steps'].get(step) != 'completed':
            raise ValueError('First complete main.py --step ' + step)
        self.verify_completed(step)

    def pin_artifact(self,path):
        path=Path(path).resolve();value=digest_file(path/'manifest.json')
        accepted=self.state.setdefault('accepted_manifests',{})
        if str(path) in accepted and accepted[str(path)]!=value:
            raise ValueError('Previously accepted artifact changed: '+str(path))
        accepted[str(path)]=value

    def verify_completed(self,step):
        if step=='industry_mapping':
            self.step_industry_mapping();return
        if step=='relations' and self.state.get('industry_inference_artifact'):
            from .industry_review import PROMPT,SCHEMA
            manifest=verify_artifact(self.state['industry_inference_artifact'])
            articles,_,_=self.articles()
            from .routing import article_pin
            if manifest.get('kind')!='evidence_checked_industry_inference' or manifest.get('prompt_sha256')!=digest(PROMPT) or manifest.get('response_schema_sha256')!=digest(SCHEMA) or manifest.get('article_pins_hash')!=digest([article_pin(a) for a in articles]):
                raise ValueError('Industry inference cache schema/prompt/source changed; rebuild relations')
            self.pin_artifact(self.state['industry_inference_artifact'])
        if step in ('validate','prepare'):
            if step=='prepare':self.step_validate()
            getattr(self,'step_'+step)();return
        if step in ('setup','encode'):
            getattr(self,'step_'+step)()
            if step=='encode':self.pin_artifact(self.state['features'])
            return
        for name,artifacts in self.state['artifacts'].items():
            if step=='review':
                from .review_reuse import validate_review_artifact
                expected=artifacts.get('review_manifest_sha256')
                if expected and digest_file(Path(artifacts['review'])/'manifest.json')!=expected:
                    raise ValueError('Review manifest changed after acceptance')
                articles,_,provenance=self.articles()
                validate_review_artifact(artifacts['review'],articles,name,provenance,self.config['review'])
                self.pin_artifact(artifacts['review'])
            elif step=='publish':
                from .routing import load_routes
                load_routes(artifacts['market_routes'],name);self.pin_artifact(artifacts['market_routes'])
            elif step=='relations':
                from .relations import load_relations
                load_relations(artifacts['relations'],self.axes[name]);self.pin_artifact(artifacts['relations'])
            elif step=='schedule':
                if digest_file(artifacts['schedule'])!=artifacts['schedule_sha256']:raise ValueError('Decision schedule changed after export')
                Schedule(self.axes[name],read_records(artifacts['schedule']))
            elif step=='panels':
                from .dataset import NewsPanel
                panel=NewsPanel(artifacts['panel']);validate_panel_settings(panel,self.config,self.axes[name])
                if panel.manifest['article_artifact_manifest_sha256']!=digest_file(Path(self.state['features'])/'manifest.json'):
                    raise ValueError('Panel features differ from current encoded snapshot')
                if panel.manifest.get('relations_manifest_sha256')!=digest_file(Path(artifacts['relations'])/'manifest.json'):
                    raise ValueError('Panel historical relations differ from current preparation')
                if (panel.manifest.get('market_routing') or {}).get('manifest_sha256')!=digest_file(Path(artifacts['market_routes'])/'manifest.json'):
                    raise ValueError('Panel market routes differ from current preparation')
                self.pin_artifact(artifacts['panel']);del panel
            elif step in ('baseline','train'):
                from .experiment import load_run_checkpoint,validate_checkpoint,load_panel
                from .protocol import load_market,resolve
                config=self.baseline_config(name);data=load_market(resolve(self.project,config['workflow']['prepared_data'][name]),self.project)
                if step=='baseline':
                    parent=load_run_checkpoint(artifacts['baseline_checkpoint']);validate_checkpoint(parent,data)
                    if parent['mode']!='baseline' or parent['smoke_only']:raise ValueError('Completed formal baseline is ineligible')
                else:
                    parent=load_run_checkpoint(Path(artifacts['fusion_training'])/'checkpoints/best_model.pt')
                    panel=load_panel(artifacts['panel'],data,parent['config']);validate_checkpoint(parent,data,panel);del panel
                del parent,data

    def run(self, step):
        if step == 'all':
            for name in STEPS:
                self.run(name)
            return
        if step == 'status':
            status=copy.deepcopy(self.state)
            status['configured_review_progress']={}
            for name,value in self.settings['review_artifacts'].items():
                if value:
                    path=self.resolve(value)/'run_status.json'
                    if path.exists():status['configured_review_progress'][name]=json.loads(path.read_text(encoding='utf-8'))
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return
        if step == 'templates':
            from .relations import mapping_templates
            path = self.root / 'mapping_templates'
            if not path.exists():
                mapping_templates(self.project, list(self.axes.values()), path)
            print('[MAPPING] ' + str(path), flush=True)
            return
        print('[PIPELINE] ' + step, flush=True)
        if self.state['steps'].get(step) == 'completed':
            try:
                self.verify_completed(step)
                self.state.setdefault('latest_checks',{})[step]={'status':'verified','at':now()}
                self.save()
            except BaseException as error:
                self.state['last_error']={'step':step,'cache_recheck':True,'type':type(error).__name__,'error':str(error),'at':now()}
                self.state.setdefault('latest_checks',{})[step]={'status':'failed','at':now()}
                self.save()
                raise
            print('[REUSE] verified completed ' + step, flush=True)
            return
        try:
            self.state.setdefault('step_source_code_sha256',{})[step]={p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}
            getattr(self, 'step_' + step)()
            product={'review':'review','publish':'market_routes','relations':'relations','panels':'panel'}.get(step)
            if product:
                for artifacts in self.state['artifacts'].values():self.pin_artifact(artifacts[product])
            if step=='encode':self.pin_artifact(self.state['features'])
            self.state['steps'][step] = 'completed'
            self.state.pop('last_error', None)
            self.save()
        except BaseException as error:
            self.state['last_error'] = {'step': step, 'type': type(error).__name__, 'error': str(error), 'at': now()}
            self.save()
            raise

    def step_setup(self):
        from .setup_news import setup_environment
        settings=dict(self.settings,model_directory=str(self.resolve(self.config['finbert']['model_dir'] or self.settings['model_directory'])))
        self.state['environment'] = setup_environment(settings)

    def step_review(self):
        from .auto_review import OllamaReviewer, auto_review
        articles, _, provenance = self.articles()
        backend = None
        for name in self.axes:
            artifact = self.settings['review_artifacts'][name]
            if artifact:
                path = self.resolve(artifact)
                manifest = verify_artifact(path)
                if manifest.get('dataset') != name:
                    raise ValueError('Review artifact dataset mismatch')
                status=json.loads((path/'run_status.json').read_text(encoding='utf-8'))
                if manifest.get('kind') not in ('automated_market_review','upstream_market_review','subset_extended_market_review_v1','completed_market_review_v1') or status.get('status')!='completed' or manifest.get('selected_articles')!=len(articles):
                    raise ValueError('Review artifact must be a completed full-snapshot review, not a sample or pending template')
            else:
                # Failed runs use new audit directories and resume per-article cache.
                path = destination(self.project, dataset=name, kind='auto_review', category='artifacts/news/routing_reviews')
                common=next((a['review'] for a in self.state['artifacts'].values() if a.get('review')),None)
                common_kind=verify_artifact(common).get('kind') if common else None
                if common and common_kind=='automated_market_review' and self.settings['review_limit'] is None:
                    from .review_reuse import rebind_review
                    rebind_review(common,articles,name,path,provenance,self.config['review'])
                    print('[REVIEW] reused common US decisions for '+name,flush=True)
                elif self.settings['analysis_source']=='upstream_only':
                    from .upstream import build_review
                    build_review(articles,name,path,provenance,self.config['review']['min_confidence'])
                    print('[REVIEW] upstream reuse; model_calls=0 for '+name,flush=True)
                else:
                    if backend is None:backend=OllamaReviewer(self.config['review'])
                    auto_review(articles, name, path, destination(self.project, self.config['review']['cache_dir']),
                                backend, self.config['review'], provenance, self.settings['review_limit'], None,
                                progress=lambda n, total, stats: print('[REVIEW] %s %d/%d calls=%d failed=%d' %
                                    (name, n, total, stats.get('model_calls', 0), stats.get('failed_articles', 0)), flush=True))
            from .review_reuse import validate_review_artifact
            validate_review_artifact(path,articles,name,provenance,self.config['review'])
            self.pin_artifact(path)
            self.state['artifacts'][name]['review'] = str(path)
            self.state['artifacts'][name]['review_manifest_sha256']=digest_file(path/'manifest.json')
            self.save()

    def semantic_proof(self,articles,reviews=None):
        proof=None
        if self.settings.get('semantic_review'):
            from .semantic_review import load_sample
            path=self.resolve(self.settings['semantic_review'])
            proof=load_sample(path,articles,reviews,self.state['snapshot_manifest_sha256'])
            self.pin_artifact(path)
        if reviews is not None and self.settings.get('sample_review'):
            from .sample_review import market_proof
            proof=market_proof(articles,reviews,proof,digest_file(self.resolve(self.settings['sample_review'])/'manifest.json'))
        return proof

    def step_publish(self):
        self.require('review')
        from .routing import publish_routes
        articles, _, provenance = self.articles()
        for name, artifacts in self.state['artifacts'].items():
            review=Path(artifacts['review'])
            if digest_file(review/'manifest.json')!=artifacts['review_manifest_sha256']:
                raise ValueError('Review manifest changed after acceptance; use a new reviewed version')
            verify_artifact(review)
            path = self.root / (name + '_market_routes')
            if not path.exists():
                reviews=read_records(Path(artifacts['review'])/'market_review.jsonl')
                proof=self.semantic_proof(articles,reviews)
                if proof is not None:
                    from .semantic_review import apply_sample
                    reviews,_=apply_sample(articles,reviews,proof['rows'],proof['confirmation'])
                publish_routes(articles,reviews,name,path,provenance,self.settings['allow_partial_routes'],semantic_audit=proof)
            verify_artifact(path)
            artifacts['market_routes'] = str(path)
            self.pin_artifact(path)

    def step_encode(self):
        import torch
        original=torch.get_num_threads()
        try:
            torch.set_num_threads(self.settings['finbert_threads'])
            return self._encode()
        finally:
            torch.set_num_threads(original)

    def _encode(self):
        from .encoder import LocalFinBert, encode_articles
        articles, _, provenance = self.articles()
        if self.settings['encoding_limit']:
            articles = articles[:self.settings['encoding_limit']]
        path = self.resolve(self.settings['features']) if self.settings['features'] else self.root / 'article_features'
        if not path.exists():
            f = self.config['finbert']
            model = self.resolve(f['model_dir'] or self.settings['model_directory'])
            installed = json.loads((model / 'download_manifest.json').read_text(encoding='utf-8'))
            for name, sha in installed['files'].items():
                if digest_file(model / name) != sha:
                    raise ValueError('Local FinBERT installation changed')
            if installed['revision'] != self.settings['model_revision']:
                raise ValueError('FinBERT revision differs from pipeline config')
            backend = LocalFinBert(model, f['device'], f['max_length'], installed['model_available_at'])
            encode_articles(articles, backend, path, destination(self.project, self.settings['cache']),
                            f['batch_size'], f['output_format'], provenance,
                            progress=lambda n,total,counts: print('[ENCODE] %d/%d cache=%d' % (n,total,counts.get('cache_hit',0)),flush=True))
        manifest = verify_artifact(path)
        model=self.resolve(self.config['finbert']['model_dir'] or self.settings['model_directory'])
        installed=json.loads((model/'download_manifest.json').read_text(encoding='utf-8'))
        validate_feature_settings(manifest,self.config,installed)
        if manifest.get('input_provenance', {}).get('snapshot_manifest_sha256') != provenance['snapshot_manifest_sha256']:
            raise ValueError('Feature artifact belongs to another frozen snapshot')
        self.state['features'] = str(path)
        self.state['encoding'] = {'articles': manifest['articles'], 'successful': manifest['successful'],
                                 'status_counts': manifest['status_counts']}
        if manifest['successful'] != len(articles):
            raise ValueError('Encoding incomplete; inspect feature manifest failures')

    def step_industry_mapping(self):
        from .industry_mapping import ensure_mapping
        result=ensure_mapping(self.project,self.config,self.base)
        if result is not None:
            path=self.resolve(self.config['industry_dictionary']['output'])
            self.pin_artifact(path)
            self.state['industry_mapping']={'directory':str(path),'datasets':result}

    def step_relations(self):
        if self.config.get('industry_dictionary'):
            self.step_industry_mapping()
        from .relations import build_relations
        articles, _, provenance = self.articles()
        paths = [self.resolve(self.settings[k]) if self.settings[k] else None for k in
                 ('security_master', 'industry_memberships', 'industry_aliases')]
        self.state['mapping_input_sha256'] = {str(p): digest_file(p) for p in paths if p}
        dictionary=None
        if self.config.get('industry_dictionary'):
            from .industry_dictionary import IndustryDictionary
            dictionary=IndustryDictionary(self.resolve(self.config['industry_dictionary']['directory']))
            for filename in ('manifest.json','industry_dictionary.jsonl'):
                p=dictionary.directory/filename;self.state['mapping_input_sha256'][str(p)]=digest_file(p)
        for parent in {p.parent for p in paths if p}:
            metadata=parent/'manifest.json'
            if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8')).get('kind')=='sec_sic_dated_historical_observations':
                if dictionary is None:
                    from .master_data import verify_observations
                    verify_observations(parent)
                else:
                    from .industry_mapping import verify_master_artifact
                    verify_master_artifact(parent)
                self.pin_artifact(parent)
        if dictionary is not None:
            from .industry_mapping import verified_mapping
            mapping_path=self.resolve(self.config['industry_dictionary']['output'])
            if not (mapping_path/'manifest.json').exists():
                raise ValueError('First run scripts/10_build_industry_mapping.py')
            verified_mapping(mapping_path,dictionary,self.axes,read_records(paths[0]) if paths[0] else [],read_records(paths[1]) if paths[1] else [])
            self.pin_artifact(mapping_path)
            paths=[mapping_path/'verified_security_master.jsonl',mapping_path/'verified_industry_memberships.jsonl',paths[2]]
            self.state['mapping_input_sha256'].update({str(p):digest_file(p) for p in paths if p})
        proof=self.semantic_proof(articles)
        if proof is not None:
            from .semantic_review import annotate_articles
            articles=annotate_articles(articles,proof)
        if self.settings['infer_industries']:
            from .industry_review import enrich_industries
            explicit_inference=self.settings.get('industry_inference_artifact')
            enriched=self.resolve(explicit_inference) if explicit_inference else self.root/'industry_inference'
            if self.state.get('industry_inference_artifact'):
                if explicit_inference and Path(self.state['industry_inference_artifact'])!=enriched:
                    raise ValueError('Configured industry artifact differs from accepted preparation')
                enriched=Path(self.state['industry_inference_artifact'])
            if explicit_inference and not (enriched/'manifest.json').exists():
                raise ValueError('Explicit industry artifact must be a completed audited inference')
            if enriched.exists() and not (enriched/'manifest.json').exists():
                enriched=self.root/('industry_inference_retry_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            if enriched.exists():
                inference=verify_artifact(enriched)
                from .industry_review import PROMPT,SCHEMA
                from .routing import article_pin
                if inference.get('kind')!='evidence_checked_industry_inference' or inference.get('prompt_sha256')!=digest(PROMPT) or inference.get('response_schema_sha256')!=digest(SCHEMA) or inference.get('article_pins_hash')!=digest([article_pin(a) for a in articles]):
                    raise ValueError('Industry artifact source/prompt/schema mismatch')
                if inference.get('taxonomy_sha256')!=digest(read_records(paths[2]) if paths[2] else []):
                    raise ValueError('Industry artifact uses a different historical taxonomy')
                if any(inference.get('settings',{}).get(k)!=self.settings[k] for k in ('industry_min_confidence','industry_max_impacts')):
                    raise ValueError('Industry artifact thresholds differ from config')
                model=inference.get('model',{})
                expected_digest=self.config['review'].get('expected_model_digest')
                if model.get('model')!=self.config['review']['model'] or (expected_digest and str(model.get('model_digest','')).removeprefix('sha256:')!=expected_digest.removeprefix('sha256:')) or model.get('request_options')!={k:self.config['review'][k] for k in ('temperature','seed','num_ctx','num_predict')}:
                    raise ValueError('Industry artifact model/decoding settings differ from config')
                previous_articles=read_records(enriched/'articles.enriched.jsonl')
                expected_quarantine={a['source_id'] for a in articles if a.get('semantic_quarantine') is True}
                cached_quarantine={a['source_id'] for a in previous_articles if a.get('semantic_quarantine') is True}
                if expected_quarantine!=cached_quarantine:
                    raise ValueError('Industry artifact uses a different semantic quarantine policy; regenerate inference')
                articles=previous_articles
                self.state['industry_inference']=inference['stats']
            else:
                articles,statistics=enrich_industries(articles,read_records(paths[2]) if paths[2] else [],
                    self.config['review'],self.settings,destination(self.project,self.settings['industry_cache']),enriched)
                self.state['industry_inference']=statistics
            if (enriched/'manifest.json').exists():
                self.state['industry_inference_artifact']=str(enriched)
                self.pin_artifact(enriched)
        if self.settings.get('industry_membership_mode')=='reference_retrospective' and self.config['panel']['mode']!='retrospective':
            raise ValueError('Reference industry membership cannot be used for observed PIT')
        for article in articles:
            if article['source_id'] in self.settings.get('industry_excluded_sources',[]):
                article['industry_excluded_proposals']=article.get('industry_impacts',[])
                article['industry_impacts']=[]
                article['industry_scope_review']={'decision':'exclude','review_method':'configured_exclusion','rationale':'Preserve configured exclusions from prior user scope reviews; no peer propagation.'}
        from .industry_scope_review import apply_scope_reviews
        scope_path=self.resolve(self.settings['industry_scope_review']) if self.settings.get('industry_scope_review') else None
        automated=None
        if self.settings.get('allow_automated_industry_scope',False):
            from .legacy_industry import automated_reviews
            snapshot_manifest=verify_artifact(self.resolve(self.config['adapted_snapshot']))
            if snapshot_manifest.get('kind')=='adapted_collector_snapshot':
                from .retrace_adapter import automated_industry_reviews
                automated=automated_industry_reviews(articles,self.settings['industry_min_confidence'])
            else:
                automated=automated_reviews(self.resolve(self.config['adapted_snapshot']),articles)
        articles,scope_summary=apply_scope_reviews(articles,scope_path,self.settings.get('require_industry_scope_review',False),self.state['snapshot_manifest_sha256'],preserve_approval_time=True,automated=automated)
        if scope_path:self.pin_artifact(scope_path)
        self.state['industry_scope_review']=scope_summary
        write_json(self.root/'industry_scope_readiness.json',scope_summary)
        for name, axis in self.axes.items():
            path = self.root / (name + '_relations')
            if not path.exists():
                build_relations(axis, articles, paths, path, provenance,policy={'version':'evidence_mapping_v3','industry_min_confidence':self.settings['industry_min_confidence']},dictionary=dictionary,reference_sources=(self.resolve(self.config['industry_dictionary']['reference_archive']),self.resolve(self.base['data']['dataset_root'])/'SP500/sp500_ticker.csv') if self.settings.get('industry_membership_mode')=='reference_retrospective' else None)
            verify_artifact(path)
            self.state['artifacts'][name]['relations'] = str(path)

    def market_channel(self):
        value=self.config.get('market_channel')
        if value is None:return None
        value=copy.deepcopy(value)
        if value['official_events']:value['official_events']=str(self.resolve(value['official_events']))
        return value

    def step_schedule(self):
        for name,axis in self.axes.items():
            path=self.root/(name+'_decision_schedule.jsonl')
            if not path.exists():export_schedule(axis,path)
            Schedule(axis,read_records(path))
            self.state['artifacts'][name].update(schedule=str(path),schedule_sha256=digest_file(path))

    def step_panels(self):
        for step in ('encode', 'publish', 'relations'):
            self.require(step)
        from .panel import build_panel
        p = self.config['panel']
        for name, axis in self.axes.items():
            artifacts = self.state['artifacts'][name]
            schedule = self.root / (name + '_decision_schedule.jsonl')
            if not schedule.exists():
                export_schedule(axis, schedule)
            expected=artifacts.get('schedule_sha256')
            if expected and digest_file(schedule)!=expected:
                raise ValueError('Decision schedule changed after export')
            relations = Path(artifacts['relations'])
            verify_artifact(relations)
            path = self.root / (name + '_panel')
            if not path.exists():
                build_panel(axis, Schedule(axis, read_records(schedule)), self.state['features'],
                            read_records(relations / 'article_stock_link.jsonl'), read_records(relations / 'industry_memberships.jsonl'),
                            path, p['mode'], p['projection'], p['lookback_sessions'], p['decay_tau_sessions'],
                            p['simulated_latency_seconds'], p['projection_seed'], p['decision_lookback'], artifacts['market_routes'],relations,components=p['projection_components'],relation_time_basis=p.get('relation_time_basis','decision_and_event'),
                            market_channel=self.market_channel())
            artifacts.update(panel=str(path), schedule=str(schedule))
            self.save()

    def step_validate(self):
        self.require('panels')
        from .dataset import NewsPanel
        reports = {}
        for name, artifacts in self.state['artifacts'].items():
            panel = NewsPanel(artifacts['panel'])
            validate_panel_settings(panel,self.config,self.axes[name])
            from .protocol import load_market
            expected=self.baseline_config(name)
            data_path=self.resolve(expected['workflow']['prepared_data'][name])
            if not data_path.exists():
                if not expected['workflow']['prepare_if_missing']:
                    raise ValueError('Prepared market data is missing: '+str(data_path))
                from .protocol import prepare_market
                prepare_market(self.project,expected,data_path)
            data=load_market(data_path,self.project)
            reports[name] = readiness(panel, self.config,data,self.state.get('features'))
            del panel
        write_json(self.root / 'readiness.json', reports)
        self.state['readiness'] = reports
        if not all(r['ready'] for r in reports.values()):
            raise ValueError('Enabled channels are not ready. See ' + str(self.root / 'readiness.json'))

    def baseline_config(self, name):
        from .configuration import compile_config
        base = copy.deepcopy(self.base)
        base['llm']['enabled'] = False
        base['data'].update(protocol=self.settings['data_protocol'], prepared_root=None)
        base['market_name'] = name
        base['output']['directory'] = None
        return compile_config(self.project, self.base_path, base, name)

    def step_baseline(self):
        self.require('validate')
        from .workflow import run_workflow
        for name, artifacts in self.state['artifacts'].items():
            if artifacts.get('baseline_checkpoint'):
                continue
            reference = self.config.get('pipeline',{}).get('baseline_checkpoints',{}).get(name,self.config.get('baseline_checkpoint'))
            if reference:
                from .experiment import load_run_checkpoint
                from .protocol import load_market
                expected = self.baseline_config(name)
                source = self.resolve(reference)
                data = load_market(self.resolve(expected['workflow']['prepared_data'][name]), self.project)
                parent = load_run_checkpoint(source)
                validate_baseline_source(parent, data, expected)
                artifacts['baseline_checkpoint'] = str(source)
                artifacts['baseline_reuse'] = {'sha256': digest_file(source), 'protocol': parent['protocol'],
                                             'review_method': 'verified_same_protocol_and_training_config'}
                print('[BASELINE] reused verified ' + parent['protocol'] + ' E0: ' + str(source), flush=True)
                self.save()
                continue
            path = self.root / (name + '_baseline.config.json')
            write_json(path, self.baseline_config(name))
            result = run_workflow(path, self.project)
            artifacts['baseline_checkpoint'] = result['best_checkpoint']
            self.save()

    def step_prepare(self):
        self.require('validate')
        self.require('baseline')
        if self.settings['review_limit'] is not None or self.settings['encoding_limit'] is not None:
            raise ValueError('Remove sample limits and rebuild full preparation before formal training')
        if baseline_signature(self.base) != self.state['baseline_parameters_sha256']:
            raise ValueError('StockMixer config changed during preparation; rebuild baseline')
        from .experiment import load_run_checkpoint, validate_checkpoint
        from .protocol import load_market, resolve
        for name, artifacts in self.state['artifacts'].items():
            config = self.baseline_config(name)
            data = load_market(resolve(self.project, config['workflow']['prepared_data'][name]), self.project)
            parent = load_run_checkpoint(artifacts['baseline_checkpoint'])
            validate_checkpoint(parent, data)
            validate_baseline_source(parent, data, config)

    def step_train(self):
        self.require('prepare')
        if not self.base['llm']['enabled']:
            raise ValueError('Enable llm.enabled in stockmixer.yaml before training fusion')
        from .workflow import run_workflow
        from .configuration import compile_config
        base = copy.deepcopy(self.base)
        for name, artifacts in self.state['artifacts'].items():
            if artifacts.get('fusion_training'):
                continue
            configured=copy.deepcopy(base)
            configured['market_name']=name
            configured['llm']['config']=str(self.llm_path)
            compile_config(self.project,self.base_path,configured,name)
            path = self.root / (name + '_fusion.config.yaml')
            path.write_text(yaml.safe_dump(configured,allow_unicode=True,sort_keys=False),encoding='utf-8')
            result = run_workflow(path, self.project)
            artifacts['fusion_training'] = result['training_directory']
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline news preparation, matched original/strict baseline, training and prediction')
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--step', choices=STEPS + ('all', 'status', 'templates'), default='all')
    parser.add_argument('--new', action='store_true', help='Create a new preparation version; never overwrite prior artifacts')
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    from .configuration import read_yaml,load_llm_config
    base_path=Path(args.config) if args.config else project/'configs/stockmixer.yaml'
    if not base_path.is_absolute():base_path=project/base_path
    base=read_yaml(base_path)
    from .configuration import boolean
    boolean(base['llm']['enabled'],'llm.enabled')
    if args.step in ('all','train') and base['llm']['enabled'] is False:
        # The public switch governs both entries. Explicit preprocessing steps may run before enabling it.
        from .workflow import run_workflow
        run_workflow(base_path,project)
        return
    llm_path=Path(base['llm']['config'])
    if not llm_path.is_absolute():llm_path=base_path.parent/llm_path
    news=load_llm_config(llm_path)
    with state_lock(destination(project,news['pipeline']['state'])):
        pipeline = Pipeline(project, args.config, args.new)
        pipeline.run(args.step)


if __name__ == '__main__':
    main()
