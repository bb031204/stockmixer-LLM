"""Completed retrace -> frozen import -> offline preparation -> unchanged training runtime."""
import argparse
import copy
import json
import os
import re
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from .configuration import boolean, load_llm_config, read_yaml
from .io import digest_file, now, write_json
from .outputs import Tee, destination
from .pipeline import Pipeline, STEPS, state_lock, validate_news_protocol

PREPARATION_STEPS = STEPS[:STEPS.index('train')]
ENTRY_STEPS = ('all', 'import', 'prepare', 'train', 'check', 'status')


def resolve(project, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(project) / path).resolve()


def completed_retrace(batch):
    """Require the published, fully completed version; a live checkpoint is insufficient."""
    batch = Path(batch).resolve()
    progress = json.loads((batch / 'progress.json').read_text(encoding='utf-8-sig'))
    live_path = batch / 'live_progress.json'
    live = json.loads(live_path.read_text(encoding='utf-8-sig')) if live_path.exists() else {}
    pending = progress.get('pending')
    if type(pending) is not int or pending < 0:
        raise ValueError('Retrace progress is missing a valid pending count')
    if pending or live.get('pending', 0):
        raise ValueError('Retrace尚未完成：published pending=%s，live pending=%s；等待全量导出和acceptance通过后重试，不会使用旧新闻启动训练。'
                         % (pending, live.get('pending', 'unavailable')))
    acceptance = json.loads((batch / 'acceptance.json').read_text(encoding='utf-8-sig'))
    if acceptance.get('passed') is not True:
        raise ValueError('Retrace全量acceptance尚未通过；不会激活新闻快照')
    path = batch / 'manifest.json'
    before = digest_file(path)
    manifest = json.loads(path.read_text(encoding='utf-8-sig'))
    if manifest.get('classification_contract') != 'dictionary-scope-replacement-1':
        raise ValueError('Expected dictionary-scope-replacement-1 retrace contract')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Retrace manifest must list accepted files')
    for name, expected in files.items():
        source = (batch / name).resolve()
        try:
            source.relative_to(batch)
        except ValueError:
            raise ValueError('Retrace manifest path escaped its batch') from None
        if not source.is_file() or digest_file(source) != expected:
            raise ValueError('Retrace accepted file changed: ' + name)
    if digest_file(path) != before:
        raise ValueError('Retrace manifest changed during verification; retry after publication finishes')
    return {'batch': str(batch), 'manifest_sha256': before, 'progress': progress,
            'acceptance_sha256': digest_file(batch / 'acceptance.json'),
            'verified_files': len(files), 'partial': False}


def activate_snapshot(project, llm_path, snapshot, pipeline_state, backup):
    """Preserve public comments; invalidate only products bound to the previous snapshot."""
    project, llm_path = Path(project).resolve(), Path(llm_path).resolve()
    original = llm_path.read_bytes()
    text = original.decode('utf-8-sig')
    expected = load_llm_config(llm_path)
    changed = copy.deepcopy(expected)
    changed['adapted_snapshot'] = Path(snapshot).relative_to(project).as_posix()
    changed['pipeline']['state'] = Path(pipeline_state).relative_to(project).as_posix()
    def replace_line(key, value, indent=''):
        nonlocal text
        pattern = r'^' + re.escape(indent + key) + r':[^\n]*$'
        text, count = re.subn(pattern, lambda m: indent + key + ': ' + value, text, flags=re.M)
        if count != 1:
            raise ValueError('Expected exactly one config field: ' + indent + key)
    replace_line('adapted_snapshot', changed['adapted_snapshot'])
    replace_line('state', changed['pipeline']['state'], '  ')
    keys = ('industry_batch_artifact', 'industry_batch_base_snapshot', 'industry_batch_legacy_snapshot',
            'industry_batch_snapshot_output', 'semantic_review', 'industry_scope_review', 'features')
    for key in keys:
        changed['pipeline'][key] = None
        if re.search(r'^  ' + re.escape(key) + ':', text, flags=re.M):
            replace_line(key, 'null', '  ')
    for key in ('panels', 'market_routes', 'review_artifacts'):
        prefix = '  ' if key == 'review_artifacts' else ''
        pattern = r'^' + prefix + key + r':[^\n]*\n(?:' + prefix + r'  (?:NASDAQ|NYSE|SP500):[^\n]*(?:\n|$))+'
        replacement = prefix + key + ':\n' + ''.join(prefix + '  ' + d + ': null\n' for d in ('NASDAQ', 'NYSE', 'SP500'))
        text, count = re.subn(pattern, lambda m: replacement, text, flags=re.M)
        if count != 1:
            raise ValueError('Cannot safely invalidate config section: ' + key)
        target = changed['pipeline'] if prefix else changed
        target[key] = {d: None for d in ('NASDAQ', 'NYSE', 'SP500')}
    backup = Path(backup)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise FileExistsError('Activation backup already exists')
    backup.write_bytes(original)
    temporary = llm_path.with_name('.' + llm_path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        if load_llm_config(temporary) != changed:
            raise ValueError('Config activation changed unexpected values')
        if llm_path.read_bytes() != original:
            raise ValueError('Config changed concurrently; activation cancelled')
        os.replace(temporary, llm_path)
    finally:
        if temporary.exists():
            temporary.unlink()


class Launcher:
    def __init__(self, project, config=None):
        self.project = Path(project).resolve()
        self.base_path = resolve(self.project, config or 'configs/stockmixer.yaml')
        self.base = read_yaml(self.base_path)
        boolean(self.base['llm']['enabled'], 'llm.enabled')
        self.llm_path = None
        self.config = None

    def load_news(self):
        self.llm_path = Path(self.base['llm']['config'])
        if not self.llm_path.is_absolute():
            self.llm_path = self.base_path.parent / self.llm_path
        self.llm_path = self.llm_path.resolve()
        self.config = load_llm_config(self.llm_path)
        settings = self.config['pipeline']
        if Path(sys.executable).resolve() != Path(settings['python']).resolve():
            raise ValueError('Use configured interpreter: ' + settings['python'])
        validate_news_protocol(self.config)
        if self.base['market_name'] not in settings['datasets']:
            raise ValueError('market_name必须包含在llm.yaml的pipeline.datasets中；请统一目标数据集配置')
        if settings['review_limit'] is not None or settings['encoding_limit'] is not None:
            raise ValueError('一键正式流程要求review_limit和encoding_limit均为null')
        self.pointer = destination(self.project, settings.get('launch_state', 'outputs/artifacts/news/launch_state.json'))
        return self.config

    def source(self):
        path = self.config['pipeline']['retrace_artifact']
        if not path:
            raise ValueError('Set pipeline.retrace_artifact to the completed upstream batch')
        return resolve(self.project, path)

    def status(self):
        result = {'llm_enabled': self.base['llm']['enabled'], 'base_config': str(self.base_path)}
        if self.config:
            result.update(datasets=self.config['pipeline']['datasets'], source=str(self.source()))
            for name in ('progress.json', 'live_progress.json'):
                path = self.source() / name
                if path.exists():
                    result[name] = json.loads(path.read_text(encoding='utf-8-sig'))
            if self.pointer.exists():
                result['launcher'] = json.loads(self.pointer.read_text(encoding='utf-8'))
            state = resolve(self.project, self.config['pipeline']['state'])
            if state.exists():
                result['preparation'] = json.loads(state.read_text(encoding='utf-8'))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    def save(self):
        self.record['updated_at'] = now()
        self.pointer.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.pointer.with_name('.' + self.pointer.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            write_json(temporary, self.record)
            os.replace(temporary, self.pointer)
        finally:
            if temporary.exists():
                temporary.unlink()
        write_json(Path(self.record['root']) / 'launcher.json', self.record)

    def import_news(self, source):
        from .retrace_adapter import import_retrace
        from .source_adapter import validate_adapted
        previous = json.loads(self.pointer.read_text(encoding='utf-8')) if self.pointer.exists() else None
        if previous and previous.get('base_config') != str(self.base_path):
            raise ValueError('launch_state属于另一份配置；请设置独立的pipeline.launch_state')
        same = previous and previous.get('source', {}).get('manifest_sha256') == source['manifest_sha256'] and previous['source']['batch'] == source['batch']
        if same:
            self.record = previous
            snapshot = Path(previous['snapshot'])
            if digest_file(snapshot / 'manifest.json') != previous['snapshot_manifest_sha256']:
                raise ValueError('Imported snapshot manifest changed; preserve it and import a new version')
            validate_adapted(snapshot)
            print('[IMPORT] verified existing full retrace snapshot: ' + str(snapshot), flush=True)
        else:
            root = destination(self.project, dataset='_'.join(self.config['pipeline']['datasets']), kind='launch', category='artifacts/news/launches')
            root.mkdir(parents=True, exist_ok=False)
            snapshot = destination(self.project, dataset='US_GLOBAL', kind='retraced', category='artifacts/news/adapted')
            result = import_retrace(self.source(), snapshot, resolve(self.project, self.base['data']['axis_dir']), False)
            after = completed_retrace(self.source())
            if after['manifest_sha256'] != source['manifest_sha256'] or after['acceptance_sha256'] != source['acceptance_sha256']:
                raise ValueError('Upstream accepted version changed during import; snapshot was not activated')
            self.record = {'kind': 'stockmixer_end_to_end_v1', 'created_at': now(), 'base_config': str(self.base_path),
                           'root': str(root), 'source': source, 'snapshot': str(snapshot), 'import_result': result,
                           'snapshot_manifest_sha256': digest_file(snapshot / 'manifest.json'),
                           'pipeline_state': str(root / 'preparation_state.json'), 'status': 'imported_pending_activation'}
            self.save()
        snapshot = Path(self.record['snapshot'])
        state = Path(self.record['pipeline_state'])
        if resolve(self.project, self.config['adapted_snapshot']) != snapshot or resolve(self.project, self.config['pipeline']['state']) != state:
            backup = destination(self.project, dataset='US_GLOBAL', kind='before_import', category='artifacts/maintenance', suffix='.yaml.bak')
            with state_lock(destination(self.project, self.config['pipeline']['state'])):
                activate_snapshot(self.project, self.llm_path, snapshot, state, backup)
            self.record['config_backup'] = str(backup)
        self.config = load_llm_config(self.llm_path)
        self.record['status'] = 'imported'
        self.record['activated_config_sha256'] = digest_file(self.llm_path)
        self.record['last_error'] = None
        self.save()

    def run(self, step='all', fresh=False):
        if step in ('all', 'train') and not self.base['llm']['enabled']:
            from .workflow import run_workflow
            return run_workflow(self.base_path, self.project)
        if not self.base['llm']['enabled'] and step == 'status':
            return self.status()
        self.load_news()
        if step == 'status':
            return self.status()
        source = completed_retrace(self.source())
        if step == 'check':
            from .configuration import compile_config
            for dataset in self.config['pipeline']['datasets']:
                base = copy.deepcopy(self.base)
                base['llm']['enabled'] = False
                base['data']['protocol'] = self.config['pipeline']['data_protocol']
                compile_config(self.project, self.base_path, base, dataset)
            print('[CHECK] 全量retrace验收和文件哈希通过；配置与行情轴通过。尚未编码、构建面板或训练。', flush=True)
            return source
        with state_lock(self.pointer):
            if not self.config['pipeline'].get('auto_import_retrace', True) and step != 'import':
                if not self.pointer.exists():
                    raise ValueError('auto_import_retrace=false；先运行scripts/18_import_news.py')
                record = json.loads(self.pointer.read_text(encoding='utf-8'))
                if record.get('source') != source or resolve(self.project, self.config['adapted_snapshot']) != Path(record['snapshot']):
                    raise ValueError('当前全量批次尚未激活；先运行scripts/18_import_news.py')
            self.import_news(source)
            root = Path(self.record['root'])
            with (root / 'launcher.log').open('a', encoding='utf-8', buffering=1) as stream:
                with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                    try:
                        print('[ENTRY] step=%s datasets=%s protocol=%s news_time=%s' %
                              (step, ','.join(self.config['pipeline']['datasets']), self.config['pipeline']['data_protocol'], self.config['panel']['mode']), flush=True)
                        self.record.update(status='running', phase=step, last_error=None)
                        self.save()
                        if step != 'import':
                            with state_lock(destination(self.project, self.config['pipeline']['state'])):
                                pipeline = Pipeline(self.project, self.base_path, fresh)
                                names = STEPS if step == 'all' else PREPARATION_STEPS if step == 'prepare' else (step,)
                                for name in names:
                                    self.record['phase'] = name
                                    self.save()
                                    pipeline.run(name)
                                self.record['artifacts'] = pipeline.state['artifacts']
                                self.record['preparation_root'] = str(pipeline.root)
                                self.record['readiness'] = pipeline.state.get('readiness')
                        self.record['status'] = 'completed' if step in ('all', 'train') else 'prepared' if step == 'prepare' else 'imported' if step == 'import' else 'phase_completed'
                        self.record['phase'] = step
                        self.record['formal_training_requested'] = step in ('all', 'train', 'baseline', 'prepare')
                        self.save()
                        print('[FILE] entry state=' + str(root / 'launcher.json'), flush=True)
                    except BaseException as error:
                        self.record['status'] = 'interrupted_or_failed'
                        self.record['last_error'] = {'phase': self.record.get('phase'), 'type': type(error).__name__, 'error': str(error), 'at': now()}
                        self.save()
                        raise
            return self.record


def main(argv=None, default_step='all'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=None, help='Default: configs/stockmixer.yaml; independent of current working directory')
    parser.add_argument('--step', choices=tuple(dict.fromkeys(ENTRY_STEPS + STEPS + ('templates',))), default=default_step)
    parser.add_argument('--new', action='store_true', help='Create fresh preparation/training artifacts; retain the full imported snapshot and FinBERT cache')
    args = parser.parse_args(argv)
    if args.new and args.step not in ('all', 'prepare'):
        parser.error('--new只适用于all或prepare；分步重跑从19_prepare_news.py --new开始')
    project = Path(__file__).resolve().parents[1]
    return Launcher(project, args.config).run(args.step, args.new)
