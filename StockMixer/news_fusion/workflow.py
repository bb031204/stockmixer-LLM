"""Config-driven training, per-epoch testing and best-checkpoint prediction."""
import argparse
import hashlib
import json
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from . import experiment
from .io import digest_file, now, write_json
from .outputs import destination, run_destination, Tee
from .configuration import configuration_sources
from .console import display_path
from .protocol import PROJECT, load_market as MarketData, configuration, prepare_market, resolve


def run_workflow(config_path=None, project=None):
    project=Path(project or PROJECT).resolve()
    source=resolve(project,config_path or 'configs/stockmixer.yaml')
    source_files=configuration_sources(source)
    source_payloads={str(path):path.read_bytes() for path in source_files}
    payload=source_payloads[str(source)]
    project,config=configuration(source,project=project)
    if any(digest_file(path)!=hashlib.sha256(value).hexdigest() for path,value in source_payloads.items()):
        raise ValueError('Configuration changed during loading; retry with a fixed config.')
    if 'workflow' not in config:
        raise ValueError('Single-entry training requires the workflow section in config.')
    settings=config['workflow']
    dataset=config['dataset']
    kind='fusion' if settings['mode']=='fusion' else 'E0'
    if settings['smoke']:
        kind+='_smoke'
    output=run_destination(project,settings['output_directory'],dataset=dataset,kind=kind)
    prediction_output=output
    if output.exists():
        raise FileExistsError('Training/prediction output already exists; choose a new run.')
    data_dir=destination(project,settings['prepared_data'][dataset])
    if data_dir==output or data_dir in output.parents or output in data_dir.parents:
        raise ValueError('Training output must be separate from prepared data.')
    report=destination(project,Path('reports')/(output.name+'_workflow'))
    report.mkdir(parents=True,exist_ok=False)
    (report/'config.source.yaml').write_bytes(payload)
    write_json(report/'config.resolved.json',config)
    for index,path in enumerate(source_files):
        (report/(str(index)+'_'+path.name)).write_bytes(source_payloads[str(path)])
    write_json(report/'configuration_sources.json',{str(path):hashlib.sha256(source_payloads[str(path)]).hexdigest() for path in source_files})
    state={'schema_version':'stockmixer_workflow_v1','status':'in_progress','phase':'preflight',
        'started_at':now(),'dataset':dataset,'mode':settings['mode'],'smoke_only':settings['smoke'],
        'configuration_source':str(source),'configuration_sha256':hashlib.sha256(payload).hexdigest(),
        'data_directory':str(data_dir),'training_directory':str(output),
        'prediction_directory':str(prediction_output) if settings['prediction']['enabled'] else None,
        'training_completed':False,'prediction_completed':False,'prediction_enabled':settings['prediction']['enabled']}

    def record(phase=None):
        if phase is not None:
            state['phase']=phase
        state['updated_at']=now()
        write_json(report/'workflow.json',state)

    record()
    with (report/'workflow.log').open('w',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            try:
                if not data_dir.exists():
                    if not settings['prepare_if_missing']:
                        raise FileNotFoundError('Prepared market data is missing: '+str(data_dir))
                    record('preparing_data')
                    prepare_market(project,config,data_dir)
                data=MarketData(data_dir,project)
                if data.axis.dataset!=dataset:
                    raise ValueError('Configured dataset differs from prepared market data.')
                prediction=settings['prediction']
                as_of=data.axis.dates[-1] if prediction['as_of']=='latest' else prediction['as_of']
                if prediction['enabled']:
                    if as_of not in data.axis.dates:
                        raise ValueError('Prediction as_of must be on the verified dataset date axis.')
                    if data.axis.dates.index(as_of)<data.split.lookback-1:
                        raise ValueError('Prediction date lacks a complete input window.')
                    if as_of<=data.axis.dates[data.axis.valid_end-1] and not prediction['allow_in_sample_replay']:
                        raise ValueError('Prediction date is within training/selection; choose a later date or explicitly allow replay.')
                state['prediction_as_of']=as_of if prediction['enabled'] else None
                baseline=None
                panel=None
                if settings['mode']=='fusion':
                    baseline=resolve(project,settings['baseline_checkpoint'])
                    panel=resolve(project,config['news']['panel'])
                    verified_panel=experiment.load_panel(panel,data,config)
                    parent=experiment.load_run_checkpoint(baseline)
                    experiment.validate_checkpoint(parent,data)
                    if parent['mode']!='baseline' or (parent['smoke_only'] and not settings['smoke']):
                        raise ValueError('Fusion requires an eligible separately trained baseline checkpoint.')
                    del verified_panel
                del data
                record('training')
                training=experiment.train(project,config,data_dir,output,smoke=settings['smoke'],
                    baseline=baseline,panel_path=panel)
                state['training_completed']=True
                state['protocol']=training['protocol']
                state['best_epoch']=training['best_epoch']
                state['final_test_evaluations']=training['final_test_evaluations']
                state['test_policy']=training.get('test_policy','final_best_once')
                state['test_evaluations']=training.get('test_evaluations',1)
                state['epoch_test_evaluations']=training.get('epoch_test_evaluations',0)
                state['metric_protocol']=training.get('metric_protocol','valid_only')
                checkpoint=output/'checkpoints/best_model.pt'
                state['best_checkpoint']=str(checkpoint)
                state['best_checkpoint_sha256']=digest_file(checkpoint)
                record('training_completed')
                if prediction['enabled']:
                    record('predicting')
                    result=experiment.predict(checkpoint,prediction_output,as_of=prediction['as_of'],
                        device=config['training']['device'],allow_replay=prediction['allow_in_sample_replay'],attach_to_training=True)
                    state['prediction_completed']=True
                    state['prediction_csv']=str(prediction_output/'predictions/predictions.csv')
                    state['prediction_rows']=result['rows']
                    state['prediction_valid_rows']=result['valid_rows']
                    state['prediction_target_date']=result['target_date']
                state['status']='completed'
                state['completed_at']=now()
                record('completed')
                if state['prediction_completed']:
                    print('[FILE  ] predictions='+str(display_path(state['prediction_csv'])),flush=True)
                print('[FILE  ] workflow='+str(display_path(report/'workflow.json')),flush=True)
            except BaseException as error:
                state['status']='interrupted_or_failed'
                state['error_type']=type(error).__name__
                state['error']=str(error)
                record()
                traceback.print_exc(file=log)
                raise
    return state


def main(argv=None):
    parser=argparse.ArgumentParser(description='Read config, train, evaluate test by policy, then predict using the best checkpoint.')
    parser.add_argument('--config',type=Path,default=None,help='Config path relative to the project; default configs/stockmixer.yaml.')
    args=parser.parse_args(argv)
    try:
        run_workflow(args.config)
    except (ValueError,OSError,ImportError,KeyError) as error:
        raise SystemExit('StockMixer workflow stopped: '+str(error)) from error


if __name__=='__main__':
    main()
