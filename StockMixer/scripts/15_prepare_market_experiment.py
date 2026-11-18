"""只准备/检查已配置的单市场新闻对照；不会启动训练、预测、Qwen或FinBERT推理。"""
import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from news_fusion.configuration import compile_config, load_llm_config, read_yaml
from news_fusion.outputs import destination
from news_fusion.pipeline import (Pipeline, prepared_inputs, state_lock,
                                 validate_baseline_source, validate_news_protocol)


def inputs(config_path):
    base = read_yaml(config_path)
    llm_path = Path(base['llm']['config'])
    if not llm_path.is_absolute():
        llm_path = config_path.parent / llm_path
    news = load_llm_config(llm_path)
    validate_news_protocol(news)
    dataset = base['market_name']
    if news['pipeline']['datasets'] != [dataset]:
        raise ValueError('This preparation entry requires pipeline.datasets to contain only market_name')
    if news['fusion']['enabled_channels'] != ['market']:
        raise ValueError('This quick-start entry prepares only the Market channel; use main.py for other experiments')

    def resolve(value):
        value = Path(value)
        return value if value.is_absolute() else PROJECT / value

    # This entry must never fall back to inference or baseline training.
    for label, value in [('baseline_checkpoint', news['baseline_checkpoint']),
                         ('features', news['pipeline']['features']),
                         ('review_artifact', news['pipeline']['review_artifacts'][dataset])]:
        if not value or not resolve(value).exists():
            raise ValueError('Precomputed ' + label + ' is required; this script never trains or runs NLP inference')
    baseline_base = copy.deepcopy(base)
    baseline_base['llm']['enabled'] = False
    baseline_base['data'].update(protocol=news['pipeline']['data_protocol'], prepared_root=None)
    expected = compile_config(PROJECT, config_path, baseline_base, dataset)
    from news_fusion.experiment import load_run_checkpoint
    from news_fusion.protocol import load_market
    parent = load_run_checkpoint(resolve(news['baseline_checkpoint']))
    data = load_market(resolve(expected['workflow']['prepared_data'][dataset]), PROJECT)
    validate_baseline_source(parent, data, expected)
    return base, news


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/stockmixer.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true', help='Rebuild/check offline associations and panels only')
    mode.add_argument('--check', action='store_true', help='Read-only readiness check (default)')
    parser.add_argument('--new', action='store_true', help='Create a new preparation version with --prepare')
    args = parser.parse_args(argv)
    if args.new and not args.prepare:
        parser.error('--new requires --prepare')
    path = args.config.resolve()
    base, news = inputs(path)
    if args.prepare:
        # Explicit allow-list: baseline here may only import the checkpoint
        # validated above. No call to main.py --step all or --step train.
        with state_lock(destination(PROJECT, news['pipeline']['state'])):
            pipeline = Pipeline(PROJECT, path, fresh=args.new)
            for step in ('review', 'publish', 'encode', 'industry_mapping', 'relations',
                         'schedule', 'panels', 'validate', 'baseline', 'prepare'):
                pipeline.run(step)
    try:
        artifacts = prepared_inputs(PROJECT, news, base['market_name'])
        compiled = compile_config(PROJECT, path, base)
    except (ValueError, OSError) as error:
        print(json.dumps({'ready': False, 'training_started': False,
                          'reason': str(error), 'next': 'Run this script with --prepare'},
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({'ready': True, 'training_started': False, 'model_inference_started': False,
        'dataset': base['market_name'], 'channels': compiled['fusion']['enabled_channels'],
        'data_protocol': news['pipeline']['data_protocol'], 'time_mode': news['panel']['mode'],
        'evidence_basis': news.get('market_channel', {}).get('evidence_basis'),
        'experiment_label': 'exploratory_retrospective_original_reproduction',
        'validation_label_overlap_possible': news['pipeline']['data_protocol'] == 'original',
        'panel': artifacts['panel'], 'baseline_checkpoint': artifacts['baseline_checkpoint'],
        'next': 'Run src/train_strict.py when you choose to start training'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit('Market preparation stopped: ' + str(error)) from error
