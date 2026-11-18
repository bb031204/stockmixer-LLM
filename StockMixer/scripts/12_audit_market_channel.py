"""市场通道审计（只读、不调用模型）：审核漏斗报告，及可选的 M3 人工评估样本 / 评分。

用法：
  python scripts/12_audit_market_channel.py                     # 漏斗报告
  python scripts/12_audit_market_channel.py --evaluation-sample  # 额外生成待人工标注的分层样本
  python scripts/12_audit_market_channel.py --score SAMPLE_DIR --labels FILLED.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from news_fusion.configuration import read_yaml, load_llm_config
from news_fusion.io import new_artifact, now, read_records, write_json
from news_fusion.market_quality import funnel
from news_fusion.offline_inputs import encoding_inputs
from news_fusion.outputs import destination
from news_fusion.time_axis import Axis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/stockmixer.yaml')
    parser.add_argument('--evaluation-sample', action='store_true')
    parser.add_argument('--per-stratum', type=int, default=40)
    parser.add_argument('--score', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args(argv)
    base = read_yaml(args.config)
    llm_path = Path(base['llm']['config'])
    config = load_llm_config(llm_path if llm_path.is_absolute() else args.config.resolve().parent/llm_path)
    if args.score:
        from news_fusion.market_evaluation import evaluate
        if not args.labels:
            raise SystemExit('--score requires --labels')
        output = destination(PROJECT, dataset='US', kind='market_evaluation_score', category='reports')
        print(json.dumps(evaluate(args.score, args.labels, output), ensure_ascii=False, indent=2))
        return
    articles, _, provenance = encoding_inputs(PROJECT, config)
    axes = Path(base['data']['axis_dir'])
    axes = axes if axes.is_absolute() else PROJECT/axes
    report = {'created_at': now(), 'model_calls': 0, 'snapshot': provenance, 'datasets': {},
              'note': 'Split assignment uses the publication date (date-only shifted to the next session); panel market_quality is authoritative.'}
    reviews = {}
    for name in config['pipeline']['datasets']:
        path = config['pipeline']['review_artifacts'][name]
        if not path:
            report['datasets'][name] = {'error': 'pipeline.review_artifacts not configured'}
            continue
        path = Path(path) if Path(path).is_absolute() else PROJECT/path
        reviews[name] = read_records(path/'market_review.jsonl')
        report['datasets'][name] = {'review_artifact': str(path),
            'funnel': funnel(articles, reviews[name], Axis.load(axes/(name+'.json')))}
    output = destination(PROJECT, dataset='_'.join(config['pipeline']['datasets']), kind='market_channel_audit', category='reports')
    with new_artifact(output) as directory:
        write_json(directory/'market_funnel.json', report)
    print(json.dumps({'output': str(output), 'datasets': {k: v.get('funnel', v) for k, v in report['datasets'].items()}}, ensure_ascii=False, indent=2))
    if args.evaluation_sample:
        from news_fusion.market_evaluation import build_sample
        first = next(iter(reviews))
        sample = destination(PROJECT, dataset='US', kind='market_evaluation_sample', category='artifacts/news/market_review_evaluation')
        build_sample(articles, reviews[first], sample, args.per_stratum, provenance=provenance)
        print('[M3] 待人工标注样本：' + str(sample) + '（复制 sample.jsonl 后填写 human_label 等字段）')


if __name__ == '__main__':
    main()
