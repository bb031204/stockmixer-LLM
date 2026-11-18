"""M1：为现有市场审核中从未审核（unreviewed_*）的记录补跑本地 Qwen 审核。

只处理这些记录；Prompt、模型 digest、阈值沿用 configs/llm.yaml 的 review 段。
第一个数据集调用模型，其余数据集复用逐文章缓存（不再请求）。结果写入新版本目录，
不修改已有审核产物。完成后把输出路径填入 llm.yaml 的 pipeline.review_artifacts，
并用 main.py --new 新建准备版本。

  python scripts/13_complete_market_review.py --check   # 只统计，不调用模型
  python scripts/13_complete_market_review.py           # 正式补审核
"""
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from news_fusion.configuration import read_yaml, load_llm_config
from news_fusion.io import read_records
from news_fusion.market_review_completion import complete_review, unreviewed_ids
from news_fusion.offline_inputs import encoding_inputs
from news_fusion.outputs import destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/stockmixer.yaml')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--limit', type=int, help='仅用于小样本预检查；正式补审核不要设置')
    args = parser.parse_args(argv)
    base = read_yaml(args.config)
    llm_path = Path(base['llm']['config'])
    config = load_llm_config(llm_path if llm_path.is_absolute() else args.config.resolve().parent/llm_path)
    if Path(sys.executable).resolve() != Path(config['pipeline']['python']).resolve():
        raise SystemExit('Use pipeline.python from config: ' + config['pipeline']['python'])
    articles, _, provenance = encoding_inputs(PROJECT, config)
    sources = {}
    for name in config['pipeline']['datasets']:
        value = config['pipeline']['review_artifacts'][name]
        if not value:
            raise SystemExit('pipeline.review_artifacts.' + name + ' is not configured')
        sources[name] = Path(value) if Path(value).is_absolute() else PROJECT/value
    if args.check:
        print(json.dumps({name: {'source': str(path), 'unreviewed': len(unreviewed_ids(read_records(path/'market_review.jsonl')))}
                          for name, path in sources.items()}, ensure_ascii=False, indent=2))
        return
    from news_fusion.auto_review import OllamaReviewer
    backend = OllamaReviewer(config['review'])
    cache = destination(PROJECT, config['review']['cache_dir'])
    results = {}
    for name, source in sources.items():
        output = destination(PROJECT, dataset=name, kind='market_review_v3', category='artifacts/news/routing_reviews')
        complete_review(source, articles, name, output, cache, backend, config['review'], provenance, args.limit,
            progress=lambda n, total, s, name=name: print('[M1] %s %d/%d calls=%d cache=%d failed=%d' % (
                name, n, total, s.get('model_calls', 0), s.get('cache_hits', 0), s.get('failed_articles', 0)), flush=True))
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        results[name] = {'output': str(output), 'decisions': manifest['decisions'], 'new_decisions': manifest['new_decisions']}
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
