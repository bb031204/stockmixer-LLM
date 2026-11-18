"""M4：把上游采集的官方宏观发布（FOMC/BLS/BEA）校验、版本化并做 FinBERT 离线编码。

本脚本不联网。输入 JSONL 由上游 U 采集，每行字段见 news_fusion/official_events.py 的 REQUIRED；
只接受首次发布文本和官方页面给出的发布时间。

  python scripts/14_build_official_events.py --source D:/financial_data_project2/data_enriched/official_events/<版本>.jsonl
完成后把输出的 features 目录填入 llm.yaml 的 market_channel.official_events。
"""
import argparse
import json
import sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from news_fusion.configuration import read_yaml, load_llm_config
from news_fusion.official_events import build_events, encode_events
from news_fusion.outputs import destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/stockmixer.yaml')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    base = read_yaml(args.config)
    llm_path = Path(base['llm']['config'])
    config = load_llm_config(llm_path if llm_path.is_absolute() else args.config.resolve().parent/llm_path)
    events = destination(PROJECT, dataset='US', kind='official_events', category='artifacts/news/official_events')
    manifest = build_events(args.source, events)
    print(json.dumps({'events': str(events), 'articles': manifest['articles'], 'event_types': manifest['event_types']}, ensure_ascii=False, indent=2))
    if args.validate_only:
        return
    from news_fusion.encoder import LocalFinBert
    from news_fusion.io import digest_file
    f = config['finbert']
    model = Path(f['model_dir'] or config['pipeline']['model_directory'])
    installed = json.loads((model/'download_manifest.json').read_text(encoding='utf-8'))
    for name, sha in installed['files'].items():
        if digest_file(model/name) != sha:
            raise SystemExit('Local FinBERT installation changed: ' + name)
    backend = LocalFinBert(model, f['device'], f['max_length'], installed['model_available_at'])
    features = events.parent/(events.name+'_features')
    encode_events(events, backend, features, destination(PROJECT, config['pipeline']['cache']), f['batch_size'])
    print('[M4] 编码完成；把 market_channel.official_events 设为：' + str(features))


if __name__ == '__main__':
    main()
