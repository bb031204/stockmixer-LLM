"""Verify the configured semantic sample and its actual human-confirmation scope."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from news_fusion.configuration import load_llm_config
from news_fusion.io import read_records,digest_file
from news_fusion.semantic_review import load_sample,apply_sample
from news_fusion.offline_inputs import encoding_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/llm.yaml')
    args=parser.parse_args();project=Path(__file__).resolve().parents[1]
    config=load_llm_config(project/args.config)
    path=config['pipeline'].get('semantic_review')
    if not path:raise ValueError('Configure pipeline.semantic_review before checking semantic samples')
    source=Path(config['adapted_snapshot']);source=source if source.is_absolute() else project/source
    articles,_,_=encoding_inputs(project,config)
    root=Path(path);root=root if root.is_absolute() else project/root
    proof=load_sample(root,articles)
    _,summary=apply_sample(articles,proof['original_reviews'],proof['rows'],proof['confirmation'])
    print(json.dumps(summary,ensure_ascii=True,indent=2))
    if config['pipeline'].get('industry_scope_review'):
        from news_fusion.industry_scope_review import apply_scope_reviews
        from news_fusion.pipeline import verify_artifact
        inference=Path(config['pipeline']['industry_inference_artifact'])
        inference=inference if inference.is_absolute() else project/inference
        verify_artifact(inference)
        proposed=read_records(inference/'articles.enriched.jsonl')
        scope=Path(config['pipeline']['industry_scope_review']);scope=scope if scope.is_absolute() else project/scope
        _,industry_summary=apply_scope_reviews(proposed,scope,config['pipeline']['require_industry_scope_review'],
            digest_file(source/'manifest.json'))
        print(json.dumps({'industry_scope':industry_summary},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
