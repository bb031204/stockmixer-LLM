"""Import a completed upstream batch using paths in configs/llm.yaml."""
import argparse
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from news_fusion.configuration import load_llm_config
from news_fusion.legacy_industry import import_batch,validate_enriched

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(PROJECT/'configs/llm.yaml'))
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    config=load_llm_config(args.config);settings=config['pipeline']
    def resolve(value):
        if not value:raise ValueError('Configure all industry_batch_* paths before importing')
        path=Path(value);return path if path.is_absolute() else PROJECT/path
    output=resolve(settings['industry_batch_snapshot_output'])
    if args.check or output.exists():print(validate_enriched(output))
    else:
        result=import_batch(resolve(settings['industry_batch_base_snapshot']),resolve(settings['industry_batch_artifact']),
            resolve(config['industry_dictionary']['directory']),output,
            legacy_snapshot=resolve(settings['industry_batch_legacy_snapshot']) if settings['industry_batch_legacy_snapshot'] else None)
        print({k:result[k] for k in ('kind','articles','batch_summary')})
    print('Import is complete. Existing experiments are unchanged; bind the new snapshot and review artifacts before preparing panels.')

if __name__=='__main__':main()
