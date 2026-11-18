"""Audit current snapshot and producer analysis without model calls or raw-data writes."""
import argparse
import json
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from news_fusion.configuration import read_yaml,load_llm_config
from news_fusion.mapping_audit import audit_mapping
from news_fusion.outputs import destination

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/stockmixer.yaml')
    args=parser.parse_args(argv)
    base=read_yaml(args.config)
    llm_path=Path(base['llm']['config'])
    if not llm_path.is_absolute():llm_path=args.config.resolve().parent/llm_path
    config=load_llm_config(llm_path)
    if Path(sys.executable).resolve()!=Path(config['pipeline']['python']).resolve():
        raise ValueError('Use pipeline.python from config')
    axes=Path(base['data']['axis_dir'])
    if not axes.is_absolute():axes=PROJECT/axes
    output=destination(PROJECT,dataset='_'.join(config['pipeline']['datasets']),kind='mapping_audit',category='reports')
    report=audit_mapping(PROJECT,config,axes,output)
    print(json.dumps({'output':str(output),'model_calls':0,'active_articles':report['active_articles'],
                      'upstream_unique_articles':report['upstream_unique_articles'],'datasets':report['datasets']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
