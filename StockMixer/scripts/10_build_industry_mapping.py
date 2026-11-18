"""Build industry dictionary to all three dataset axes; never infer missing historical facts."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from news_fusion.configuration import read_yaml,load_llm_config
from news_fusion.industry_mapping import ensure_mapping

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=PROJECT/'configs/stockmixer.yaml')
    args=parser.parse_args(argv);base=read_yaml(args.config);p=Path(base['llm']['config'])
    config=load_llm_config(p if p.is_absolute() else args.config.resolve().parent/p)
    if Path(sys.executable).resolve()!=Path(config['pipeline']['python']).resolve():raise ValueError('Use pipeline.python')
    result=ensure_mapping(PROJECT,config,base)
    print(json.dumps({'output':config['industry_dictionary']['output'],'datasets':result},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
