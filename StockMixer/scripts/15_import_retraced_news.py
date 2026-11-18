"""Import versioned upstream news, optionally activate a fresh preparation configuration."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from news_fusion.configuration import read_yaml,load_llm_config
from news_fusion.retrace_adapter import import_retrace

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/stockmixer.yaml')
    parser.add_argument('--batch',type=Path)
    parser.add_argument('--allow-partial',action='store_true',help='Diagnostic import only; excluded pending articles remain explicit')
    parser.add_argument('--activate',action='store_true',help='Switch llm.yaml to this new snapshot and invalidate old derived caches')
    args=parser.parse_args()
    if args.activate and args.allow_partial:parser.error('Partial diagnostics cannot activate training configuration')
    base=read_yaml(args.config)
    cp=Path(base['llm']['config']);cp=cp if cp.is_absolute() else args.config.resolve().parent/cp
    config=load_llm_config(cp)
    if Path(sys.executable).resolve()!=Path(config['pipeline']['python']).resolve():raise ValueError('Use pipeline.python')
    def resolve(v):
        p=Path(v);return p if p.is_absolute() else PROJECT/p
    batch=args.batch or resolve(config['pipeline']['retrace_artifact'])
    from news_fusion.outputs import destination
    output=destination(PROJECT,dataset='US_GLOBAL',kind='retraced',category='artifacts/news/adapted')
    result=import_retrace(batch,output,resolve(base['data']['axis_dir']),args.allow_partial)
    if args.activate:
        if args.allow_partial:raise ValueError('Partial diagnostics cannot activate training configuration')
        from datetime import datetime
        import shutil,yaml
        tag=datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(cp,cp.with_name('llm.before_retrace_'+tag+'.yaml.bak'))
        # Preserve comments and change only the fields that cannot share a prior snapshot.
        import re
        content=cp.read_text(encoding='utf-8')
        content=re.sub(r'^adapted_snapshot:.*$', 'adapted_snapshot: '+output.relative_to(PROJECT).as_posix(),content,flags=re.M)
        for key in ('industry_batch_artifact','industry_batch_base_snapshot','industry_batch_legacy_snapshot','industry_batch_snapshot_output','semantic_review','industry_scope_review','features'):
            content=re.sub(r'^  '+key+r':.*$', '  '+key+': null',content,flags=re.M)
        content=re.sub(r'^  state:.*$', '  state: outputs/artifacts/news/'+tag+'_retrace_state.json',content,flags=re.M)
        for section in ('review_artifacts','panels','market_routes'):
            pattern=(r'(?m)^  review_artifacts:\n(?:    .*(?:\n|$))+' if section=='review_artifacts' else r'(?m)^'+section+r':[^\n]*\n(?:  (?:NASDAQ|NYSE|SP500):[^\n]*(?:\n|$))+')
            prefix='  ' if section=='review_artifacts' else ''
            replacement=prefix+section+':\n'+''.join(prefix+'  '+n+': null\n' for n in ('NASDAQ','NYSE','SP500'))
            content=re.sub(pattern,replacement,content)
        # Validate the staged configuration before committing it.
        temporary=cp.with_suffix('.yaml.tmp');temporary.write_text(content,encoding='utf-8')
        load_llm_config(temporary);temporary.replace(cp)
    print(json.dumps({'output':str(output),'result':result,'activated':args.activate},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
