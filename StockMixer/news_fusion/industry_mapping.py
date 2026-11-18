"""Exact dataset-axis tables, separating dated SEC evidence from undated reference labels."""
import csv
import json
import tarfile
import io
from collections import Counter,defaultdict
from pathlib import Path
from .industry_dictionary import IndustryDictionary
from .io import digest,digest_file,file_manifest,new_artifact,now,read_records,write_records,write_json,verify_files
from .relations import validate_master
from .time_axis import Axis

_CHECKED_SEC={}

def verify_master_artifact(directory):
    directory=Path(directory)
    path=directory/'manifest.json'
    if not path.exists():return
    before=digest_file(path);manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('kind')!='sec_sic_dated_historical_observations':return
    verify_files(directory,manifest)
    # Always hash source evidence; only parsing identical evidence is cached within this process.
    if _CHECKED_SEC.get(str(directory))!=before:
        from .master_data import verify_observations
        verify_observations(directory)
        _CHECKED_SEC[str(directory)]=before
    if digest_file(path)!=before:raise ValueError('Historical primary manifest changed during validation')

def master_inputs(project,config):
    def path(v):
        p=Path(v);return p if p.is_absolute() else Path(project)/p
    names=('security_master','industry_memberships','industry_aliases')
    paths=[path(config['pipeline'][k]) if config['pipeline'][k] else None for k in names]
    pins={str(p):digest_file(p) for p in paths if p}
    for parent in {p.parent for p in paths if p}:
        if (parent/'manifest.json').exists():pins[str(parent/'manifest.json')]=digest_file(parent/'manifest.json')
    for parent in {p.parent for p in paths if p}:verify_master_artifact(parent)
    tables=[read_records(p) if p else [] for p in paths]
    if any(digest_file(p)!=h for p,h in pins.items()):raise ValueError('Historical inputs changed while being read')
    return tables,pins

def reference_inputs(archive,sp,dictionary):
    """Parse the exact bytes whose hashes are recorded, never reopen a live version."""
    archive=Path(archive);sp=Path(sp)
    archive_bytes=archive.read_bytes();sp_bytes=sp.read_bytes()
    import hashlib
    archive_hash=hashlib.sha256(archive_bytes).hexdigest();sp_hash=hashlib.sha256(sp_bytes).hexdigest()
    references={};company_names={}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes),mode='r:gz') as a:
        for dataset in ('NASDAQ','NYSE'):
            data=json.loads(a.extractfile('relation/sector_industry/'+dataset+'_industry_ticker.json').read());by_ticker=defaultdict(list)
            for label,tickers in data.items():
                if label.casefold() in ('n/a','na','unknown','unspecified'):continue
                rows=[r for r in dictionary.rows if r['industry_standard']=='TRSR-INDUSTRY' and r['name']==label and r['source_sha256']==archive_hash]
                if len(rows)!=1:raise ValueError('Reference labels differ from pinned dictionary')
                for ticker in tickers:by_ticker[ticker].append(rows[0]['key'])
            references[dataset]=by_ticker
    by_ticker=defaultdict(list)
    for item in csv.DictReader(io.StringIO(sp_bytes.decode('utf-8-sig'))):
        rows=[r for r in dictionary.rows if r['industry_standard']=='SP500-SECTOR' and r['name']==item['Sector'] and r['source_sha256']==sp_hash]
        if len(rows)!=1:raise ValueError('SP500 sector source differs from dictionary')
        by_ticker[item['Symbol']].append(rows[0]['key']);company_names[item['Symbol']]=item['Name']
    references['SP500']=by_ticker
    return references,company_names,{'archive_sha256':archive_hash,'sp500_csv_sha256':sp_hash}


def build_mapping(project,config,base):
    project=Path(project).resolve();settings=config['industry_dictionary']
    def path(v):
        p=Path(v);return p if p.is_absolute() else project/p
    dictionary=IndustryDictionary(path(settings['directory']))
    sp=path(base['data']['dataset_root'])/'SP500/sp500_ticker.csv'
    references,company_names,reference_pins=reference_inputs(path(settings['reference_archive']),sp,dictionary)
    tables,input_pins=master_inputs(project,config)
    master_pins={config['pipeline'][k]:input_pins[str(path(config['pipeline'][k]))] for k in ('security_master','industry_memberships') if config['pipeline'][k]}
    from .outputs import destination
    output=destination(project,settings['output']);summary={};all_rows=[];all_members=[]
    with new_artifact(output) as directory:
        reference_all=[]
        for dataset in ('NASDAQ','NYSE','SP500'):
            axis=Axis.load(path(base['data']['axis_dir'])/(dataset+'.json'))
            securities,members,_=validate_master(axis,*tables);rows=[]
            for i,uid in enumerate(axis.stock_uids):
                ticker=axis.source.get('tickers',[s.split(':',1)[1] for s in axis.stock_uids])[i]
                keys=sorted(set(references[dataset].get(ticker,[])))
                if len(keys)>1:raise ValueError('Reference primary industry conflict for '+uid)
                observations=[m for m in members if m['stock_uid']==uid]
                for m in observations:
                    key=(m['industry_standard'],m['industry_version'],m['industry_code'])
                    if key not in dictionary.by_code:raise ValueError('Verified membership code is absent from dictionary')
                    all_members.append(dict(m,industry_key=dictionary.by_code[key]['key']))
                row={'dataset':dataset,'stock_uid':uid,'array_index':i,'dataset_ticker':ticker,
                     'reference_company_name':company_names.get(ticker) if dataset=='SP500' else None,
                     'reference_industry_key':keys[0] if keys else None,
                     'reference_industry_name':dictionary.by_key[keys[0]]['name'] if keys else None,
                     'reference_status':'undated_reference_not_training_eligible' if keys else 'unmatched',
                     'verified_observation_count':len(observations),
                     'verified_industry_keys':sorted({dictionary.by_code[(m['industry_standard'],m['industry_version'],m['industry_code'])]['key'] for m in observations}),
                     'continuous_historical_membership_verified':False}
                rows.append(row)
            assert len(rows)==len(axis.stock_uids) and [r['array_index'] for r in rows]==list(range(len(rows)))
            write_records(directory/(dataset+'_stock_industry.jsonl'),rows)
            grouped=defaultdict(list)
            for row in rows:
                if row['reference_industry_key']:
                    grouped[row['reference_industry_key']].append({k:row[k] for k in ('stock_uid','array_index','dataset_ticker')})
            write_json(directory/(dataset+'_industry_to_stocks.json'),{'dataset':dataset,'status':'undated_reference_not_training_eligible',
                'dictionary_sha256':dictionary.sha256,'industries':dict(grouped)})
            with (directory/(dataset+'_stock_industry.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
                for r in rows:writer.writerow(dict(r,verified_industry_keys=json.dumps(r['verified_industry_keys'],ensure_ascii=False)))
            from .reference_membership import memberships as reference_members
            reference_rows=reference_members(axis,dictionary,path(settings['reference_archive']),sp,dictionary.manifest['created_at'])
            reference_all.extend(reference_rows)
            write_records(directory/(dataset+'_reference_experiment_memberships.jsonl'),reference_rows)
            summary[dataset]={'stocks':len(rows),'reference_matched_stocks':sum(bool(r['reference_industry_key']) for r in rows),
                'reference_unmatched_stocks':sum(not r['reference_industry_key'] for r in rows),
                'stocks_with_verified_dated_observations':sum(bool(r['verified_observation_count']) for r in rows),
                'stock_axis_hash':axis.stock_hash,'date_axis_hash':axis.date_hash}
            all_rows.extend(rows)
        write_records(directory/'reference_experiment_memberships.jsonl',reference_all)
        write_records(directory/'stock_industry_index.jsonl',all_rows)
        write_records(directory/'verified_industry_memberships.jsonl',all_members)
        write_records(directory/'verified_security_master.jsonl',tables[0])
        if any(digest_file(p)!=h for p,h in input_pins.items()):raise ValueError('Historical inputs changed during mapping publication')
        if digest_file(path(settings['reference_archive']))!=reference_pins['archive_sha256'] or digest_file(sp)!=reference_pins['sp500_csv_sha256']:
            raise ValueError('Reference sources changed during mapping publication')
        write_json(directory/'manifest.json',{'kind':'dataset_industry_axis_mapping','created_at':now(),
            'dictionary_sha256':dictionary.sha256,'dictionary_version':dictionary.manifest['version'],
            'datasets':summary,'master_input_sha256':master_pins,
            'reference_sources':reference_pins,
            'undated_references_enabled_for_training':False,'model_modified':False,'finbert_inference_run':False,
            'files':file_manifest(directory)})
    return summary


def verified_mapping(directory,dictionary,axes,securities,memberships,references=None,company_names=None):
    """A display/reference row can never promote itself into an accepted membership."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_files(directory,manifest,required=('stock_industry_index.jsonl','verified_security_master.jsonl','verified_industry_memberships.jsonl'))
    if manifest.get('kind')!='dataset_industry_axis_mapping' or manifest.get('dictionary_sha256')!=dictionary.sha256 or manifest.get('undated_references_enabled_for_training') is not False:
        raise ValueError('Mapping dictionary/version or reference training policy mismatch')
    if read_records(directory/'verified_security_master.jsonl')!=securities:
        raise ValueError('Mapping securities differ from verified historical input')
    expected=[];views=[]
    index=read_records(directory/'stock_industry_index.jsonl')
    if any(r.get('dataset') not in manifest.get('datasets',{}) for r in index):raise ValueError('Mapping index has an undeclared dataset')
    for name,axis in axes.items():
        metadata=manifest.get('datasets',{}).get(name,{})
        if metadata.get('stock_axis_hash')!=axis.stock_hash or metadata.get('date_axis_hash')!=axis.date_hash:
            raise ValueError('Mapping axis mismatch: '+name)
        rows=[r for r in index if r['dataset']==name]
        if len(rows)!=len(axis.stock_uids) or any(type(r.get('array_index')) is not int for r in rows) or [(r['stock_uid'],r['array_index']) for r in rows]!=list(zip(axis.stock_uids,range(len(axis.stock_uids)))):
            raise ValueError('One-to-one stock mapping order mismatch')
        _,members,_=validate_master(axis,securities,memberships,[])
        views.append((name,axis,rows,members))
        # Builder groups observations by stock axis order.
        for uid in axis.stock_uids:
            for m in [v for v in members if v['stock_uid']==uid]:
                key=(m['industry_standard'],m['industry_version'],m['industry_code'])
                if key not in dictionary.by_code:raise ValueError('Historical industry code absent from dictionary')
                expected.append(dict(m,industry_key=dictionary.by_code[key]['key']))
    if [r for r in read_records(directory/'verified_industry_memberships.jsonl') if r['dataset'] in axes]!=expected:
        raise ValueError('Mapping membership is not independently supported by dated input evidence')
    for name,axis,rows,members in views:
        required=(name+'_stock_industry.jsonl',name+'_stock_industry.csv',name+'_industry_to_stocks.json')
        if not set(required)<=set(manifest['files']):raise ValueError('Mapping display/proof tables are incomplete')
        if read_records(directory/required[0])!=rows:raise ValueError('Per-market JSONL differs from the stock index')
        grouped=defaultdict(list)
        for i,row in enumerate(rows):
            ticker=axis.source.get('tickers',[s.split(':',1)[1] for s in axis.stock_uids])[i]
            if row.get('dataset_ticker')!=ticker:raise ValueError('Mapping dataset ticker differs from the stock axis')
            obs=[m for m in members if m['stock_uid']==axis.stock_uids[i]]
            keys=sorted({dictionary.by_code[(m['industry_standard'],m['industry_version'],m['industry_code'])]['key'] for m in obs})
            if type(row.get('verified_observation_count')) is not int or row['verified_observation_count']!=len(obs) or row.get('verified_industry_keys')!=keys or row.get('continuous_historical_membership_verified') is not False:
                raise ValueError('Mapping display overstates historical verification')
            key=row.get('reference_industry_key')
            if key is not None and key not in dictionary.by_key:raise ValueError('Unknown reference industry key')
            if row.get('reference_industry_name')!=(dictionary.by_key[key]['name'] if key else None) or row.get('reference_status')!=('undated_reference_not_training_eligible' if key else 'unmatched'):
                raise ValueError('Reference label/status differs from the dictionary')
            if references is not None:
                choices=sorted(set(references[name].get(ticker,[])))
                if len(choices)>1 or key!=(choices[0] if choices else None):raise ValueError('Reference stock industry differs from original dataset evidence')
                if row.get('reference_company_name')!=((company_names or {}).get(ticker) if name=='SP500' else None):
                    raise ValueError('Reference company name differs from original dataset evidence')
            if key:grouped[key].append({k:row[k] for k in ('stock_uid','array_index','dataset_ticker')})
        expected_reverse={'dataset':name,'status':'undated_reference_not_training_eligible','dictionary_sha256':dictionary.sha256,'industries':dict(grouped)}
        if json.loads((directory/required[2]).read_text(encoding='utf-8'))!=expected_reverse:raise ValueError('Industry reverse index differs from stock rows')
        with (directory/required[1]).open(encoding='utf-8-sig',newline='') as f:
            csv_rows=list(csv.DictReader(f))
        cells=lambda r:{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else '' if v is None else str(v) for k,v in r.items()}
        if csv_rows!=[cells(r) for r in rows]:raise ValueError('Mapping CSV differs from JSON stock rows')
        counts={'stocks':len(rows),'reference_matched_stocks':sum(bool(r['reference_industry_key']) for r in rows),
            'reference_unmatched_stocks':sum(not r['reference_industry_key'] for r in rows),
            'stocks_with_verified_dated_observations':sum(bool(r['verified_observation_count']) for r in rows)}
        if any(manifest['datasets'][name].get(k)!=v for k,v in counts.items()):raise ValueError('Mapping coverage counts contradict stock rows')
    return manifest


def ensure_mapping(project,config,base):
    project=Path(project).resolve();settings=config.get('industry_dictionary')
    if settings is None:return None
    def path(v):
        p=Path(v);return p if p.is_absolute() else project/p
    from .outputs import destination
    output=destination(project,settings['output'])
    if not output.exists():return build_mapping(project,config,base)
    dictionary=IndustryDictionary(path(settings['directory']))
    axes={n:Axis.load(path(base['data']['axis_dir'])/(n+'.json')) for n in config['pipeline']['datasets']}
    tables,_=master_inputs(project,config)
    references,names,ref_pins=reference_inputs(path(settings['reference_archive']),path(base['data']['dataset_root'])/'SP500/sp500_ticker.csv',dictionary)
    manifest=verified_mapping(output,dictionary,axes,tables[0],tables[1],references,names)
    if manifest.get('reference_sources')!=ref_pins:
        raise ValueError('Reference source changed; select a new mapping output version')
    return manifest['datasets']
