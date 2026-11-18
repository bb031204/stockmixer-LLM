import copy,csv,json,tempfile,unittest
from collections import defaultdict
from pathlib import Path
from news_fusion.industry_dictionary import IndustryDictionary
from news_fusion.industry_mapping import verified_mapping
from news_fusion.io import digest,write_json,write_records,file_manifest
from tests_news.test_historical_relations import axis,security

ROOT=Path(__file__).resolve().parents[1]

class MappingViewTests(unittest.TestCase):
    def setUp(self):
        self.dictionary=IndustryDictionary(ROOT/'news_assets/industry_dictionary')
        self.a=axis();self.a.stock_hash=digest(self.a.stock_uids);self.a.date_hash=digest(['2014-01-02'])
        self.keys=[r['key'] for r in self.dictionary.rows if r['industry_standard']=='TRSR-INDUSTRY'][:3]
        self.rows=[]
        for i,uid in enumerate(self.a.stock_uids):
            self.rows.append({'dataset':'NASDAQ','stock_uid':uid,'array_index':i,'dataset_ticker':uid.split(':')[1],
                'reference_company_name':None,'reference_industry_key':self.keys[i],
                'reference_industry_name':self.dictionary.by_key[self.keys[i]]['name'],
                'reference_status':'undated_reference_not_training_eligible','verified_observation_count':0,
                'verified_industry_keys':[],'continuous_historical_membership_verified':False})

    def package(self,p,rows):
        write_records(p/'stock_industry_index.jsonl',rows);write_records(p/'NASDAQ_stock_industry.jsonl',rows)
        write_records(p/'verified_security_master.jsonl',[security()]);write_records(p/'verified_industry_memberships.jsonl',[])
        with (p/'NASDAQ_stock_industry.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            for row in rows:w.writerow(dict(row,verified_industry_keys=json.dumps(row['verified_industry_keys'])))
        grouped=defaultdict(list)
        for row in rows:grouped[row['reference_industry_key']].append({k:row[k] for k in ('stock_uid','array_index','dataset_ticker')})
        write_json(p/'NASDAQ_industry_to_stocks.json',{'dataset':'NASDAQ','status':'undated_reference_not_training_eligible',
            'dictionary_sha256':self.dictionary.sha256,'industries':dict(grouped)})
        self.manifest={'kind':'dataset_industry_axis_mapping','dictionary_sha256':self.dictionary.sha256,
            'undated_references_enabled_for_training':False,'datasets':{'NASDAQ':{'stock_axis_hash':self.a.stock_hash,
                'date_axis_hash':self.a.date_hash,'stocks':2,'reference_matched_stocks':2,'reference_unmatched_stocks':0,
                'stocks_with_verified_dated_observations':0}},'files':file_manifest(p)}
        write_json(p/'manifest.json',self.manifest)
    def check(self,p,references=None):return verified_mapping(p,self.dictionary,{'NASDAQ':self.a},[security()],[],references=references,company_names={})
    def rehash(self,p):
        self.manifest['files']=file_manifest(p);write_json(p/'manifest.json',self.manifest)
    def test_consistent_views_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p,self.rows);self.check(p)
    def test_rehashed_csv_cannot_disagree_with_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p,self.rows)
            source=(p/'NASDAQ_stock_industry.csv').read_text(encoding='utf-8-sig')
            (p/'NASDAQ_stock_industry.csv').write_text(source.replace('ABC','WRONG'),encoding='utf-8-sig');self.rehash(p)
            with self.assertRaisesRegex(ValueError,'CSV differs'):self.check(p)
    def test_rehashed_reverse_index_cannot_broadcast_to_a_different_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.package(p,self.rows)
            r=json.loads((p/'NASDAQ_industry_to_stocks.json').read_text(encoding='utf-8'))
            r['industries'][self.keys[0]][0]['stock_uid']='NASDAQ:XYZ'
            write_json(p/'NASDAQ_industry_to_stocks.json',r);self.rehash(p)
            with self.assertRaisesRegex(ValueError,'reverse index differs'):self.check(p)
    def test_consistent_but_wrong_reference_label_is_checked_against_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=copy.deepcopy(self.rows)
            rows[0]['reference_industry_key']=self.keys[2];rows[0]['reference_industry_name']=self.dictionary.by_key[self.keys[2]]['name']
            self.package(p,rows)
            refs={'NASDAQ':{'ABC':[self.keys[0]],'XYZ':[self.keys[1]]}}
            with self.assertRaisesRegex(ValueError,'original dataset evidence'):self.check(p,refs)
    def test_boolean_array_index_is_not_an_integer_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=copy.deepcopy(self.rows);rows[1]['array_index']=True;self.package(p,rows)
            with self.assertRaisesRegex(ValueError,'order mismatch'):self.check(p)

if __name__=='__main__':unittest.main()
