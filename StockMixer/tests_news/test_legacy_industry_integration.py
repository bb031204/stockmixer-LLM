"""Synthetic temporary fixtures only; no model/network calls or production artifacts."""
import copy,json,tempfile,unittest,zipfile
from pathlib import Path
from tests_news.test_snapshot import fixture
from news_fusion import legacy_industry as m
from news_fusion.snapshot import freeze_current
from news_fusion.inputs import normalize_article,load_articles
from news_fusion.industry_dictionary import IndustryDictionary
from news_fusion.io import write_json,write_records,read_records,file_manifest,digest_file
from news_fusion.source_adapter import producer_hash
from news_fusion.industry_scope_review import apply_scope_reviews

PROJECT=Path(__file__).resolve().parents[1]

class Tests(unittest.TestCase):
    def test_complete_import_independent_replay_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,parent,frozen,raw,articles=fixture(root)
            summary='Semiconductor manufacturers face supply shortages.'
            for sid in raw:
                raw[sid].update(summary=summary,主要内容=summary,时间='2013-01-02')
                articles[sid].update(summary=summary,source_record_hash=producer_hash(raw[sid]),record_version=producer_hash(raw[sid]))
                articles[sid]['input_text_hash']=normalize_article(articles[sid])['input_text_hash']
                if sid in ('a','b'):write_json(source/'US'/(sid+'.json'),raw[sid])
            write_records(parent/'articles.jsonl',list(articles.values()))
            proofs=read_records(parent/'provenance.jsonl')
            for p in proofs:p['record_hash']=producer_hash(raw[p['source_id']])
            write_records(parent/'provenance.jsonl',proofs);write_records(parent/'review_records.jsonl',[])
            metadata=json.loads((parent/'manifest.json').read_text(encoding='utf-8'));files=file_manifest(parent);files.pop('manifest.json',None)
            metadata['files']=files;write_json(parent/'manifest.json',metadata);freeze_current(source,parent,frozen)
            original={p.name:p.read_bytes() for p in (source/'US').glob('*.json')}
            dictionary=IndustryDictionary(PROJECT/'news_assets/industry_dictionary');batch=root/'batch';batch.mkdir();(batch/'analysis').mkdir()
            settings={'dictionary_sha256':dictionary.sha256,'prompt_sha256':producer_hash('synthetic-only'),
              'candidate_limit':12,'active_standards':['SEC-SIC'],'min_confidence':.85}
            fingerprint=producer_hash(settings);index=[];values=[]
            with zipfile.ZipFile(batch/'raw_news.zip','w') as archive:
                for sid in ('a','b'):
                    path=source/'US'/(sid+'.json');archive.writestr('US/'+sid+'.json',path.read_bytes())
                    index.append({'source_id':sid,'path':str(path),'archive_path':'US/'+sid+'.json','sha256':digest_file(path),'record_hash':producer_hash(raw[sid])})
                    row=dictionary.by_key['SEC-SIC:3674']
                    topic={'industry_key':row['key'],'quote':summary,'scope':'industry_wide','scope_quote':summary,
                        'mechanism':'Supply shortages limit production across independent manufacturers.','direction':'negative','confidence':.95,
                        'industry_standard':row['industry_standard'],'industry_version':row['industry_version'],'industry_code':row['industry_code'],
                        'industry_name_standard':row['name'],'dictionary_sha256':dictionary.sha256,'dictionary_version':dictionary.manifest['version'],
                        'evidence_basis':'stored_summary_only','original_evidence_verified':False,'industry_scope_status':'automated_evidence_gate_passed','propagation_approved':True}
                    impact=dict(topic,industry_name_original=row['name'],industry_name=row['name'],evidence=summary,industry_scope_quote=summary,
                        evidence_type='inferred',impact_role='affected',horizon='unspecified',llm_inferred_at='2026-01-01T00:00:00Z')
                    value={'source_id':sid,'record_hash':producer_hash(raw[sid]),'summary_sha256':producer_hash(summary),'fingerprint':fingerprint,
                        'text_basis':'stored_summary_only','historical_observed_pit_ready':False,'dictionary':{'sha256':dictionary.sha256,'version':dictionary.manifest['version']},
                        'candidate_keys':[r['key'] for r in dictionary.candidates(summary,12,['SEC-SIC'],headline=raw[sid]['title'])],
                        'industries':[topic],'industry_impacts':[impact],'status':'completed','llm_inferred_at':'2026-01-01T00:00:00Z','feature_available_at':'2026-01-02T00:00:00Z'}
                    values.append(value);write_json(batch/'analysis'/(sid+'.json'),value)
            write_records(batch/'input_index.jsonl',index);write_records(batch/'industry_classifications.jsonl',values)
            write_json(batch/'run.json',{'kind':m.BATCH_KIND,'fingerprint':fingerprint,'settings':settings,'unique_articles':2})
            (batch/'industry_prompt.txt').write_text('synthetic-only',encoding='utf-8')
            write_json(batch/'manifest.json',{'kind':m.BATCH_KIND,'fingerprint':fingerprint,'pending':0,'total':2,'processed':2,
              'status_counts':{'completed':2},'articles_with_dictionary_mentions':2,'articles_with_industry_impacts':2,'industry_impacts':2,
              'files':{p.relative_to(batch).as_posix():digest_file(p) for p in batch.rglob('*') if p.is_file()}})
            output=root/'enriched';m.import_batch(frozen,batch,PROJECT/'news_assets/industry_dictionary',output)
            self.assertTrue(m.validate_enriched(output)['valid'])
            normalized,_=load_articles([output/'articles.jsonl']);automatic=m.automated_reviews(output,normalized)
            reviewed,counts=apply_scope_reviews(normalized,required=True,automated=automatic,preserve_approval_time=True)
            self.assertEqual(counts['automated_evidence_gated_samples'],2);self.assertEqual(counts['human_confirmed_samples'],0)
            self.assertEqual(original,{p.name:p.read_bytes() for p in (source/'US').glob('*.json')})
            changed=read_records(output/'articles.jsonl');changed[0]['industry_impacts'][0]['confidence']=.1;write_records(output/'articles.jsonl',changed)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'));manifest['files']['articles.jsonl']=digest_file(output/'articles.jsonl');write_json(output/'manifest.json',manifest)
            with self.assertRaises(ValueError):m.validate_enriched(output)

if __name__=='__main__':unittest.main()
