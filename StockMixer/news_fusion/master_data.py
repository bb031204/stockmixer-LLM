"""Build dated SEC-SIC observations; never invent continuous historical intervals."""
import argparse
import hashlib
import json
import re
import sys
import time
import shutil
import zipfile
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.parts, self.rows, self.row, self.cell, self.link = [], [], None, None, None
        self.hidden = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.cell = {'parts': [], 'href': None}
        if tag == 'a' and self.cell is not None:
            self.cell['href'] = attrs.get('href')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        if tag in ('td', 'th') and self.cell is not None:
            self.cell['text'] = ' '.join(' '.join(self.cell.pop('parts')).split())
            if self.row is not None:
                self.row.append(self.cell)
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, value):
        if not self.hidden:
            self.parts.append(value)
            if self.cell is not None:
                self.cell['parts'].append(value)

    @property
    def text(self):
        return ' '.join(' '.join(self.parts).split())


def sha(value):
    return hashlib.sha256(value).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def validate_settings(settings):
    expected={'cache','output','offline_only','request_interval_seconds','timeout_seconds','maximum_document_bytes',
              'maximum_filings_per_issuer','maximum_documents_per_filing','user_agent','issuers'}
    if not isinstance(settings,dict) or set(settings)!=expected:
        raise ValueError('historical_master has missing/unknown settings')
    if type(settings['offline_only']) is not bool:
        raise ValueError('historical_master.offline_only must be boolean')
    for key in ('cache','output','user_agent'):
        if not isinstance(settings[key],str) or not settings[key].strip():raise ValueError('Invalid historical_master.'+key)
    for key,low,high in [('timeout_seconds',1,120),('maximum_document_bytes',1024,100000000),('maximum_documents_per_filing',1,5)]:
        if type(settings[key]) is not int or not low<=settings[key]<=high:raise ValueError('Invalid historical_master.'+key)
    delay=settings['request_interval_seconds']
    if isinstance(delay,bool) or not isinstance(delay,(float,int)) or not .2<=delay<=60:
        raise ValueError('SEC request interval must be between 0.2 and 60 seconds')
    limit=settings['maximum_filings_per_issuer']
    if limit is not None and (type(limit) is not int or limit<1):raise ValueError('Invalid historical filing limit')
    if not isinstance(settings['issuers'],list) or not settings['issuers']:raise ValueError('Historical issuers are required')
    seen=set()
    for issuer in settings['issuers']:
        if not isinstance(issuer,dict) or set(issuer)!={'cik','ticker','exchange','names'}:
            raise ValueError('Unknown/missing issuer setting')
        if not isinstance(issuer['cik'],str) or not re.fullmatch(r'\d{1,10}',issuer['cik']):raise ValueError('Invalid historical CIK')
        if not isinstance(issuer['ticker'],str) or not issuer['ticker'] or issuer['exchange'] not in ('NASDAQ','NYSE'):
            raise ValueError('Invalid historical ticker/exchange candidate')
        if not isinstance(issuer['names'],list) or not issuer['names'] or not all(isinstance(v,str) and v.strip() for v in issuer['names']):
            raise ValueError('Issuer names must be explicit nonempty strings')
        identity=(int(issuer['cik']),issuer['ticker'],issuer['exchange'])
        if identity in seen:raise ValueError('Duplicate issuer candidate')
        seen.add(identity)


def canonical_index(url):
    parsed = urlsplit(url)
    match = re.fullmatch(r'/Archives/edgar/data/(\d+)/(?:\d+/)?(\d{10}-\d{2}-\d{6})-index\.html?', parsed.path)
    if parsed.scheme != 'https' or parsed.hostname != 'www.sec.gov' or not match:
        raise ValueError('Only an official SEC filing index is allowed')
    cik, accession = match.groups()
    return 'https://www.sec.gov/Archives/edgar/data/%s/%s/%s-index.htm' % (cik, accession.replace('-', ''), accession)


def fetch(url, cache, settings):
    import requests
    if urlsplit(url).hostname != 'www.sec.gov' or urlsplit(url).scheme != 'https':
        raise ValueError('Non-SEC evidence URL rejected')
    key = sha(url.encode())
    html_path, metadata = cache / (key + '.html'), cache / (key + '.json')
    if html_path.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        body = html_path.read_bytes()
        if saved['url'] != url or saved['sha256'] != sha(body):
            raise ValueError('Evidence cache checksum mismatch')
        return body.decode('utf-8', errors='replace'), saved
    if settings.get('offline_only'):
        raise ValueError('Evidence is not present in offline cache')
    time.sleep(settings['request_interval_seconds'])
    response = requests.get(url, headers={'User-Agent': settings['user_agent']},
                            timeout=settings['timeout_seconds'], allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200 or len(response.content) > settings['maximum_document_bytes']:
        raise ValueError('Invalid/oversized SEC evidence response')
    body = response.content
    value = {'url': url, 'fetched_at': utcnow(), 'sha256': sha(body), 'bytes': len(body)}
    html_path.write_bytes(body)
    write_json(metadata, value)
    return body.decode('utf-8', errors='replace'), value


def index_fields(source, cik):
    page = Page(source)
    day = re.search(r'Filing Date\s+(\d{4}-\d{2}-\d{2})', page.text)
    block = re.search(r'\bSIC(?:</acronym>)?\s*:\s*(.*?)(?:</p>|</div>|</td>|<br\s*/?>|$)', source, re.I | re.S)
    sic_text = Page(block.group(1)).text if block else ''
    sic = re.fullmatch(r'(\d{4})\s+(.+)', sic_text)
    issuer = re.search(r'([\w][\w\s.,/&()\\\-]+)\s+\(Filer\)\s+CIK\s*:\s*0*' + str(int(cik)) + r'\b', page.text)
    if not day or not sic or not issuer:
        raise ValueError('Historical filing date/CIK/SIC evidence is incomplete')
    docs = []
    for row in page.rows:
        if len(row) >= 4 and row[2].get('href') and row[3]['text'] in ('8-K', '10-K', '10-Q', 'EX-99.1', 'EX-99.01'):
            docs.append((row[2]['href'], row[3]['text']))
    # Press releases usually state the contemporaneous exchange and ticker.
    docs.sort(key=lambda item: (not item[1].startswith('EX-99'), item[1]))
    return day.group(1), sic.group(1), sic.group(2).strip(), docs


def trading_quote(text, ticker, exchange):
    market = r'(?:NASDAQ|Nasdaq)' if exchange == 'NASDAQ' else r'(?:NYSE|New York Stock Exchange)'
    pattern = re.compile(market + r'[^.;]{0,100}?(?<![\w])' + re.escape(ticker) + r'(?![\w])', re.I)
    match = pattern.search(text)
    if not match:
        # Modern covers have symbol before the exchange.
        pattern = re.compile(r'(?<![\w])' + re.escape(ticker) + r'(?![\w])[^.;]{0,60}?' + market, re.I)
        match = pattern.search(text)
    if not match:
        return None
    return text[max(0, match.start() - 100):min(len(text), match.end() + 100)]



def issuer_owned_ticker(text, name, ticker, exchange):
    """An issuer must directly own the exchange/ticker declaration, not merely occur nearby."""
    market=r'(?:NASDAQ|Nasdaq)' if exchange=='NASDAQ' else r'(?:NYSE|New York Stock Exchange)'
    pattern=re.escape(name)+r'\s*[.,]?\s*\(\s*'+market+r'\s*:?\s*'+re.escape(ticker)+r'\s*\)'
    return re.search(pattern,text,re.I) is not None

def same_filer(source,name):
    def normalized(value):
        # SEC display suffixes /NEW/, /DE/, /PR/ are not part of the issuer name.
        value=re.sub(r'\s*/[A-Za-z]{2,4}/\s*$','',value)
        value=re.sub(r'[^a-z0-9 ]',' ',value.casefold());parts=value.split()
        while parts and parts[-1] in ('inc','incorporated','corp','corporation','company','co','ltd'):parts.pop()
        return ' '.join(parts)
    for match in re.finditer(r'<span[^>]*class=[\'"]companyName[\'"][^>]*>(.*?)</span>',source,re.I|re.S):
        text=Page(match.group(1)).text
        if '(Filer)' in text and normalized(text.split('(Filer)')[0])==normalized(name):return True
    return False

def generate(project, config):
    settings = config['historical_master']
    validate_settings(settings)
    project=Path(project).resolve()
    cache = (project / settings['cache']).resolve()
    output = (project / settings['output']).resolve()
    for path in (cache,output):
        if project/'outputs' not in path.parents:raise ValueError('Historical outputs/cache must stay inside project outputs')
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = project / config['adapted_snapshot']
    if output.exists():
        manifest=verify_observations(output)
        if manifest.get('settings')!=settings or manifest.get('snapshot_manifest_sha256')!=sha((snapshot/'manifest.json').read_bytes()):
            raise ValueError('Historical settings/snapshot changed; select a new historical_master.output')
        print('[REUSE] verified SEC-SIC evidence artifact: '+str(output),flush=True)
        return manifest
    output.mkdir(parents=True)
    articles = {a['source_id']: a for a in records(snapshot / 'articles.jsonl')}
    axes = {name: json.loads((project / 'news_assets/axes' / (name + '.json')).read_text(encoding='utf-8'))
            for name in config['pipeline']['datasets']}
    raw = {}
    from .io import verify_files
    from .source_adapter import producer_hash
    source_manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    verify_files(snapshot,source_manifest,required=('articles.jsonl','raw_news.zip'))
    with zipfile.ZipFile(snapshot/'raw_news.zip') as archive:
        for name in archive.namelist():
            if not name.endswith('.json'):continue
            value=json.loads(archive.read(name).decode('utf-8-sig'))
            sid=value.get('source_id')
            if sid in articles:
                if producer_hash(value)!=articles[sid]['source_record_hash']:
                    raise ValueError('Archived raw source differs from frozen article version')
                if sid in raw and producer_hash(raw[sid])!=producer_hash(value):
                    raise ValueError('Conflicting original source records')
                raw[sid]=value
    securities, memberships, aliases, audit = [], [], [], []
    for issuer in settings['issuers']:
        selected = [(sid, a, raw[sid]) for sid, a in articles.items() if sid in raw
                    and '/%s/' % issuer['cik'] in raw[sid].get('url', '')]
        selected.sort(key=lambda item: (item[1]['published_at'], item[0]))
        if settings['maximum_filings_per_issuer'] is not None:
            selected = selected[:settings['maximum_filings_per_issuer']]
        for sid, article, source in selected:
            item = {'source_id': sid, 'cik': issuer['cik'], 'status': 'pending', 'at': utcnow()}
            try:
                url = canonical_index(source['url'])
                body, index_meta = fetch(url, cache, settings)
                day, sic, sic_name, documents = index_fields(body, issuer['cik'])
                publication = article['published_at'][:10]
                if day != publication:
                    raise ValueError('Filing/publication date mismatch; do not guess the effective date')
                quote = None
                proof = None
                for href, _ in documents[:settings['maximum_documents_per_filing']]:
                    doc_url = urljoin(url, href)
                    doc_body, doc_meta = fetch(doc_url, cache, settings)
                    doc_text = Page(doc_body).text
                    if not any(name.casefold() in doc_text.casefold() for name in issuer['names']):
                        continue
                    quote = trading_quote(doc_text, issuer['ticker'], issuer['exchange'])
                    if not re.search(r'common\s+(?:stock|shares)', doc_text, re.I):
                        quote = None
                    if quote:
                        proof = doc_meta
                        break
                if not quote:
                    raise ValueError('No contemporaneous issuer/exchange/ticker evidence in selected filing documents')
                until = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
                known = utcnow()
                common = {'valid_from': day, 'valid_to': until, 'known_from': known,
                    'mapping_source': url, 'evidence': quote, 'reviewed_by': 'sec_document_evidence_validator_v1',
                    'review_method': 'deterministic_primary_document_validation', 'verification_status': 'verified',
                    'validity_basis': 'dated_filing_observation_only_no_interval_interpolation',
                    'filing_index_sha256': index_meta['sha256'], 'security_document': proof,
                    'source_article_id': sid}
                for name, axis in axes.items():
                    if issuer['ticker'] not in axis['tickers']:
                        continue
                    index = axis['tickers'].index(issuer['ticker'])
                    security_id = 'SEC-CIK:%010d:COMMON:%s' % (int(issuer['cik']), issuer['ticker'])
                    row = dict(common, dataset=name, stock_uid=axis['stock_uids'][index], array_index=index,
                        dataset_ticker=issuer['ticker'], ticker=issuer['ticker'], exchange=issuer['exchange'],
                        security_id=security_id, company_name=issuer['names'][0],
                        aliases=[n for n in issuer['names'][1:] if n.casefold() in doc_text.casefold()],
                        cik='%010d' % int(issuer['cik']), security_type='common_equity',
                        security_id_scope='local issuer/common-equity identity, not a vendor permanent security ID')
                    securities.append(row)
                    memberships.append(dict(row, industry_standard='SEC-SIC', industry_version='EDGAR-SIC',
                        industry_code=sic, industry_name=sic_name, exposure_weight=1.0,
                        evidence='SEC dated filing index reports SIC %s %s. %s' % (sic, sic_name, quote)))
                aliases.append(dict(common, alias=sic_name, industry_standard='SEC-SIC',
                    industry_version='EDGAR-SIC', industry_code=sic, industry_name=sic_name,
                    evidence='SEC historical filing index names SIC %s as %s.' % (sic, sic_name)))
                item.update(status='verified_observation', filed_at=day, ticker=issuer['ticker'], sic=sic,
                            filing_index=index_meta, security_document=proof, security_quote=quote)
            except Exception as error:
                item.update(error=str(error), error_type=type(error).__name__)
            audit.append(item)
            print('[SEC] %s %s %s' % (issuer['ticker'], article['published_at'], item['status']), flush=True)
            write_rows(output / 'audit.jsonl', audit)
    def unique(rows, fields):
        result = {}
        for row in rows:
            key = tuple(row.get(k) for k in fields)
            if key in result and any(result[key].get(k) != row.get(k) for k in ('security_id','industry_code')):
                raise ValueError('Conflicting observations for one identity/date')
            result.setdefault(key, row)
        return list(result.values())
    securities = unique(securities, ('dataset','stock_uid','valid_from','valid_to'))
    memberships = unique(memberships, ('dataset','stock_uid','valid_from','industry_code'))
    aliases = unique(aliases, ('alias','industry_code','valid_from'))
    for item in audit:
        if item['status']!='verified_observation':continue
        for proof in (item['filing_index'],item['security_document']):
            key=sha(proof['url'].encode())
            for extension in ('.html','.json'):
                shutil.copy2(cache/(key+extension),output/('evidence_'+key+extension))
    for filename, rows in [('security_master.jsonl',securities),('industry_memberships.jsonl',memberships),('industry_aliases.jsonl',aliases)]:
        write_rows(output / filename, rows)
    manifest = {'kind':'sec_sic_dated_historical_observations', 'created_at':utcnow(),
        'settings':settings, 'snapshot_manifest_sha256':sha((snapshot/'manifest.json').read_bytes()),
        'securities':len(securities),'memberships':len(memberships),'aliases':len(aliases),
        'verified_filings':sum(a['status']=='verified_observation' for a in audit),
        'pending_filings':sum(a['status']=='pending' for a in audit),
        'not_full_universe':True, 'not_continuous_history':True,
        'no_current_ticker_backfill':True, 'no_assumed_industry_intervals':True,
        'source_code_sha256':sha(Path(__file__).read_bytes()),
        'files':{p.name:sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}}
    write_json(output/'manifest.json',manifest)
    verify_observations(output)
    print(json.dumps({k:v for k,v in manifest.items() if k in ('securities','memberships','aliases','verified_filings','pending_filings')}))
    return manifest


def verify_observations(directory):
    """Recheck local primary evidence, not only a verified label or source URL."""
    from .io import verify_files
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='sec_sic_dated_historical_observations':
        raise ValueError('Not a SEC-SIC observation artifact')
    verify_files(directory,manifest,required=('security_master.jsonl','industry_memberships.jsonl','industry_aliases.jsonl','audit.jsonl'))
    audits={a['source_id']:a for a in records(directory/'audit.jsonl') if a['status']=='verified_observation'}
    parsed={}
    for name in ('security_master.jsonl','industry_memberships.jsonl','industry_aliases.jsonl'):
        for row in records(directory/name):
            proof=audits.get(row['source_article_id'])
            if proof is None:raise ValueError('Verified row lacks a successful original filing audit')
            docs=[]
            for document in (proof['filing_index'],proof['security_document']):
                key=sha(document['url'].encode());p=directory/('evidence_'+key+'.html')
                if sha(p.read_bytes())!=document['sha256']:
                    raise ValueError('SEC primary evidence SHA mismatch')
                if key not in parsed:parsed[key]=p.read_text(encoding='utf-8',errors='replace')
                docs.append(parsed[key])
            day,sic,sic_name,_=index_fields(docs[0],proof['cik'])
            end=(date.fromisoformat(day)+timedelta(days=1)).isoformat()
            if row['valid_from']!=day or row['valid_to']!=end:
                raise ValueError('Historical observation was extended beyond its evidenced date')
            if row.get('industry_code',sic)!=sic or row.get('industry_name',sic_name)!=sic_name:
                raise ValueError('SEC-SIC row differs from historical filing classification')
            if name!='industry_aliases.jsonl':
                if row['ticker']!=proof['ticker'] or row['company_name'].casefold() not in Page(docs[1]).text.casefold():
                    raise ValueError('Issuer/ticker row differs from primary evidence')
                if not trading_quote(Page(docs[1]).text,row['ticker'],row['exchange']):
                    raise ValueError('Historical exchange/ticker evidence missing')
                if row.get('security_type')=='listed_security_type_unspecified':
                    # Exact historical issuer-owned symbol is sufficient for matching the dataset
                    # ticker; do not claim that the source verified its share class.
                    if proof.get('identity_evidence_basis')!='issuer_owned_exchange_ticker_declaration_share_class_unspecified':
                        raise ValueError('Unspecified-class security lacks its distinct evidence policy')
                    if not issuer_owned_ticker(Page(docs[1]).text,row['company_name'],row['ticker'],row['exchange']):
                        raise ValueError('Ticker declaration does not belong to the named issuer')
                    if not same_filer(docs[0],row['company_name']):
                        raise ValueError('Named issuer differs from the historical SIC filer')
                    expected='SEC-CIK:%010d:LISTED-SECURITY:%s'%(int(proof['cik']),proof['ticker'])
                else:
                    if not re.search(r'common\s+(?:stock|shares)',Page(docs[1]).text,re.I):
                        raise ValueError('Common-equity identity lacks explicit share-class evidence')
                    expected='SEC-CIK:%010d:COMMON:%s'%(int(proof['cik']),proof['ticker'])
                if row['security_id']!=expected:raise ValueError('Local security identity differs from its issuer/type proof')
    return manifest


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/llm.yaml')
    parser.add_argument('--project',default=None)
    args=parser.parse_args()
    project=Path(args.project).resolve() if args.project else Path(__file__).resolve().parents[1]
    config=yaml.safe_load((project/args.config).read_text(encoding='utf-8'))
    generate(project,config)


if __name__=='__main__':
    main()
