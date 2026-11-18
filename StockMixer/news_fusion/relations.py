"""Evidence-bound historical identity resolution. No LLM ticker/name guessing."""
import csv
import re
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, write_json, write_records,verify_files


def normalized(value):
    return unicodedata.normalize('NFKC', value or '').strip().casefold()


def require_industry_timing(manifest,mode,channels):
    if mode=='observed_pit' and 'industry' in channels and (manifest.get('mapping_policy') or {}).get('version')!='evidence_mapping_v3':
        raise ValueError('Rebuild the industry panel with evidence_mapping_v3 to bind inference and approval availability times')


def publication_day(article):
    from .time_axis import timestamp
    value = article.get('published_at')
    if not isinstance(value, str):
        raise ValueError('Article lacks publication date')
    return value if len(value) == 10 else timestamp(value).tz_convert('America/New_York').strftime('%Y-%m-%d')


def valid_on(row, day):
    return row['valid_from'] <= day and (not row.get('valid_to') or day < row['valid_to'])


def overlaps(left,right):
    return max(left['valid_from'],right['valid_from'])<min(left.get('valid_to') or '9999-12-31',right.get('valid_to') or '9999-12-31')


def verified(row, label):
    if row.get('verification_status') != 'verified':
        return False
    for field in ('mapping_source', 'evidence', 'reviewed_by', 'known_from', 'valid_from'):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(label + ' lacks ' + field)
    date.fromisoformat(row['valid_from'])
    if row.get('valid_to'):
        if not isinstance(row['valid_to'],str):raise ValueError(label+' valid_to must be an ISO date string')
        date.fromisoformat(row['valid_to'])
        if row['valid_to'] <= row['valid_from']:
            raise ValueError(label + ' has an empty historical validity interval')
    from .time_axis import timestamp
    if timestamp(row['known_from']) > timestamp(now()):
        raise ValueError(label + ' has future review time')
    if row.get('known_to') and timestamp(row['known_to'])<=timestamp(row['known_from']):
        raise ValueError(label+' has a reversed knowledge-time interval')
    if len(row['evidence'].strip()) < 8:
        raise ValueError(label + ' lacks substantive evidence')
    return True


def axis_identity(axis, row):
    if row.get('dataset') != axis.dataset:
        return False
    uid = row.get('stock_uid')
    if uid not in axis.stock_uids:
        raise ValueError('Unknown stock_uid in ' + axis.dataset + ': ' + str(uid))
    index = axis.stock_uids.index(uid)
    if type(row.get('array_index')) is not int or row['array_index'] != index:
        raise ValueError('Identity row/index mismatch: ' + str(uid))
    ticker = axis.source['tickers'][index] if 'tickers' in axis.source else uid.split(':', 1)[1]
    if row.get('dataset_ticker', row.get('ticker')) != ticker:
        raise ValueError('Identity dataset ticker mismatch: ' + uid)
    return True


def validate_master(axis, securities, memberships, aliases):
    security_rows, member_rows, alias_rows = [], [], []
    for row in securities:
        if not verified(row, 'security') or not axis_identity(axis, row):
            continue
        for field in ('ticker', 'exchange', 'security_id', 'company_name'):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise ValueError('Verified security lacks ' + field)
        if row['exchange'] not in ('NASDAQ', 'NYSE', 'AMEX', 'NYSEAMERICAN'):
            raise ValueError('Unsupported verified exchange')
        if not isinstance(row.get('aliases', []), list) or not all(isinstance(s, str) and s for s in row.get('aliases', [])):
            raise ValueError('Security aliases must be explicit strings')
        security_rows.append(row)
    for row in memberships:
        if not verified(row, 'industry membership') or not axis_identity(axis, row):
            continue
        for field in ('industry_standard', 'industry_version', 'industry_code', 'industry_name', 'security_id'):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise ValueError('Verified industry membership lacks ' + field)
        weight = row.get('exposure_weight', 1.0)
        from .validation import finite_number
        finite_number(weight, 'exposure_weight', strictly_positive=True)
        if weight > 1:
            raise ValueError('Membership exposure_weight must be <= 1')
        owners = [s for s in security_rows if s['stock_uid'] == row['stock_uid'] and s['security_id'] == row['security_id']
                  and s['valid_from'] <= row['valid_from'] and
                  (not s.get('valid_to') or (row.get('valid_to') and row['valid_to'] <= s['valid_to']))]
        if not owners:
            raise ValueError('Membership has no matching historical verified security')
        member_rows.append(row)
    for row in aliases:
        if not verified(row, 'industry alias'):
            continue
        for field in ('alias', 'industry_standard', 'industry_version', 'industry_code'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError('Verified industry alias lacks ' + field)
        alias_rows.append(row)
    by_code={};by_uid={}
    for row in security_rows:
        code=(normalized(row['ticker']),row['exchange'])
        for other in by_code.get(code,[]):
            if overlaps(row,other) and (row['stock_uid'],row['security_id'])!=(other['stock_uid'],other['security_id']):
                raise ValueError('Conflicting historical ticker ownership intervals: '+row['ticker'])
        for other in by_uid.get(row['stock_uid'],[]):
            if overlaps(row,other) and row['security_id']!=other['security_id']:
                raise ValueError('One array row has conflicting historical security identities')
        by_code.setdefault(code,[]).append(row);by_uid.setdefault(row['stock_uid'],[]).append(row)
    by_industry={}
    for row in member_rows:
        key=(row['stock_uid'],row['industry_standard'],row['industry_version'])
        for other in by_industry.get(key,[]):
            if normalized(row['industry_standard'])=='gics' and overlaps(row,other) and row['industry_code']!=other['industry_code']:
                raise ValueError('Overlapping conflicting primary GICS classifications')
        by_industry.setdefault(key,[]).append(row)
    return security_rows, member_rows, alias_rows


def resolve_relations(axis, articles, securities, memberships, aliases, policy=None, dictionary=None):
    securities, memberships, aliases = validate_master(axis, securities, memberships, aliases)
    links, unresolved, counts = [], [], Counter()
    seen = set()
    for article in articles:
        if article.get('semantic_quarantine') is True:
            counts['semantic_quarantine']+=1
            unresolved.append({'source_id':article['source_id'],'channel':'all','reason':'semantic_quarantine'})
            continue
        if article.get('input_text_hash')!=digest([article.get('title') or '',article.get('summary') or '']):
            raise ValueError('Relation article text differs from its bound input_text_hash')
        day = publication_day(article)
        for entity in article.get('entities', []):
            if not isinstance(entity, dict):
                continue
            name = entity.get('name', entity.get('名称'))
            code = entity.get('ticker', entity.get('股票代码'))
            exchange = entity.get('exchange', entity.get('交易所'))
            if any(value is not None and not isinstance(value,str) for value in (name,code,exchange)):
                unresolved.append({'source_id':article['source_id'],'channel':'company','reason':'invalid_entity_field_type'})
                counts['invalid_entity_field_type']+=1
                continue
            exchange = exchange.upper().strip() if isinstance(exchange, str) else None
            if not name and not code:
                continue
            candidates = [s for s in securities if valid_on(s, day)]
            if policy is not None and entity.get('security_id') is not None:
                candidates=[s for s in candidates if s['security_id']==entity['security_id']]
            if code:
                # Preserve punctuation/share class; do not translate integer IDs to array positions.
                candidates = [s for s in candidates if normalized(s['ticker']) == normalized(str(code))]
            if exchange:
                candidates = [s for s in candidates if s['exchange'] == exchange]
            if name:
                candidates = [s for s in candidates if normalized(name) in
                              {normalized(v) for v in [s['company_name'], *s.get('aliases', [])]}]
                # Upstream entities are suggestions. A verified master alone does not prove an article mentions them.
                source=normalized((article.get('title') or '')+'\n'+(article.get('summary') or ''))
                if not re.search(r'(?<!\w)'+re.escape(normalized(name))+r'(?!\w)',source):
                    candidates=[]
            else:
                text = (article.get('title') or '') + '\n' + (article.get('summary') or '')
                explicit = re.search(r'(?:NASDAQ|NYSE|AMEX)\s*:\s*' + re.escape(str(code)) + r'(?![\w.\-])', text, re.I)
                if not explicit:
                    candidates = []
            owners = {(s['stock_uid'], s['security_id']) for s in candidates}
            if len(owners) != 1:
                reason = 'ambiguous_identity' if owners else 'unverified_or_unmatched_identity'
                unresolved.append({'source_id': article['source_id'], 'channel': 'company', 'name': name,
                                   'ticker': code, 'exchange': exchange, 'reason': reason})
                counts[reason] += 1
                continue
            owner = candidates[0]
            key = (article['source_id'], owner['stock_uid'])
            if key in seen:
                continue
            seen.add(key)
            link = dict(owner, source_id=article['source_id'], channel='company', evidence_type='explicit',
                        article_record_version=article.get('record_version'), article_input_text_hash=article['input_text_hash'])
            if policy is not None:
                link.update(mapping_method='exact_historical_identity_and_literal_article_mention',mapping_confidence=None,
                    source_confidence=entity.get('confidence'),mapping_status='verified_deterministic',
                    mapping_evidence_hash=digest(owner))
            links.append(link)
            counts['company_links'] += 1
        # Industry inference must have a quote/mechanism. Bare legacy tags are only candidates.
        for raw_impact in article.get('industry_impacts', []):
            impact=raw_impact
            if policy is not None:
                from .upstream import industry_impact,confidence
                try:impact=industry_impact(raw_impact)
                except ValueError:
                    unresolved.append({'source_id':article['source_id'],'channel':'industry','reason':'invalid_or_conflicting_industry_schema'})
                    counts['invalid_or_conflicting_industry_schema']+=1
                    continue
                score=confidence(impact.get('confidence'))
                if (impact.get('confidence') is not None and score is None) or (impact.get('evidence_type')=='inferred' and (score is None or score<policy['industry_min_confidence'])):
                    unresolved.append({'source_id':article['source_id'],'channel':'industry','reason':'missing_or_low_industry_source_confidence','source_confidence':score})
                    counts['missing_or_low_industry_source_confidence']+=1
                    continue
            if not isinstance(impact, dict):
                continue
            name = impact.get('industry_name', impact.get('industry', impact.get('行业')))
            quote = impact.get('evidence', impact.get('证据'))
            kind = impact.get('evidence_type')
            mechanism = impact.get('mechanism')
            text = (article.get('title') or '') + '\n' + (article.get('summary') or '')
            selected=None
            if dictionary is not None and impact.get('industry_key'):
                selected=dictionary.by_key.get(impact['industry_key'])
                if selected is None or not dictionary.selection_valid(impact['industry_key'],name) or not dictionary.evidence_matches(impact['industry_key'],quote or '') or impact.get('dictionary_sha256')!=dictionary.sha256 or tuple(impact.get(k) for k in ('industry_standard','industry_version','industry_code'))!=tuple(selected[k] for k in ('industry_standard','industry_version','industry_code')):
                    unresolved.append({'source_id':article['source_id'],'channel':'industry','reason':'industry_dictionary_version_or_code_mismatch'})
                    counts['industry_dictionary_version_or_code_mismatch']+=1
                    continue
            original_proof=False
            if selected is not None:
                from .upstream import industry_impact as adapt_impact
                proof=article.get('upstream_analysis') or {};audit=proof.get('audit') or {}
                original_proof=(proof.get('record_hash')==article.get('source_record_hash') and audit.get('record_hash_matches') is True
                    and re.fullmatch('[a-f0-9]{64}',str(audit.get('sha256'))) is not None
                    and re.fullmatch('[a-f0-9]{64}',str(proof.get('content_hash'))) is not None
                    and (proof.get('industry_dictionary') or {}).get('sha256')==dictionary.sha256
                    and impact.get('original_evidence_verified') is True
                    and impact.get('original_evidence_quote')==quote
                    and impact.get('evidence_basis')=='original_article_quote_verified_before_raw_cleanup'
                    and any(digest(adapt_impact(raw))==digest(impact) for raw in proof.get('raw_industry_impacts',[])))
            if policy is not None and ((kind=='explicit' and selected is None and (not isinstance(quote,str) or not isinstance(name,str) or not re.search(r'(?<!\w)'+re.escape(normalized(name))+r'(?!\w)',normalized(quote)))) or (kind=='inferred' and (not isinstance(mechanism,str) or len(mechanism.strip())<8))):
                unresolved.append({'source_id':article['source_id'],'channel':'industry','industry':name,
                                   'reason':'invalid_industry_evidence_type_or_mechanism'})
                counts['invalid_industry_evidence_type_or_mechanism']+=1
                continue
            if selected is not None:
                matches=[dict(alias=name,industry_standard=selected['industry_standard'],industry_version=selected['industry_version'],
                    industry_code=selected['industry_code'],valid_from=day,valid_to=None,known_from=dictionary.manifest['created_at'],
                    mapping_source=selected['source'],evidence=selected['name'],reviewed_by='deterministic_dictionary_code_validator',
                    verification_status='verified',validity_basis='article_association_not_historical_security_membership')]
            else:
                matches = [a for a in aliases if valid_on(a, day) and normalized(a['alias']) == normalized(name)]
            codes = {(a['industry_standard'], a['industry_version'], a['industry_code']) for a in matches}
            if len(codes) != 1 or kind not in ('explicit', 'inferred') or not isinstance(quote, str) or len(quote) < 8 or (quote not in text and not original_proof) or (kind == 'inferred' and not mechanism):
                unresolved.append({'source_id': article['source_id'], 'channel': 'industry', 'industry': name,
                                   'reason': 'unverified_industry_alias_or_article_evidence'})
                counts['unverified_industry_alias_or_article_evidence'] += 1
                continue
            alias = matches[0]
            if selected is not None:
                alias.update(industry_key=selected['key'],dictionary_sha256=dictionary.sha256,
                    article_evidence_basis='bound_upstream_original_quote' if original_proof else 'pinned_summary_quote')
            if policy is not None:
                declared=tuple(impact.get(k) for k in ('industry_standard','industry_version','industry_code'))
                if any(v is not None for v in declared) and declared!=next(iter(codes)):
                    unresolved.append({'source_id':article['source_id'],'channel':'industry','reason':'conflicting_or_incomplete_industry_classification','declared':list(declared)})
                    counts['conflicting_or_incomplete_industry_classification']+=1
                    continue
            from .time_axis import timestamp
            known=alias['known_from']
            if impact.get('llm_inferred_at'):
                known=max(timestamp(known),timestamp(impact['llm_inferred_at'])).isoformat()
            if policy is not None and policy.get('version')=='evidence_mapping_v3':
                times=[known,article.get('llm_inferred_at'),article.get('effective_feature_available_at'),article.get('feature_available_at'),article.get('commit_verified_at'),
                    (article.get('industry_scope_review') or {}).get('reviewed_at')]
                known=max(timestamp(value) for value in times if value).isoformat()
            links.append(dict(alias, source_id=article['source_id'], dataset=axis.dataset, channel='industry',known_from=known,
                              evidence_type=kind, article_evidence=quote, mechanism=mechanism,
                              impact_metadata={k:impact[k] for k in ('event_type','impact_role','direction','confidence','importance','horizon','llm_inferred_at','model_digest','prompt_sha256','review_method') if k in impact},
                              article_record_version=article.get('record_version'), article_input_text_hash=article['input_text_hash']))
            if policy is not None:
                links[-1].update(source_confidence=score,mapping_confidence=None,mapping_status='verified_deterministic',
                    mapping_method='exact_dated_industry_alias',mapping_evidence_hash=digest(alias))
            counts['industry_links'] += 1
    return links, memberships, unresolved, dict(counts)


def build_relations(axis, articles, paths, output, provenance, policy=None, dictionary=None, reference_sources=None):
    if provenance.get('mode')!='verified_snapshot' or re.fullmatch('[a-f0-9]{64}',str(provenance.get('snapshot_manifest_sha256'))) is None:
        raise ValueError('Historical relations require a pinned verified snapshot')
    tables = [read_records(path) if path else [] for path in paths]
    links, members, unresolved, counts = resolve_relations(axis, articles, *tables, policy=policy,dictionary=dictionary)
    if reference_sources is not None:
        if dictionary is None:raise ValueError('Reference membership requires a pinned dictionary')
        from .reference_membership import memberships as reference_members,add_references
        refs=reference_members(axis,dictionary,*reference_sources,dictionary.manifest['created_at'])
        members=add_references(members,refs)
    with new_artifact(output) as directory:
        if reference_sources is not None:
            import shutil
            for p,n in zip(reference_sources,('reference_archive.tar.gz','reference_sp500.csv')):shutil.copy2(p,directory/n)
        write_records(directory / 'article_stock_link.jsonl', links)
        write_records(directory / 'industry_memberships.jsonl', members)
        write_records(directory / 'unmatched.jsonl', unresolved)
        write_records(directory/'article_bindings.jsonl',articles)
        for name,rows in zip(('security_master.jsonl','industry_master.jsonl','industry_aliases.jsonl'),tables):
            write_records(directory/name,rows)
        if dictionary is not None:
            import shutil
            shutil.copy2(dictionary.directory/'industry_dictionary.jsonl',directory/'industry_dictionary.jsonl')
            shutil.copy2(dictionary.directory/'manifest.json',directory/'industry_dictionary_manifest.json')
            for name in dictionary.manifest.get('files',{}):
                if name!='industry_dictionary.jsonl':shutil.copy2(dictionary.directory/name,directory/name)
        manifest = {'reference_memberships_enabled':reference_sources is not None,'dictionary_sources_embedded':dictionary is not None,'industry_dictionary_sha256':dictionary.sha256 if dictionary is not None else None,'mapping_policy':policy,'kind': 'historical_news_relations', 'created_at': now(), 'dataset': axis.dataset,
                    'stock_axis_hash': axis.stock_hash, 'trading_dates_hash': axis.date_hash,
                    'source_input': provenance, 'input_sha256': {str(p): digest_file(p) for p in paths if p},
                    'counts': counts, 'unmatched': len(unresolved), 'memberships': len(members),
                    'files': file_manifest(directory)}
        write_json(directory / 'manifest.json', manifest)
    return manifest


def load_relations(directory,axis,manifest_name='manifest.json',prefix=''):
    import json
    directory=Path(directory)
    manifest=json.loads((directory/manifest_name).read_text(encoding='utf-8'))
    if manifest.get('kind')!='historical_news_relations' or manifest.get('dataset')!=axis.dataset or manifest.get('stock_axis_hash')!=axis.stock_hash or manifest.get('trading_dates_hash')!=axis.date_hash:
        raise ValueError('Historical relation artifact dataset/axis mismatch')
    if manifest.get('source_input',{}).get('mode')!='verified_snapshot':
        raise ValueError('Historical relation source snapshot is not verified')
    required=('article_stock_link.jsonl','industry_memberships.jsonl','article_bindings.jsonl',
              'security_master.jsonl','industry_master.jsonl','industry_aliases.jsonl','unmatched.jsonl')
    files=manifest.get('files',{})
    if not set(required)<=set(files) or any(Path(k).name!=k for k in files):
        raise ValueError('Historical relation proof files are missing')
    verify_files(directory,{'files':{prefix+k:v for k,v in files.items()}},required=[prefix+k for k in required])
    def records(name):return read_records(directory/(prefix+name))
    bindings=records('article_bindings.jsonl')
    dictionary=None
    if manifest.get('industry_dictionary_sha256'):
        from .industry_dictionary import IndustryDictionary
        dictionary=IndustryDictionary(directory,manifest_name=prefix+'industry_dictionary_manifest.json',dictionary_name=prefix+'industry_dictionary.jsonl',verify_sources=manifest.get('dictionary_sources_embedded',False))
        if dictionary.sha256!=manifest['industry_dictionary_sha256']:raise ValueError('Relation dictionary changed')
    links,members,unmatched,counts=resolve_relations(axis,bindings,records('security_master.jsonl'),
        records('industry_master.jsonl'),records('industry_aliases.jsonl'),policy=manifest.get('mapping_policy'),dictionary=dictionary)
    if manifest.get('reference_memberships_enabled'):
        from .reference_membership import memberships as reference_members,add_references
        members=add_references(members,reference_members(axis,dictionary,directory/(prefix+'reference_archive.tar.gz'),directory/(prefix+'reference_sp500.csv'),dictionary.manifest['created_at']))
    if digest(links)!=digest(records('article_stock_link.jsonl')) or digest(members)!=digest(records('industry_memberships.jsonl')) or digest(unmatched)!=digest(records('unmatched.jsonl')) or counts!=manifest['counts']:
        raise ValueError('Historical relations differ from their independently re-resolved evidence')
    return links,members,manifest


def mapping_templates(project, axes, output):
    """Editable templates are pending; dataset-span dates are not proof of listing validity."""
    import json
    with new_artifact(output) as directory:
        rows = []
        for axis in axes:
            for i, uid in enumerate(axis.stock_uids):
                rows.append({'dataset': axis.dataset, 'stock_uid': uid, 'array_index': i,
                             'dataset_ticker': uid.split(':', 1)[1], 'ticker': uid.split(':', 1)[1],
                             'exchange': None, 'security_id': None, 'company_name': None, 'aliases': [],
                             'valid_from': None, 'valid_to': None, 'known_from': None,
                             'mapping_source': None, 'evidence': None, 'reviewed_by': None,
                             'verification_status': 'pending'})
        write_records(directory / 'security_master.jsonl', rows)
        write_records(directory / 'industry_memberships.jsonl', [])
        write_records(directory / 'industry_aliases.jsonl', [])
        write_json(directory / 'manifest.json', {'kind': 'pending_mapping_templates', 'created_at': now(),
                   'securities': len(rows), 'verified_rows': 0, 'files': file_manifest(directory)})
