"""Evidence-bound market reviews; no scope/market tag is automatic approval."""
import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .inputs import text
from .time_axis import timestamp

SCHEMA="stockmixer_market_routes_v1"
BINDING_FIELDS=("source_id","input_text_hash","source_record_hash","published_at","publication_precision","market","impact_scope")
FILES=("market_route_bindings.jsonl","market_route_reviews.jsonl","market_routes.jsonl")


def binding(article):
    result={key:article.get(key) for key in BINDING_FIELDS}
    # Optional fields only: legacy pinned review hashes remain replayable.
    if 'upstream_analysis' in article or 'market_scope' in article or 'affected_markets' in article:
        for key in ('market_scope','affected_markets','scope_evidence','upstream_analysis',
                    'qwen_model_revision','prompt_version','llm_inferred_at'):
            if key in article:result[key]=article[key]
    return result


def article_pin(article):
    return digest(binding(article))


def keyed(rows):
    result={}
    for row in rows:
        if not isinstance(row,dict):
            raise ValueError("Market review rows must be mappings.")
        sid=row.get("source_id")
        if not isinstance(sid,str) or not sid.strip() or sid in result:
            raise ValueError("Market review identities must be nonempty and unique.")
        result[sid]=row
    return result


def review_template(articles,dataset,output,provenance):
    rows=[]
    for article in articles:
        rows.append({"source_id":article["source_id"],"dataset":dataset,"article_pin":article_pin(article),
            "title":article["title"],"summary":article["summary"],"published_at":article.get("published_at"),
            "candidate_market_scope":"market" in article.get("impact_scope",[]),
            "decision":"pending","evidence_quote":None,"rationale":None,"reviewed_by":None,"reviewed_at":None})
    keyed(rows)
    with new_artifact(output) as directory:
        write_records(directory/"market_review.jsonl",rows)
        write_json(directory/"manifest.json",{"schema_version":SCHEMA,"kind":"market_review_template","created_at":now(),
            "dataset":dataset,"target_market":"US","articles":len(rows),"pending":len(rows),
            "source_input":provenance,"files":file_manifest(directory)})
    return {"dataset":dataset,"articles":len(rows),"pending":len(rows),"output":str(output)}


def checked_reviews(articles,reviews,allow_partial=False,dataset=None):
    by_id=keyed(articles);reviewed=keyed(reviews)
    if set(by_id)!=set(reviewed):
        raise ValueError("Review must cover exactly the selected snapshot; retain pending rows for partial review.")
    counts=Counter();routes=[]
    for sid,a in by_id.items():
        row=reviewed[sid]
        if a.get("input_text_hash")!=digest([text(a.get("title")),text(a.get("summary"))]):
            raise ValueError("Market review binding text differs from its content hash.")
        if dataset is not None and row.get("dataset")!=dataset:
            raise ValueError("Market review target dataset mismatch.")
        if row.get("article_pin")!=article_pin(a):
            raise ValueError("Market review article version/pin mismatch: "+sid)
        for field in ("title","summary","published_at"):
            if field in row and row[field]!=a.get(field):
                raise ValueError("Read-only review text/time changed: "+sid)
        if row.get('review_method')=='upstream_evidence_mapping_v1':
            from .upstream import make_reviews
            minimum=row.get('upstream_min_confidence')
            if isinstance(minimum,bool) or not isinstance(minimum,(float,int)) or not 0<=minimum<=1:
                raise ValueError('Invalid upstream review threshold')
            expected=make_reviews([a],row['dataset'],minimum)[0]
            if expected!=row:raise ValueError('Upstream decision differs from pinned evidence')
        elif row.get('scope_claims') is not None:
            from .upstream import scope_claims
            minimum=row.get('upstream_min_confidence')
            if isinstance(minimum,bool) or not isinstance(minimum,(float,int)) or not 0<=minimum<=1:
                raise ValueError('Invalid upstream scope threshold')
            claims,error=scope_claims(a,minimum)
            if error or claims!=row['scope_claims']:
                raise ValueError('Typed scope claims require upstream evidence replay')
        decision=row.get("decision");counts[decision]+=1
        if decision not in ("pending","include","exclude"):
            raise ValueError("Review decision must be pending/include/exclude.")
        if decision=="pending":continue
        if not isinstance(row.get("reviewed_by"),str) or not row["reviewed_by"].strip() or len(text(row.get("rationale")))<8:
            raise ValueError("Resolved reviews need reviewer identity and a substantive rationale.")
        reviewed_at=timestamp(row.get("reviewed_at"))
        if reviewed_at.to_pydatetime()>datetime.now(timezone.utc):
            raise ValueError("Market review completion cannot be in the future.")
        if decision=="include":
            quote=text(row.get("evidence_quote"))
            source=text(a.get("title",""))+" "+text(a.get("summary",""))
            from .source_proof import bound_quote
            if len(quote)<8 or (quote.casefold() not in source.casefold() and not (row.get('scope_claims') and bound_quote(a,'scope_evidence',quote))):
                raise ValueError("Included market route needs a verbatim evidence quote from its pinned article.")
            publication=a.get("published_at")
            if not isinstance(publication,str) or not publication:
                raise ValueError("Market route requires a publication date.")
            routes.append({"source_id":sid,"article_pin":article_pin(a),"channel":"market","target_market":"US",
                "verification_status":"verified","mapping_source":"evidence_bound_market_review_v1",
                "valid_from":publication[:10] if len(publication)==10 else timestamp(publication).tz_convert("America/New_York").strftime("%Y-%m-%d"),"valid_to":None,"reviewed_at":reviewed_at.isoformat(),
                "reviewed_by":row["reviewed_by"],"evidence_quote":quote,"rationale":row["rationale"]})
            if row.get('scope_claims'):
                routes[-1].update(scope_claims=row['scope_claims'],mapping_method='upstream_typed_scope_reuse',
                    source_confidence=row['scope_claims'][0]['source_confidence'],mapping_confidence=None)
    if counts["pending"] and not allow_partial:
        raise ValueError("Market review still has pending rows; resolve them or explicitly publish with --allow-partial.")
    if counts["include"]+counts["exclude"]==0:
        if not reviews or not all(r.get('review_method')=='upstream_evidence_mapping_v1' for r in reviews) or not allow_partial:
            raise ValueError("No reviewed decisions; pending templates are not publishable.")
        # A fully pending upstream snapshot publishes zero routes, never fabricated coverage.
    return routes,dict(counts)


def publish_routes(articles,reviews,dataset,output,provenance,allow_partial=False,semantic_audit=None):
    if provenance.get("mode")!="verified_snapshot":
        raise ValueError("Market review publication requires a verified snapshot.")
    semantic_summary=None
    if semantic_audit is not None:
        from .semantic_review import apply_sample
        expected,semantic_summary=apply_sample(articles,semantic_audit['original_reviews'],semantic_audit['rows'],semantic_audit['confirmation'])
        if digest(expected)!=digest(reviews):raise ValueError('Semantic overlay differs from its bound evidence')
        semantic_summary['source_manifest_sha256']=semantic_audit['manifest_sha256']
    routes,counts=checked_reviews(articles,reviews,allow_partial,dataset)
    created=now()
    for route in routes:
        route.update(dataset=dataset,known_from=max(timestamp(created),timestamp(route["reviewed_at"])).isoformat())
    bindings=[dict(binding(a),title=a["title"],summary=a["summary"]) for a in articles]
    with new_artifact(output) as directory:
        write_records(directory/FILES[0],bindings)
        write_records(directory/FILES[1],reviews)
        write_records(directory/FILES[2],routes)
        if semantic_audit is not None:
            write_records(directory/'semantic_reviews.jsonl',semantic_audit['rows'])
            write_records(directory/'semantic_original_reviews.jsonl',semantic_audit['original_reviews'])
            write_json(directory/'semantic_user_confirmation.json',semantic_audit['confirmation'])
        write_json(directory/"manifest.json",{"schema_version":SCHEMA,"kind":"reviewed_market_routes","created_at":created,
            "dataset":dataset,"target_market":"US","source_input":provenance,"articles":len(articles),
            "decisions":counts,"accepted_routes":len(routes),"allow_partial":bool(allow_partial),
            "availability_basis":"review_and_publication_completion","semantic_review":semantic_summary,"files":file_manifest(directory)})
    load_routes(output,dataset)
    return {"dataset":dataset,"accepted_routes":len(routes),"decisions":counts,"output":str(output)}


def load_routes(directory,dataset,manifest_name="manifest.json"):
    directory=Path(directory)
    manifest=json.loads((directory/manifest_name).read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=SCHEMA or manifest.get("kind")!="reviewed_market_routes" or manifest.get("dataset")!=dataset or manifest.get("target_market")!="US":
        raise ValueError("Reviewed market-route artifact schema/dataset mismatch.")
    if manifest.get("source_input",{}).get("mode")!="verified_snapshot":
        raise ValueError("Market route source snapshot is not pinned.")
    if re.fullmatch(r"[a-f0-9]{64}",str(manifest["source_input"].get("snapshot_manifest_sha256"))) is None:
        raise ValueError("Market route source snapshot digest is missing.")
    verify_files(directory,manifest,required=FILES)
    articles=read_records(directory/FILES[0]);reviews=read_records(directory/FILES[1]);routes=read_records(directory/FILES[2])
    if manifest.get('semantic_review') is not None:
        from .semantic_review import apply_sample
        original=read_records(directory/'semantic_original_reviews.jsonl')
        sample=read_records(directory/'semantic_reviews.jsonl')
        confirmation=json.loads((directory/'semantic_user_confirmation.json').read_text(encoding='utf-8'))
        replay,summary=apply_sample(articles,original,sample,confirmation)
        claimed=dict(manifest['semantic_review']);claimed.pop('source_manifest_sha256',None)
        if digest(replay)!=digest(reviews) or summary!=claimed:
            raise ValueError('Semantic market review does not replay from its original decisions')
    expected,counts=checked_reviews(articles,reviews,manifest.get("allow_partial") is True,dataset)
    for route in expected:
        route.update(dataset=dataset,known_from=max(timestamp(manifest["created_at"]),timestamp(route["reviewed_at"])).isoformat())
    if routes!=expected or manifest.get("decisions")!=counts or manifest.get("articles")!=len(articles) or manifest.get("accepted_routes")!=len(routes):
        raise ValueError("Market routes differ from their evidence-bound review decisions.")
    return routes,articles,manifest


def feature_routes(directory,axis,articles,feature_manifest):
    routes,bindings,manifest=load_routes(directory,axis.dataset)
    by_id=keyed(bindings)
    provenance=feature_manifest.get("input_provenance") or {}
    snapshot_hash=provenance.get("snapshot_manifest_sha256")
    if snapshot_hash and snapshot_hash!=manifest["source_input"]["snapshot_manifest_sha256"]:
        raise ValueError("FinBERT features and market reviews belong to different snapshots.")
    for article in articles:
        if article["source_id"] not in by_id or article_pin(article)!=article_pin(by_id[article["source_id"]]):
            raise ValueError("FinBERT article version differs from reviewed market bindings.")
    encoded_ids={a["source_id"] for a in articles}
    return [route for route in routes if route["source_id"] in encoded_ids],manifest


def assert_panel_routes(directory,axis,manifest):
    directory=Path(directory)
    metadata=manifest.get("market_routing")
    if not isinstance(metadata,dict) or not metadata.get("manifest_sha256"):
        raise ValueError("Market routing flag lacks a pinned review artifact; rebuild the panel.")
    if digest_file(directory/"market_routing_manifest.json")!=metadata["manifest_sha256"]:
        raise ValueError("Panel market-routing manifest digest mismatch.")
    routes,bindings,review=load_routes(directory,axis.dataset,"market_routing_manifest.json")
    allowed={row["source_id"]:row for row in routes}
    if metadata.get("accepted_routes")!=len(routes) or metadata.get("source_snapshot_manifest_sha256")!=review["source_input"]["snapshot_manifest_sha256"]:
        raise ValueError("Panel routing metadata differs from its review evidence.")
    scoped=any(any(c['scope']!='US_EQUITY' for c in r.get('scope_claims',[])) for r in routes)
    owners=[]
    if scoped:
        if manifest.get('market_layout')!='stock_date' or not manifest.get('historical_relations_verified'):
            raise ValueError('Exchange-specific market scope requires per-stock historical mapping')
        from .relations import load_relations,validate_master
        load_relations(directory,axis,'relations_manifest.json','relation_')
        owners,_,_=validate_master(axis,read_records(directory/'relation_security_master.jsonl'),[],[])
    articles=keyed(bindings)
    quality=manifest.get('market_quality')
    if quality is not None:
        # 市场通道专项：新闻路由只能是已审核路由的子集；官方事件路由按固定规则生成并单独落盘。
        official=read_records(directory/'official_market_routes.jsonl')
        if (quality.get('official') or {}).get('routes_sha256',digest([]))!=digest(official):
            raise ValueError('Panel official market routes differ from their recorded digest')
        if any(r.get('mapping_source')!='official_release_rule_v1' or r['source_id'] in allowed for r in official):
            raise ValueError('Official market routes must use the official rule and not collide with news routes')
        allowed=dict(allowed,**{r['source_id']:r for r in official})
    for row in read_records(directory/"provenance.jsonl"):
        if row.get("channel")=="market" and row.get("source_id") not in allowed:
            raise ValueError("Panel contains an unreviewed market article.")
        if scoped and row.get('channel')=='market':
            from .relations import publication_day
            from .upstream import market_owners
            route=allowed[row['source_id']]
            claims=route.get('scope_claims') or [{'scope':'US_EQUITY'}]
            permitted={axis.stock_uids[i] for claim in claims for i,_ in
                       market_owners(claim['scope'],owners,axis,publication_day(articles[row['source_id']]))}
            if row.get('stock_uid') not in permitted:
                raise ValueError('Market scope propagated outside verified historical exchange membership')
    return review
