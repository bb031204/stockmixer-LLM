"""Read formal title/summary only; never infer tickers, language or timestamps."""
import re
import unicodedata
from datetime import date, datetime
from collections import Counter
from .io import digest, read_records

def text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip() if isinstance(value, str) else ""

def normalize_article(row):
    source = row.get("source_id")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Article missing nonempty source_id; no identity is guessed.")
    title = text(row.get("title", row.get("标题", "")))
    summary = text(row.get("summary", row.get("主要内容", "")))
    if title.casefold() == summary.casefold():
        title = ""
    language = str(row.get("language", "unknown")).lower()
    cjk = len(re.findall(r"[\u3400-\u9fff]", summary))
    latin = len(re.findall(r"[A-Za-z]", summary))
    status = "ready"
    if not summary:
        status = "empty_text"
    elif language not in ("en", "en-us", "en-gb") or cjk > max(2, latin):
        status = "unsupported_language"
    if "<html" in summary.lower() or "<script" in summary.lower():
        status = "invalid_text"
    scopes = row.get("impact_scope", row.get("影响范围", []))
    if isinstance(scopes, str):
        scopes = [scopes]
    elif isinstance(scopes,tuple) or hasattr(scopes,"tolist"):
        scopes = list(scopes)
    names = {"公司":"company", "行业":"industry", "市场":"market"}
    scopes = [names.get(s,s) for s in scopes if isinstance(s,str)] if isinstance(scopes,list) else []
    result = {key:row.get(key) for key in ("published_at", "publication_precision", "first_seen_at",
        "fetched_at", "feature_available_at", "availability_basis", "importance", "importance_verified",
        "qwen_model_revision", "prompt_version", "record_version", "llm_inferred_at", "extracted_at",
        "commit_verified_at", "effective_feature_available_at", "llm_time_required", "availability_comparison")}
    result={k:(v.isoformat() if isinstance(v,(date,datetime)) else v) for k,v in result.items()}
    result["input_schema_version"]=row.get("schema_version")
    for key in ("adapter_schema_version", "source_schema_version", "source_record_hash", "source_feature_available_at",
                "source_availability_basis", "upstream_observed_evidence_complete", "language_basis",
                "availability_mapping_reason", "producer_code_version", "producer_schema_version", "qwen_model_name",
                "analysis_method", "source_verification", "adapter_status", "adapter_errors"):
        if key in row:
            result[key]=row[key]
    if result["published_at"] is None and isinstance(row.get("时间"),str):
        result["published_at"]=row["时间"]
    result.update(source_id=source.strip(), title=title, summary=summary, language=language,
        market=row.get("market"), impact_scope=scopes, input_status=status, input_text_hash=digest([title,summary]))
    return result

def load_articles(paths):
    articles, seen, stats = [], {}, Counter()
    for path in paths:
        for raw in read_records(path):
            if raw.get("adapter_schema_version")=="stockmixer_source_adapter_v1" and raw.get("adapter_status")!="ready":
                raise ValueError("Adapted record requires review; encode articles.jsonl only.")
            row = normalize_article(raw)
            if row["source_id"] in seen:
                if digest(row) != seen[row["source_id"]]:
                    raise ValueError("Conflicting versions for one source_id; select an as-of snapshot.")
                stats["duplicate_source_id"] += 1
                continue
            seen[row["source_id"]] = digest(row)
            articles.append(row)
            stats[row["input_status"]] += 1
    if not articles:
        raise ValueError("No formal news records found; no synthetic real-data artifact will be generated.")
    return articles, dict(stats)
