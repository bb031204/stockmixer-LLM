"""Read-only collector snapshot adaptation. No imports from the collector or NLP calls."""
import bisect
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .inputs import normalize_article
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .time_axis import Axis

ADAPTER_SCHEMA = "stockmixer_source_adapter_v1"
IMMUTABLE_TEXT_FIELDS = ("source_id", "url", "时间", "分类", "关键词", "主要内容")
SYSTEM_FIELDS = ("first_seen_at", "fetched_at", "llm_inferred_at", "extracted_at",
                 "commit_verified_at", "feature_available_at")


def producer_hash(value):
    """Exact collector digest convention; deliberately differs from news_fusion.io.digest."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant in source snapshot.")


def parse_json(raw):
    return json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)


def contained(root, relative):
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError:
        raise ValueError("Source snapshot path escaped its declared root.") from None
    return result


class Snapshot:
    def __init__(self, root, markets, allow_append_only=False):
        self.root = Path(root).resolve()
        self.markets = tuple(markets)
        if not self.markets or len(set(self.markets)) != len(self.markets) or not set(self.markets) <= {"US", "GLOBAL"}:
            raise ValueError("Adaptation markets must be a unique selection of US/GLOBAL.")
        self.hashes = {}
        self.started_at = now()
        self.allow_append_only = allow_append_only
        self.new_files = []
        self.membership = self.list_inputs()
        self.originals = None

    def list_inputs(self):
        result = {}
        for market in self.markets:
            folder = self.root / "data" / market
            if not folder.is_dir():
                raise FileNotFoundError("Source market directory is missing: " + market)
            result[market] = sorted(p.relative_to(self.root).as_posix() for p in folder.glob("*.json") if p.is_file())
        return result

    def read(self, relative):
        path = contained(self.root, relative)
        raw = path.read_bytes()
        hashed = hashlib.sha256(raw).hexdigest()
        key = path.relative_to(self.root).as_posix()
        if key in self.hashes and self.hashes[key] != hashed:
            raise ValueError("Source file changed during snapshot read.")
        self.hashes[key] = hashed
        return raw

    def json(self, relative):
        return parse_json(self.read(relative))

    def verify(self):
        current = self.list_inputs()
        old = {p for files in self.membership.values() for p in files}
        new = {p for files in current.values() for p in files}
        if old - new or (new - old and not self.allow_append_only):
            raise ValueError("Source directory membership changed; retry a stable snapshot.")
        self.new_files = sorted(new - old)
        for relative, expected in self.hashes.items():
            if digest_file(contained(self.root, relative)) != expected:
                raise ValueError("Source content changed; incomplete snapshot will not be published.")

    def archive_originals(self):
        if self.originals is not None:
            return self.originals
        self.originals = {}
        for filename in ("data/US_original.zip", "data/GLOBAL_original.zip"):
            path = self.root / filename
            if not path.is_file():
                continue
            # Read archive bytes from the same hashed snapshot. Nothing is extracted.
            import io
            with zipfile.ZipFile(io.BytesIO(self.read(filename))) as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.lower().endswith(".json"):
                        continue
                    if member.file_size > 4 * 1024 * 1024:
                        raise ValueError("Unexpected oversized original JSON in archive.")
                    row = parse_json(archive.read(member))
                    identity = row.get("source_id")
                    if not isinstance(identity, str):
                        continue
                    if identity in self.originals and producer_hash(self.originals[identity]) != producer_hash(row):
                        raise ValueError("Conflicting original versions in archives.")
                    self.originals[identity] = row
        return self.originals


def choose_audit(snapshot, row):
    identity = row["source_id"]
    candidates, matches = [], []
    for family, relative in (("retraced", "audit/" + identity + ".json"),
                              ("production", ".runtime/simple/audit/" + identity + ".json"),
                              ("enriched", "data_enriched/analysis/" + identity + ".json")):
        if not (snapshot.root / relative).is_file():
            continue
        audit = snapshot.json(relative)
        descriptor = {"family": family, "path": relative, "sha256": snapshot.hashes[relative],
                      "record_hash_matches": audit.get("record_hash") == producer_hash(row)}
        candidates.append(descriptor)
        if descriptor["record_hash_matches"]:
            if audit.get("source_id") != identity:
                raise ValueError("Matching audit hash has an inconsistent source_id.")
            published = audit.get("published_at", audit.get("source_published_at"))
            if published is not None and published != row.get("时间"):
                raise ValueError("Matching audit has a contradictory source publication.")
            if audit.get("url") is not None and audit["url"] != row.get("url"):
                raise ValueError("Matching audit has a contradictory source URL.")
            matches.append((audit, descriptor))
    if len(matches) > 1:
        # Never choose a later timestamp or overwrite one family with another implicitly.
        raise ValueError("Ambiguous matching audit versions; select a producer snapshot first.")
    if not matches:
        return {}, None, candidates
    return matches[0][0], matches[0][1], candidates


def resolve_language(snapshot, row, audit, descriptor):
    value = row.get("language") or audit.get("language")
    if isinstance(value, str) and value.strip():
        return value.lower().strip(), "current_record" if row.get("language") else "matching_current_audit", []
    if descriptor and descriptor["family"] == "enriched":
        old_path = ".runtime/simple/audit/" + row["source_id"] + ".json"
        if (snapshot.root / old_path).is_file():
            old = snapshot.json(old_path)
            original = snapshot.archive_originals().get(row["source_id"])
            if (original and snapshot.hashes[old_path] == audit.get("matching_audit_hash")
                    and old.get("record_hash") == audit.get("original_record_hash") == producer_hash(original)
                    and all(original.get(k) == row.get(k) for k in IMMUTABLE_TEXT_FIELDS)
                    and isinstance(old.get("language"), str) and old["language"]):
                return old["language"].lower(), "legacy_audit_verified_unchanged_summary", []
    return "unknown", "unresolved", ["language_evidence_missing"]


def utc_value(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp_type_invalid")
    if result.tzinfo is None:
        raise ValueError("timestamp_timezone_missing")
    return result.astimezone(timezone.utc)


def publication(value, stated_precision):
    if not isinstance(value, str):
        raise ValueError("publication_missing")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        date.fromisoformat(value)
        if stated_precision not in (None, "date"):
            raise ValueError("publication_precision_conflict")
        return value, "date", None
    stamp = utc_value(value)
    precision = "second" if re.search(r"T\d{2}:\d{2}:\d{2}", value) else "minute"
    if stated_precision not in (None, precision):
        raise ValueError("publication_precision_conflict")
    return value, precision, stamp


def adapt_times(row, audit, descriptor):
    info = row.get("时间信息") or {}
    errors, warnings, evidence, parsed = [], [], {}, {}
    try:
        published, precision, published_stamp = publication(row.get("时间"), info.get("publication_precision"))
    except (ValueError, TypeError) as error:
        published, precision, published_stamp = row.get("时间"), info.get("publication_precision"), None
        errors.append("publication_invalid:" + type(error).__name__)
    result = {"published_at": published, "publication_precision": precision}
    for key in SYSTEM_FIELDS:
        sources = []
        if info.get(key) is not None:
            sources.append(("record_time_information", info[key]))
        if audit.get(key) is not None:
            sources.append(("matching_audit", audit[key]))
        evidence[key] = [{"source": source, "value": value} for source, value in sources]
        values = []
        for source, value in sources:
            try:
                values.append(utc_value(value))
            except (ValueError, TypeError, OverflowError):
                errors.append(key + ":invalid_timestamp")
        # Preserve all evidence; canonical timestamps select the conservative later value.
        parsed[key] = max(values) if values else None
        result[key] = parsed[key].isoformat() if parsed[key] else None
        if len(set(values)) > 1:
            warnings.append(key + ":sources_differ_use_later")
    raw_availability = info.get("feature_available_at")
    basis = info.get("availability_basis") or "unknown"
    result["source_feature_available_at"] = raw_availability
    result["source_availability_basis"] = basis
    result["llm_time_required"] = bool(audit.get("model"))
    necessary = [value for value in parsed.values() if value is not None]
    if published_stamp is not None:
        necessary.append(published_stamp)
    effective = max(necessary) if necessary and parsed["feature_available_at"] else None
    result["feature_available_at"] = effective.isoformat() if effective else None
    result["effective_feature_available_at"] = result["feature_available_at"]
    if effective and parsed["feature_available_at"] and effective > parsed["feature_available_at"]:
        warnings.append("feature_available_at_raised_to_latest_evidence")
    for key in ("first_seen_at", "fetched_at", "feature_available_at"):
        if parsed[key] is None:
            warnings.append(key + ":unknown")
    if result["llm_time_required"] and parsed["llm_inferred_at"] is None:
        warnings.append("llm_inferred_at:unknown")
    if effective is None:
        errors.append("feature_availability_unknown")
    if basis not in ("extraction", "verified_audit", "migration", "observed", "retrospective_retrace_current_inference_not_historical_PIT"):
        errors.append("availability_basis_unknown")
    upstream_observed = bool(descriptor and not errors and basis in ("extraction", "verified_audit", "observed")
        and all(parsed[k] for k in ("first_seen_at", "fetched_at", "commit_verified_at", "feature_available_at"))
        and (not result["llm_time_required"] or parsed["llm_inferred_at"]))
    # An extraction is only reclassified with matching content AND complete measured timing evidence.
    result["availability_basis"] = "verified_audit" if upstream_observed else basis
    result["upstream_observed_evidence_complete"] = upstream_observed
    result["availability_mapping_reason"] = "hash_matched_audit_and_measured_commit" if upstream_observed else "source_basis_preserved"
    result["date_only_policy"] = "next_trading_session" if precision == "date" else "verified_decision_cutoff_required"
    return result, {"errors": sorted(set(errors)), "warnings": sorted(set(warnings)), "evidence": evidence}


def adapt_one(snapshot, row, locations):
    audit, descriptor, candidates = choose_audit(snapshot, row)
    errors = []
    if descriptor is None:
        errors.append("matching_current_audit_missing")
    if audit.get("status") in ("failed", "error", "needs_review"):
        errors.append("matching_audit_not_completed")
    language, language_basis, language_errors = resolve_language(snapshot, row, audit, descriptor)
    errors.extend(language_errors)
    market = "GLOBAL" if "GLOBAL" in locations else "US"
    canonical = audit.get("canonical_market")
    if canonical in ("US", "GLOBAL"):
        if canonical != market:
            errors.append("canonical_market_conflicts_with_views")
        market = canonical
    elif canonical is not None:
        errors.append("unsupported_canonical_market")
    times, check = adapt_times(row, audit, descriptor)
    errors.extend(check["errors"])
    scopes = row.get("影响范围") or []
    if not isinstance(scopes, list) or not set(scopes) <= {"company", "industry", "market"}:
        errors.append("impact_scope_invalid")
        scopes = []
    record = {"schema_version": ADAPTER_SCHEMA, "source_schema_version": "public-title-11" if len(row) == 11 else "legacy_public_json",
        "source_id": row["source_id"], "title": row.get("title"), "summary": row.get("主要内容"),
        "language": language, "language_basis": language_basis, "market": market, "impact_scope": scopes,
        "record_version": producer_hash(row), "source_record_hash": producer_hash(row),
        "qwen_model_name": audit.get("model"), "qwen_model_revision": audit.get("model_digest"),
        "prompt_version": audit.get("prompt_version"), "producer_code_version": audit.get("code_version"),
        "producer_schema_version": audit.get("schema_version"), "analysis_method": (audit.get("analysis") or {}).get("method"),
        "source_verification": audit.get("source_verification", "not_revalidated_by_adapter"),
        "availability_comparison": "strictly_before_cutoff",
        "entities": row.get("股票实体") or [], "industries": row.get("涉及行业") or [],
        "events": audit.get("events") or [], "industry_impacts": audit.get("industry_impacts") or [],
        "security_mapping_status": "unresolved", "historical_industry_mapping_status": "unresolved",
        "producer_quality_flags": audit.get("quality_flags", audit.get("relation_quality_flags", [])),
        "adapter_schema_version": ADAPTER_SCHEMA, **times}
    # Reuse only analysis bound to the exact producer record; no NLP calls here.
    record['upstream_analysis'] = {'record_hash': producer_hash(row), 'audit': descriptor,
        'schema_version': audit.get('schema_version'), 'raw_industry_impacts': audit.get('industry_impacts') or [],
        'content_hash':audit.get('content_hash'),'industry_dictionary':audit.get('industry_dictionary')}
    if descriptor and audit.get('evidence_pack'):
        pin = audit['evidence_pack']
        relative = pin.get('path')
        # Retrace exports historically pin paths relative to the batch, rather than adapted/.
        if descriptor['family'] == 'retraced' and isinstance(relative,str) and relative.startswith('adapted/'):
            relative = relative[len('adapted/'):]
        if not isinstance(relative,str):
            raise ValueError('Missing evidence pack path')
        pack_raw = snapshot.read(relative)
        if hashlib.sha256(pack_raw).hexdigest() != pin.get('sha256'):
            raise ValueError('Upstream evidence pack hash mismatch')
        pack = parse_json(pack_raw)
        if (pack.get('record_hash') != producer_hash(row) or pack.get('source_id') != row['source_id']
                or pack.get('source_body_hash') != audit.get('source_body_hash')):
            raise ValueError('Upstream evidence pack record/body binding mismatch')
        record['upstream_analysis']['bound_source_evidence'] = {
            'audit_json': snapshot.read(descriptor['path']).decode('utf-8'),
            'pack_json': pack_raw.decode('utf-8')}
    for key in ('market_scope', 'affected_markets', 'scope_evidence'):
        if key in row and key in audit and row[key] != audit[key]:
            errors.append('conflicting_upstream_' + key)
        if key in audit or key in row:
            record[key] = audit.get(key, row.get(key))
    from .upstream import industry_impact
    try:
        record['industry_impacts'] = [industry_impact(v) for v in record['industry_impacts']]
    except ValueError:
        errors.append('conflicting_upstream_industry_schema')
    if descriptor and audit.get('market_scope_version')=='market-proof-1':
        record['industry_pending_impacts']=[i for i in record['industry_impacts'] if i.get('propagation_approved') is not True]
        record['industry_impacts']=[i for i in record['industry_impacts'] if i.get('propagation_approved') is True]
    normalized = normalize_article(record)
    if normalized["input_status"] != "ready":
        errors.append("encoder_input:" + normalized["input_status"])
    record["adapter_status"] = "ready" if not errors else "review_required"
    record["adapter_errors"] = sorted(set(errors))
    record["input_text_hash"] = normalized["input_text_hash"]
    provenance = {"source_id": row["source_id"], "record_hash": producer_hash(row), "source_views": sorted(locations),
        "selected_audit": descriptor, "audit_candidates": candidates, "language_basis": language_basis,
        "version_resolution": "exact_producer_record_hash_only", "security_mapping_status": "unresolved"}
    if language_basis == "legacy_audit_verified_unchanged_summary":
        provenance["language_proof"] = {"original_record_hash": audit.get("original_record_hash"),
            "old_audit_sha256": audit.get("matching_audit_hash"), "unchanged_fields": list(IMMUTABLE_TEXT_FIELDS),
            "archive_files": [p for p in snapshot.hashes if p.endswith(".zip")]}
    check.update(source_id=row["source_id"], errors=sorted(set(errors)),
                 effective_feature_available_at=times["effective_feature_available_at"],
                 upstream_observed_evidence_complete=times["upstream_observed_evidence_complete"])
    return record, provenance, check


def validate_adapted(directory, source_root=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ADAPTER_SCHEMA or manifest.get("kind") != "adapted_collector_snapshot":
        raise ValueError("Unsupported adapted source schema.")
    verify_files(directory, manifest)
    source = Path(source_root or manifest["source_root"]).resolve()
    snapshot = json.loads((directory / "source_snapshot.json").read_text(encoding="utf-8"))
    for relative, expected in snapshot["files"].items():
        if digest_file(contained(source, relative)) != expected:
            raise ValueError("Source has changed since adaptation: " + relative)
    reader = Snapshot(source, snapshot["markets"])
    old = {p for files in snapshot["market_files"].values() for p in files}
    current = {p for files in reader.membership.values() for p in files}
    if old - current or (current - old and not snapshot.get("allow_append_only", False)):
        raise ValueError("Source membership has changed since adaptation.")
    rows = read_records(directory / "articles.jsonl")
    if len({row["source_id"] for row in rows}) != len(rows) or len(rows) != manifest["ready_articles"]:
        raise ValueError("Adapted source identity/count mismatch.")
    if any(row["adapter_status"] != "ready" or normalize_article(row)["input_status"] != "ready" for row in rows):
        raise ValueError("Adapted encoder input validation failed.")
    return {"valid": True, "source_hashes_unchanged": True, "ready_articles": len(rows),
            "source_files_verified": len(snapshot["files"]), "new_source_files_outside_snapshot": len(current - old)}


def adapt_snapshot(source_root, output, axes_dir=None, markets=("GLOBAL", "US"), progress=None):
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    try:
        output.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("Read-only adapter output must be outside the source project.")
    if output.exists():
        raise FileExistsError("Choose a new adapted snapshot directory; previous artifacts are not overwritten.")
    # A live collector may append new articles. Freeze this run's file list, verify every included input,
    # and report later additions separately rather than stopping the producer or mixing them into this version.
    snapshot = Snapshot(source_root, markets, allow_append_only=True)
    rows, locations, physical, aliases = {}, defaultdict(set), Counter(), defaultdict(list)
    for market, files in snapshot.membership.items():
        for relative in files:
            row = snapshot.json(relative)
            identity = row.get("source_id")
            if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
                raise ValueError("Source ID must be a safe, nonempty collector identifier.")
            if identity in rows and producer_hash(rows[identity]) != producer_hash(row):
                raise ValueError("Conflicting copies for one source_id; no latest-version guessing.")
            rows[identity] = row
            locations[identity].add(market)
            aliases[identity].append({"path": relative, "sha256": snapshot.hashes[relative]})
            physical[market] += 1
    if not rows:
        raise ValueError("No source JSON records found.")
    records, provenance, checks = [], [], []
    for index, identity in enumerate(sorted(rows), start=1):
        record, proof, check = adapt_one(snapshot, rows[identity], locations[identity])
        records.append(record)
        provenance.append(proof)
        checks.append(check)
        if progress and (index % 100 == 0 or index == len(rows)):
            progress(index, len(rows))
    # Attach aliases from the actual source IDs, not from filename-derived identities.
    for proof in provenance:
        proof["source_files"] = sorted(aliases[proof["source_id"]], key=lambda value: value["path"])
    content = defaultdict(list)
    for record in records:
        content[record["input_text_hash"]].append(record["source_id"])
    duplicate_content = [{"input_text_hash": key, "source_ids": ids, "action": "retain_distinct_ids_for_asof_dedup"}
                         for key, ids in sorted(content.items()) if len(ids) > 1]
    ready = [r for r in records if r["adapter_status"] == "ready"]
    review = [r for r in records if r["adapter_status"] != "ready"]
    datasets, axis_hashes = {}, {}
    if axes_dir is not None:
        for name in ("NASDAQ", "NYSE", "SP500"):
            axis = Axis.load(Path(axes_dir) / (name + ".json"))
            axis_hashes[name] = digest_file(Path(axes_dir) / (name + ".json"))
            counts = Counter()
            for record in ready:
                day = record["published_at"][:10]
                if not axis.dates[0] <= day <= axis.dates[-1]:
                    continue
                counts["publication_in_range"] += 1
                availability = record.get("effective_feature_available_at")
                counts["availability_within_dataset_end_upper_bound"] += bool(availability and availability[:10] <= axis.dates[-1])
                if record["publication_precision"] == "date":
                    index = bisect.bisect_right(axis.dates, day)
                    counts["date_only_with_next_dataset_session"] += index < len(axis.dates)
            datasets[name] = {"first_date": axis.dates[0], "last_date": axis.dates[-1], **dict(counts),
                "full_observed_pit_ready": False, "reason": "FinBERT/model/mapping availability and verified decision schedule still required",
                "session_assignment": "date-only uses strictly next session; exact-time cutoff not guessed"}
    snapshot.verify()
    with new_artifact(output) as directory:
        write_records(directory / "articles.jsonl", ready)
        write_records(directory / "review_records.jsonl", review)
        write_records(directory / "provenance.jsonl", provenance)
        write_records(directory / "time_checks.jsonl", checks)
        write_records(directory / "duplicate_content_groups.jsonl", duplicate_content)
        source_manifest = {"source_root": str(source_root), "markets": list(markets),
                           "snapshot_started_at": snapshot.started_at, "allow_append_only": True,
                           "market_files": snapshot.membership, "files": dict(sorted(snapshot.hashes.items()))}
        write_json(directory / "source_snapshot.json", source_manifest)
        manifest = {"schema_version": ADAPTER_SCHEMA, "kind": "adapted_collector_snapshot", "created_at": now(),
            "source_root": str(source_root), "source_read_only": True, "input_physical_files": dict(physical),
            "snapshot_policy": "frozen_input_file_list; included files must not change; later additions are excluded",
            "new_source_files_outside_snapshot_at_verification": len(snapshot.new_files),
            "unique_articles": len(records), "ready_articles": len(ready), "review_required": len(review),
            "duplicate_views_removed": sum(physical.values()) - len(records), "content_duplicate_groups": len(duplicate_content),
            "deduplication": "source_id+identical_record_hash; distinct IDs are never globally merged using future evidence",
            "language_basis_counts": dict(Counter(r["language_basis"] for r in records)),
            "audit_family_counts": dict(Counter((p["selected_audit"] or {}).get("family", "missing") for p in provenance)),
            "source_availability_basis_counts": dict(Counter(r["source_availability_basis"] for r in records)),
            "canonical_availability_basis_counts": dict(Counter(r["availability_basis"] for r in records)),
            "llm_time_known": sum(bool(r["llm_inferred_at"]) for r in records),
            "upstream_observed_evidence_complete": sum(r["upstream_observed_evidence_complete"] for r in records),
            "warning_counts": dict(Counter(w for check in checks for w in check["warnings"])),
            "error_counts": dict(Counter(e for check in checks for e in check["errors"])), "datasets": datasets,
            "model_inference_run": False, "baseline_modified": False, "source_modules_imported": False,
            "source_snapshot_hash": digest(source_manifest), "axis_file_sha256": axis_hashes,
            "adapter_code_sha256": {p.name: digest_file(p) for p in Path(__file__).parent.glob("*.py")},
            "historical_security_or_industry_mapping_created": False, "files": file_manifest(directory)}
        write_json(directory / "manifest.json", manifest)
        # Fail atomically if a live collector changed any record or evidence while output was written.
        snapshot.verify()
    return manifest
