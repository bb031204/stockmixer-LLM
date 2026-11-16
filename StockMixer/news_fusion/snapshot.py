"""Freeze exact current news membership while inheriting an immutable adapted parent."""
import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from .inputs import normalize_article
from .io import digest, digest_file, file_manifest, new_artifact, now, read_records, verify_files, write_json, write_records
from .source_adapter import ADAPTER_SCHEMA, parse_json, producer_hash
from .outputs import destination, PROJECT

SCHEMA="stockmixer_current_snapshot_v1"
REQUIRED=("articles.jsonl","provenance.jsonl","time_checks.jsonl","raw_news.zip","raw_index.jsonl",
          "selection_ledger.jsonl","selection.json","parent_artifact.zip","excluded_parent_articles.jsonl",
          "excluded_parent_provenance.jsonl","duplicate_content_groups.jsonl")


def keyed(rows,label):
    result={}
    for row in rows:
        sid=row.get("source_id")
        if not isinstance(sid,str) or not sid.strip() or sid in result:
            raise ValueError(label+": source_id is missing or duplicated.")
        result[sid]=row
    return result


def jsonl_bytes(payload):
    return [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]


def input_paths(root):
    result=[]
    for scope in ("US","GLOBAL"):
        folder=root/scope
        if not folder.is_dir():
            raise FileNotFoundError("Missing current input directory: "+str(folder))
        for path in folder.rglob("*.json"):
            if path.is_file():
                path.resolve().relative_to(root)
                result.append(path)
    return sorted(result)


def parent_payload(parent):
    manifest=json.loads((parent/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=ADAPTER_SCHEMA or manifest.get("kind")!="adapted_collector_snapshot":
        raise ValueError("An intact original adapted parent snapshot is required.")
    verify_files(parent,manifest,required=("articles.jsonl","provenance.jsonl","time_checks.jsonl","source_snapshot.json"))
    payload={name:(parent/name).read_bytes() for name in ("manifest.json",*sorted(manifest["files"]))}
    if hashlib.sha256(payload["manifest.json"]).hexdigest()!=digest_file(parent/"manifest.json"):
        raise ValueError("Parent manifest changed during read.")
    for name,expected in manifest["files"].items():
        if hashlib.sha256(payload[name]).hexdigest()!=expected:
            raise ValueError("Parent artifact changed during read: "+name)
    return manifest,payload


def check_current(root,index):
    expected={row["path"]:row["file_sha256"] for row in index}
    paths=input_paths(root)
    if {p.relative_to(root).as_posix() for p in paths}!=set(expected):
        raise ValueError("Current input membership changed; do not publish a mixed snapshot.")
    for path in paths:
        if digest_file(path)!=expected[path.relative_to(root).as_posix()]:
            raise ValueError("Current source changed while freezing: "+str(path))


def freeze_current(root,parent,output):
    root,parent,output=(Path(p).resolve() for p in (root,parent,output))
    # A derived directory inside adapted/ is allowed; US/GLOBAL raw inputs cannot be overwritten.
    for raw in (root/"US",root/"GLOBAL",parent):
        try:
            output.relative_to(raw)
        except ValueError:
            pass
        else:
            raise ValueError("Snapshot output must be separate from raw input and parent artifacts.")
    manifest,payload=parent_payload(parent)
    parent_rows=keyed(jsonl_bytes(payload["articles.jsonl"]),"parent articles")
    parent_proof=keyed(jsonl_bytes(payload["provenance.jsonl"]),"parent provenance")
    parent_times=keyed(jsonl_bytes(payload["time_checks.jsonl"]),"parent time checks")
    if len(parent_rows)!=manifest["ready_articles"]:
        raise ValueError("Parent ready-article count mismatch.")
    raw_bytes,index,versions,aliases={},[],{},defaultdict(list)
    for path in input_paths(root):
        relative=path.relative_to(root).as_posix()
        data=path.read_bytes(); row=parse_json(data)
        sid=row.get("source_id")
        if not isinstance(sid,str) or not sid.strip():
            raise ValueError("Current source_id is missing.")
        version=producer_hash(row)
        if sid in versions and versions[sid]!=version:
            raise ValueError("Conflicting current versions for source_id: "+sid)
        versions[sid]=version; aliases[sid].append(relative); raw_bytes[relative]=data
        index.append({"path":relative,"market_view":PurePosixPath(relative).parts[0],"source_id":sid,
                      "file_sha256":hashlib.sha256(data).hexdigest(),"source_record_hash":version})
    if not versions:
        raise ValueError("No real current news articles; no empty selection is published.")
    added=sorted(set(versions)-set(parent_rows))
    changed=sorted(sid for sid in versions if sid in parent_rows and parent_rows[sid].get("source_record_hash")!=versions[sid])
    if added or changed:
        raise ValueError("New/changed current articles require fresh evidence adaptation; new="+str(added)+" changed="+str(changed))
    selected,proofs,times,ledger=[],[],[],[]
    for sid in sorted(versions):
        article=parent_rows[sid]
        if article.get("adapter_status")!="ready" or normalize_article(article)["input_status"]!="ready":
            raise ValueError("Selected parent article is not encoding-ready: "+sid)
        if sid not in parent_proof or sid not in parent_times or parent_proof[sid].get("record_hash")!=versions[sid]:
            raise ValueError("Missing/mismatched inherited evidence: "+sid)
        selected.append(article); proofs.append(parent_proof[sid]); times.append(parent_times[sid])
        ledger.append({"source_id":sid,"source_record_hash":versions[sid],"current_paths":sorted(aliases[sid]),
            "parent_article_hash":digest(article),"parent_provenance_hash":digest(parent_proof[sid]),
            "action":"exact_hash_match; inherit unchanged article and historical parent evidence"})
    excluded=sorted(set(parent_rows)-set(versions))
    content=defaultdict(list)
    for row in selected:
        content[row["input_text_hash"]].append(row["source_id"])
    duplicates=[{"input_text_hash":key,"source_ids":ids,"action":"retain IDs; deduplicate causally at panel time"}
                for key,ids in sorted(content.items()) if len(ids)>1]
    check_current(root,index)
    with new_artifact(output) as directory:
        with zipfile.ZipFile(directory/"raw_news.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,data in sorted(raw_bytes.items()):
                archive.writestr(name,data)
        with zipfile.ZipFile(directory/"parent_artifact.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for name,data in sorted(payload.items()):
                archive.writestr(name,data)
        for name,rows in (("articles.jsonl",selected),("provenance.jsonl",proofs),("time_checks.jsonl",times),
                          ("raw_index.jsonl",index),("selection_ledger.jsonl",ledger),
                          ("excluded_parent_articles.jsonl",[parent_rows[sid] for sid in excluded]),
                          ("excluded_parent_provenance.jsonl",[parent_proof[sid] for sid in excluded]),
                          ("duplicate_content_groups.jsonl",duplicates)):
            write_records(directory/name,rows)
        selection={"current_source_root":str(root),"parent_directory":str(parent),
            "parent_manifest_sha256":hashlib.sha256(payload["manifest.json"]).hexdigest(),
            "parent_articles":len(parent_rows),"selected_articles":len(selected),"excluded_parent_ids":excluded,
            "new_current_ids":[],"changed_current_ids":[],"raw_files":len(index),
            "physical_market_files":{scope:sum(row["market_view"]==scope for row in index) for scope in ("US","GLOBAL")},
            "selection_rule":"current membership AND exact producer source_record_hash; no text-only matching",
            "evidence_basis":"inherited intact parent artifact; upstream external files are referenced, not newly re-authenticated",
            "time_policy":"all inherited timestamps unchanged; frozen_at is not historical feature availability"}
        write_json(directory/"selection.json",selection)
        result={"schema_version":SCHEMA,"kind":"frozen_current_selection","created_at":now(),"source_read_only":True,
            "source_root":str(root),"parent_source_root":manifest.get("source_root"),"ready_articles":len(selected),
            "parent_manifest_sha256":selection["parent_manifest_sha256"],"articles":len(selected),"raw_files":len(index),
            "excluded_parent_articles":len(excluded),"source_membership_hash":digest(index),
            "selected_article_hash":digest(selected),"language_basis_counts":dict(Counter(row.get("language_basis") for row in selected)),
            "time_mode":"retrospective","model_inference_run":False,"historical_observed_pit_ready":False,
            "publication_range":[min(row["published_at"] for row in selected),max(row["published_at"] for row in selected)],
            "code_sha256":{name:digest_file(Path(__file__).parent/name) for name in ("snapshot.py","source_adapter.py","inputs.py","io.py")},
            "files":file_manifest(directory)}
        write_json(directory/"manifest.json",result)
        validate_snapshot(directory)
        check_current(root,index)
        # Detect parent edits after reading, without reading/reinterpreting a live producer's newer state.
        if any(digest_file(parent/name)!=hashlib.sha256(data).hexdigest() for name,data in payload.items()):
            raise ValueError("Parent snapshot changed while freezing.")
    return result


def validate_snapshot(directory,current_root=None):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version")!=SCHEMA or manifest.get("kind")!="frozen_current_selection":
        raise ValueError("Unsupported current snapshot schema.")
    if manifest.get("time_mode")!="retrospective" or manifest.get("model_inference_run") is not False or manifest.get("historical_observed_pit_ready") is not False:
        raise ValueError("Freezing an inherited snapshot cannot create observed-PIT/model-inference evidence.")
    verify_files(directory,manifest,required=REQUIRED)
    selection=json.loads((directory/"selection.json").read_text(encoding="utf-8"))
    rows=read_records(directory/"articles.jsonl"); selected=keyed(rows,"selected articles")
    proofs=keyed(read_records(directory/"provenance.jsonl"),"selected evidence")
    times=keyed(read_records(directory/"time_checks.jsonl"),"selected time checks")
    ledger=keyed(read_records(directory/"selection_ledger.jsonl"),"selection ledger")
    index=read_records(directory/"raw_index.jsonl")
    with zipfile.ZipFile(directory/"parent_artifact.zip") as archive:
        parent_bytes=archive.read("manifest.json"); parent_manifest=json.loads(parent_bytes)
        expected={"manifest.json",*parent_manifest["files"]}
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=expected:
            raise ValueError("Archived parent membership mismatch.")
        for name,hashed in parent_manifest["files"].items():
            if Path(name).name!=name or hashlib.sha256(archive.read(name)).hexdigest()!=hashed:
                raise ValueError("Archived parent file hash mismatch.")
        if hashlib.sha256(parent_bytes).hexdigest()!=manifest["parent_manifest_sha256"] or selection["parent_manifest_sha256"]!=manifest["parent_manifest_sha256"]:
            raise ValueError("Parent manifest pin mismatch.")
        parents=keyed(jsonl_bytes(archive.read("articles.jsonl")),"archived parent articles")
        parent_proofs=keyed(jsonl_bytes(archive.read("provenance.jsonl")),"archived parent evidence")
        parent_times=keyed(jsonl_bytes(archive.read("time_checks.jsonl")),"archived parent times")
    aliases=defaultdict(list); versions={}
    paths=[row["path"] for row in index]
    if len(set(paths))!=len(paths):
        raise ValueError("Duplicated frozen raw path.")
    with zipfile.ZipFile(directory/"raw_news.zip") as archive:
        if len(archive.namelist())!=len(paths) or set(archive.namelist())!=set(paths):
            raise ValueError("Frozen raw archive membership mismatch.")
        for item in index:
            path=PurePosixPath(item["path"])
            if path.is_absolute() or ".." in path.parts or path.parts[0] not in ("US","GLOBAL"):
                raise ValueError("Invalid frozen raw path.")
            payload=archive.read(item["path"]); raw=parse_json(payload); sid=raw["source_id"]; version=producer_hash(raw)
            if hashlib.sha256(payload).hexdigest()!=item["file_sha256"] or sid!=item["source_id"] or version!=item["source_record_hash"]:
                raise ValueError("Frozen raw bytes/identity/version mismatch.")
            if sid in versions and versions[sid]!=version:
                raise ValueError("Conflicting raw versions in frozen snapshot.")
            versions[sid]=version; aliases[sid].append(item["path"])
    if set(selected)!=set(versions) or set(proofs)!=set(selected) or set(times)!=set(selected) or set(ledger)!=set(selected):
        raise ValueError("Frozen selection identity set mismatch.")
    for sid,row in selected.items():
        if row!=parents.get(sid) or proofs[sid]!=parent_proofs.get(sid) or times[sid]!=parent_times.get(sid):
            raise ValueError("Inherited article/evidence/time fields were modified.")
        entry=ledger[sid]
        if row.get("source_record_hash")!=versions[sid] or entry["source_record_hash"]!=versions[sid] or entry["parent_article_hash"]!=digest(row) or entry["parent_provenance_hash"]!=digest(proofs[sid]) or entry["current_paths"]!=sorted(aliases[sid]):
            raise ValueError("Frozen selection proof mismatch.")
        if row.get("adapter_status")!="ready" or normalize_article(row)["input_status"]!="ready":
            raise ValueError("Frozen article is not encoding-ready.")
    excluded=keyed(read_records(directory/"excluded_parent_articles.jsonl"),"excluded articles")
    excluded_proofs=keyed(read_records(directory/"excluded_parent_provenance.jsonl"),"excluded evidence")
    omitted=set(parents)-set(selected)
    if set(excluded)!=omitted or set(excluded_proofs)!=omitted or selection["excluded_parent_ids"]!=sorted(omitted):
        raise ValueError("Excluded parent set mismatch.")
    if any(excluded[sid]!=parents[sid] or excluded_proofs[sid]!=parent_proofs[sid] for sid in omitted):
        raise ValueError("Excluded parent records were modified.")
    if manifest["articles"]!=len(selected) or manifest["raw_files"]!=len(index) or manifest["excluded_parent_articles"]!=len(omitted) or selection["parent_articles"]!=len(parents) or manifest["source_membership_hash"]!=digest(index) or manifest["selected_article_hash"]!=digest(rows) or selection["selected_articles"]!=len(selected):
        raise ValueError("Frozen snapshot counts/digests mismatch.")
    if current_root is not None:
        check_current(Path(current_root).resolve(),index)
    return {"valid":True,"articles":len(selected),"raw_files":len(index),"excluded_parent_articles":len(omitted),
            "parent_evidence_preserved":True,"timestamps_preserved":True,"raw_bytes_verified":True,
            "current_inputs_verified":current_root is not None,"historical_observed_pit_ready":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    freeze=commands.add_parser("freeze")
    for name in ("source-root","parent"):
        freeze.add_argument("--"+name,type=Path,required=True)
    freeze.add_argument("--output",type=Path)
    validate=commands.add_parser("validate")
    validate.add_argument("--artifact",type=Path,required=True); validate.add_argument("--current-root",type=Path)
    args=parser.parse_args()
    result=freeze_current(args.source_root,args.parent,destination(PROJECT,args.output,kind="snapshot",category="artifacts/news/adapted")) if args.command=="freeze" else validate_snapshot(args.artifact,args.current_root)
    print(json.dumps({key:value for key,value in result.items() if key not in ("files","code_sha256")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
