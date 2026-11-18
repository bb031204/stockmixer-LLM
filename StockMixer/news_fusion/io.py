import hashlib
import json
import os
import shutil
import uuid
import time
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SCHEMA = "stockmixer_news_v1"

def now():
    return datetime.now(timezone.utc).isoformat()

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def read_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).replace({np.nan: None}).to_dict("records")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, list) else [value]
    raise ValueError("Expected JSON, JSONL or Parquet input.")

def write_records(path, rows):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    else:
        raise ValueError("Output table format must be .jsonl or .parquet.")

def publish_directory(temporary,output):
    """Retry transient Windows sharing/access failures without exposing a partial target."""
    for attempt in range(12):
        if output.exists():raise FileExistsError("Output appeared during processing.")
        try:
            os.rename(temporary,output)
            return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==11:raise
            time.sleep(min(.1*(2**attempt),1.0))


@contextmanager
def new_artifact(output):
    """Do not overwrite existing artifacts or publish partial results."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists; select a new version directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Python's Windows mkdtemp may repeatedly retry access-denied errors. Fail once instead.
    temporary = output.parent / ("news-staging-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        yield temporary
        if output.exists():
            raise FileExistsError("Output appeared during processing.")
        publish_directory(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

def file_manifest(directory):
    return {p.name: digest_file(p) for p in Path(directory).iterdir()
            if p.is_file() and p.name != "manifest.json"}

def verify_files(directory, manifest, required=()):
    files=manifest.get("files")
    if not isinstance(files,dict) or not files or not set(required)<=set(files):
        raise ValueError("Artifact manifest omits required file hashes.")
    root=Path(directory).resolve()
    for name, expected in files.items():
        if not isinstance(name,str) or name in (".","..","") or Path(name).name!=name:
            raise ValueError("Invalid artifact filename.")
        path=(root/name).resolve()
        if path.parent!=root or not isinstance(expected,str) or re.fullmatch(r"[0-9a-f]{64}",expected) is None:
            raise ValueError("Invalid artifact file hash/path: "+name)
        if digest_file(path) != expected:
            raise ValueError("Artifact file hash mismatch: " + Path(name).name)
