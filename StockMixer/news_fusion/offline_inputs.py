"""Select immutable, verified news inputs before constructing an NLP backend."""
import json
from pathlib import Path
from .inputs import load_articles
from .io import digest_file


def encoding_inputs(project, config, explicit=None):
    project=Path(project).resolve()
    def resolve(value):
        path=Path(value)
        return (path if path.is_absolute() else project/path).resolve()
    provenance={"mode":"explicit_files"}
    if explicit is None:
        snapshot=resolve(config["adapted_snapshot"])
        manifest_path=snapshot/"manifest.json"
        before=digest_file(manifest_path)
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind")=="frozen_industry_enriched_selection_v1":
            from .legacy_industry import validate_enriched
            validate_enriched(snapshot)
        elif manifest.get("kind")=="frozen_current_selection":
            from .snapshot import validate_snapshot
            validate_snapshot(snapshot)
        elif manifest.get("kind")=="adapted_collector_snapshot":
            from .source_adapter import validate_adapted
            validate_adapted(snapshot)
        else:
            raise ValueError("adapted_snapshot must be a verified frozen/adapted news artifact.")
        paths=[snapshot/"articles.jsonl"]
        provenance={"mode":"verified_snapshot","snapshot_directory":str(snapshot),
                    "snapshot_manifest_sha256":before}
    else:
        paths=[resolve(value) for value in explicit]
        if not paths:
            raise ValueError("Explicit encoding input cannot be empty.")
    hashes={str(path):digest_file(path) for path in paths}
    if explicit is None and hashes[str(paths[0])]!=manifest.get("files",{}).get("articles.jsonl"):
        raise ValueError("Snapshot articles differ from the verified manifest.")
    articles,counts=load_articles(paths)
    if not counts.get("ready"):
        raise ValueError("No encoding-ready articles. Use the verified adapted snapshot, not unadapted raw JSON.")
    if any(digest_file(path)!=hashed for path,hashed in hashes.items()):
        raise ValueError("Encoding inputs changed while being read.")
    if explicit is None and digest_file(manifest_path)!=before:
        raise ValueError("Snapshot manifest changed while being verified.")
    provenance.update(input_files_sha256=hashes,articles=len(articles),ready_articles=counts.get("ready",0))
    return articles,counts,provenance
