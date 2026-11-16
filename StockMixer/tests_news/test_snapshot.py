import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from news_fusion import snapshot
from news_fusion.inputs import normalize_article
from news_fusion.io import file_manifest, read_records, write_json, write_records
from news_fusion.source_adapter import ADAPTER_SCHEMA, producer_hash


def fixture(root):
    source,parent,output=root/"current",root/"parent",root/"frozen"
    (source/"US").mkdir(parents=True); (source/"GLOBAL").mkdir(); parent.mkdir()
    raw,articles,proofs,times={},{},[],[]
    for sid in ("a","b","c"):
        row={"source_id":sid,"title":"Economic announcement","summary":"A central bank publishes economic forecasts.",
             "published_at":"2013-01-02","language":"en"}
        raw[sid]=row
        value=normalize_article(row)
        value.pop("input_status")
        value.update(adapter_schema_version=ADAPTER_SCHEMA,adapter_status="ready",source_record_hash=producer_hash(row),
            feature_available_at="2026-11-12T12:00:00Z",llm_inferred_at=None,language_basis="synthetic evidence fixture")
        articles[sid]=value
        proofs.append({"source_id":sid,"record_hash":producer_hash(row),"selected_audit":{"test_only":True}})
        times.append({"source_id":sid,"warnings":["unknown llm time retained"],"llm_inferred_at":None})
    for sid in ("a","b"):
        write_json(source/"US"/(sid+".json"),raw[sid])
    write_records(parent/"articles.jsonl",list(articles.values()))
    write_records(parent/"provenance.jsonl",proofs)
    write_records(parent/"time_checks.jsonl",times)
    write_json(parent/"source_snapshot.json",{"files":{},"fixture_only":True})
    write_json(parent/"manifest.json",{"schema_version":ADAPTER_SCHEMA,"kind":"adapted_collector_snapshot",
        "ready_articles":3,"source_root":"unused external reference","files":file_manifest(parent)})
    return source,parent,output,raw,articles


class FrozenSelectionTests(unittest.TestCase):
    def test_exact_subset_retains_excluded_records_bytes_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,raw,articles=fixture(Path(temporary))
            original={path.name:path.read_bytes() for path in (source/"US").iterdir()}
            result=snapshot.freeze_current(source,parent,output)
            self.assertEqual((result["articles"],result["excluded_parent_articles"]),(2,1))
            self.assertEqual(read_records(output/"articles.jsonl"),[articles["a"],articles["b"]])
            self.assertEqual(read_records(output/"excluded_parent_articles.jsonl"),[articles["c"]])
            self.assertEqual(original,{path.name:path.read_bytes() for path in (source/"US").iterdir()})
            self.assertTrue(snapshot.validate_snapshot(output,source)["current_inputs_verified"])
            # Validation does not depend on a still-existing external producer or parent directory.
            parent.rename(Path(temporary)/"parent_offline")
            source.rename(Path(temporary)/"current_offline")
            self.assertTrue(snapshot.validate_snapshot(output)["timestamps_preserved"])

    def test_same_id_identical_views_deduplicated_distinct_content_ids_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,raw,_=fixture(Path(temporary))
            write_json(source/"GLOBAL/a.json",raw["a"])
            result=snapshot.freeze_current(source,parent,output)
            self.assertEqual((result["articles"],result["raw_files"]),(2,3))
            self.assertEqual(len(read_records(output/"duplicate_content_groups.jsonl")),1)
            self.assertTrue(snapshot.validate_snapshot(output)["raw_bytes_verified"])

    def test_conflicting_current_versions_abort_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,raw,_=fixture(Path(temporary))
            write_json(source/"GLOBAL/a.json",dict(raw["a"],summary="Different economic report."))
            with self.assertRaisesRegex(ValueError,"Conflicting"):
                snapshot.freeze_current(source,parent,output)
            self.assertFalse(output.exists())

    def test_new_or_changed_source_requires_new_adaptation(self):
        for kind in ("new","changed"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temporary:
                source,parent,output,raw,_=fixture(Path(temporary))
                if kind=="new":
                    write_json(source/"US/d.json",dict(raw["a"],source_id="d"))
                else:
                    write_json(source/"US/a.json",dict(raw["a"],title="Changed title"))
                with self.assertRaisesRegex(ValueError,"fresh evidence"):
                    snapshot.freeze_current(source,parent,output)
                self.assertFalse(output.exists())

    def test_rehashed_timestamp_edit_is_rejected_against_archived_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,_,_=fixture(Path(temporary))
            snapshot.freeze_current(source,parent,output)
            rows=read_records(output/"articles.jsonl")
            rows[0]["feature_available_at"]="2013-01-02T00:00:00Z"
            write_records(output/"articles.jsonl",rows)
            manifest=json.loads((output/"manifest.json").read_text())
            manifest["files"]=file_manifest(output); write_json(output/"manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"modified"):
                snapshot.validate_snapshot(output)

    def test_live_change_while_writing_does_not_publish_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,raw,_=fixture(Path(temporary))
            original=snapshot.new_artifact
            @contextlib.contextmanager
            def concurrent_writer(path):
                with original(path) as directory:
                    write_json(source/"US/a.json",dict(raw["a"],title="Concurrent writer changed input"))
                    yield directory
            with patch.object(snapshot,"new_artifact",concurrent_writer),self.assertRaisesRegex(ValueError,"changed"):
                snapshot.freeze_current(source,parent,output)
            self.assertFalse(output.exists())

    def test_existing_output_and_raw_directory_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,parent,output,_,_=fixture(Path(temporary))
            with self.assertRaises(ValueError):
                snapshot.freeze_current(source,parent,source/"US/output")
            snapshot.freeze_current(source,parent,output)
            before=(output/"manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                snapshot.freeze_current(source,parent,output)
            self.assertEqual(before,(output/"manifest.json").read_bytes())


if __name__=="__main__":
    unittest.main()
