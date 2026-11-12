import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from news_fusion.inputs import load_articles, normalize_article
from news_fusion.io import digest_file, read_records, write_json, new_artifact
from news_fusion.source_adapter import Snapshot, adapt_snapshot, producer_hash, validate_adapted
from news_fusion.time_axis import Axis, Schedule


class SourceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "producer"
        for relative in ("data/US", "data/GLOBAL", ".runtime/simple/audit", "data_enriched/analysis"):
            (self.source / relative).mkdir(parents=True)

    def row(self, identity="article"):
        return {"source_id": identity, "url": "https://example.test/" + identity, "title": None,
            "时间": "2020-03-05", "分类": "earnings", "关键词": ["revenue"],
            "主要内容": "Revenue increased at the company during the reporting period.",
            "时间信息": {"publication_precision": "date", "first_seen_at": "2020-03-05T18:00:00Z",
                "fetched_at": "2020-03-05T18:01:00Z", "feature_available_at": "2020-03-05T19:00:00Z",
                "availability_basis": "extraction"},
            "股票实体": [{"名称": "Example Company", "股票代码": None, "交易所": None}],
            "涉及行业": [{"名称": "Technology", "依据": "explicit"}], "影响范围": ["company"]}

    def production(self, row=None, market="US"):
        row = row or self.row()
        write_json(self.source / "data/US" / (row["source_id"] + ".json"), row)
        if market == "GLOBAL":
            write_json(self.source / "data/GLOBAL" / (row["source_id"] + ".json"), row)
        audit = {"source_id": row["source_id"], "record_hash": producer_hash(row),
            "source_published_at": row["时间"], "canonical_market": market, "language": "en",
            "model": "qwen3:8b", "model_digest": "fixture-model", "schema_version": "1.1",
            "llm_inferred_at": "2020-03-05T18:30:00Z", "extracted_at": "2020-03-05T18:59:00Z",
            "commit_verified_at": "2020-03-05T19:00:00Z", "feature_available_at": "2020-03-05T19:00:00Z"}
        path = self.source / ".runtime/simple/audit" / (row["source_id"] + ".json")
        write_json(path, audit)
        return row, audit, path

    def enriched(self, change_summary=False):
        original = self.row()
        old_audit = {"source_id": original["source_id"], "record_hash": producer_hash(original), "language": "en"}
        old_path = self.source / ".runtime/simple/audit/article.json"
        write_json(old_path, old_audit)
        with zipfile.ZipFile(self.source / "data/US_original.zip", "w") as archive:
            archive.writestr("US/original.json", json.dumps(original, ensure_ascii=False))
        row = json.loads(json.dumps(original))
        row["时间信息"].update(availability_basis="migration", feature_available_at="2026-11-12T12:00:00Z")
        row["涉及行业"].append({"名称": "Software", "依据": "inferred"})
        if change_summary:
            row["主要内容"] = "A different summary not authenticated by the old language audit."
        write_json(self.source / "data/US/article.json", row)
        audit = {"source_id": row["source_id"], "record_hash": producer_hash(row), "status": "completed",
            "canonical_market": "US", "original_record_hash": producer_hash(original),
            "matching_audit_hash": digest_file(old_path), "published_at": row["时间"],
            "feature_available_at": "2026-11-12T12:00:00Z", "commit_verified_at": "2026-11-12T12:00:00Z",
            "llm_inferred_at": None, "model": None, "analysis": {"method": "verified_legacy_metadata"}}
        write_json(self.source / "data_enriched/analysis/article.json", audit)
        return row

    def adapt(self, name="adapted"):
        return adapt_snapshot(self.source, self.root / name)

    def test_view_dedup_and_original_files_unchanged(self):
        self.production(market="GLOBAL")
        before = {p: digest_file(p) for p in self.source.rglob("*") if p.is_file()}
        manifest = self.adapt()
        self.assertEqual(manifest["unique_articles"], 1)
        self.assertEqual(manifest["duplicate_views_removed"], 1)
        self.assertEqual(manifest["ready_articles"], 1)
        rows, _ = load_articles([self.root / "adapted/articles.jsonl"])
        self.assertEqual(rows[0]["market"], "GLOBAL")
        self.assertEqual(rows[0]["input_status"], "ready")
        self.assertTrue(all(digest_file(p) == h for p, h in before.items()))
        self.assertTrue(validate_adapted(self.root / "adapted")["source_hashes_unchanged"])

    def test_conflicting_copies_fail_without_publishing(self):
        row, _, _ = self.production(market="GLOBAL")
        row["主要内容"] = "Conflicting content."
        write_json(self.source / "data/GLOBAL/article.json", row)
        with self.assertRaises(ValueError):
            self.adapt()
        self.assertFalse((self.root / "adapted").exists())

    def test_enriched_language_restored_only_through_matching_original(self):
        self.enriched()
        manifest = self.adapt()
        row = read_records(self.root / "adapted/articles.jsonl")[0]
        self.assertEqual(row["language"], "en")
        self.assertEqual(row["language_basis"], "legacy_audit_verified_unchanged_summary")
        self.assertEqual(row["availability_basis"], "migration")
        self.assertFalse(row["upstream_observed_evidence_complete"])
        self.assertIsNone(row["llm_inferred_at"])
        self.assertEqual(manifest["audit_family_counts"], {"enriched": 1})
        proof = read_records(self.root / "adapted/provenance.jsonl")[0]
        self.assertIn("language_proof", proof)

    def test_changed_summary_does_not_inherit_old_language(self):
        self.enriched(change_summary=True)
        manifest = self.adapt()
        self.assertEqual(manifest["ready_articles"], 0)
        self.assertEqual(manifest["review_required"], 1)
        row = read_records(self.root / "adapted/review_records.jsonl")[0]
        self.assertEqual(row["language"], "unknown")
        with self.assertRaises(ValueError):
            load_articles([self.root / "adapted/review_records.jsonl"])

    def test_audit_filename_does_not_authenticate_a_different_version(self):
        row, audit, path = self.production()
        audit["record_hash"] = "wrong"
        write_json(path, audit)
        manifest = self.adapt()
        self.assertEqual(manifest["review_required"], 1)
        self.assertEqual(manifest["ready_articles"], 0)

    def test_two_matching_audits_are_not_resolved_by_latest_timestamp(self):
        row, audit, _ = self.production()
        write_json(self.source / "data_enriched/analysis/article.json", audit)
        with self.assertRaises(ValueError):
            self.adapt()

    def test_distinct_ids_with_same_text_are_retained_for_asof_dedup(self):
        self.production(self.row("one"))
        self.production(self.row("two"))
        manifest = self.adapt()
        self.assertEqual(manifest["ready_articles"], 2)
        self.assertEqual(manifest["content_duplicate_groups"], 1)

    def test_later_qwen_completion_moves_effective_time_forward(self):
        row, audit, path = self.production()
        original_time = row["时间信息"]["feature_available_at"]
        audit["llm_inferred_at"] = "2020-03-06T22:00:00Z"
        write_json(path, audit)
        self.adapt()
        canonical = read_records(self.root / "adapted/articles.jsonl")[0]
        self.assertEqual(canonical["source_feature_available_at"], original_time)
        self.assertEqual(canonical["feature_available_at"], "2020-03-06T22:00:00+00:00")
        normalized, _ = load_articles([self.root / "adapted/articles.jsonl"])
        self.assertEqual(normalized[0]["llm_inferred_at"], canonical["llm_inferred_at"])
        self.assertEqual(normalized[0]["source_record_hash"], canonical["source_record_hash"])
        raw = json.loads((self.source / "data/US/article.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["时间信息"]["feature_available_at"], original_time)

    def test_date_only_is_not_assigned_an_invented_time(self):
        self.production()
        self.adapt()
        canonical = read_records(self.root / "adapted/articles.jsonl")[0]
        self.assertEqual(canonical["published_at"], "2020-03-05")
        self.assertEqual(canonical["date_only_policy"], "next_trading_session")

    def test_bad_system_timezone_is_review_required(self):
        row = self.row()
        row["时间信息"]["fetched_at"] = "2020-03-05T18:00:00"
        self.production(row)
        self.assertEqual(self.adapt()["review_required"], 1)

    def test_output_cannot_overwrite_producer_or_existing_artifacts(self):
        self.production()
        with self.assertRaises(ValueError):
            adapt_snapshot(self.source, self.source / "derived")
        self.adapt()
        with self.assertRaises(FileExistsError):
            self.adapt()

    def test_staging_access_denied_is_not_retried_indefinitely(self):
        mkdir = Path.mkdir
        attempts = []
        def deny_stage(path, *args, **kwargs):
            if path.name.startswith("news-staging-"):
                attempts.append(path)
                raise PermissionError("Synthetic denied staging directory")
            return mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", deny_stage):
            with self.assertRaises(PermissionError):
                with new_artifact(self.root / "output"):
                    self.fail("No publication is allowed on staging failure")
        self.assertEqual(len(attempts), 1)

    def test_validation_detects_changed_source(self):
        self.production()
        self.adapt()
        (self.source / "data/US/article.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_adapted(self.root / "adapted")

    def test_snapshot_membership_change_is_detected(self):
        self.production()
        snapshot = Snapshot(self.source, ["US", "GLOBAL"])
        write_json(self.source / "data/US/new.json", self.row("new"))
        with self.assertRaises(ValueError):
            snapshot.verify()

    def test_live_append_is_excluded_from_the_frozen_snapshot(self):
        self.production()
        snapshot = Snapshot(self.source, ["US", "GLOBAL"], allow_append_only=True)
        snapshot.json("data/US/article.json")
        write_json(self.source / "data/US/new.json", self.row("new"))
        snapshot.verify()
        self.assertEqual(snapshot.new_files, ["data/US/new.json"])
        self.assertNotIn("data/US/new.json", snapshot.membership["US"])

    def test_input_change_during_output_creation_is_not_published(self):
        self.production()
        original_verify = Snapshot.verify
        count = 0
        def mutate_at_second_verify(snapshot):
            nonlocal count
            count += 1
            if count == 2:
                (self.source / "data/US/article.json").write_text("{}", encoding="utf-8")
            original_verify(snapshot)
        with patch.object(Snapshot, "verify", mutate_at_second_verify):
            with self.assertRaises(ValueError):
                self.adapt()
        self.assertFalse((self.root / "adapted").exists())


class AdaptedPITTests(unittest.TestCase):
    def schedule(self):
        days = ["2020-03-05", "2020-03-06", "2020-03-09", "2020-03-10"]
        axis = Axis("TEST", ["one"], days, 2, 3, {})
        return Schedule(axis, [{"date": day, "decision_cutoff_at": day + ("T21:00:00Z" if i < 2 else "T20:00:00Z")}
                               for i, day in enumerate(days)])

    def article(self):
        return {"published_at": "2020-03-05T18:00:00Z", "availability_basis": "verified_audit",
            **{k: "2020-03-05T19:00:00Z" for k in ("first_seen_at", "fetched_at", "feature_available_at",
                "finbert_completed_at", "news_feature_available_at", "model_available_at")},
            "llm_inferred_at": "2020-03-05T18:30:00Z", "llm_time_required": True}

    def test_optional_explicit_completion_and_commit_times_are_honored(self):
        for key in ("llm_inferred_at", "commit_verified_at", "extracted_at", "effective_feature_available_at"):
            article = self.article()
            article[key] = "2020-03-06T22:00:00Z"
            self.assertEqual(self.schedule().first_session(article, "observed_pit"), 2)

    def test_required_unknown_qwen_time_is_rejected_for_observed(self):
        article = self.article()
        article["llm_inferred_at"] = None
        with self.assertRaises(ValueError):
            self.schedule().first_session(article, "observed_pit")

    def test_producer_strict_cutoff_contract_is_preserved(self):
        article = self.article()
        article.update(llm_inferred_at="2020-03-05T21:00:00Z", availability_comparison="strictly_before_cutoff")
        self.assertEqual(self.schedule().first_session(article, "observed_pit"), 1)


if __name__ == "__main__":
    unittest.main()
