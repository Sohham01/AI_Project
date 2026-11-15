"""Focused, offline tests for the annotation review-manifest sidecar."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.services.annotation.manifest as manifest_module
from app.services.annotation.manifest import (
    MANIFEST_VERSION,
    ManifestValidationError,
    build_manifest_records,
    read_manifest,
    validate_candidate_manifest_join,
    validate_candidate_records,
    validate_manifest,
    validate_manifest_for_training,
    write_manifest,
)


class AnnotationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self.temporary_directory.name, name)

    @staticmethod
    def candidate(document_id: str = "doc-1", **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "document_id": document_id,
            "text": "PRIVATE SOURCE TEXT SHOULD NEVER APPEAR IN AN ERROR",
            "normalized_text": "private normalized text",
            "response_type": "",
            "post_id": "post-1",
            "comment_id": "comment-1",
            "parent_comment_id": "parent-1",
            "subreddit": "approved-community",
            "score": "4",
            "created_at": "2026-01-02T03:04:05+00:00",
            "source_url": "https://source.invalid/post-1",
            "source_provider": "synthetic-provider",
            "source_mode": "direct",
            "parent_context_status": "available",
            "context_snapshot_ref": "snapshot-1",
            "source_record_hash": "sha256:source-1",
            "language": "en",
            "script": "latin",
            "is_code_mixed": "false",
            "detected_languages": "en",
        }
        row.update(overrides)
        return row

    def built(self, *rows: dict[str, object]) -> list[dict[str, object]]:
        return list(build_manifest_records(list(rows) or [self.candidate()]))

    def test_valid_join_copies_source_facts_and_unsets_review_state(self) -> None:
        candidate = self.candidate()
        records = self.built(candidate)
        record = records[0]

        self.assertEqual(record["manifest_version"], MANIFEST_VERSION)
        self.assertEqual(record["document_id"], "doc-1")
        self.assertEqual(record["source"]["source_subreddit"], "approved-community")
        self.assertEqual(record["source"]["source_created_at"], candidate["created_at"])
        self.assertEqual(record["source"]["source_url"], candidate["source_url"])
        self.assertNotIn("text", record["source"])
        self.assertEqual(record["review"]["status"], "unreviewed")
        self.assertIsNone(record["review"]["response_label"])
        self.assertIsNone(record["adjudication"]["label"])
        self.assertEqual(record["dataset_approval"]["status"], "pending")
        self.assertIsNone(record["dataset_approval"]["approval_ref"])

        report = validate_candidate_manifest_join([candidate], records)
        self.assertTrue(report.valid, report.as_dict())

    def test_missing_and_duplicate_candidate_ids_are_errors(self) -> None:
        missing = validate_candidate_records([self.candidate(" ")])
        self.assertFalse(missing.valid)
        self.assertIn("missing_candidate_document_id", {issue.code for issue in missing.errors})

        duplicate = validate_candidate_records(
            [self.candidate("same"), self.candidate("same", comment_id="comment-2")]
        )
        self.assertFalse(duplicate.valid)
        self.assertIn(
            "duplicate_candidate_document_id",
            {issue.code for issue in duplicate.errors},
        )

        records = self.built(self.candidate("same"), self.candidate("other", comment_id="comment-2"))
        records[1]["document_id"] = "same"
        manifest_report = validate_manifest(records)
        self.assertIn(
            "duplicate_manifest_document_id",
            {issue.code for issue in manifest_report.errors},
        )

    def test_orphan_and_missing_join_ids_are_errors(self) -> None:
        candidates = [self.candidate("doc-1")]
        records = self.built(self.candidate("doc-2"))
        report = validate_candidate_manifest_join(candidates, records)
        codes = {issue.code for issue in report.errors}
        self.assertIn("missing_manifest_for_candidate", codes)
        self.assertIn("orphan_manifest_document_id", codes)

    def test_shared_source_contradiction_and_exact_timestamp_equality(self) -> None:
        candidate = self.candidate()
        records = self.built(candidate)
        records[0]["source"]["source_url"] = "https://source.invalid/other"
        report = validate_candidate_manifest_join([candidate], records)
        self.assertIn("source_field_conflict", {issue.code for issue in report.errors})

        timestamp_records = self.built(candidate)
        timestamp_records[0]["source"]["source_created_at"] = "2026-01-02T03:04:05Z"
        timestamp_report = validate_candidate_manifest_join([candidate], timestamp_records)
        self.assertIn(
            "source_timestamp_conflict",
            {issue.code for issue in timestamp_report.errors},
        )

    def test_missing_optional_provenance_warns_but_identity_is_required(self) -> None:
        sparse = {"document_id": "sparse-1", "text": "private"}
        report = validate_candidate_records([sparse])
        self.assertTrue(report.valid, report.as_dict())
        self.assertTrue(any(issue.code == "missing_optional_provenance" for issue in report.warnings))

        built = build_manifest_records([sparse])
        self.assertIsNone(built[0]["source"]["source_url"])
        self.assertTrue(built.warnings)

        with self.assertRaises(ManifestValidationError):
            build_manifest_records([{"document_id": "", "text": "private"}])

    def test_policy_required_provenance_is_an_error_not_an_invented_value(self) -> None:
        sparse = {"document_id": "required-source-1", "text": "private"}
        report = validate_candidate_records(
            [sparse],
            required_source_fields=("source_record_hash", "source_url"),
        )
        self.assertIn("missing_required_provenance", report.error_codes)
        self.assertEqual(
            {issue.field for issue in report.errors if issue.code == "missing_required_provenance"},
            {"source_record_hash", "source_url"},
        )
        self.assertNotIn(
            "source_url",
            {
                issue.field
                for issue in report.warnings
                if issue.code == "missing_optional_provenance"
            },
        )

        target = self.path("required-source.jsonl")
        with self.assertRaises(ManifestValidationError):
            write_manifest(
                target,
                candidates=[sparse],
                required_source_fields=("source_record_hash",),
            )
        self.assertFalse(os.path.exists(target))

    def test_blank_response_labels_are_review_only(self) -> None:
        candidate = self.candidate(response_type="")
        records = self.built(candidate)
        normal_report = validate_manifest(records)
        self.assertTrue(normal_report.valid, normal_report.as_dict())
        self.assertTrue(any(issue.code == "blank_response_label" for issue in normal_report.warnings))

        approval_report = validate_manifest_for_training(
            records,
            dataset_approval={"status": "approved"},
        )
        self.assertIn("blank_response_label", {issue.code for issue in approval_report.errors})

    def test_review_and_dataset_approval_are_separate_gates(self) -> None:
        records = self.built(self.candidate())
        record = records[0]
        record["review"].update(
            {"status": "approved_for_training", "response_label": "agreement"}
        )
        record["dataset_approval"]["training_inclusion"] = "include"

        without_dataset_approval = validate_manifest_for_training(records)
        self.assertIn(
            "dataset_approval_required",
            {issue.code for issue in without_dataset_approval.errors},
        )

        missing_dataset_version = validate_manifest_for_training(
            records,
            dataset_approval={"status": "approved"},
        )
        self.assertIn(
            "approved_missing_dataset_version",
            {issue.code for issue in missing_dataset_version.errors},
        )

        approved = validate_manifest_for_training(
            records,
            dataset_approval={"status": "approved", "dataset_version": "v1"},
        )
        self.assertTrue(approved.valid, approved.as_dict())

    def test_excluded_and_adjudication_required_are_not_approved(self) -> None:
        excluded = self.built(self.candidate("excluded"))[0]
        excluded["review"].update(
            {
                "status": "excluded",
                "exclusion_reason": "insufficient_context",
                "response_label": "agreement",
            }
        )
        excluded["dataset_approval"]["training_inclusion"] = "exclude"
        excluded_report = validate_manifest_for_training(
            [excluded], dataset_approval={"status": "approved"}
        )
        self.assertIn("record_excluded", {issue.code for issue in excluded_report.errors})

        needs_adjudication = self.built(self.candidate("adjudication"))[0]
        needs_adjudication["review"].update(
            {"status": "adjudication_required", "response_label": "agreement"}
        )
        needs_adjudication["adjudication"]["status"] = "required"
        needs_adjudication["dataset_approval"]["training_inclusion"] = "include"
        adjudication_report = validate_manifest_for_training(
            [needs_adjudication], dataset_approval={"status": "approved"}
        )
        self.assertIn("adjudication_required", {issue.code for issue in adjudication_report.errors})

    def test_manifest_identity_validation_catches_blank_and_duplicate_ids(self) -> None:
        records = self.built(self.candidate("one"), self.candidate("two", comment_id="comment-2"))
        records[0]["document_id"] = ""
        records[1]["document_id"] = "one"
        report = validate_manifest(records)
        codes = {issue.code for issue in report.errors}
        self.assertIn("missing_manifest_document_id", codes)
        # The changed second row is unique relative to the blank first row.
        self.assertNotIn("duplicate_manifest_document_id", codes)

        records[0]["document_id"] = "one"
        duplicate_report = validate_manifest(records)
        self.assertIn(
            "duplicate_manifest_document_id",
            {issue.code for issue in duplicate_report.errors},
        )

    def test_legacy_candidate_csv_without_new_context_columns_is_supported(self) -> None:
        source_path = self.path("legacy-candidates.csv")
        with open(source_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["document_id", "text", "response_type", "post_id", "comment_id"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "document_id": "legacy-1",
                    "text": "legacy private text",
                    "response_type": "",
                    "post_id": "post-legacy",
                    "comment_id": "comment-legacy",
                }
            )

        records = build_manifest_records(source_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["document_id"], "legacy-1")
        self.assertEqual(records[0]["source"]["post_id"], "post-legacy")
        self.assertIsNone(records[0]["source"]["source_url"])
        self.assertTrue(records.warnings)

        target = self.path("legacy-manifest.jsonl")
        write_manifest(target, candidate_csv_path=source_path)
        self.assertEqual(read_manifest(target)[0]["document_id"], "legacy-1")

    def test_fresh_creation_and_existing_destination_preservation(self) -> None:
        target = self.path("manifest.jsonl")
        records = self.built(self.candidate())
        write_manifest(target, records)
        original = Path(target).read_bytes()
        self.assertTrue(original)

        with self.assertRaises(FileExistsError):
            write_manifest(target, records, overwrite=True)
        self.assertEqual(Path(target).read_bytes(), original)

    def test_validation_happens_before_destination_creation(self) -> None:
        target = self.path("invalid.jsonl")
        invalid = [{"manifest_version": MANIFEST_VERSION, "document_id": ""}]
        with self.assertRaises(ManifestValidationError):
            write_manifest(target, invalid)
        self.assertFalse(os.path.exists(target))

    def test_partial_low_level_write_is_cleaned_up(self) -> None:
        target = self.path("partial.jsonl")
        records = self.built(
            self.candidate("one"), self.candidate("two", comment_id="comment-2")
        )
        real_write = manifest_module.os.write
        calls = {"count": 0}

        def fail_after_first_write(fd: int, payload: bytes) -> int:
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("synthetic write failure")
            return real_write(fd, payload)

        with patch.object(manifest_module.os, "write", side_effect=fail_after_first_write):
            with self.assertRaises(OSError):
                write_manifest(target, records)
        self.assertFalse(os.path.exists(target))

    def test_fsync_failure_is_cleaned_up(self) -> None:
        target = self.path("fsync-failure.jsonl")
        records = self.built(self.candidate())
        with patch.object(manifest_module.os, "fsync", side_effect=OSError("synthetic fsync failure")):
            with self.assertRaises(OSError):
                write_manifest(target, records)
        self.assertFalse(os.path.exists(target))

    def test_stable_serialization_and_read_back_round_trip(self) -> None:
        first_target = self.path("first.jsonl")
        second_target = self.path("second.jsonl")
        records = self.built(
            self.candidate("one"), self.candidate("two", comment_id="comment-2")
        )
        write_manifest(first_target, records)
        # Rebuild from mappings with a different insertion order at the input
        # boundary; canonical JSON must remain byte-stable.
        reordered = [dict(reversed(list(record.items()))) for record in records]
        write_manifest(second_target, reordered)
        self.assertEqual(Path(first_target).read_bytes(), Path(second_target).read_bytes())
        self.assertEqual(read_manifest(first_target), read_manifest(second_target))

    def test_privacy_safe_validation_and_read_errors_never_include_source_text(self) -> None:
        secret = "TOP_SECRET_SOURCE_SENTENCE_9f4c"
        with self.assertRaises(ManifestValidationError) as context:
            build_manifest_records([{"document_id": "", "text": secret}])
        self.assertNotIn(secret, str(context.exception))
        self.assertNotIn(secret, json.dumps(context.exception.report.as_dict()))

        malformed_path = self.path("malformed.jsonl")
        Path(malformed_path).write_text(
            '{"document_id":"doc-1","text":"' + secret + '"\n',
            encoding="utf-8",
        )
        with self.assertRaises(ManifestValidationError) as read_context:
            read_manifest(malformed_path)
        self.assertNotIn(secret, str(read_context.exception))
        self.assertNotIn(secret, json.dumps(read_context.exception.report.as_dict()))


if __name__ == "__main__":
    unittest.main()
