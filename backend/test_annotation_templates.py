"""Focused contract tests for the annotation templates and CSV loaders.

All row fixtures in this module are synthetic and are written below a temporary
directory.  The repository's annotation files are never used as loader input.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
for import_root in (str(BACKEND_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


from app.services.nlp.issue_identification.models.annotation import (  # noqa: E402
    CSV_COLUMNS as ISSUE_CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV as DEFAULT_ISSUE_CSV,
    AnnotationCorpus,
)
from app.services.nlp.response_classification.models.annotation import (  # noqa: E402
    CSV_COLUMNS as RESPONSE_CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV as DEFAULT_RESPONSE_CSV,
    ResponseCorpus,
)
from app.services.nlp.sentiment_emotion.models.annotation import (  # noqa: E402
    DEFAULT_EMOTION_CSV,
    DEFAULT_SENTIMENT_CSV,
    EMOTION_CSV_COLUMNS,
    SENTIMENT_CSV_COLUMNS,
    EmotionCorpus,
    SentimentCorpus,
)
from app.services.nlp.solution_detection.models.annotation import (  # noqa: E402
    CSV_COLUMNS as SOLUTION_CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV as DEFAULT_SOLUTION_CSV,
    SolutionCorpus,
)
from ml.configs import TaskConfig, TaskType  # noqa: E402


class AnnotationTemplateContractTests(unittest.TestCase):
    """Keep blank templates aligned with the executable schema contracts."""

    @staticmethod
    def _write_csv(directory: str, name: str, columns, rows) -> str:
        path = Path(directory) / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    @staticmethod
    def _row(columns, **values):
        row = {column: "" for column in columns}
        row.update(values)
        return row

    def test_header_only_templates_match_canonical_columns(self):
        templates = {
            "sentiment_annotations.template.csv": SENTIMENT_CSV_COLUMNS,
            "emotion_annotations.template.csv": EMOTION_CSV_COLUMNS,
            "issue_annotations.template.csv": ISSUE_CSV_COLUMNS,
            "response_annotations.template.csv": RESPONSE_CSV_COLUMNS,
            "solution_annotations.template.csv": SOLUTION_CSV_COLUMNS,
        }
        template_dir = PROJECT_ROOT / "data" / "annotations" / "templates"

        for name, columns in templates.items():
            with self.subTest(template=name):
                with (template_dir / name).open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(
                    rows,
                    [list(columns)],
                    "templates must contain exactly their canonical header and no data rows",
                )

    def test_task_types_and_default_dataset_paths(self):
        expected = (
            (TaskType.SENTIMENT, "sentiment", "sentiment_annotations.csv", DEFAULT_SENTIMENT_CSV),
            (TaskType.EMOTION, "emotion", "emotion_annotations.csv", DEFAULT_EMOTION_CSV),
            (TaskType.ISSUE, "issue", "issue_annotations.csv", DEFAULT_ISSUE_CSV),
            (TaskType.RESPONSE, "response", "response_annotations.csv", DEFAULT_RESPONSE_CSV),
            (TaskType.SOLUTION, "solution", "solution_annotations.csv", DEFAULT_SOLUTION_CSV),
        )

        self.assertEqual(
            tuple(task.value for task in TaskType),
            tuple(value for _task, value, _filename, _schema_path in expected),
        )

        for task, value, filename, schema_path in expected:
            with self.subTest(task=task.value):
                config = TaskConfig(task_name=value)
                expected_config_path = f"data/annotations/{filename}"
                self.assertEqual(config.dataset_path, expected_config_path)
                self.assertEqual(
                    os.path.normpath(schema_path),
                    os.path.normpath(expected_config_path),
                )

    def test_representative_rows_load_from_temporary_synthetic_fixtures(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sentiment_path = self._write_csv(
                temporary_directory,
                "sentiment.csv",
                SENTIMENT_CSV_COLUMNS,
                [
                    self._row(
                        SENTIMENT_CSV_COLUMNS,
                        document_id="sent-1",
                        text="the placement result was excellent",
                        sentiment="positive",
                        language="en",
                        script="latin",
                        detected_languages="en",
                    )
                ],
            )
            sentiment = SentimentCorpus.from_csv(sentiment_path)
            sentiment.validate()
            self.assertEqual(len(sentiment), 1)
            self.assertEqual(sentiment.annotations[0].sentiment, "positive")

            emotion_path = self._write_csv(
                temporary_directory,
                "emotion.csv",
                EMOTION_CSV_COLUMNS,
                [
                    self._row(
                        EMOTION_CSV_COLUMNS,
                        document_id="emotion-1",
                        text="I am anxious but grateful",
                        emotions="anxiety|gratitude",
                    ),
                    self._row(
                        EMOTION_CSV_COLUMNS,
                        document_id="emotion-2",
                        text="the timetable was posted",
                        emotions="",
                    ),
                ],
            )
            emotion = EmotionCorpus.from_csv(emotion_path)
            emotion.validate()
            self.assertEqual(len(emotion), 2)
            self.assertEqual(emotion.annotations[0].emotions, ["anxiety", "gratitude"])
            self.assertEqual(emotion.annotations[1].emotions, [])

            issue_path = self._write_csv(
                temporary_directory,
                "issue.csv",
                ISSUE_CSV_COLUMNS,
                [
                    self._row(
                        ISSUE_CSV_COLUMNS,
                        document_id="issue-1",
                        text="fees and hostel maintenance are difficult",
                        labels="fees|hostel",
                    ),
                    self._row(
                        ISSUE_CSV_COLUMNS,
                        document_id="issue-2",
                        text="the campus opened on Monday",
                        labels="",
                    ),
                ],
            )
            issue = AnnotationCorpus.from_csv(issue_path)
            issue.validate()
            self.assertEqual(len(issue), 2)
            self.assertEqual(issue.annotations[0].labels, ["fees", "hostel"])
            self.assertEqual(issue.annotations[1].labels, [])

            response_path = self._write_csv(
                temporary_directory,
                "response.csv",
                RESPONSE_CSV_COLUMNS,
                [
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="response-1",
                        text="same bro",
                        response_type="agreement",
                    ),
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="response-2",
                        text="you should contact the office",
                        response_type="advice",
                    ),
                ],
            )
            response = ResponseCorpus.from_csv(response_path)
            response.validate()
            self.assertEqual(response.labels, ["agreement", "advice"])

            solution_path = self._write_csv(
                temporary_directory,
                "solution.csv",
                SOLUTION_CSV_COLUMNS,
                [
                    self._row(
                        SOLUTION_CSV_COLUMNS,
                        document_id="solution-1",
                        text="apply before the deadline",
                        is_solution="true",
                        solution_type="scholarship",
                    ),
                    self._row(
                        SOLUTION_CSV_COLUMNS,
                        document_id="solution-2",
                        text="same issue here",
                        is_solution="false",
                    ),
                ],
            )
            solution = SolutionCorpus.from_csv(solution_path)
            solution.validate()
            self.assertEqual(solution.is_solutions, [True, False])

    def test_invalid_response_label_is_rejected_by_corpus_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                temporary_directory,
                "invalid-response.csv",
                RESPONSE_CSV_COLUMNS,
                [
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="response-bad",
                        text="not a canonical role",
                        response_type="not_a_v1_role",
                    )
                ],
            )
            corpus = ResponseCorpus.from_csv(path)

            # The current loader parses CSV shape first; explicit corpus
            # validation is the rejection boundary for the fixed label taxonomy.
            with self.assertRaisesRegex(ValueError, "not one of the six"):
                corpus.validate()

    def test_invalid_or_blank_solution_booleans_are_rejected(self):
        for token in ("maybe", ""):
            with self.subTest(token=token or "blank"):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    path = self._write_csv(
                        temporary_directory,
                        "invalid-solution.csv",
                        SOLUTION_CSV_COLUMNS,
                        [
                            self._row(
                                SOLUTION_CSV_COLUMNS,
                                document_id="solution-bad",
                                text="a synthetic row",
                                is_solution=token,
                            )
                        ],
                    )
                    with self.assertRaisesRegex(ValueError, "is_solution"):
                        SolutionCorpus.from_csv(path)

    def test_duplicate_response_ids_and_conflicting_labels_are_surfaced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                temporary_directory,
                "duplicate-response.csv",
                RESPONSE_CSV_COLUMNS,
                [
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="same-label",
                        text="same label, first row",
                        response_type="agreement",
                    ),
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="same-label",
                        text="same label, duplicate row",
                        response_type="agreement",
                    ),
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="conflicting-label",
                        text="first conflicting row",
                        response_type="opinion",
                    ),
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="conflicting-label",
                        text="second conflicting row",
                        response_type="advice",
                    ),
                ],
            )
            corpus = ResponseCorpus.from_csv(path)
            corpus.validate()

            self.assertEqual(
                corpus.duplicate_document_ids(),
                ["same-label", "conflicting-label"],
            )
            self.assertEqual(corpus.conflicting_document_ids(), ["conflicting-label"])
            with self.assertRaisesRegex(ValueError, "conflicting-label"):
                corpus.validate_unique_document_ids()

    def test_missing_response_document_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                temporary_directory,
                "missing-response-id.csv",
                RESPONSE_CSV_COLUMNS,
                [
                    self._row(
                        RESPONSE_CSV_COLUMNS,
                        document_id="",
                        text="same bro",
                        response_type="agreement",
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "document_id"):
                ResponseCorpus.from_csv(path)

    def test_review_markers_are_not_current_training_csv_fields(self):
        """Review/approval state belongs in a sidecar, not a training target CSV."""

        for marker in ("review_status", "approval_status"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, RESPONSE_CSV_COLUMNS)
                self.assertNotIn(marker, SOLUTION_CSV_COLUMNS)

        # The loader may ignore an explicitly supplied sidecar marker, but it
        # must not expose it as a parsed training field or use it as a label.
        columns = [*RESPONSE_CSV_COLUMNS, "review_status"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = self._write_csv(
                temporary_directory,
                "response-with-review-marker.csv",
                columns,
                [
                    self._row(
                        columns,
                        document_id="response-reviewed",
                        text="same bro",
                        response_type="agreement",
                        review_status="approved_for_training",
                    )
                ],
            )
            corpus = ResponseCorpus.from_csv(path)
            corpus.validate()

        annotation = corpus.annotations[0]
        self.assertEqual(annotation.response_type, "agreement")
        self.assertNotIn("review_status", annotation.model_dump())
        self.assertNotIn("approval_status", annotation.model_dump())


if __name__ == "__main__":
    unittest.main()
