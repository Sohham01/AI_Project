"""
Comprehensive tests for the issue identification stage
(TF-IDF -> One-vs-Rest Logistic Regression, multi-label).

Tests cover:
- Multi-label annotations and the annotation CSV schema
- Single-label and multiple-simultaneous-label predictions
- Configurable probability threshold
- Determinism
- Empty / blank input text
- Invalid labels and invalid configuration
- Duplicate document ids
- Input text is never modified
- Model metadata and version
- Small / invalid training data
- Probability output
- Unknown / unseen labels
- Serialization of results
- Multilingual and code-mixed text (English, Hinglish, Romanized Hindi,
  Bengali, Odia, emoji placeholder tokens)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.issue_identification import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    AnnotationCorpus,
    IssueAnnotation,
    IssueDocument,
    IssueEvaluationReport,
    IssueIdentification,
    IssueIdentificationResult,
    IssueModelConfig,
    IssuePrediction,
    IssueTrainingReport,
    TfidfLogisticIssueClassifier,
    evaluate_predictions,
)


# ---------------------------------------------------------------------------
# Synthetic annotation fixture.
#
# SYNTHETIC, illustrative data written to exercise the architecture. It is not
# collected from Reddit and must never be quoted as project performance.
# Labels are language-neutral on purpose: `fees` covers an English, a Hinglish
# and a Bengali complaint alike, because the taxonomy is defined by annotators
# and is not tied to any one language.
# ---------------------------------------------------------------------------
FIXTURE = [
    # --- fees ---
    ("a01", "the fees are way too high and unaffordable for most students", ["fees"], "en", "latin", False),
    ("a02", "fee structure is ridiculous and the tuition fee hike is unacceptable", ["fees"], "en", "latin", False),
    ("a03", "fees ka bhaari hai aur scholarship se bach bhi nahi milta", ["fees"], "hi", "latin", True),
    ("a04", "IIM fees badhte badhte bahut zyada ho gaye hain", ["fees"], "hi", "latin", True),
    ("a05", "the annual fee is too expensive compared to other institutes", ["fees"], "en", "latin", False),
    # --- roi ---
    ("a06", "the ROI of this MBA is not worth it at all", ["roi"], "en", "latin", False),
    ("a07", "return on investment is poor for an MBA programme", ["roi"], "en", "latin", False),
    ("a08", "ROI questionable hai MBA ka return justify nahi karta", ["roi"], "hi", "latin", True),
    ("a09", "the return on investment is questionable given the fees", ["roi", "fees"], "en", "latin", False),
    # --- placement_stress ---
    ("a10", "placement season is extremely stressful for students", ["placement_stress"], "en", "latin", False),
    ("a11", "placement season pressure and PPO uncertainty is stressful", ["placement_stress"], "en", "latin", False),
    ("a12", "placement season bahut stressful hai PPO ka tension", ["placement_stress"], "hi", "latin", True),
    ("a13", "campus placement stress is really high every year", ["placement_stress"], "en", "latin", False),
    # --- hostel ---
    ("a14", "hostel rooms are too small and cramped for two students", ["hostel"], "en", "latin", False),
    ("a15", "hostel room maintenance is poor and the room is very small", ["hostel"], "en", "latin", False),
    ("a16", "hostel room chhota hai aur maintenance bahut kharab", ["hostel"], "hi", "latin", True),
    ("a17", "the hostel room size and maintenance are both problems", ["hostel", "mess_food"], "en", "latin", False),
    # --- mess_food ---
    ("a18", "mess food quality is really bad every single day", ["mess_food"], "en", "latin", False),
    ("a19", "the hostel mess serves very bad and stale food", ["mess_food"], "en", "latin", False),
    ("a20", "mess ka khana kharab hai quality bahut kharab", ["mess_food"], "hi", "latin", True),
    ("a21", "mess food is unhealthy and the cooking quality is poor", ["mess_food"], "en", "latin", False),
    # --- exam_pressure ---
    ("a22", "CAT exam pressure is immense and preparation is stressful", ["exam_pressure"], "en", "latin", False),
    ("a23", "CAT exam preparation pressure is really high for students", ["exam_pressure"], "en", "latin", False),
    ("a24", "CAT exam ki preparation pressure bahut zyada hai", ["exam_pressure"], "hi", "latin", True),
    # --- native script ---
    ("a25", "ফি অনেক বেশি এবং ROI কোনো কাজে আসে না", ["fees", "roi"], "bn", "bengali", False),
    ("a26", "ছাত্রদের জন্য প্লেসমেন্ট চাপ অত্যন্ত বেশি", ["placement_stress"], "bn", "bengali", False),
    ("a27", "ଫି ବହୁତ ଅଧିକ ଏବଂ ROI ଉପଯୋଗୀ ନାହିଁ", ["fees", "roi"], "or", "odia", False),
    ("a28", "ହସ୍ଟେଲ ରେଣ୍ଟର ବହୁତ ଛୋଟ ଏବଂ ଖାବା ଖରାପ", ["hostel", "mess_food"], "or", "odia", False),
    # --- emoji placeholder tokens (from the preprocessing stage) ---
    ("a29", "fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>", ["fees", "roi", "placement_stress"], "en", "latin", False),
    # --- negatives: no issue expressed ---
    ("a30", "campus fest was amazing and the food was great", [], "en", "latin", False),
    ("a31", "IIM Sambalpur library has a great collection of books", [], "en", "latin", False),
    ("a32", "the new academic building is beautiful and spacious", [], "en", "latin", False),
    ("a33", "maine naya laptop liya aur course bhi achha hai", [], "hi", "latin", True),
    # --- multi-label overlaps ---
    ("a34", "fees are high and the ROI is poor", ["fees", "roi"], "en", "latin", False),
    ("a35", "mess food is bad and the hostel room is small", ["mess_food", "hostel"], "en", "latin", False),
    ("a36", "placement stress is high and the fees are also high", ["placement_stress", "fees"], "en", "latin", False),
]


def build_annotations() -> list:
    return [
        IssueAnnotation(
            document_id=doc_id,
            text=text,
            labels=list(labels),
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=[language, "en"] if is_code_mixed else [language],
        )
        for doc_id, text, labels, language, script, is_code_mixed in FIXTURE
    ]


def build_documents() -> list:
    return [
        IssueDocument(
            document_id=doc_id,
            text=text,
            original_text=text,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=[language, "en"] if is_code_mixed else [language],
        )
        for doc_id, text, _labels, language, script, is_code_mixed in FIXTURE
    ]


#: A deliberately different annotation set, used to prove the label space is
#: learned from data and not hard-coded. Its taxonomy has nothing to do with
#: the IIM fixtures above.
ALTERNATE_FIXTURE = [
    ("o1", "library timing is inconvenient", ["library_hours"]),
    ("o2", "library closes too early inconvenient", ["library_hours"]),
    ("o3", "gym equipment is missing", ["gym_equipment"]),
    ("o4", "the gym has no new equipment", ["gym_equipment"]),
    ("o5", "library hours and gym equipment both bad", ["library_hours", "gym_equipment"]),
    ("o6", "campus wifi is very slow", ["wifi"]),
    ("o7", "the wifi in campus is slow and unreliable", ["wifi"]),
]


def build_alternate_annotations() -> list:
    return [
        IssueAnnotation(document_id=doc_id, text=text, labels=list(labels))
        for doc_id, text, labels in ALTERNATE_FIXTURE
    ]


class TrainedClassifierMixin:
    """Trains one classifier per test class, deterministically."""

    def setUp(self):
        self.annotations = build_annotations()
        self.classifier = TfidfLogisticIssueClassifier()
        self.report = self.classifier.train(self.annotations)
        self.documents = build_documents()


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


class TestIssueIdentificationInterface(unittest.TestCase):
    def test_interface_is_abstract(self):
        with self.assertRaises(TypeError):
            IssueIdentification()

    def test_classifier_implements_interface(self):
        self.assertIsInstance(TfidfLogisticIssueClassifier(), IssueIdentification)

    def test_labels_empty_before_training(self):
        self.assertEqual(TfidfLogisticIssueClassifier().labels, [])

    def test_is_trained_false_before_training(self):
        self.assertFalse(TfidfLogisticIssueClassifier().is_trained)

    def test_predict_before_train_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            TfidfLogisticIssueClassifier().predict(
                [IssueDocument(document_id="x", text="fees too high")]
            )
        self.assertIn("not been trained", str(ctx.exception))

    def test_evaluate_before_train_raises(self):
        with self.assertRaises(RuntimeError):
            TfidfLogisticIssueClassifier().evaluate([])

    def test_result_models_exported(self):
        for model in (IssuePrediction, IssueIdentificationResult, IssueTrainingReport,
                      IssueEvaluationReport, IssueDocument, IssueAnnotation):
            self.assertTrue(hasattr(model, "model_fields"))


# ---------------------------------------------------------------------------
# Multi-label annotations
# ---------------------------------------------------------------------------


class TestIssueAnnotation(unittest.TestCase):
    def test_multi_label_annotation(self):
        annotation = IssueAnnotation(
            document_id="a1",
            text="fees are way too high and ROI feels questionable",
            labels=["fees", "roi"],
        )
        self.assertEqual(len(annotation.labels), 2)
        self.assertTrue(annotation.is_multi_label)

    def test_single_label_annotation(self):
        annotation = IssueAnnotation(
            document_id="a2",
            text="placement season is extremely stressful",
            labels=["placement_stress"],
        )
        self.assertFalse(annotation.is_multi_label)

    def test_three_label_annotation(self):
        annotation = IssueAnnotation(
            document_id="a3",
            text="hostel mess food and fees",
            labels=["hostel", "mess_food", "fees"],
        )
        self.assertEqual(len(annotation.labels), 3)

    def test_zero_label_annotation_allowed(self):
        """A document with no issue is a valid negative training example."""
        annotation = IssueAnnotation(
            document_id="a4", text="campus fest was great", labels=[]
        )
        self.assertEqual(annotation.labels, [])
        self.assertFalse(annotation.is_multi_label)

    def test_labels_are_not_forced_to_one(self):
        corpus = AnnotationCorpus(annotations=build_annotations())
        multi = [a for a in corpus if len(a.labels) > 1]
        self.assertGreaterEqual(len(multi), 5)

    def test_normalised_labels_deduplicated_and_sorted(self):
        annotation = IssueAnnotation(
            document_id="a5", text="x", labels=["roi", "fees", "roi", " fees "]
        )
        self.assertEqual(annotation.normalised_labels(), ["fees", "roi"])

    def test_invalid_empty_label_rejected(self):
        annotation = IssueAnnotation(document_id="a6", text="x", labels=["fees", "  "])
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_labels()
        self.assertIn("empty issue label", str(ctx.exception))

    def test_invalid_unseparated_multi_label_rejected(self):
        annotation = IssueAnnotation(
            document_id="a7", text="x", labels=["fees|roi"]
        )
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_labels()
        self.assertIn("must be split", str(ctx.exception))

    def test_corpus_validate_detects_bad_label(self):
        corpus = AnnotationCorpus(
            annotations=[IssueAnnotation(document_id="a8", text="x", labels=[""])]
        )
        with self.assertRaises(ValueError):
            corpus.validate()

    def test_annotation_to_issue_document(self):
        annotation = IssueAnnotation(
            document_id="a9",
            text="fees high",
            labels=["fees"],
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=3,
        )
        document = annotation.to_issue_document()
        self.assertEqual(document.document_id, "a9")
        self.assertEqual(document.text, "fees high")
        self.assertEqual(document.language, "hi")
        self.assertTrue(document.is_code_mixed)
        self.assertEqual(document.detected_languages, ["hi", "en"])
        self.assertEqual(document.topic_id, 3)


class TestAnnotationCsvSchema(unittest.TestCase):
    def test_default_annotation_path_is_documented(self):
        self.assertEqual(
            DEFAULT_ANNOTATION_CSV,
            os.path.join("data", "annotations", "issue_annotations.csv"),
        )

    def test_csv_columns(self):
        self.assertEqual(
            CSV_COLUMNS,
            ["document_id", "text", "labels", "post_id", "comment_id",
             "language", "script", "is_code_mixed", "detected_languages"],
        )

    def test_csv_round_trip(self):
        original = AnnotationCorpus(annotations=build_annotations())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "issue_annotations.csv")
            original.to_csv(path)
            self.assertTrue(os.path.exists(path))
            loaded = AnnotationCorpus.from_csv(path)
        self.assertEqual(len(loaded), len(original))
        self.assertEqual(loaded.labels, original.labels)
        self.assertEqual(
            [a.document_id for a in loaded], [a.document_id for a in original]
        )
        self.assertEqual(
            [a.labels for a in loaded], [a.labels for a in original]
        )

    def test_csv_parses_pipe_separated_multi_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "issue_annotations.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,labels\n")
                handle.write('ann-1,"fees are way too high and ROI feels questionable",fees|roi\n')
                handle.write('ann-2,"hostel rooms are too small and mess food is bad",hostel|mess_food\n')
                handle.write('ann-3,"placement season is extremely stressful",placement_stress\n')
            corpus = AnnotationCorpus.from_csv(path)
        self.assertEqual(corpus.labels, ["fees", "hostel", "mess_food",
                                         "placement_stress", "roi"])
        self.assertEqual(corpus.annotations[0].labels, ["fees", "roi"])
        self.assertEqual(corpus.annotations[1].labels, ["hostel", "mess_food"])

    def test_csv_parses_optional_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "issue_annotations.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,labels,is_code_mixed,detected_languages,topic_id\n")
                handle.write('x1,"fees high",fees,true,hi|en,3\n')
            corpus = AnnotationCorpus.from_csv(path)
        annotation = corpus.annotations[0]
        self.assertTrue(annotation.is_code_mixed)
        self.assertEqual(annotation.detected_languages, ["hi", "en"])
        self.assertEqual(annotation.topic_id, 3)

    def test_csv_missing_required_columns_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text\nx1,fees high\n")
            with self.assertRaises(ValueError) as ctx:
                AnnotationCorpus.from_csv(path)
        self.assertIn("labels", str(ctx.exception))

    def test_csv_missing_file_raises_with_format_hint(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            AnnotationCorpus.from_csv("no_such_dir/no_such_file.csv")
        message = str(ctx.exception)
        self.assertIn("document_id", message)
        self.assertIn("|", message)

    def test_csv_invalid_topic_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad_topic.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,labels,topic_id\nx1,fees high,fees,abc\n")
            with self.assertRaises(ValueError) as ctx:
                AnnotationCorpus.from_csv(path)
        self.assertIn("topic_id", str(ctx.exception))

    def test_csv_row_missing_document_id_rejected(self):
        with self.assertRaises(ValueError):
            IssueAnnotation.from_csv_row({"document_id": "", "text": "x", "labels": "a"})

    def test_csv_blank_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "issue_annotations.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,labels\n")
                handle.write("x1,fees high,fees\n")
                handle.write(",,\n")
                handle.write("x2,roi poor,roi\n")
            corpus = AnnotationCorpus.from_csv(path)
        self.assertEqual(len(corpus), 2)

    def test_label_counts(self):
        counts = AnnotationCorpus(annotations=build_annotations()).label_counts()
        self.assertEqual(counts["fees"], 11)
        self.assertEqual(list(counts), sorted(counts))

    def test_labels_sorted_deterministically(self):
        labels = AnnotationCorpus(annotations=build_annotations()).labels
        self.assertEqual(labels, sorted(labels))


# ---------------------------------------------------------------------------
# Training data adequacy
# ---------------------------------------------------------------------------


class TestTrainingDataValidation(unittest.TestCase):
    def test_empty_training_set_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train([])
        self.assertIn("no training data", str(ctx.exception))
        self.assertIn("issue_annotations.csv", str(ctx.exception))

    def test_none_training_set_rejected(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier().train(None)

    def test_too_few_documents_rejected(self):
        annotations = build_annotations()[:3]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train(annotations)
        self.assertIn("insufficient training data", str(ctx.exception))

    def test_single_label_rejected(self):
        annotations = [
            IssueAnnotation(document_id=f"s{i}", text=f"fees are high number {i}",
                            labels=["fees"])
            for i in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train(annotations)
        self.assertIn("multi-label", str(ctx.exception))

    def test_rare_label_rejected(self):
        annotations = build_annotations() + [
            IssueAnnotation(document_id="rare", text="a brand new issue appeared",
                            labels=["brand_new_issue"])
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train(annotations)
        self.assertIn("brand_new_issue", str(ctx.exception))
        self.assertIn("positive example", str(ctx.exception))

    def test_rare_label_allowed_when_config_relaxed(self):
        annotations = build_annotations() + [
            IssueAnnotation(document_id="rare", text="a brand new issue appeared",
                            labels=["brand_new_issue"])
        ]
        classifier = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(min_positive_examples_per_label=1)
        )
        report = classifier.train(annotations)
        self.assertIn("brand_new_issue", report.labels)

    def test_degenerate_all_positive_label_rejected(self):
        """One-vs-Rest needs a negative example per label."""
        annotations = [
            IssueAnnotation(document_id=f"u{i}", text=f"issue text number {i}",
                            labels=["universal"] + (["common"] if i < 3 else []))
            for i in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train(annotations)
        self.assertIn("degenerate", str(ctx.exception))
        self.assertIn("universal", str(ctx.exception))

    def test_wrong_record_type_rejected(self):
        with self.assertRaises(TypeError):
            TfidfLogisticIssueClassifier().train(["not an annotation"])

    def test_blank_text_everywhere_rejected(self):
        annotations = [
            IssueAnnotation(document_id=f"b{i}", text="  ", labels=["fees" if i % 2 else "roi"])
            for i in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticIssueClassifier().train(annotations)
        self.assertIn("TF-IDF", str(ctx.exception))

    def test_min_training_documents_configurable(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(
                config=IssueModelConfig(min_training_documents=1000)
            ).train(build_annotations())


class TestInvalidConfiguration(unittest.TestCase):
    def test_invalid_max_features(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(max_features=0))

    def test_invalid_min_df(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(min_df=1.5))

    def test_invalid_max_df(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(max_df=0.0))

    def test_invalid_ngram_range(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(ngram_range=(2, 1)))

    def test_invalid_norm(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(norm="l3"))

    def test_invalid_token_pattern(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(
                config=IssueModelConfig(token_pattern="(?u)\\w+(")
            )

    def test_language_specific_stopwords_rejected(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(stop_words="hindi"))

    def test_invalid_c(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(C=0))

    def test_invalid_max_iter(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(max_iter=0))

    def test_invalid_class_weight(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(class_weight="magic"))

    def test_invalid_min_positive_examples(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(
                config=IssueModelConfig(min_positive_examples_per_label=0)
            )

    def test_invalid_config_threshold(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(config=IssueModelConfig(threshold=1.0))

    def test_invalid_constructor_threshold(self):
        with self.assertRaises(ValueError):
            TfidfLogisticIssueClassifier(threshold=0)

    def test_invalid_call_threshold(self):
        classifier = TfidfLogisticIssueClassifier()
        classifier.train(build_annotations())
        with self.assertRaises(ValueError):
            classifier.predict(build_documents(), threshold=1.5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTraining(TrainedClassifierMixin, unittest.TestCase):
    def test_labels_learned_from_annotation_data(self):
        self.assertEqual(
            self.report.labels,
            ["exam_pressure", "fees", "hostel", "mess_food",
             "placement_stress", "roi"],
        )

    def test_labels_are_exactly_the_annotated_label_set(self):
        self.assertEqual(
            set(self.report.labels),
            set(AnnotationCorpus(annotations=self.annotations).labels),
        )

    def test_labels_not_hard_coded(self):
        """A different annotation set must yield a different label set."""
        report = TfidfLogisticIssueClassifier().train(build_alternate_annotations())
        self.assertEqual(report.labels, ["gym_equipment", "library_hours", "wifi"])
        # And the IIM-specific labels are gone
        for label in ("fees", "roi", "hostel", "mess_food", "placement_stress"):
            self.assertNotIn(label, report.labels)

    def test_positive_counts_reported(self):
        counts = self.report.positive_examples_per_label
        self.assertEqual(counts["fees"], 11)
        self.assertEqual(counts["roi"], 8)
        self.assertEqual(counts["exam_pressure"], 3)
        self.assertEqual(set(counts), set(self.report.labels))

    def test_multi_label_documents_counted(self):
        self.assertGreaterEqual(self.report.n_multi_label_documents, 5)

    def test_documents_without_labels_counted(self):
        self.assertEqual(self.report.n_documents_without_labels, 4)

    def test_feature_count_reported(self):
        self.assertGreater(self.report.n_features, 100)

    def test_report_is_usable(self):
        self.assertTrue(self.report.is_usable)
        self.assertEqual(self.report.n_labels, 6)
        self.assertEqual(self.report.n_training_documents, len(FIXTURE))

    def test_classifier_exposes_labels_after_training(self):
        self.assertTrue(self.classifier.is_trained)
        self.assertEqual(self.classifier.labels, self.report.labels)

    def test_retraining_replaces_label_set(self):
        self.classifier.train(build_alternate_annotations())
        self.assertEqual(self.classifier.labels,
                         ["gym_equipment", "library_hours", "wifi"])

    def test_no_convergence_warnings_on_fixture(self):
        self.assertEqual(self.report.warnings, [])

    def test_timestamp_opt_in(self):
        self.assertIsNone(self.report.trained_at)
        classifier = TfidfLogisticIssueClassifier(include_timestamp=True)
        report = classifier.train(self.annotations)
        self.assertIn("+00:00", report.trained_at)


class TestDuplicateDocumentIds(TrainedClassifierMixin, unittest.TestCase):
    def test_duplicate_training_ids_reported(self):
        classifier = TfidfLogisticIssueClassifier()
        report = classifier.train(self.annotations + [self.annotations[0]])
        self.assertTrue(
            any("duplicate_document_ids" in w for w in report.warnings)
        )

    def test_duplicate_training_rows_all_used(self):
        classifier = TfidfLogisticIssueClassifier()
        report = classifier.train(self.annotations + [self.annotations[0]])
        self.assertEqual(report.n_training_documents, len(FIXTURE) + 1)

    def test_no_duplicate_warning_when_ids_unique(self):
        self.assertFalse(
            any("duplicate_document_ids" in w for w in self.report.warnings)
        )

    def test_corpus_reports_duplicates(self):
        corpus = AnnotationCorpus(annotations=self.annotations + [self.annotations[0]])
        self.assertEqual(corpus.duplicate_document_ids(), ["a01"])

    def test_duplicate_prediction_ids_preserved(self):
        docs = [
            IssueDocument(document_id="dup", text="fees are way too high"),
            IssueDocument(document_id="dup", text="fees are way too high"),
            IssueDocument(document_id="other", text="roi is poor"),
        ]
        result = self.classifier.predict(docs)
        self.assertEqual(len(result.predictions), 3)
        self.assertEqual(
            [p.document_id for p in result.predictions], ["dup", "dup", "other"]
        )

    def test_duplicate_documents_get_consistent_predictions(self):
        docs = [
            IssueDocument(document_id="dup", text="fees are way too high"),
            IssueDocument(document_id="dup", text="fees are way too high"),
        ]
        result = self.classifier.predict(docs)
        first, second = result.predictions
        self.assertEqual(first.issue_probabilities, second.issue_probabilities)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestSingleLabelPrediction(TrainedClassifierMixin, unittest.TestCase):
    def test_single_label_prediction(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="placement season is extremely stressful")]
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.predicted_issue_labels, ["placement_stress"])

    def test_prediction_for_helper(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        )
        self.assertIsNotNone(result.prediction_for("q"))
        self.assertIsNone(result.prediction_for("missing"))

    def test_no_issue_predicted_for_positive_text(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="campus fest was amazing and the food was great")]
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.predicted_issue_labels, [])
        self.assertFalse(prediction.has_issue)

    def test_has_issue_property(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        )
        self.assertTrue(result.predictions[0].has_issue)


class TestMultipleSimultaneousPredictions(TrainedClassifierMixin, unittest.TestCase):
    def test_three_labels_at_once(self):
        result = self.classifier.predict([
            IssueDocument(
                document_id="q",
                text="fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>",
            )
        ])
        prediction = result.predictions[0]
        self.assertEqual(
            prediction.predicted_issue_labels,
            ["fees", "placement_stress", "roi"],
        )
        self.assertTrue(prediction.is_multi_label)

    def test_two_labels_at_once(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q",
                          text="the return on investment is questionable given the fees")
        ])
        self.assertEqual(
            result.predictions[0].predicted_issue_labels, ["fees", "roi"]
        )

    def test_multi_label_counter(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q1", text="fees <EMOJI_fire> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>"),
            IssueDocument(document_id="q2", text="placement season is extremely stressful"),
        ])
        self.assertEqual(result.n_multi_label_documents, 1)

    def test_independent_sigmoids_not_softmax(self):
        """Sigmoid outputs are independent: they need not sum to 1."""
        result = self.classifier.predict([
            IssueDocument(document_id="q",
                          text="fees <EMOJI_fire> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>")
        ])
        total = sum(result.predictions[0].issue_probabilities.values())
        self.assertGreater(total, 1.0)

    def test_label_frequency(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q1", text="fees <EMOJI_fire> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>"),
            IssueDocument(document_id="q2", text="fees are way too high"),
        ])
        self.assertEqual(result.label_frequency()["fees"], 2)


class TestConfigurableThreshold(TrainedClassifierMixin, unittest.TestCase):
    TEXT = "hostel rooms are too small and mess food is bad"

    def _labels_at(self, threshold):
        return self.classifier.predict(
            [IssueDocument(document_id="q", text=self.TEXT)], threshold=threshold
        ).predictions[0].predicted_issue_labels

    def test_default_threshold_is_half(self):
        self.assertEqual(self.classifier.threshold, 0.5)
        self.assertEqual(self.classifier.config.threshold, 0.5)

    def test_constructor_threshold(self):
        classifier = TfidfLogisticIssueClassifier(threshold=0.3)
        classifier.train(self.annotations)
        self.assertEqual(classifier.threshold, 0.3)
        result = classifier.predict([IssueDocument(document_id="q", text=self.TEXT)])
        self.assertEqual(result.threshold, 0.3)
        self.assertEqual(result.predictions[0].predicted_issue_labels,
                         ["hostel", "mess_food"])

    def test_call_threshold_reveals_second_label(self):
        self.assertEqual(self._labels_at(0.5), [])
        self.assertEqual(self._labels_at(0.45), ["hostel"])
        self.assertEqual(self._labels_at(0.3), ["hostel", "mess_food"])

    def test_lower_threshold_is_a_superset(self):
        high = set(self._labels_at(0.5))
        low = set(self._labels_at(0.2))
        self.assertTrue(high.issubset(low))

    def test_threshold_does_not_change_probabilities(self):
        a = self.classifier.predict(
            [IssueDocument(document_id="q", text=self.TEXT)], threshold=0.5
        ).predictions[0].issue_probabilities
        b = self.classifier.predict(
            [IssueDocument(document_id="q", text=self.TEXT)], threshold=0.3
        ).predictions[0].issue_probabilities
        self.assertEqual(a, b)

    def test_call_threshold_does_not_mutate_default(self):
        self.classifier.predict(
            [IssueDocument(document_id="q", text=self.TEXT)], threshold=0.1
        )
        self.assertEqual(self.classifier.threshold, 0.5)

    def test_labels_above_helper_rethresholds_without_inference(self):
        prediction = self.classifier.predict(
            [IssueDocument(document_id="q", text=self.TEXT)]
        ).predictions[0]
        self.assertEqual(prediction.labels_above(0.3), ["hostel", "mess_food"])
        self.assertEqual(prediction.labels_above(0.99), [])


class TestProbabilityOutput(TrainedClassifierMixin, unittest.TestCase):
    def test_probabilities_cover_every_known_label(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        )
        self.assertEqual(
            set(result.predictions[0].issue_probabilities), set(self.report.labels)
        )

    def test_probabilities_in_unit_interval(self):
        result = self.classifier.predict(self.documents)
        for prediction in result.predictions:
            for probability in prediction.issue_probabilities.values():
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)

    def test_probabilities_present_even_when_nothing_predicted(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="campus fest was amazing")]
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.predicted_issue_labels, [])
        self.assertEqual(len(prediction.issue_probabilities), len(self.report.labels))

    def test_confidence_is_max_probability(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        )
        prediction = result.predictions[0]
        self.assertAlmostEqual(
            prediction.confidence, max(prediction.issue_probabilities.values()), places=5
        )

    def test_confidence_in_unit_interval(self):
        result = self.classifier.predict(self.documents)
        for prediction in result.predictions:
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)

    def test_top_label(self):
        prediction = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        ).predictions[0]
        self.assertEqual(prediction.top_label(), "fees")

    def test_top_label_none_without_labels(self):
        self.assertIsNone(IssuePrediction(document_id="q").top_label())

    def test_probabilities_rounded_for_reproducibility(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")]
        )
        for probability in result.predictions[0].issue_probabilities.values():
            self.assertEqual(probability, round(probability, 6))


class TestEmptyText(TrainedClassifierMixin, unittest.TestCase):
    def test_empty_text_document(self):
        result = self.classifier.predict([IssueDocument(document_id="q", text="")])
        self.assertEqual(result.predictions[0].predicted_issue_labels, [])
        self.assertEqual(result.predictions[0].document_id, "q")

    def test_whitespace_only_text(self):
        result = self.classifier.predict([IssueDocument(document_id="q", text="   \t  ")])
        self.assertEqual(result.predictions[0].predicted_issue_labels, [])

    def test_empty_document_list(self):
        result = self.classifier.predict([])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.predictions, [])

    def test_none_document_list(self):
        result = self.classifier.predict(None)
        self.assertEqual(result.n_documents, 0)

    def test_empty_text_still_reports_known_labels(self):
        result = self.classifier.predict([IssueDocument(document_id="q", text="")])
        self.assertEqual(result.known_labels, self.report.labels)

    def test_punctuation_only_text(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="!!! ??? ...")]
        )
        self.assertEqual(result.predictions[0].predicted_issue_labels, [])

    def test_counters(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q1", text=""),
            IssueDocument(document_id="q2", text="the fees are way too high"),
        ])
        self.assertEqual(result.n_documents_without_issue, 1)
        self.assertEqual(result.n_multi_label_documents, 0)


class TestDeterminism(TrainedClassifierMixin, unittest.TestCase):
    def test_repeated_predictions_identical(self):
        first = self.classifier.predict(self.documents)
        second = self.classifier.predict(self.documents)
        self.assertEqual(
            [p.issue_probabilities for p in first.predictions],
            [p.issue_probabilities for p in second.predictions],
        )

    def test_repeated_training_identical(self):
        other = TfidfLogisticIssueClassifier()
        other.train(build_annotations())
        a = self.classifier.predict(self.documents)
        b = other.predict(self.documents)
        self.assertEqual(
            [p.predicted_issue_labels for p in a.predictions],
            [p.predicted_issue_labels for p in b.predictions],
        )
        self.assertEqual(self.classifier.labels, other.labels)

    def test_label_ordering_is_stable(self):
        self.assertEqual(self.classifier.labels, sorted(self.classifier.labels))
        self.assertEqual(
            list(self.classifier.predict(self.documents).known_labels),
            sorted(self.report.labels),
        )

    def test_random_state_pinned(self):
        self.assertEqual(IssueModelConfig().random_state, 42)

    def test_timestamp_excluded_by_default(self):
        result = self.classifier.predict(self.documents)
        self.assertIsNone(result.trained_at)


class TestUnknownUnseenLabels(TrainedClassifierMixin, unittest.TestCase):
    def test_unseen_candidate_labels_reported(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")],
            candidate_labels=["fees", "brand_new_issue", "another_new_one"],
        )
        self.assertEqual(result.unseen_labels,
                         ["another_new_one", "brand_new_issue"])

    def test_unseen_labels_never_predicted(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")],
            candidate_labels=["brand_new_issue"],
        )
        for prediction in result.predictions:
            self.assertNotIn("brand_new_issue", prediction.predicted_issue_labels)
            self.assertNotIn("brand_new_issue", prediction.issue_probabilities)

    def test_unseen_labels_warned(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")],
            candidate_labels=["brand_new_issue"],
        )
        self.assertTrue(any("unseen_labels" in w for w in result.warnings))

    def test_known_candidate_labels_not_reported_unseen(self):
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text="the fees are way too high")],
            candidate_labels=self.report.labels,
        )
        self.assertEqual(result.unseen_labels, [])

    def test_predictions_restricted_to_trained_labels(self):
        result = self.classifier.predict(self.documents)
        for prediction in result.predictions:
            for label in prediction.predicted_issue_labels:
                self.assertIn(label, self.report.labels)

    def test_labels_are_language_neutral(self):
        """No language-specific label variants are invented."""
        for label in self.report.labels:
            for language in ("hindi", "bengali", "odia", "english", "hi", "bn", "or"):
                self.assertNotIn(language, label)


# ---------------------------------------------------------------------------
# Input integrity
# ---------------------------------------------------------------------------


class TestInputNotModified(TrainedClassifierMixin, unittest.TestCase):
    def test_annotations_unchanged_by_train(self):
        before = [a.model_dump() for a in self.annotations]
        TfidfLogisticIssueClassifier().train(self.annotations)
        self.assertEqual(before, [a.model_dump() for a in self.annotations])

    def test_annotations_unchanged_after_prediction(self):
        before = [a.model_dump() for a in self.annotations]
        self.classifier.predict(self.documents)
        self.assertEqual(before, [a.model_dump() for a in self.annotations])

    def test_documents_unchanged_by_predict(self):
        before = [d.model_dump() for d in self.documents]
        self.classifier.predict(self.documents)
        self.assertEqual(before, [d.model_dump() for d in self.documents])

    def test_input_text_returned_verbatim(self):
        text = "fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire>"
        result = self.classifier.predict(
            [IssueDocument(document_id="q", text=text, original_text=text)]
        )
        self.assertEqual(result.predictions[0].text, text)
        self.assertEqual(result.predictions[0].original_text, text)

    def test_source_list_not_reordered(self):
        self.classifier.predict(self.documents)
        self.assertEqual([d.document_id for d in self.documents],
                         [d.document_id for d in build_documents()])

    def test_no_preprocessing_imported(self):
        """The stage must not depend on the preprocessing implementation."""
        import app.services.nlp.issue_identification.tfidf_logistic_classifier as module
        source = open(module.__file__, encoding="utf-8").read()
        self.assertNotIn("app.services.nlp.preprocessing", source)
        self.assertNotIn("MultilingualTextPreprocessor", source)
        self.assertNotIn("normalize_", source)


# ---------------------------------------------------------------------------
# Metadata and serialization
# ---------------------------------------------------------------------------


class TestModelMetadata(TrainedClassifierMixin, unittest.TestCase):
    def test_model_name_and_version(self):
        self.assertEqual(self.classifier.name, "tfidf_logistic_multilabel")
        self.assertEqual(self.classifier.version, "v1")
        self.assertEqual(self.report.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(self.report.model_version, "v1")

    def test_result_carries_metadata(self):
        result = self.classifier.predict(self.documents)
        self.assertEqual(result.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(result.model_version, "v1")
        self.assertEqual(result.threshold, 0.5)

    def test_metadata_on_empty_result(self):
        result = self.classifier.predict([])
        self.assertEqual(result.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(result.model_version, "v1")

    def test_config_snapshot_contains_all_knobs(self):
        config = self.report.config
        for key in ("threshold", "tfidf", "logistic_regression", "data_guards", "features"):
            self.assertIn(key, config)
        for key in ("max_features", "min_df", "max_df", "ngram_range", "stop_words",
                    "lowercase", "sublinear_tf", "token_pattern", "norm",
                    "use_idf", "smooth_idf"):
            self.assertIn(key, config["tfidf"])
        for key in ("C", "max_iter", "solver", "class_weight", "random_state"):
            self.assertIn(key, config["logistic_regression"])

    def test_config_is_json_serializable(self):
        import json
        json.dumps(self.report.config)
        json.dumps(self.report.model_dump())
        json.dumps(self.classifier.predict(self.documents).model_dump())

    def test_library_versions_recorded(self):
        self.assertIn("scikit-learn", self.report.library_versions)
        self.assertIn("numpy", self.report.library_versions)

    def test_config_reflects_configurable_values(self):
        classifier = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(C=0.5, max_iter=250, max_features=500, ngram_range=(1, 1)),
            threshold=0.3,
        )
        report = classifier.train(self.annotations)
        self.assertEqual(report.config["logistic_regression"]["C"], 0.5)
        self.assertEqual(report.config["logistic_regression"]["max_iter"], 250)
        self.assertEqual(report.config["tfidf"]["max_features"], 500)
        self.assertEqual(report.config["tfidf"]["ngram_range"], [1, 1])
        self.assertEqual(report.config["threshold"], 0.3)
        self.assertEqual(report.threshold, 0.3)


class TestSerialization(TrainedClassifierMixin, unittest.TestCase):
    def test_result_model_dump_round_trip(self):
        result = self.classifier.predict(self.documents)
        restored = IssueIdentificationResult.model_validate(result.model_dump())
        self.assertEqual(
            [p.predicted_issue_labels for p in restored.predictions],
            [p.predicted_issue_labels for p in result.predictions],
        )

    def test_result_json_round_trip(self):
        result = self.classifier.predict(self.documents)
        restored = IssueIdentificationResult.model_validate_json(
            result.model_dump_json()
        )
        self.assertEqual(
            [p.issue_probabilities for p in restored.predictions],
            [p.issue_probabilities for p in result.predictions],
        )
        self.assertEqual(
            restored.predictions[0].predicted_issue_labels,
            result.predictions[0].predicted_issue_labels,
        )

    def test_training_report_serializes(self):
        restored = IssueTrainingReport.model_validate_json(
            self.report.model_dump_json()
        )
        self.assertEqual(restored.labels, self.report.labels)
        self.assertEqual(
            restored.positive_examples_per_label, self.report.positive_examples_per_label
        )

    def test_evaluation_report_serializes(self):
        report = self.classifier.evaluate(self.annotations)
        restored = IssueEvaluationReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored.micro_f1, report.micro_f1)
        self.assertEqual(restored.per_label_f1, report.per_label_f1)

    def test_prediction_serializes(self):
        prediction = self.classifier.predict(
            self.documents
        ).predictions[0]
        restored = IssuePrediction.model_validate_json(prediction.model_dump_json())
        self.assertEqual(restored.predicted_issue_labels,
                         prediction.predicted_issue_labels)

    def test_annotation_corpus_csv_is_persistence_path(self):
        """The documented CSV round-trips into a trainable corpus."""
        corpus = AnnotationCorpus(annotations=self.annotations)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "issue_annotations.csv")
            corpus.to_csv(path)
            reloaded = AnnotationCorpus.from_csv(path)
        report = TfidfLogisticIssueClassifier().train(reloaded)
        self.assertEqual(report.labels, self.report.labels)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation(TrainedClassifierMixin, unittest.TestCase):
    def test_meaningful_evaluation_on_adequate_data(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertTrue(report.is_meaningful)
        self.assertIsNone(report.reason)
        self.assertEqual(report.n_documents, len(FIXTURE))
        self.assertEqual(report.n_labels, 6)

    def test_metrics_present_and_bounded(self):
        report = self.classifier.evaluate(self.annotations)
        for value in (report.micro_f1, report.macro_f1, report.micro_precision,
                      report.macro_precision, report.macro_recall,
                      report.subset_accuracy, report.hamming_accuracy):
            self.assertIsNotNone(value)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_per_label_f1_for_every_label(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertEqual(set(report.per_label_f1), set(self.report.labels))
        self.assertEqual(set(report.per_label_support), set(self.report.labels))
        self.assertEqual(set(report.per_label_precision), set(self.report.labels))
        self.assertEqual(set(report.per_label_recall), set(self.report.labels))

    def test_support_matches_annotation_counts(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertEqual(report.per_label_support["fees"], 11)
        self.assertEqual(report.per_label_support["roi"], 8)

    def test_classification_report_present(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertIsNotNone(report.classification_report)
        for label in self.report.labels:
            self.assertIn(label, report.classification_report)

    def test_subset_accuracy_is_strictest(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertLessEqual(report.subset_accuracy, report.micro_f1)

    def test_threshold_affects_evaluation(self):
        low = self.classifier.evaluate(self.annotations, threshold=0.1)
        high = self.classifier.evaluate(self.annotations, threshold=0.95)
        self.assertGreaterEqual(low.macro_recall, high.macro_recall)

    def test_evaluation_summary_text(self):
        text = self.classifier.evaluate(self.annotations).format_summary()
        self.assertIn("micro-F1", text)
        self.assertIn("subset accuracy", text)

    def test_too_small_evaluation_set_withholds_metrics(self):
        """No fabricated numbers when there is not enough data."""
        report = self.classifier.evaluate(self.annotations[:4])
        self.assertFalse(report.is_meaningful)
        self.assertIn("at least", report.reason)
        for value in (report.micro_f1, report.macro_f1, report.micro_precision,
                      report.macro_precision, report.macro_recall,
                      report.subset_accuracy, report.hamming_accuracy,
                      report.hamming_loss, report.average_precision):
            self.assertIsNone(value)
        self.assertEqual(report.per_label_f1, {})

    def test_empty_evaluation_set_withholds_metrics(self):
        report = self.classifier.evaluate([])
        self.assertFalse(report.is_meaningful)
        self.assertIsNone(report.micro_f1)

    def test_withheld_summary_is_explicit(self):
        text = self.classifier.evaluate(self.annotations[:4]).format_summary()
        self.assertIn("NOT MEANINGFUL", text)
        self.assertIn("withheld", text)

    def test_support_still_reported_when_withheld(self):
        report = self.classifier.evaluate(self.annotations[:4])
        self.assertEqual(sum(report.per_label_support.values()),
                         sum(len(set(a.labels)) for a in self.annotations[:4]))

    def test_unpredicted_label_warns(self):
        report = evaluate_predictions(
            [[1, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0],
             [0, 0], [1, 0], [0, 0], [1, 0]],
            [[1, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0], [0, 0], [1, 0],
             [0, 0], [1, 0], [0, 0], [1, 0]],
            ["a", "b"],
        )
        self.assertTrue(report.is_meaningful)
        self.assertTrue(any("never predicted" in w for w in report.warnings))
        self.assertEqual(report.per_label_f1["b"], 0.0)
        self.assertLess(report.macro_f1, 1.0)

    def test_evaluate_predictions_shape_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([[1, 0]], [[1, 0, 0]], ["a", "b", "c"])

    def test_evaluate_predictions_label_count_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate_predictions([[1, 0]], [[1, 0]], ["a"])

    def test_single_label_evaluation_withheld(self):
        report = evaluate_predictions(
            [[1]] * 12, [[1]] * 12, ["only_label"]
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("multi-label", report.reason)

    def test_evaluation_accepts_extra_gold_labels_gracefully(self):
        extended = self.annotations + [
            IssueAnnotation(document_id="extra", text="some new complaint",
                            labels=["not_in_model"])
        ]
        report = self.classifier.evaluate(extended)
        self.assertTrue(report.is_meaningful)
        self.assertEqual(report.n_labels, 6)
        self.assertNotIn("not_in_model", report.per_label_f1)


# ---------------------------------------------------------------------------
# Multilingual / code-mixed / emoji
# ---------------------------------------------------------------------------


class TestMultilingualCodeMixed(TrainedClassifierMixin, unittest.TestCase):
    def test_all_languages_present_in_fixture(self):
        scripts = {d.script for d in self.documents}
        self.assertTrue({"latin", "devanagari", "bengali", "odia"} & scripts)
        languages = {d.language for d in self.documents}
        self.assertTrue({"en", "hi", "bn", "or"} <= languages)

    def test_predictions_valid_for_every_language(self):
        result = self.classifier.predict(self.documents)
        self.assertEqual(result.n_documents, len(self.documents))
        for prediction in result.predictions:
            self.assertTrue(set(prediction.issue_probabilities) <= set(self.report.labels))

    def test_code_mixed_metadata_round_trips(self):
        result = self.classifier.predict(self.documents)
        for source, prediction in zip(self.documents, result.predictions):
            self.assertEqual(prediction.language, source.language)
            self.assertEqual(prediction.script, source.script)
            self.assertEqual(prediction.is_code_mixed, source.is_code_mixed)
            self.assertEqual(prediction.detected_languages, source.detected_languages)

    def test_code_mixed_documents_are_predicted(self):
        mixed = [d for d in self.documents if d.is_code_mixed]
        self.assertGreaterEqual(len(mixed), 5)
        result = self.classifier.predict(mixed)
        self.assertEqual(result.n_documents, len(mixed))

    def test_hinglish_uses_shared_label_space(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q", text="mess ka khana kharab hai quality bahut kharab",
                          language="hi", script="latin", is_code_mixed=True,
                          detected_languages=["hi", "en"]),
        ])
        self.assertIn("mess_food", result.predictions[0].predicted_issue_labels)

    def test_romanized_hindi_uses_shared_label_space(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q", text="fees ka bhaari hai aur scholarship se bach bhi nahi milta",
                          language="hi", script="latin", is_code_mixed=True,
                          detected_languages=["hi", "en"]),
        ])
        self.assertIn("fees", result.predictions[0].predicted_issue_labels)

    def test_native_script_not_translated(self):
        docs = [IssueDocument(document_id="q1", text="ফি অনেক বেশি এবং ROI কোনো কাজে আসে না",
                              language="bn", script="bengali"),
                IssueDocument(document_id="q2", text="ଫି ବହୁତ ଅଧିକ ଏବଂ ROI ଉପଯୋଗୀ ନାହିଁ",
                              language="or", script="odia")]
        result = self.classifier.predict(docs)
        for source, prediction in zip(docs, result.predictions):
            self.assertEqual(prediction.text, source.text)
            self.assertEqual(prediction.language, source.language)
        self.assertIn("fees", result.predictions[0].predicted_issue_labels)
        self.assertIn("roi", result.predictions[0].predicted_issue_labels)
        self.assertIn("fees", result.predictions[1].predicted_issue_labels)

    def test_no_language_specific_stopword_removal(self):
        self.assertIsNone(self.classifier.config.stop_words)
        self.assertIsNone(self.report.config["tfidf"]["stop_words"])

    def test_romanized_function_words_survive(self):
        """No language-specific stop-word removal, so code-switch markers stay."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), token_pattern=self.classifier.config.token_pattern
        )
        vectorizer.fit([d.text for d in self.documents])
        vocab = set(vectorizer.vocabulary_)
        for term in ("hai", "ka", "ki", "aur", "nahi", "bahut", "kharab", "zyada"):
            self.assertIn(term, vocab)

    def test_domain_acronyms_preserved(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), token_pattern=self.classifier.config.token_pattern
        )
        vectorizer.fit([d.text for d in self.documents])
        vocab = set(vectorizer.vocabulary_)
        for term in ("iim", "roi", "cat", "mba", "p"):
            if term == "p":
                continue
            self.assertIn(term, vocab)

    def test_emoji_placeholder_tokens_handled(self):
        result = self.classifier.predict([
            IssueDocument(document_id="q",
                          text="fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>"),
        ])
        prediction = result.predictions[0]
        self.assertEqual(
            prediction.predicted_issue_labels, ["fees", "placement_stress", "roi"]
        )
        self.assertIn("<EMOJI_loudly_crying_face>", prediction.text)


class TestPlaceholderHandling(TrainedClassifierMixin, unittest.TestCase):
    EMOJI_TEXT = ("fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> "
                  "placement stress <EMOJI_pensive_face>")

    def test_placeholders_dropped_from_features_by_default(self):
        classifier = TfidfLogisticIssueClassifier()
        classifier.train(self.annotations)
        self.assertNotIn("emoji_fire", classifier._vectorizer.vocabulary_)

    def test_placeholder_dropping_documented_in_config(self):
        self.assertTrue(self.classifier.config.drop_placeholder_tokens)
        self.assertTrue(
            self.report.config["features"]["drop_placeholder_tokens"]
        )

    def test_placeholder_text_returned_verbatim(self):
        result = self.classifier.predict([IssueDocument(document_id="q", text=self.EMOJI_TEXT)])
        self.assertEqual(result.predictions[0].text, self.EMOJI_TEXT)

    def test_placeholder_keeping_does_not_crash(self):
        classifier = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(drop_placeholder_tokens=False)
        )
        classifier.train(self.annotations)
        self.assertIn("emoji_fire", classifier._vectorizer.vocabulary_)
        result = classifier.predict([IssueDocument(document_id="q", text=self.EMOJI_TEXT)])
        self.assertIn("fees", result.predictions[0].predicted_issue_labels)
        self.assertIn("roi", result.predictions[0].predicted_issue_labels)


class TestTopicFeatures(unittest.TestCase):
    def setUp(self):
        self.annotations = build_annotations()
        self.documents = build_documents()

    def test_topic_features_off_by_default(self):
        classifier = TfidfLogisticIssueClassifier()
        report = classifier.train(self.annotations)
        self.assertFalse(report.config["features"]["use_topic_features"])

    def test_topic_metadata_accepted_and_echoed(self):
        classifier = TfidfLogisticIssueClassifier()
        classifier.train(self.annotations)
        docs = [
            IssueDocument(document_id="q", text="fees are way too high",
                          topic_id=2, topic_label="Topic 2",
                          topic_keywords=["fees", "roi", "scholarship"])
        ]
        prediction = classifier.predict(docs).predictions[0]
        self.assertEqual(prediction.topic_id, 2)
        self.assertEqual(prediction.topic_label, "Topic 2")

    def test_topic_features_can_be_enabled(self):
        classifier = TfidfLogisticIssueClassifier(
            config=IssueModelConfig(use_topic_features=True)
        )
        report = classifier.train(self.annotations)
        self.assertTrue(report.config["features"]["use_topic_features"])
        result = classifier.predict(
            [IssueDocument(document_id="q", text="a short vague comment",
                           topic_id=1, topic_keywords=["fees"])]
        )
        self.assertEqual(result.n_documents, 1)
        self.assertIn("fees", result.predictions[0].issue_probabilities)

    def test_topic_feature_tokens_do_not_collide_with_vocabulary(self):
        from app.services.nlp.issue_identification.tfidf_logistic_classifier import (
            TfidfLogisticIssueClassifier as C,
        )
        classifier = C(config=IssueModelConfig(use_topic_features=True))
        text = classifier._featurize_document(
            IssueDocument(document_id="q", text="fees", topic_id=3,
                          topic_keywords=["fees"])
        )
        self.assertIn("topicid3", text)
        self.assertIn("topickeyfees", text)


if __name__ == "__main__":
    unittest.main()
