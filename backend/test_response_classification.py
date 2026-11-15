"""
Comprehensive tests for the response classification stage.

Tests cover:
- The six assignment-defined response types: the exact label set, canonical
  ordering, and rejection of anything outside it
- Annotation models and the annotation CSV schema (round-trip, located errors,
  BOM, blank lines, missing columns)
- Training refusals: empty corpus, missing labels, too few documents, fewer than
  two classes, a class with too little evidence, unusable TF-IDF vocabulary,
  conflicting duplicate document_ids
- Prediction: one class per comment, six probabilities forming a valid
  distribution, confidence, the confidence floor and abstention, determinism
- Context independence: ``parent_comment_text`` / ``topic_id`` / ``issue_labels``
  do not change a V1 prediction
- Metadata and original-text preservation, input immutability, serialization
- Class weighting (default "balanced", disable, reject) and imbalance warnings
- Evaluation: accuracy, macro/micro/weighted F1, per-class metrics, confusion
  matrix, classification report, and the meaningful-data guards
- Multilingual and code-mixed text: Hinglish, Romanized Hindi, Bengali, Odia,
  emoji placeholders
- Architectural isolation: no preprocessing, no database, no coupling to the
  completed stages
"""

import ast
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.response_classification import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    EXPECTED_NUMBER_OF_RESPONSE_TYPES,
    MIN_CLASSES_FOR_MEANINGFUL_METRICS,
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    RESPONSE_TYPE_CODES,
    RESPONSE_TYPE_DESCRIPTIONS,
    RESPONSE_TYPES,
    ResponseAnnotation,
    ResponseClassification,
    ResponseClassificationResult,
    ResponseCorpus,
    ResponseDocument,
    ResponseEvaluationReport,
    ResponseModelConfig,
    ResponsePrediction,
    ResponseTrainingReport,
    TfidfLogisticResponseClassifier,
    evaluate_response_classification,
    is_valid_response_type,
    response_type_order,
    unknown_response_types,
    validate_response_type,
)

PACKAGE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app",
    "services",
    "nlp",
    "response_classification",
)

#: The assignment's taxonomy, spelled out independently of the package constant
#: so a change to the package cannot silently redefine the test's expectation.
ASSIGNMENT_RESPONSE_TYPES = (
    "agreement",
    "personal_experience",
    "opinion",
    "advice",
    "solution",
    "counterargument",
)


# ---------------------------------------------------------------------------
# Synthetic annotation fixtures.
#
# SYNTHETIC, illustrative data written to exercise the architecture. It was not
# collected from Reddit, it was not annotated by real annotators, and no metric
# in this file is project performance.
#
# The text is deliberately keyed on the lexical markers each role tends to use
# ("same", "i faced", "i think", "you should", "apply", "actually") because that
# is what a bag-of-words baseline can actually learn from a handful of examples.
# A larger, real annotation set would show far more genuine overlap.
# ---------------------------------------------------------------------------

# (document_id, preprocessed text, response_type, language, script, code-mixed)
TRAINING_FIXTURE = [
    # --- A. agreement ---
    ("a01", "same bro exactly", "agreement", "en", "latin", False),
    ("a02", "true that, i agree with you", "agreement", "en", "latin", False),
    ("a03", "exactly this is what i said", "agreement", "en", "latin", False),
    ("a04", "yes that is right, same here", "agreement", "en", "latin", False),
    # --- B. personal experience ---
    ("p01", "i faced this exact problem last year", "personal_experience", "en", "latin", False),
    ("p02", "in my own case it happened too", "personal_experience", "en", "latin", False),
    ("p03", "i went through this myself", "personal_experience", "en", "latin", False),
    ("p04", "i remember facing this at my college", "personal_experience", "en", "latin", False),
    # --- C. opinion ---
    ("o01", "i think this college is better", "opinion", "en", "latin", False),
    ("o02", "in my view the campus is fine", "opinion", "en", "latin", False),
    ("o03", "honestly i believe it is good", "opinion", "en", "latin", False),
    ("o04", "i feel this place is underrated", "opinion", "en", "latin", False),
    # --- D. advice ---
    ("d01", "you should contact the placement office", "advice", "en", "latin", False),
    ("d02", "you ought to talk to the professor", "advice", "en", "latin", False),
    ("d03", "i would advise you to start early", "advice", "en", "latin", False),
    ("d04", "you must check the scholarship portal", "advice", "en", "latin", False),
    # --- E. solution ---
    ("s01", "apply for the scholarship on the portal", "solution", "en", "latin", False),
    ("s02", "register on the website before monday", "solution", "en", "latin", False),
    ("s03", "use the hostel wifi to download it", "solution", "en", "latin", False),
    ("s04", "submit the form on the intranet page", "solution", "en", "latin", False),
    # --- F. counterargument ---
    ("c01", "actually the fees are not that high", "counterargument", "en", "latin", False),
    ("c02", "but that is not true here", "counterargument", "en", "latin", False),
    ("c03", "however the fees rose last year", "counterargument", "en", "latin", False),
    ("c04", "that is wrong, i disagree entirely", "counterargument", "en", "latin", False),
]

#: A genuinely held-out set: disjoint ids, disjoint text, three comments per
#: response type.  18 comments is above MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS (10)
#: and every class has support >= 2, so the metrics are computable - but they
#: still describe a hand-written toy corpus, not Reddit.
HELDOUT_FIXTURE = [
    ("h01", "exactly, same thing here", "agreement", "en", "latin", False),
    ("h02", "yes i agree completely", "agreement", "en", "latin", False),
    ("h03", "correct, that matches my point", "agreement", "en", "latin", False),
    ("h04", "i faced the same issue in 2023", "personal_experience", "en", "latin", False),
    ("h05", "this happened to me as well", "personal_experience", "en", "latin", False),
    ("h06", "i experienced this at my campus", "personal_experience", "en", "latin", False),
    ("h07", "i think this college is underrated", "opinion", "en", "latin", False),
    ("h08", "in my opinion the fees are fair", "opinion", "en", "latin", False),
    ("h09", "i believe the mess food is good", "opinion", "en", "latin", False),
    ("h10", "you should talk to the hostel warden", "advice", "en", "latin", False),
    ("h11", "you ought to apply for the bursary", "advice", "en", "latin", False),
    ("h12", "i would advise checking the syllabus", "advice", "en", "latin", False),
    ("h13", "register on the scholarship portal", "solution", "en", "latin", False),
    ("h14", "submit the form on the intranet", "solution", "en", "latin", False),
    ("h15", "download the form from the website", "solution", "en", "latin", False),
    ("h16", "actually the fees are quite low here", "counterargument", "en", "latin", False),
    ("h17", "but that is wrong about this place", "counterargument", "en", "latin", False),
    ("h18", "however i disagree with that claim", "counterargument", "en", "latin", False),
]

#: Code-mixed / non-Latin-script training rows, so the multilingual section has
#: something real to test against.  Each class keeps a recognisable marker.
MULTILINGUAL_FIXTURE = [
    ("m01", "haan bilkul same baat hai", "agreement", "hi", "latin", True),
    ("m02", "sahi hai, main bhi yahi sochta hoon", "agreement", "hi", "latin", True),
    ("m03", "हां बिल्कुल यही बात है", "agreement", "hi", "devanagari", False),
    ("m04", "pichle saal mujhe bhi yahi problem aayi thi", "personal_experience", "hi", "latin", True),
    ("m05", "mere saath bhi yeh hua tha", "personal_experience", "hi", "latin", True),
    ("m06", "गत वर्ष मुझे भी यही समस्या हुई थी", "personal_experience", "hi", "devanagari", False),
    ("m07", "मुझे लगता है यह कॉलेज बेहतर है", "opinion", "hi", "devanagari", False),
    ("m08", "mere nazariye ke hisaab se yeh theek hai", "opinion", "hi", "latin", True),
    ("m09", "হ্যাঁ ঠিক একই কথা", "agreement", "bn", "bengali", False),
    ("m10", "আমার একই অভিজ্ঞতা হয়েছিল", "personal_experience", "bn", "bengali", False),
    ("m11", "আমার মতে এটি ভালো কলেজ", "opinion", "bn", "bengali", False),
    ("m12", "ଆମେ ସେବାଟା ଭଲ ଲାଗୁଛି", "opinion", "or", "oriya", False),
    ("m13", "ଗତ ବର୍ଷ ମୁଁକୁ ମିଛି", "personal_experience", "or", "oriya", False),
    ("m14", "ହଁ ଠିକ କଥା", "agreement", "or", "oriya", False),
    # advice / solution / counterargument in Hinglish and one non-Latin set
    ("m15", "aapko placement office se baat karni chahiye", "advice", "hi", "latin", True),
    ("m16", "aapko yahan apply karna chahiye", "advice", "hi", "latin", True),
    ("m17", "scholarship ke liye portal par apply karo", "solution", "hi", "latin", True),
    ("m18", "intranet par form submit kar do", "solution", "hi", "latin", True),
    ("m19", "lekin fees itne zyada nahi hain", "counterargument", "hi", "latin", True),
    ("m20", "par yeh baat galat hai", "counterargument", "hi", "latin", True),
    ("m21", "আপনার প্লেসমেন্ট অফিসে যোগাযোগ করা উচিত", "advice", "bn", "bengali", False),
    ("m22", "বৃত্তির জন্য আবেদন করুন", "solution", "bn", "bengali", False),
    ("m23", "কিন্তু ফি এত বেশি নয়", "counterargument", "bn", "bengali", False),
    ("m24", "ଆପଣ ଆବେଦନ କରିବେ", "advice", "or", "oriya", False),
    ("m25", "ପୋର୍ଟାଲରେ ଫର୍ମ ଜମା ଦିଅନ୍ତୁ", "solution", "or", "oriya", False),
    ("m26", "କିନ୍ତୁ ଫି ଏତେ ଅଧିକ ନାହିଁ", "counterargument", "or", "oriya", False),
]


def _imbalanced_corpus(majority="agreement", majority_count=10, minority_count=3):
    """
    A deliberately skewed six-class corpus: ``majority_count`` rows of one role
    and ``minority_count`` of each other role.

    Used instead of a perfectly balanced set because ``class_weight="balanced"``
    computes equal weights on a balanced corpus, making the two fitted models
    identical - so the class-weight and imbalance tests would prove nothing.
    """
    rows = [
        (f"maj{index}", f"same bro exactly {index}", "agreement", "en", "latin", False)
        for index in range(majority_count)
    ]
    for response_type, base in (
        ("personal_experience", "i faced this"),
        ("opinion", "i think this"),
        ("advice", "you should"),
        ("solution", "apply now"),
        ("counterargument", "actually that is wrong"),
    ):
        for index in range(minority_count):
            rows.append(
                (
                    f"{response_type[:2]}{index}",
                    f"{base} {index}",
                    response_type,
                    "en",
                    "latin",
                    False,
                )
            )
    return rows


def _annotations(fixture):
    return [
        ResponseAnnotation(
            document_id=document_id,
            text=text,
            response_type=response_type,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
        )
        for document_id, text, response_type, language, script, is_code_mixed in fixture
    ]


def _documents(fixture):
    return [
        ResponseDocument(
            document_id=document_id,
            text=text,
            original_text=text,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
        )
        for document_id, text, _role, language, script, is_code_mixed in fixture
    ]


class ResponseStageTestCase(unittest.TestCase):
    """Shared, once-per-class-model setup: a trained six-class classifier."""

    @classmethod
    def setUpClass(cls):
        cls.annotations = _annotations(TRAINING_FIXTURE)
        cls.classifier = TfidfLogisticResponseClassifier()
        cls.report = cls.classifier.train(cls.annotations)


# ===========================================================================
# The label set
# ===========================================================================


class TestResponseLabels(unittest.TestCase):
    def test_labels_are_exactly_the_six_assignment_classes(self):
        self.assertEqual(
            tuple(RESPONSE_TYPES),
            ASSIGNMENT_RESPONSE_TYPES,
            "the V1 taxonomy must be exactly the six assignment response types",
        )

    def test_label_set_has_no_duplicates_and_is_six_long(self):
        self.assertEqual(len(RESPONSE_TYPES), 6)
        self.assertEqual(len(set(RESPONSE_TYPES)), 6)
        self.assertEqual(EXPECTED_NUMBER_OF_RESPONSE_TYPES, 6)

    def test_labels_are_ordered_as_the_assignment_lists_them(self):
        """A-F order, so the probability vector and confusion matrix read A-F."""
        self.assertEqual(list(RESPONSE_TYPES)[0], "agreement")
        self.assertEqual(list(RESPONSE_TYPES)[-1], "counterargument")
        self.assertEqual(
            [RESPONSE_TYPE_CODES[name] for name in RESPONSE_TYPES],
            ["A", "B", "C", "D", "E", "F"],
        )

    def test_every_label_has_a_description(self):
        for name in RESPONSE_TYPES:
            self.assertIn(name, RESPONSE_TYPE_DESCRIPTIONS)
            self.assertTrue(RESPONSE_TYPE_DESCRIPTIONS[name].strip())

    def test_is_valid_response_type(self):
        for name in RESPONSE_TYPES:
            self.assertTrue(is_valid_response_type(name))
        for value in ("", "   ", None, "Agreement", "personal experience", "unknown"):
            self.assertFalse(is_valid_response_type(value))
        self.assertFalse(is_valid_response_type(7))

    def test_validate_response_type_normalises_whitespace(self):
        self.assertEqual(validate_response_type("  advice  "), "advice")

    def test_validate_response_type_rejects_out_of_taxonomy_values(self):
        for value in ("sarcasm", "agreement|advice", "personal-experience"):
            with self.assertRaises(ValueError) as ctx:
                validate_response_type(value, document_id="ann-1")
            message = str(ctx.exception)
            self.assertIn("ann-1", message)
            # The error must name the six valid values, or it is not actionable.
            for name in RESPONSE_TYPES:
                self.assertIn(name, message)

    def test_validate_response_type_rejects_blank(self):
        with self.assertRaises(ValueError):
            validate_response_type("   ", document_id="ann-2")
        with self.assertRaises(ValueError):
            validate_response_type(None)

    def test_unknown_response_types_helper(self):
        self.assertEqual(
            unknown_response_types(["advice", "sarcasm", "opinion", "sarcasm", None, ""]),
            ["sarcasm"],
        )

    def test_blank_values_are_not_reported_as_unknown_labels(self):
        """A blank cell is a missing label, not a seventh class."""
        self.assertEqual(unknown_response_types(["", "   ", None]), [])

    def test_response_type_order_is_canonical(self):
        shuffled = ["solution", "agreement", "counterargument", "opinion", "advice", "personal_experience"]
        self.assertEqual(
            response_type_order(shuffled), list(ASSIGNMENT_RESPONSE_TYPES)
        )
        self.assertEqual(response_type_order(["advice", "agreement"]), ["agreement", "advice"])
        self.assertEqual(response_type_order([]), [])

    def test_response_type_order_keeps_unknown_values_visible(self):
        """Bypassing validation must not silently drop a label from reporting."""
        self.assertEqual(
            response_type_order(["advice", "sarcasm"]), ["advice", "sarcasm"]
        )


# ===========================================================================
# Annotation model + CSV
# ===========================================================================


class TestResponseAnnotation(unittest.TestCase):
    def test_annotation_fields(self):
        annotation = ResponseAnnotation(
            document_id="ann-001",
            text="same bro",
            response_type="agreement",
            post_id="post-1",
            comment_id="c-101",
            language="en",
            script="latin",
            is_code_mixed=True,
            detected_languages=["en", "hi"],
        )
        self.assertEqual(annotation.normalised_response_type(), "agreement")
        self.assertEqual(annotation.validate_response_type(), "agreement")
        self.assertEqual(annotation.post_id, "post-1")
        self.assertEqual(annotation.detected_languages, ["en", "hi"])

    def test_annotation_defaults(self):
        annotation = ResponseAnnotation(document_id="a", response_type="opinion")
        self.assertEqual(annotation.text, "")
        self.assertIsNone(annotation.post_id)
        self.assertIsNone(annotation.comment_id)
        self.assertIsNone(annotation.topic_id)
        self.assertEqual(annotation.language, "unknown")
        self.assertEqual(annotation.script, "unknown")
        self.assertFalse(annotation.is_code_mixed)
        self.assertEqual(annotation.detected_languages, [])

    def test_response_type_is_required(self):
        with self.assertRaises(Exception):
            ResponseAnnotation(document_id="a", text="hi")

    def test_annotation_rejects_invalid_label(self):
        annotation = ResponseAnnotation(
            document_id="ann-9", text="x", response_type="sarcasm"
        )
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_response_type()
        self.assertIn("sarcasm", str(ctx.exception))

    def test_to_response_document_projects_metadata(self):
        annotation = ResponseAnnotation(
            document_id="ann-002",
            text="apply for the scholarship",
            response_type="solution",
            post_id="post-1",
            comment_id="c-102",
            language="en",
            script="latin",
            is_code_mixed=True,
            detected_languages=["en", "hi"],
            topic_id=3,
        )
        document = annotation.to_response_document()
        self.assertIsInstance(document, ResponseDocument)
        self.assertEqual(document.document_id, "ann-002")
        self.assertEqual(document.text, "apply for the scholarship")
        self.assertEqual(document.original_text, "apply for the scholarship")
        self.assertEqual(document.post_id, "post-1")
        self.assertEqual(document.comment_id, "c-102")
        self.assertEqual(document.topic_id, 3)
        self.assertTrue(document.is_code_mixed)

    def test_csv_row_round_trip(self):
        original = ResponseAnnotation(
            document_id="ann-003",
            text="actually the fees are not that high",
            response_type="counterargument",
            post_id="post-2",
            comment_id="c-200",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        restored = ResponseAnnotation.from_csv_row(original.to_csv_row())
        self.assertEqual(restored.document_id, original.document_id)
        self.assertEqual(restored.text, original.text)
        self.assertEqual(restored.response_type, original.response_type)
        self.assertEqual(restored.post_id, original.post_id)
        self.assertEqual(restored.detected_languages, original.detected_languages)
        self.assertEqual(restored.is_code_mixed, True)

    def test_csv_row_uses_documented_columns(self):
        row = ResponseAnnotation(document_id="a", text="t", response_type="opinion").to_csv_row()
        self.assertEqual(set(row), set(CSV_COLUMNS))
        self.assertEqual(list(row), CSV_COLUMNS)

    def test_from_csv_row_requires_id_and_type(self):
        with self.assertRaises(ValueError):
            ResponseAnnotation.from_csv_row({"document_id": "", "text": "x", "response_type": "opinion"})
        with self.assertRaises(ValueError):
            ResponseAnnotation.from_csv_row({"document_id": "a", "text": "x", "response_type": ""})

    def test_from_csv_row_rejects_non_integer_topic_id(self):
        with self.assertRaises(ValueError) as ctx:
            ResponseAnnotation.from_csv_row(
                {"document_id": "a", "text": "x", "response_type": "opinion", "topic_id": "three"}
            )
        self.assertIn("topic_id", str(ctx.exception))

    def test_from_csv_row_parses_bool_spellings(self):
        for value, expected in (("true", True), ("YES", True), ("1", True), ("false", False), ("0", False)):
            row = ResponseAnnotation.from_csv_row(
                {"document_id": "a", "text": "x", "response_type": "opinion", "is_code_mixed": value}
            )
            self.assertEqual(row.is_code_mixed, expected, value)

    def test_from_csv_row_parses_optional_topic_id(self):
        row = ResponseAnnotation.from_csv_row(
            {"document_id": "a", "text": "x", "response_type": "opinion", "topic_id": "7"}
        )
        self.assertEqual(row.topic_id, 7)
        blank = ResponseAnnotation.from_csv_row(
            {"document_id": "a", "text": "x", "response_type": "opinion", "topic_id": ""}
        )
        self.assertIsNone(blank.topic_id)


class TestResponseCorpus(unittest.TestCase):
    def test_corpus_basics(self):
        corpus = ResponseCorpus(annotations=_annotations(TRAINING_FIXTURE))
        self.assertEqual(len(corpus), len(TRAINING_FIXTURE))
        self.assertEqual(len(list(corpus)), len(TRAINING_FIXTURE))
        self.assertEqual(corpus.document_ids[0], "a01")
        corpus.validate()

    def test_class_counts_are_canonically_ordered(self):
        corpus = ResponseCorpus(annotations=_annotations(TRAINING_FIXTURE))
        counts = corpus.class_counts()
        self.assertEqual(len(counts), 6)
        for name, count in counts.items():
            self.assertEqual(count, 4, name)
        self.assertEqual(list(counts), list(RESPONSE_TYPES))

    def test_labels_and_missing_response_types(self):
        corpus = ResponseCorpus(
            annotations=_annotations(
                [r for r in TRAINING_FIXTURE if r[2] in ("agreement", "advice")]
            )
        )
        self.assertEqual(corpus.labels, ["agreement", "advice"])
        self.assertEqual(
            corpus.missing_response_types(),
            ["personal_experience", "opinion", "solution", "counterargument"],
        )

    def test_duplicate_document_ids_detected(self):
        corpus = ResponseCorpus(
            annotations=[
                ResponseAnnotation(document_id="d1", text="same bro", response_type="agreement"),
                ResponseAnnotation(document_id="d1", text="same bro again", response_type="agreement"),
            ]
        )
        self.assertEqual(corpus.duplicate_document_ids(), ["d1"])
        self.assertEqual(corpus.conflicting_document_ids(), [])

    def test_conflicting_document_ids_detected(self):
        """Same comment, two roles: a self-contradictory corpus, not a duplicate."""
        corpus = ResponseCorpus(
            annotations=[
                ResponseAnnotation(document_id="d1", text="x", response_type="agreement"),
                ResponseAnnotation(document_id="d1", text="x", response_type="advice"),
                ResponseAnnotation(document_id="d2", text="y", response_type="opinion"),
                ResponseAnnotation(document_id="d2", text="y", response_type="opinion"),
            ]
        )
        self.assertEqual(corpus.conflicting_document_ids(), ["d1"])
        self.assertEqual(corpus.duplicate_document_ids(), ["d1", "d2"])
        with self.assertRaises(ValueError) as ctx:
            corpus.validate_unique_document_ids()
        self.assertIn("d1", str(ctx.exception))

    def test_unique_document_ids_passes_on_clean_corpus(self):
        corpus = ResponseCorpus(annotations=_annotations(TRAINING_FIXTURE))
        corpus.validate_unique_document_ids()

    def test_from_records(self):
        corpus = ResponseCorpus.from_records(
            [
                {"document_id": "a", "text": "same", "response_type": "agreement"},
                {"document_id": "b", "text": "apply now", "response_type": "solution"},
            ]
        )
        self.assertEqual(len(corpus), 2)


class TestResponseAnnotationCsv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, content, encoding="utf-8"):
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding=encoding, newline="") as handle:
            handle.write(content)
        return path

    def test_round_trip(self):
        corpus = ResponseCorpus(annotations=_annotations(TRAINING_FIXTURE))
        path = os.path.join(self.tmpdir, "response_annotations.csv")
        corpus.to_csv(path)
        self.assertTrue(os.path.exists(path))

        with open(path, "r", encoding="utf-8") as handle:
            header = handle.readline().strip()
        self.assertEqual(header, ",".join(CSV_COLUMNS))

        restored = ResponseCorpus.from_csv(path)
        self.assertEqual(len(restored), len(corpus))
        self.assertEqual(restored.labels, corpus.labels)
        for original, loaded in zip(corpus.annotations, restored.annotations):
            self.assertEqual(loaded.document_id, original.document_id)
            self.assertEqual(loaded.text, original.text)
            self.assertEqual(loaded.response_type, original.response_type)

    def test_default_annotation_path(self):
        self.assertEqual(
            DEFAULT_ANNOTATION_CSV,
            os.path.join("data", "annotations", "response_annotations.csv"),
        )

    def test_missing_file_names_expected_columns(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ResponseCorpus.from_csv(os.path.join(self.tmpdir, "nope.csv"))
        message = str(ctx.exception)
        self.assertIn("document_id", message)
        self.assertIn("response_type", message)

    def test_missing_required_column(self):
        path = self._write(
            "bad.csv", "document_id,text\nann-1,same bro\n"
        )
        with self.assertRaises(ValueError) as ctx:
            ResponseCorpus.from_csv(path)
        self.assertIn("response_type", str(ctx.exception))

    def test_empty_file_has_no_header(self):
        path = self._write("empty.csv", "")
        with self.assertRaises(ValueError) as ctx:
            ResponseCorpus.from_csv(path)
        self.assertIn("header", str(ctx.exception))

    def test_blank_lines_are_skipped(self):
        path = self._write(
            "blanks.csv",
            "document_id,text,response_type\n"
            "ann-1,same bro,agreement\n"
            "\n"
            "ann-2,apply now,solution\n",
        )
        corpus = ResponseCorpus.from_csv(path)
        self.assertEqual(len(corpus), 2)

    def test_utf8_bom_is_tolerated(self):
        path = self._write(
            "bom.csv",
            "document_id,text,response_type\nann-1,same bro,agreement\n",
            encoding="utf-8-sig",
        )
        corpus = ResponseCorpus.from_csv(path)
        self.assertEqual(corpus.annotations[0].document_id, "ann-1")

    def test_malformed_row_reports_file_and_line(self):
        path = self._write(
            "located.csv",
            "document_id,text,response_type\n"
            "ann-1,same bro,agreement\n"
            "ann-2,,,\n",
        )
        with self.assertRaises(ValueError) as ctx:
            ResponseCorpus.from_csv(path)
        self.assertIn(":3", str(ctx.exception))

    def test_extra_columns_are_ignored(self):
        path = self._write(
            "extra.csv",
            "document_id,text,response_type,annotator,confidence\n"
            "ann-1,same bro,agreement,alice,0.9\n",
        )
        corpus = ResponseCorpus.from_csv(path)
        self.assertEqual(corpus.annotations[0].response_type, "agreement")

    def test_non_latin_text_survives_csv_round_trip(self):
        corpus = ResponseCorpus(annotations=_annotations(MULTILINGUAL_FIXTURE))
        path = os.path.join(self.tmpdir, "multi.csv")
        corpus.to_csv(path)
        restored = ResponseCorpus.from_csv(path)
        self.assertEqual(len(restored), len(MULTILINGUAL_FIXTURE))
        for original, loaded in zip(corpus.annotations, restored.annotations):
            self.assertEqual(loaded.text, original.text)

    def test_corpus_to_csv_creates_parent_directories(self):
        corpus = ResponseCorpus(
            annotations=[
                ResponseAnnotation(document_id="a", text="same bro", response_type="agreement")
            ]
        )
        path = os.path.join(self.tmpdir, "nested", "deeper", "out.csv")
        corpus.to_csv(path)
        self.assertTrue(os.path.exists(path))


# ===========================================================================
# Input document
# ===========================================================================


class TestResponseDocument(unittest.TestCase):
    def test_defaults(self):
        document = ResponseDocument(document_id="d")
        self.assertEqual(document.text, "")
        self.assertIsNone(document.post_id)
        self.assertIsNone(document.comment_id)
        self.assertIsNone(document.original_text)
        self.assertEqual(document.language, "unknown")
        self.assertEqual(document.script, "unknown")
        self.assertFalse(document.is_code_mixed)
        self.assertEqual(document.detected_languages, [])
        self.assertIsNone(document.topic_id)
        self.assertEqual(document.topic_keywords, [])
        self.assertEqual(document.issue_labels, [])
        self.assertEqual(document.issue_probabilities, {})
        self.assertIsNone(document.parent_comment_text)
        self.assertFalse(document.has_context)

    def test_context_is_optional_and_detected(self):
        self.assertFalse(ResponseDocument(document_id="d", topic_id=2).has_context is False)
        self.assertTrue(ResponseDocument(document_id="d", topic_id=2).has_context)
        self.assertTrue(
            ResponseDocument(document_id="d", parent_comment_text="parent").has_context
        )
        self.assertTrue(
            ResponseDocument(document_id="d", issue_labels=["fees"]).has_context
        )
        # A comment alone is a valid V1 input.
        self.assertFalse(ResponseDocument(document_id="d", text="same bro").has_context)

    def test_list_fields_do_not_alias_across_documents(self):
        first = ResponseDocument(document_id="a", issue_labels=["fees"])
        second = ResponseDocument(document_id="b")
        first.issue_labels.append("hostel")
        self.assertEqual(second.issue_labels, [])


# ===========================================================================
# Training
# ===========================================================================


class TestResponseTraining(ResponseStageTestCase):
    def test_model_name_and_version(self):
        self.assertEqual(self.classifier.name, "tfidf_logistic_multiclass")
        self.assertEqual(self.classifier.version, "v1")
        self.assertEqual(self.report.model_name, "tfidf_logistic_multiclass")
        self.assertEqual(self.report.model_version, "v1")

    def test_training_report_type_and_learned_classes(self):
        self.assertIsInstance(self.report, ResponseTrainingReport)
        self.assertEqual(self.report.n_training_documents, len(TRAINING_FIXTURE))
        self.assertEqual(self.report.n_classes, 6)
        self.assertEqual(self.report.labels, list(RESPONSE_TYPES))
        self.assertEqual(self.report.expected_labels, list(RESPONSE_TYPES))
        self.assertEqual(self.report.missing_response_types, [])
        self.assertTrue(self.report.covers_full_taxonomy)
        self.assertTrue(self.report.is_usable)

    def test_report_records_evidence_per_class(self):
        for name in RESPONSE_TYPES:
            self.assertEqual(self.report.examples_per_class[name], 4, name)
        self.assertAlmostEqual(
            self.report.majority_class_share,
            4 / len(TRAINING_FIXTURE),
            places=6,
        )
        self.assertGreater(self.report.n_features, 0)
        self.assertEqual(self.report.classes_per_document, 1)

    def test_report_carries_configuration_metadata(self):
        self.assertEqual(self.report.config["logistic_regression"]["solver"], "lbfgs")
        self.assertEqual(self.report.config["logistic_regression"]["random_state"], 42)
        self.assertEqual(self.report.config["logistic_regression"]["class_weight"], "balanced")
        self.assertEqual(self.report.config["tfidf"]["ngram_range"], [1, 2])
        self.assertIn("scikit-learn", self.report.library_versions)
        self.assertIsNone(self.report.trained_at)

    def test_trained_flag_and_labels_property(self):
        self.assertTrue(self.classifier.is_trained)
        self.assertEqual(self.classifier.labels, list(RESPONSE_TYPES))
        self.assertEqual(self.classifier.missing_response_types, [])

    def test_timestamp_is_opt_in(self):
        classifier = TfidfLogisticResponseClassifier(include_timestamp=True)
        report = classifier.train(self.annotations)
        self.assertIsNotNone(report.trained_at)

    def test_train_is_repeatable(self):
        second = TfidfLogisticResponseClassifier()
        first_report = second.train(self.annotations)
        third_report = second.train(self.annotations)
        self.assertEqual(
            first_report.examples_per_class, third_report.examples_per_class
        )
        first_docs = [ResponseDocument(document_id="q", text="same bro")]
        a = second.predict(first_docs, threshold=0.01).predictions[0]
        second.train(self.annotations)
        b = second.predict(first_docs, threshold=0.01).predictions[0]
        self.assertEqual(a.response_probabilities, b.response_probabilities)

    def test_subset_corpus_trains_but_reports_missing_types(self):
        subset = _annotations(
            [row for row in TRAINING_FIXTURE if row[2] in ("agreement", "opinion")]
        )
        classifier = TfidfLogisticResponseClassifier()
        report = classifier.train(subset)
        self.assertEqual(report.labels, ["agreement", "opinion"])
        self.assertEqual(
            report.missing_response_types,
            ["personal_experience", "advice", "solution", "counterargument"],
        )
        self.assertFalse(report.covers_full_taxonomy)
        self.assertTrue(
            any("no training examples" in warning for warning in report.warnings),
            report.warnings,
        )
        # The model must not invent a class it was never taught.
        self.assertEqual(classifier.labels, ["agreement", "opinion"])

    def test_require_all_response_types_refuses_partial_corpus(self):
        subset = _annotations(
            [row for row in TRAINING_FIXTURE if row[2] in ("agreement", "opinion")]
        )
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(require_all_response_types=True)
        )
        with self.assertRaises(ValueError) as ctx:
            classifier.train(subset)
        message = str(ctx.exception)
        self.assertIn("incomplete taxonomy", message)
        for name in ("personal_experience", "advice", "solution", "counterargument"):
            self.assertIn(name, message)

    def test_require_all_response_types_accepts_full_corpus(self):
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(require_all_response_types=True)
        )
        report = classifier.train(self.annotations)
        self.assertTrue(report.covers_full_taxonomy)

    def test_imbalance_warning_when_one_class_dominates(self):
        """
        A majority role must be reported, because it makes accuracy worthless.

        The guard is exercised with a lowered ``imbalance_warning_share`` rather
        than an absurdly lopsided corpus: 10 of 25 comments being one role is
        already enough to inflate accuracy, and building a 41:10 corpus to cross
        the default 0.8 would only prove arithmetic.
        """
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(imbalance_warning_share=0.35)
        )
        report = classifier.train(_annotations(_imbalanced_corpus()))
        self.assertEqual(report.majority_class, "agreement")
        self.assertEqual(report.majority_class_share, 0.4)
        self.assertTrue(
            any("class imbalance" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_imbalance_warning_threshold_is_strictly_greater_than(self):
        """A share exactly at the threshold is not yet 'imbalanced'."""
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(imbalance_warning_share=0.4)
        )
        report = classifier.train(_annotations(_imbalanced_corpus()))
        self.assertEqual(report.majority_class_share, 0.4)
        self.assertFalse(
            any("class imbalance" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_no_imbalance_warning_on_balanced_corpus(self):
        self.assertFalse(
            any("class imbalance" in warning for warning in self.report.warnings),
            self.report.warnings,
        )


class TestResponseTrainingRefusals(unittest.TestCase):
    def test_empty_corpus(self):
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(ValueError) as ctx:
            classifier.train([])
        message = str(ctx.exception)
        self.assertIn("no training data", message)
        self.assertIn(DEFAULT_ANNOTATION_CSV, message)

    def test_none_corpus(self):
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(ValueError):
            classifier.train(None)

    def test_wrong_record_type(self):
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(TypeError) as ctx:
            classifier.train([{"document_id": "a", "response_type": "opinion"}])
        self.assertIn("ResponseAnnotation", str(ctx.exception))

    def test_invalid_label_refuses_training(self):
        rows = list(TRAINING_FIXTURE) + [
            ("x1", "some new thing", "sarcasm", "en", "latin", False)
        ]
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        self.assertIn("sarcasm", str(ctx.exception))

    def test_blank_label_refuses_training(self):
        classifier = TfidfLogisticResponseClassifier()
        rows = list(TRAINING_FIXTURE)
        rows[0] = ("a01", "same bro", "   ", "en", "latin", False)
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        self.assertIn("a01", str(ctx.exception))

    def test_too_few_documents(self):
        classifier = TfidfLogisticResponseClassifier()
        rows = _annotations(TRAINING_FIXTURE[:3])
        with self.assertRaises(ValueError) as ctx:
            classifier.train(rows)
        self.assertIn("insufficient training data", str(ctx.exception))

    def test_single_class_is_insufficient(self):
        classifier = TfidfLogisticResponseClassifier()
        rows = [
            ("a1", "same bro", "agreement", "en", "latin", False),
            ("a2", "true that", "agreement", "en", "latin", False),
            ("a3", "exactly right", "agreement", "en", "latin", False),
            ("a4", "yes i agree", "agreement", "en", "latin", False),
            ("a5", "correct indeed", "agreement", "en", "latin", False),
        ]
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        message = str(ctx.exception)
        self.assertIn("insufficient label variation", message)
        self.assertIn("at least 2", message)

    def test_class_with_too_few_examples(self):
        """One example of a role is not enough to learn it; two is the floor."""
        rows = [row for row in TRAINING_FIXTURE if row[0] != "c04"]
        rows = [row for row in rows if row[0] not in ("c02", "c03")]
        self.assertEqual(
            len([row for row in rows if row[2] == "counterargument"]), 1
        )
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        message = str(ctx.exception)
        self.assertIn("insufficient examples per class", message)
        self.assertIn("counterargument", message)
        self.assertIn("(1)", message)

    def test_min_examples_per_class_is_configurable(self):
        rows = [row for row in TRAINING_FIXTURE if row[0] != "c04"]
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(min_examples_per_class=1)
        )
        report = classifier.train(_annotations(rows))
        self.assertEqual(report.examples_per_class["counterargument"], 3)

    def test_conflicting_duplicate_document_ids_refuse_training(self):
        rows = list(TRAINING_FIXTURE) + [
            ("a01", "same bro exactly", "advice", "en", "latin", False)
        ]
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        message = str(ctx.exception)
        self.assertIn("annotation contract violated", message)
        self.assertIn("a01", message)
        self.assertIn("single-label", message)

    def test_identical_duplicates_train_but_warn(self):
        """A repeated row with the same label is not a contradiction, just skew."""
        rows = list(TRAINING_FIXTURE) + [
            ("a01", "same bro exactly", "agreement", "en", "latin", False)
        ]
        classifier = TfidfLogisticResponseClassifier()
        report = classifier.train(_annotations(rows))
        self.assertTrue(
            any("duplicate document_ids" in warning for warning in report.warnings),
            report.warnings,
        )
        self.assertEqual(report.examples_per_class["agreement"], 5)

    def test_empty_vocabulary_is_refused(self):
        """Text made only of placeholders yields no features, which is an error."""
        rows = [
            (f"e{i}", "<URL>", "agreement" if i < 2 else "opinion", "en", "latin", False)
            for i in range(6)
        ]
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(drop_placeholder_tokens=True)
        )
        with self.assertRaises(ValueError) as ctx:
            classifier.train(_annotations(rows))
        self.assertIn("no usable TF-IDF features", str(ctx.exception))

    def test_min_df_above_max_df_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(min_df=3, max_df=0.5))
        self.assertIn("min_df", str(ctx.exception))

    def test_predict_before_training_refuses(self):
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(RuntimeError) as ctx:
            classifier.predict([ResponseDocument(document_id="d", text="same bro")])
        message = str(ctx.exception)
        self.assertIn("has not been trained", message)
        for name in RESPONSE_TYPES:
            self.assertIn(name, message)

    def test_evaluate_before_training_refuses(self):
        classifier = TfidfLogisticResponseClassifier()
        with self.assertRaises(RuntimeError):
            classifier.evaluate(_annotations(HELDOUT_FIXTURE))


# ===========================================================================
# Prediction
# ===========================================================================


class TestResponsePrediction(ResponseStageTestCase):
    def setUp(self):
        # A low floor so the arg-max is observable on a small toy corpus; the
        # abstain path is tested separately at the default 0.5 floor.
        self.docs = [
            ResponseDocument(document_id="t1", text="same bro exactly"),
            ResponseDocument(document_id="t2", text="i faced this exact problem last year"),
            ResponseDocument(document_id="t3", text="i think this college is better"),
            ResponseDocument(document_id="t4", text="you should contact the placement office"),
            ResponseDocument(document_id="t5", text="apply for the scholarship on the portal"),
            ResponseDocument(document_id="t6", text="actually the fees are not that high"),
        ]
        self.result = self.classifier.predict(self.docs, threshold=0.01)

    def test_result_type_and_metadata(self):
        self.assertIsInstance(self.result, ResponseClassificationResult)
        self.assertEqual(self.result.model_name, "tfidf_logistic_multiclass")
        self.assertEqual(self.result.model_version, "v1")
        self.assertEqual(self.result.n_documents, len(self.docs))
        self.assertEqual(self.result.known_response_types, list(RESPONSE_TYPES))
        self.assertEqual(self.result.missing_response_types, [])
        self.assertIn("scikit-learn", self.result.library_versions)

    def test_each_comment_gets_exactly_one_class(self):
        for prediction in self.result.predictions:
            self.assertIsInstance(prediction, ResponsePrediction)
            self.assertIsInstance(prediction.response_type, str)
            self.assertIn(prediction.response_type, RESPONSE_TYPES)

    def test_all_six_response_types_are_predicted(self):
        """One recognisable comment per role, each assigned to that role."""
        predicted = {p.document_id: p.response_type for p in self.result.predictions}
        expected = {
            "t1": "agreement",
            "t2": "personal_experience",
            "t3": "opinion",
            "t4": "advice",
            "t5": "solution",
            "t6": "counterargument",
        }
        for document_id, response_type in expected.items():
            self.assertEqual(predicted[document_id], response_type, document_id)

    def test_probability_vector_has_all_six_values(self):
        for prediction in self.result.predictions:
            self.assertEqual(
                list(prediction.response_probabilities), list(RESPONSE_TYPES)
            )
            self.assertEqual(len(prediction.response_probabilities), 6)

    def test_probabilities_form_a_valid_distribution(self):
        """Sum to 1 within rounding tolerance; every value a probability."""
        for prediction in self.result.predictions:
            values = list(prediction.response_probabilities.values())
            for value in values:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            # 6-dp rounding means the sum can land 1e-6 either side of 1.
            self.assertTrue(
                math.isclose(sum(values), 1.0, abs_tol=1e-4),
                f"{prediction.document_id} sums to {sum(values)}",
            )

    def test_response_type_is_the_arg_max(self):
        for prediction in self.result.predictions:
            top = max(
                sorted(prediction.response_probabilities.items()),
                key=lambda kv: kv[1],
            )[0]
            self.assertEqual(prediction.response_type, top)
            self.assertEqual(prediction.top_response_type(), top)

    def test_confidence_matches_winning_probability(self):
        for prediction in self.result.predictions:
            self.assertAlmostEqual(
                prediction.confidence,
                prediction.response_probabilities[prediction.response_type],
                places=6,
            )
            self.assertGreater(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)

    def test_result_level_helpers(self):
        self.assertEqual(
            self.result.response_type_frequency(), {name: 1 for name in RESPONSE_TYPES}
        )
        self.assertGreater(self.result.average_confidence(), 0.0)
        first = self.result.prediction_for("t1")
        self.assertIsNotNone(first)
        self.assertIsNone(self.result.prediction_for("does-not-exist"))

    def test_unpredicted_response_types_reported(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="only", text="same bro exactly")],
            threshold=0.01,
        )
        self.assertEqual(
            result.unpredicted_response_types,
            [
                "personal_experience",
                "opinion",
                "advice",
                "solution",
                "counterargument",
            ],
        )

    def test_probability_of_unknown_class_is_none(self):
        prediction = self.result.predictions[0]
        self.assertIsNone(prediction.probability_of("sarcasm"))

    def test_response_type_at_re_derives_without_inference(self):
        prediction = self.result.predictions[0]
        # Moving the floor above the winning probability abstains.
        self.assertIsNone(prediction.response_type_at(0.999))
        self.assertEqual(prediction.response_type_at(0.0), "agreement")

    def test_has_response_type_flag(self):
        self.assertTrue(self.result.predictions[0].has_response_type)


class TestResponseAbstention(ResponseStageTestCase):
    """
    The confidence floor is a floor on a single softmax winner.

    Note what these tests do *not* claim: a 0.5 floor looks high for a six-way
    softmax, and on a real corpus it usually would be.  This toy corpus is
    artificially separable, so the model does clear 0.5 and commits.  The
    mechanism is what is tested - and when it abstains, the full distribution
    stays available so a consumer can decide differently.
    """

    def test_default_floor_commits_on_a_separable_toy_corpus(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="same bro exactly")]
        )
        prediction = result.predictions[0]
        self.assertEqual(self.classifier.threshold, 0.5)
        self.assertEqual(result.threshold, 0.5)
        self.assertFalse(prediction.abstained)
        self.assertEqual(prediction.response_type, "agreement")
        self.assertEqual(result.n_abstained, 0)
        self.assertGreater(prediction.confidence, 0.5)

    def test_raising_the_floor_produces_an_abstention(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="same bro exactly")], threshold=0.9
        )
        prediction = result.predictions[0]
        self.assertTrue(prediction.abstained)
        self.assertIsNone(prediction.response_type)
        self.assertEqual(result.n_abstained, 1)
        # The distribution is still there, so a consumer can see what it leaned to.
        self.assertEqual(len(prediction.response_probabilities), 6)
        self.assertEqual(prediction.top_response_type(), "agreement")
        self.assertGreater(prediction.confidence, 0.0)

    def test_abstained_prediction_is_still_flagged_not_dropped(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="apply for the scholarship")],
            threshold=0.9,
        )
        self.assertEqual(len(result.predictions), 1)
        self.assertTrue(result.predictions[0].abstained)
        self.assertFalse(result.predictions[0].has_response_type)
        # An abstention is not an assignment, so it is left out of the frequency.
        self.assertEqual(result.response_type_frequency(), {})

    def test_lowered_floor_recovers_the_arg_max(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="same bro exactly")], threshold=0.01
        )
        self.assertFalse(result.predictions[0].abstained)
        self.assertEqual(result.predictions[0].response_type, "agreement")
        self.assertEqual(result.n_abstained, 0)

    def test_threshold_override_is_per_call(self):
        docs = [ResponseDocument(document_id="d", text="same bro exactly")]
        strict = self.classifier.predict(docs, threshold=0.9)
        loose = self.classifier.predict(docs, threshold=0.01)
        self.assertTrue(strict.predictions[0].abstained)
        self.assertFalse(loose.predictions[0].abstained)
        # The stored default is untouched.
        self.assertEqual(self.classifier.threshold, 0.5)
        self.assertEqual(self.classifier.config.threshold, 0.5)

    def test_invalid_threshold_rejected(self):
        docs = [ResponseDocument(document_id="d", text="same bro")]
        for bad in (0.0, 1.0, -0.5, 2, "high", True):
            with self.assertRaises(ValueError):
                self.classifier.predict(docs, threshold=bad)

    def test_explicit_none_threshold_means_use_the_default(self):
        """``None`` is the documented "no override" sentinel, not an error."""
        docs = [ResponseDocument(document_id="d", text="same bro exactly")]
        result = self.classifier.predict(docs, threshold=None)
        self.assertEqual(result.threshold, self.classifier.threshold)


class TestResponsePredictionEdgeCases(ResponseStageTestCase):
    def test_empty_input(self):
        result = self.classifier.predict([], threshold=0.01)
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.average_confidence(), 0.0)
        self.assertEqual(result.response_type_frequency(), {})
        self.assertEqual(result.known_response_types, list(RESPONSE_TYPES))

    def test_none_input(self):
        self.assertEqual(self.classifier.predict(None, threshold=0.01).n_documents, 0)

    def test_empty_text_still_gets_a_distribution(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="blank", text="")], threshold=0.01
        )
        prediction = result.predictions[0]
        self.assertEqual(len(prediction.response_probabilities), 6)
        self.assertTrue(
            math.isclose(
                sum(prediction.response_probabilities.values()), 1.0, abs_tol=1e-4
            )
        )
        self.assertEqual(prediction.text, "")

    def test_placeholder_only_text_does_not_crash(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="ph", text="<URL> <MENTION>")], threshold=0.01
        )
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(len(result.predictions[0].response_probabilities), 6)

    def test_out_of_vocabulary_text_does_not_crash(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="oov", text="zzzqqq xyzzy frobnicate")],
            threshold=0.01,
        )
        self.assertEqual(len(result.predictions[0].response_probabilities), 6)

    def test_duplicate_document_ids_in_input_are_preserved(self):
        docs = [
            ResponseDocument(document_id="same", text="same bro exactly"),
            ResponseDocument(document_id="same", text="apply for the scholarship"),
        ]
        result = self.classifier.predict(docs, threshold=0.01)
        self.assertEqual(len(result.predictions), 2)


class TestDeterminism(ResponseStageTestCase):
    def test_repeated_prediction_is_identical(self):
        docs = [ResponseDocument(document_id="d", text="same bro exactly")]
        first = self.classifier.predict(docs, threshold=0.01)
        second = self.classifier.predict(docs, threshold=0.01)
        self.assertEqual(first.model_dump_json(), second.model_dump_json())

    def test_two_classifiers_trained_the_same_way_agree(self):
        docs = [ResponseDocument(document_id="d", text="apply for the scholarship")]
        a = TfidfLogisticResponseClassifier()
        a.train(self.annotations)
        b = TfidfLogisticResponseClassifier()
        b.train(self.annotations)
        self.assertEqual(
            a.predict(docs, threshold=0.01).model_dump_json(),
            b.predict(docs, threshold=0.01).model_dump_json(),
        )

    def test_probability_key_order_is_canonical(self):
        """Two differently-ordered corpora must still give the same key order."""
        forward = TfidfLogisticResponseClassifier()
        forward.train(_annotations(TRAINING_FIXTURE))
        reversed_corpus = TfidfLogisticResponseClassifier()
        reversed_corpus.train(_annotations(list(reversed(TRAINING_FIXTURE))))
        self.assertEqual(forward.labels, reversed_corpus.labels)
        self.assertEqual(forward.labels, list(RESPONSE_TYPES))
        self.assertEqual(reversed_corpus.labels, list(RESPONSE_TYPES))

    def test_no_timestamp_by_default_keeps_output_stable(self):
        self.assertIsNone(self.result_timestamp())
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="same bro")], threshold=0.01
        )
        self.assertIsNone(result.trained_at)

    def result_timestamp(self):
        return self.classifier.training_report.trained_at


# ===========================================================================
# Context independence (V1 is text-only)
# ===========================================================================


class TestContextIndependence(ResponseStageTestCase):
    def test_parent_comment_text_does_not_change_the_prediction(self):
        base = ResponseDocument(
            document_id="d", text="same bro exactly", original_text="same bro exactly"
        )
        with_parent = ResponseDocument(
            document_id="d",
            text="same bro exactly",
            original_text="same bro exactly",
            parent_comment_text="the fees are far too high for this course",
        )
        a = self.classifier.predict([base], threshold=0.01).predictions[0]
        b = self.classifier.predict([with_parent], threshold=0.01).predictions[0]
        self.assertEqual(a.response_type, b.response_type)
        self.assertEqual(a.response_probabilities, b.response_probabilities)

    def test_issue_and_topic_metadata_do_not_change_the_prediction(self):
        base = ResponseDocument(document_id="d", text="apply for the scholarship")
        enriched = ResponseDocument(
            document_id="d",
            text="apply for the scholarship",
            topic_id=7,
            topic_label="fees",
            topic_keywords=["fees", "roi"],
            issue_labels=["fees"],
            issue_probabilities={"fees": 0.9},
        )
        a = self.classifier.predict([base], threshold=0.01).predictions[0]
        b = self.classifier.predict([enriched], threshold=0.01).predictions[0]
        self.assertEqual(a.response_type, b.response_type)
        self.assertEqual(a.response_probabilities, b.response_probabilities)

    def test_context_is_still_echoed_back(self):
        document = ResponseDocument(
            document_id="d",
            text="apply for the scholarship",
            topic_id=7,
            issue_labels=["fees"],
            parent_comment_text="the fees are far too high",
        )
        result = self.classifier.predict([document], threshold=0.01)
        self.assertEqual(result.n_documents_with_context, 1)
        prediction = result.predictions[0]
        self.assertEqual(prediction.topic_id, 7)
        self.assertEqual(prediction.issue_labels, ["fees"])
        self.assertEqual(prediction.parent_comment_text, "the fees are far too high")

    def test_result_config_declares_context_unused(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="d", text="same bro")], threshold=0.01
        )
        self.assertEqual(
            result.config["context"],
            {
                "uses_parent_comment": False,
                "uses_topic_metadata": False,
                "uses_issue_metadata": False,
            },
        )

    def test_classifier_has_no_context_feature_flags(self):
        """V1 must not ship a dead switch implying context support."""
        source = TfidfLogisticResponseClassifier.__doc__ or ""
        self.assertIn("not** read", source)
        self.assertFalse(
            hasattr(self.classifier.config, "use_parent_features")
            or hasattr(self.classifier.config, "use_issue_features")
            or hasattr(self.classifier.config, "use_topic_features")
        )


# ===========================================================================
# Metadata preservation
# ===========================================================================


class TestMetadataPreservation(ResponseStageTestCase):
    def test_all_metadata_is_echoed_unchanged(self):
        document = ResponseDocument(
            document_id="c-999",
            text="apply for the scholarship on the portal",
            original_text="Apply for the scholarship on the portal!!!",
            post_id="post-77",
            comment_id="c-999",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=3,
            topic_label="fees",
            topic_keywords=["fees", "scholarship"],
            issue_labels=["fees"],
            issue_probabilities={"fees": 0.8},
            parent_comment_text="scholarships are not announced",
        )
        prediction = self.classifier.predict([document], threshold=0.01).predictions[0]
        self.assertEqual(prediction.document_id, document.document_id)
        self.assertEqual(prediction.text, document.text)
        self.assertEqual(prediction.original_text, document.original_text)
        self.assertEqual(prediction.post_id, document.post_id)
        self.assertEqual(prediction.comment_id, document.comment_id)
        self.assertEqual(prediction.language, document.language)
        self.assertEqual(prediction.script, document.script)
        self.assertEqual(prediction.is_code_mixed, document.is_code_mixed)
        self.assertEqual(prediction.detected_languages, document.detected_languages)
        self.assertEqual(prediction.topic_id, document.topic_id)
        self.assertEqual(prediction.topic_label, document.topic_label)
        self.assertEqual(prediction.topic_keywords, document.topic_keywords)
        self.assertEqual(prediction.issue_labels, document.issue_labels)
        self.assertEqual(prediction.issue_probabilities, document.issue_probabilities)
        self.assertEqual(prediction.parent_comment_text, document.parent_comment_text)

    def test_original_text_is_never_normalised(self):
        """Punctuation, casing and placeholders survive untouched."""
        original = "Same BRO, apply at <URL> now... <EMOJI_tada>"
        document = ResponseDocument(
            document_id="c-1",
            text="same bro apply at <url> now <emoji_tada>",
            original_text=original,
        )
        before = document.model_dump()
        prediction = self.classifier.predict([document], threshold=0.01).predictions[0]
        self.assertEqual(prediction.original_text, original)
        self.assertEqual(prediction.text, document.text)
        # And the input object itself was not mutated.
        self.assertEqual(document.model_dump(), before)

    def test_input_documents_are_not_mutated(self):
        documents = [
            ResponseDocument(
                document_id="d",
                text="same bro exactly",
                issue_labels=["fees"],
                detected_languages=["en"],
            )
        ]
        before = [document.model_dump() for document in documents]
        self.classifier.predict(documents, threshold=0.01)
        self.assertEqual([document.model_dump() for document in documents], before)

    def test_prediction_lists_do_not_alias_the_document(self):
        document = ResponseDocument(
            document_id="d", text="same bro", detected_languages=["en"], issue_labels=["fees"]
        )
        prediction = self.classifier.predict([document], threshold=0.01).predictions[0]
        prediction.detected_languages.append("hi")
        prediction.issue_labels.append("hostel")
        self.assertEqual(document.detected_languages, ["en"])
        self.assertEqual(document.issue_labels, ["fees"])

    def test_placeholder_stripping_does_not_affect_echoed_text(self):
        """The model copy drops placeholders; the document keeps them."""
        document = ResponseDocument(
            document_id="d",
            text="same bro exactly <URL> <MENTION>",
            original_text="same bro exactly <URL> <MENTION>",
        )
        prediction = self.classifier.predict([document], threshold=0.01).predictions[0]
        self.assertIn("<URL>", prediction.text)
        self.assertIn("<MENTION>", prediction.text)

    def test_placeholders_can_be_kept_when_configured(self):
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(drop_placeholder_tokens=False)
        )
        classifier.train(self.annotations)
        document = ResponseDocument(document_id="d", text="same bro exactly <URL>")
        prediction = classifier.predict([document], threshold=0.01).predictions[0]
        self.assertIn("<URL>", prediction.text)
        self.assertEqual(len(prediction.response_probabilities), 6)


# ===========================================================================
# Multilingual, code-mixing, emoji placeholders
# ===========================================================================


class TestMultilingualAndCodeMixing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.annotations = _annotations(TRAINING_FIXTURE) + _annotations(
            MULTILINGUAL_FIXTURE
        )
        cls.classifier = TfidfLogisticResponseClassifier()
        cls.report = cls.classifier.train(cls.annotations)

    def test_trains_on_mixed_scripts(self):
        self.assertEqual(self.report.n_classes, 6)
        self.assertEqual(self.report.missing_response_types, [])
        self.assertGreater(self.report.n_features, 0)

    def test_probabilities_valid_for_every_language(self):
        texts = [
            ("romanized-hi", "pichle saal mujhe bhi yahi problem aayi thi"),
            ("hinglish", "aapko placement office se baat karni chahiye"),
            ("devanagari", "गत वर्ष मुझे भी यही समस्या हुई थी"),
            ("bengali", "আমার একই অভিজ্ঞতা হয়েছিল"),
            ("oriya", "ଗତ ବର୍ଷ ମୁଁକୁ ମିଛି"),
        ]
        documents = [
            ResponseDocument(document_id=label, text=text) for label, text in texts
        ]
        result = self.classifier.predict(documents, threshold=0.01)
        self.assertEqual(result.n_documents, 5)
        for prediction in result.predictions:
            values = list(prediction.response_probabilities.values())
            self.assertEqual(len(values), 6)
            self.assertTrue(
                math.isclose(sum(values), 1.0, abs_tol=1e-4),
                f"{prediction.document_id} sums to {sum(values)}",
            )
            self.assertIn(prediction.response_type, RESPONSE_TYPES)

    def test_romanized_hindi_personal_experience(self):
        result = self.classifier.predict(
            [
                ResponseDocument(
                    document_id="rh",
                    text="pichle saal mujhe bhi yahi problem aayi thi",
                )
            ],
            threshold=0.01,
        )
        self.assertEqual(
            result.predictions[0].response_type, "personal_experience"
        )

    def test_bengali_agreement_and_experience(self):
        result = self.classifier.predict(
            [
                ResponseDocument(document_id="bn1", text="হ্যাঁ ঠিক একই কথা"),
                ResponseDocument(document_id="bn2", text="আমার একই অভিজ্ঞতা হয়েছিল"),
            ],
            threshold=0.01,
        )
        predicted = {p.document_id: p.response_type for p in result.predictions}
        self.assertEqual(predicted["bn1"], "agreement")
        self.assertEqual(predicted["bn2"], "personal_experience")

    def test_oriya_opinion(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="or", text="ଆମେ ସେବାଟା ଭଲ ଲାଗୁଛି")],
            threshold=0.01,
        )
        self.assertEqual(result.predictions[0].response_type, "opinion")

    def test_code_mixed_text_is_flagged_and_preserved(self):
        document = ResponseDocument(
            document_id="cm",
            text="aapko yahan apply karna chahiye",
            original_text="Aapko yahan apply karna chahiye!",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        prediction = self.classifier.predict([document], threshold=0.01).predictions[0]
        self.assertTrue(prediction.is_code_mixed)
        self.assertEqual(prediction.detected_languages, ["hi", "en"])
        self.assertEqual(prediction.text, document.text)

    def test_token_pattern_keeps_non_latin_scripts(self):
        """The default token pattern must not silently drop Devanagari/Odia."""
        self.assertIn("(?u)", self.classifier.config.token_pattern)
        self.assertIn("\\w", self.classifier.config.token_pattern)

    def test_emoji_placeholders_are_handled(self):
        for text in (
            "same bro exactly <EMOJI_tada>",
            "i faced this too <EMOJI_loudly_crying_face>",
            "you should talk to the office <EMOJI_point_right> <MENTION>",
        ):
            result = self.classifier.predict(
                [ResponseDocument(document_id="e", text=text)], threshold=0.01
            )
            prediction = result.predictions[0]
            self.assertEqual(len(prediction.response_probabilities), 6)
            self.assertEqual(prediction.text, text)

    def test_placeholder_text_does_not_distort_other_comments(self):
        """Stripped placeholders must not leak into neighbouring documents."""
        docs = [
            ResponseDocument(document_id="with", text="same bro exactly <URL>"),
            ResponseDocument(document_id="without", text="same bro exactly"),
        ]
        result = self.classifier.predict(docs, threshold=0.01)
        a, b = result.predictions
        self.assertTrue(
            math.isclose(
                a.response_probabilities["agreement"],
                b.response_probabilities["agreement"],
                abs_tol=0.15,
            ),
            "a comment's URL token should not shift the role distribution much",
        )

    def test_mixed_script_corpus_serialises_cleanly(self):
        result = self.classifier.predict(
            [ResponseDocument(document_id="bn", text="আমার মতে এটি ভালো কলেজ")],
            threshold=0.01,
        )
        payload = result.model_dump_json()
        self.assertEqual(json.loads(payload)["n_documents"], 1)


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization(ResponseStageTestCase):
    def setUp(self):
        self.documents = [
            ResponseDocument(
                document_id="s1",
                text="apply for the scholarship on the portal",
                original_text="Apply for the scholarship!!",
                post_id="p1",
                comment_id="c1",
                language="hi",
                script="latin",
                is_code_mixed=True,
                detected_languages=["hi", "en"],
                topic_id=2,
                issue_labels=["fees"],
                parent_comment_text="fees are high",
            )
        ]
        self.result = self.classifier.predict(self.documents, threshold=0.01)

    def test_result_round_trips_through_json(self):
        restored = ResponseClassificationResult.model_validate_json(
            self.result.model_dump_json()
        )
        self.assertEqual(restored, self.result)
        self.assertEqual(
            restored.predictions[0].response_probabilities,
            self.result.predictions[0].response_probabilities,
        )
        self.assertEqual(restored.predictions[0].parent_comment_text, "fees are high")

    def test_result_is_json_encodable(self):
        payload = json.loads(self.result.model_dump_json())
        self.assertEqual(payload["model_name"], "tfidf_logistic_multiclass")
        self.assertEqual(payload["known_response_types"], list(RESPONSE_TYPES))
        self.assertEqual(
            sorted(payload["predictions"][0]["response_probabilities"]),
            sorted(RESPONSE_TYPES),
        )

    def test_training_report_round_trips_through_json(self):
        restored = ResponseTrainingReport.model_validate_json(
            self.report.model_dump_json()
        )
        self.assertEqual(restored, self.report)
        self.assertEqual(restored.labels, list(RESPONSE_TYPES))
        self.assertTrue(restored.covers_full_taxonomy)

    def test_prediction_model_validates_standalone(self):
        prediction = self.result.predictions[0]
        restored = ResponsePrediction.model_validate_json(
            prediction.model_dump_json()
        )
        self.assertEqual(restored, prediction)
        self.assertEqual(restored.response_type, prediction.response_type)

    def test_document_round_trips(self):
        document = self.documents[0]
        restored = ResponseDocument.model_validate_json(document.model_dump_json())
        self.assertEqual(restored, document)

    def test_annotation_round_trips(self):
        annotation = self.annotations[0]
        restored = ResponseAnnotation.model_validate_json(annotation.model_dump_json())
        self.assertEqual(restored, annotation)

    def test_evaluation_report_round_trips(self):
        report = self.classifier.evaluate(_annotations(HELDOUT_FIXTURE), threshold=0.01)
        restored = ResponseEvaluationReport.model_validate_json(report.model_dump_json())
        self.assertEqual(restored, report)


# ===========================================================================
# Class weighting
# ===========================================================================


class TestClassWeighting(unittest.TestCase):
    def setUp(self):
        self.annotations = _annotations(TRAINING_FIXTURE)

    def test_default_is_balanced(self):
        classifier = TfidfLogisticResponseClassifier()
        self.assertEqual(classifier.config.class_weight, "balanced")
        self.assertEqual(
            classifier.config_dict()["logistic_regression"]["class_weight"], "balanced"
        )

    def test_balanced_can_be_disabled(self):
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(class_weight=None)
        )
        report = classifier.train(self.annotations)
        self.assertIsNone(classifier.config.class_weight)
        self.assertIsNone(report.config["logistic_regression"]["class_weight"])
        self.assertTrue(report.is_usable)

    def test_invalid_class_weight_rejected(self):
        for bad in ("balanced_weights", "uniform", 1, True):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticResponseClassifier(ResponseModelConfig(class_weight=bad))
            self.assertIn("class_weight", str(ctx.exception))

    def test_class_weight_changes_the_fitted_model(self):
        """
        On an imbalanced corpus, balancing shifts mass off the majority role.

        The fixture is deliberately skewed, because on a perfectly balanced
        corpus ``"balanced"`` computes equal weights and the two models are
        identical - so testing it there would prove nothing.
        """
        rows = _imbalanced_corpus()
        annotations = _annotations(rows)

        document = ResponseDocument(
            document_id="r", text="actually the fees are not that high"
        )
        balanced = TfidfLogisticResponseClassifier()
        balanced.train(annotations)
        unweighted = TfidfLogisticResponseClassifier(
            ResponseModelConfig(class_weight=None)
        )
        unweighted.train(annotations)

        with_balancing = balanced.predict([document], threshold=0.01).predictions[0]
        without = unweighted.predict([document], threshold=0.01).predictions[0]

        self.assertEqual(
            with_balancing.response_type, without.response_type, "counterargument"
        )
        self.assertNotEqual(
            with_balancing.response_probabilities, without.response_probabilities
        )
        # The documented effect: the rare role gains mass, the dominant one loses it.
        self.assertGreater(
            with_balancing.response_probabilities["counterargument"],
            without.response_probabilities["counterargument"],
        )
        self.assertLess(
            with_balancing.response_probabilities["agreement"],
            without.response_probabilities["agreement"],
        )

    def test_class_weight_is_recorded_in_the_training_report(self):
        classifier = TfidfLogisticResponseClassifier(
            ResponseModelConfig(class_weight=None)
        )
        report = classifier.train(self.annotations)
        self.assertIn("class_weight", report.config["logistic_regression"])


# ===========================================================================
# Configuration validation
# ===========================================================================


class TestConfigValidation(unittest.TestCase):
    def test_defaults_match_the_documented_baseline(self):
        config = ResponseModelConfig()
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.solver, "lbfgs")
        self.assertEqual(config.C, 4.0)
        self.assertEqual(config.ngram_range, (1, 2))
        self.assertIsNone(config.stop_words)
        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(config.class_weight, "balanced")
        self.assertTrue(config.lowercase)
        self.assertTrue(config.sublinear_tf)
        self.assertEqual(config.norm, "l2")
        self.assertTrue(config.drop_placeholder_tokens)
        self.assertFalse(config.require_all_response_types)

    def test_max_features_must_be_positive(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticResponseClassifier(ResponseModelConfig(max_features=bad))
            self.assertIn("max_features", str(ctx.exception))

    def test_df_bounds_are_checked(self):
        for kwargs in (
            {"min_df": -1},
            {"max_df": -1},
            {"min_df": 0.0},
            {"max_df": 1.5},
            {"min_df": "two"},
        ):
            with self.assertRaises(ValueError):
                TfidfLogisticResponseClassifier(ResponseModelConfig(**kwargs))

    def test_ngram_range_is_checked(self):
        for bad in ((0, 2), (2, 1), (-1, 3)):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticResponseClassifier(ResponseModelConfig(ngram_range=bad))
            self.assertIn("ngram_range", str(ctx.exception))

    def test_norm_is_checked(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(norm="l3"))
        self.assertIn("norm", str(ctx.exception))

    def test_token_pattern_is_compiled(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(token_pattern="([unclosed"))
        self.assertIn("token_pattern", str(ctx.exception))

    def test_stop_words_restrictions(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words="hindi"))
        self.assertIn("stop_words", str(ctx.exception))
        with self.assertRaises(ValueError):
            TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words=42))
        with self.assertRaises(ValueError):
            TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words=[1, 2]))
        # None and the built-in English list are both allowed.
        TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words=None))
        TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words="english"))
        TfidfLogisticResponseClassifier(ResponseModelConfig(stop_words=["the", "a"]))

    def test_regularisation_and_iterations(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(C=0))
        self.assertIn("C", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(ResponseModelConfig(max_iter=0))
        self.assertIn("max_iter", str(ctx.exception))

    def test_data_guard_bounds(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(
                ResponseModelConfig(min_training_documents=1)
            )
        self.assertIn("min_training_documents", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticResponseClassifier(
                ResponseModelConfig(min_examples_per_class=0)
            )
        self.assertIn("min_examples_per_class", str(ctx.exception))

    def test_imbalance_share_bounds(self):
        # Note: a bool is not tested here.  ``imbalance_warning_share`` is a
        # ``float`` field, and pydantic coerces ``True`` to ``1.0`` before this
        # validator runs, so it is a valid value by the time it is seen.
        for bad in (0.0, 1.5, -0.2, "half"):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticResponseClassifier(
                    ResponseModelConfig(imbalance_warning_share=bad)
                )
            self.assertIn("imbalance_warning_share", str(ctx.exception))

    def test_threshold_bounds(self):
        # A numeric string is coerced by pydantic to the float 0.5, so it is a
        # legitimate value by the time this validator sees it; only genuinely
        # unparseable input raises here.
        for bad in (0.0, 1.0, 1.5, -0.1, "high", True):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticResponseClassifier(ResponseModelConfig(threshold=bad))
            self.assertIn("threshold", str(ctx.exception))

    def test_config_dict_is_json_serialisable(self):
        classifier = TfidfLogisticResponseClassifier()
        payload = json.loads(json.dumps(classifier.config_dict()))
        self.assertEqual(payload["logistic_regression"]["random_state"], 42)
        self.assertIn("data_guards", payload)
        self.assertIn("features", payload)


# ===========================================================================
# Evaluation
# ===========================================================================


class TestResponseEvaluation(ResponseStageTestCase):
    def setUp(self):
        self.heldout = _annotations(HELDOUT_FIXTURE)
        self.report_eval = self.classifier.evaluate(self.heldout, threshold=0.01)

    def test_report_type(self):
        self.assertIsInstance(self.report_eval, ResponseEvaluationReport)

    def test_held_out_set_is_genuinely_disjoint(self):
        """Guards the evaluation itself, not just the metrics."""
        training_ids = {row[0] for row in TRAINING_FIXTURE}
        training_texts = {row[1] for row in TRAINING_FIXTURE}
        for document_id, text, _role, *_rest in HELDOUT_FIXTURE:
            self.assertNotIn(document_id, training_ids)
            self.assertNotIn(text, training_texts)
        self.assertGreaterEqual(len(HELDOUT_FIXTURE), MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS)

    def test_metrics_are_meaningful_on_a_sized_set(self):
        self.assertTrue(
            self.report_eval.is_meaningful, self.report_eval.reason
        )
        self.assertIsNone(self.report_eval.reason)
        self.assertEqual(self.report_eval.n_documents, len(HELDOUT_FIXTURE))
        self.assertEqual(self.report_eval.n_classes, 6)
        self.assertEqual(self.report_eval.class_order, list(RESPONSE_TYPES))

    def test_all_required_metrics_are_reported(self):
        for name in (
            "accuracy",
            "macro_f1",
            "micro_f1",
            "weighted_f1",
            "macro_precision",
            "micro_precision",
            "weighted_precision",
            "macro_recall",
            "micro_recall",
            "weighted_recall",
        ):
            value = getattr(self.report_eval, name)
            self.assertIsNotNone(value, f"{name} must be reported")
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_per_class_metrics_cover_all_six(self):
        for name in RESPONSE_TYPES:
            self.assertIn(name, self.report_eval.per_class_f1)
            self.assertIn(name, self.report_eval.per_class_precision)
            self.assertIn(name, self.report_eval.per_class_recall)
            self.assertEqual(self.report_eval.per_class_support[name], 3, name)
        # Predicted counts need not match support - that gap *is* the confusion
        # matrix - but every prediction must be accounted for exactly once.
        self.assertEqual(
            sum(self.report_eval.per_class_predicted.values()),
            self.report_eval.n_documents,
        )
        self.assertEqual(
            set(self.report_eval.per_class_predicted), set(RESPONSE_TYPES)
        )

    def test_micro_f1_equals_accuracy_for_single_label(self):
        """
        For single-label multiclass data, micro-F1 and accuracy are the same
        quantity.  A divergence would mean the aggregates and the per-class
        metrics were computed over different data.
        """
        self.assertAlmostEqual(
            self.report_eval.micro_f1, self.report_eval.accuracy, places=6
        )
        self.assertAlmostEqual(
            self.report_eval.micro_precision, self.report_eval.accuracy, places=6
        )
        self.assertAlmostEqual(
            self.report_eval.micro_recall, self.report_eval.accuracy, places=6
        )

    def test_confusion_matrix_shape_and_totals(self):
        matrix = self.report_eval.confusion_matrix
        self.assertIsNotNone(matrix)
        self.assertEqual(len(matrix), 6)
        for row in matrix:
            self.assertEqual(len(row), 6)
        # No abstentions at this floor, so every comment lands in the matrix.
        total = sum(sum(row) for row in matrix)
        self.assertEqual(total, self.report_eval.n_documents)

    def test_confusion_matrix_diagonal_is_the_correct_count(self):
        matrix = self.report_eval.confusion_matrix
        correct = sum(matrix[i][i] for i in range(6))
        self.assertEqual(
            round(correct / self.report_eval.n_documents, 6), self.report_eval.accuracy
        )

    def test_confusion_matrix_row_sums_match_support(self):
        for index, name in enumerate(self.report_eval.class_order):
            row_total = sum(self.report_eval.confusion_matrix[index])
            self.assertEqual(
                row_total, self.report_eval.per_class_support[name], name
            )

    def test_classification_report_present(self):
        report = self.report_eval.classification_report
        self.assertIsNotNone(report)
        for name in RESPONSE_TYPES:
            self.assertIn(name, report)
            for metric in ("precision", "recall", "f1-score", "support"):
                self.assertIn(metric, report[name])

    def test_evaluation_is_above_chance_on_this_fixture(self):
        """
        Not a claim about Reddit - just a check that the metric plumbing is not
        returning a trivially-zero result on a separable toy corpus.
        """
        self.assertGreaterEqual(self.report_eval.accuracy, 0.6)
        self.assertGreaterEqual(self.report_eval.macro_f1, 0.5)

    def test_evaluation_at_default_floor_counts_abstentions(self):
        report = self.classifier.evaluate(self.heldout)
        self.assertGreater(report.n_abstained, 0)
        self.assertTrue(
            any("abstained" in warning for warning in report.warnings), report.warnings
        )
        # Abstentions count as misses, so accuracy drops rather than being
        # computed on the surviving rows only.
        self.assertLessEqual(report.accuracy, 1.0)
        matrix_total = sum(sum(row) for row in report.confusion_matrix)
        self.assertEqual(
            matrix_total, report.n_documents - report.n_abstained
        )

    def test_evaluation_on_training_data_warns_about_memorisation(self):
        report = self.classifier.evaluate(self.annotations, threshold=0.01)
        self.assertTrue(
            any("memorisation" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_evaluation_warns_on_partial_overlap(self):
        mixed = list(HELDOUT_FIXTURE) + TRAINING_FIXTURE[:2]
        report = self.classifier.evaluate(_annotations(mixed), threshold=0.01)
        self.assertTrue(
            any("inflates these metrics" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_evaluate_empty_gold_set(self):
        report = self.classifier.evaluate([], threshold=0.01)
        self.assertFalse(report.is_meaningful)
        self.assertIsNone(report.accuracy)
        self.assertIsNone(report.macro_f1)

    def test_evaluate_none(self):
        self.assertFalse(self.classifier.evaluate(None).is_meaningful)

    def test_confusion_matrix_as_rows_renders(self):
        rows = self.report_eval.confusion_matrix_as_rows()
        self.assertEqual(len(rows), 6)
        for name in RESPONSE_TYPES:
            self.assertTrue(any(row.startswith(f"    {name}") for row in rows))

    def test_format_summary_states_the_numbers(self):
        summary = self.report_eval.format_summary()
        self.assertIn("Response classification evaluation", summary)
        self.assertIn("accuracy", summary)
        self.assertIn("macro-F1", summary)
        self.assertIn("weighted-F1", summary)
        self.assertIn("confusion matrix", summary)
        self.assertNotIn("None", summary)

    def test_format_summary_says_when_not_meaningful(self):
        report = evaluate_response_classification(["advice"], ["advice"], list(RESPONSE_TYPES))
        summary = report.format_summary()
        self.assertIn("NOT MEANINGFUL", summary)
        self.assertIn("withheld", summary)


class TestResponseEvaluationGuards(unittest.TestCase):
    def test_too_few_documents_withholds_metrics(self):
        report = evaluate_response_classification(
            ["advice", "agreement", "advice"],
            ["advice", "agreement", "solution"],
            list(RESPONSE_TYPES),
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("only 3", report.reason)
        for name in ("accuracy", "macro_f1", "micro_f1", "weighted_f1"):
            self.assertIsNone(getattr(report, name), name)
        self.assertIsNone(report.confusion_matrix)
        self.assertIsNone(report.classification_report)
        # Support is still reported: it is a fact about the data.  Every class
        # seen in either gold or predictions gets a column, so a class that was
        # only ever predicted shows support 0 rather than vanishing.
        self.assertEqual(
            report.per_class_support, {"agreement": 1, "advice": 2, "solution": 0}
        )
        self.assertEqual(
            report.per_class_predicted, {"agreement": 1, "advice": 1, "solution": 1}
        )

    def test_single_class_withholds_metrics(self):
        truth = ["advice"] * 12
        report = evaluate_response_classification(truth, truth, list(RESPONSE_TYPES))
        self.assertFalse(report.is_meaningful)
        self.assertIn("response type", report.reason)
        self.assertIsNone(report.macro_f1)

    def test_unscoreable_gold_is_counted_not_dropped(self):
        gold = ["advice", "agreement", "opinion", "solution"] * 3 + [None, None]
        predicted = ["advice", "agreement", "opinion", "solution"] * 3 + ["advice", "agreement"]
        report = evaluate_response_classification(
            gold, predicted, ["agreement", "opinion", "solution"]
        )
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(report.n_unscoreable, 2)
        self.assertTrue(
            any("gold role the model cannot emit" in w for w in report.warnings),
            report.warnings,
        )
        # The two unscoreable rows are misses, so accuracy cannot be 1.0.
        self.assertLess(report.accuracy, 1.0)
        self.assertEqual(report.n_documents, 14)

    def test_all_unscoreable_gold_withholds_metrics(self):
        # Two distinct predicted classes, so the "too few classes" guard does not
        # fire first and the zero-support guard is the one under test.
        gold = [None] * 12
        predicted = ["advice", "solution"] * 6
        report = evaluate_response_classification(
            gold, predicted, list(RESPONSE_TYPES)
        )
        self.assertEqual(len(report.class_order), 2)
        self.assertFalse(report.is_meaningful)
        self.assertIn("no evaluation comment has a gold response type", report.reason)
        self.assertEqual(report.n_unscoreable, 12)
        self.assertIsNone(report.accuracy)
        self.assertIsNone(report.macro_f1)
        self.assertIsNone(report.confusion_matrix)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            evaluate_response_classification(["advice"], ["advice", "opinion"], list(RESPONSE_TYPES))
        self.assertIn("entries", str(ctx.exception))

    def test_thin_class_support_is_warned_about(self):
        gold = ["agreement"] * 10 + ["advice"]
        predicted = ["agreement"] * 10 + ["agreement"]
        report = evaluate_response_classification(gold, predicted, list(RESPONSE_TYPES))
        self.assertTrue(
            any("unstable" in warning for warning in report.warnings), report.warnings
        )

    def test_never_predicted_class_is_warned_about(self):
        gold = ["agreement"] * 6 + ["advice"] * 6
        predicted = ["agreement"] * 12
        report = evaluate_response_classification(gold, predicted, list(RESPONSE_TYPES))
        self.assertTrue(
            any("never predicted" in warning for warning in report.warnings),
            report.warnings,
        )
        self.assertLess(report.macro_f1, report.accuracy)

    def test_empty_inputs(self):
        report = evaluate_response_classification([], [], list(RESPONSE_TYPES))
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_documents, 0)

    def test_evaluate_handles_a_role_the_model_never_saw(self):
        """
        A model trained on a subset must report, not crash, on the roles it lacks.

        12 held-out comments, 3 of them an unscoreable ``advice``: enough to pass
        the size guard so the unscoreable rows actually reach the metric
        computation rather than being hidden behind a withheld report.
        """
        subset = _annotations(
            [row for row in TRAINING_FIXTURE if row[2] in ("agreement", "opinion")]
        )
        classifier = TfidfLogisticResponseClassifier()
        report_train = classifier.train(subset)
        self.assertEqual(report_train.labels, ["agreement", "opinion"])

        heldout = _annotations(
            [
                row
                for row in HELDOUT_FIXTURE
                if row[2] in ("agreement", "opinion", "advice", "solution")
            ]
        )
        report = classifier.evaluate(heldout, threshold=0.01)
        self.assertTrue(
            any("cannot emit" in warning for warning in report.warnings),
            report.warnings,
        )
        self.assertEqual(report.n_documents, len(heldout))
        # Every advice / solution comment has a gold role the model cannot produce.
        self.assertEqual(report.n_unscoreable, 6)
        self.assertTrue(report.is_meaningful, report.reason)
        # They are counted as misses, so accuracy cannot be perfect.
        self.assertLess(report.accuracy, 1.0)

    def test_withheld_report_still_counts_abstentions_and_unscoreable(self):
        """Counts are facts about the data, so they survive a withheld report."""
        report = evaluate_response_classification(
            [None, "advice", "agreement", None],
            ["advice", None, "agreement", "advice"],
            list(RESPONSE_TYPES),
        )
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_unscoreable, 2)
        self.assertEqual(report.n_abstained, 1)
        self.assertEqual(report.n_documents, 4)


# ===========================================================================
# Architectural isolation
# ===========================================================================


def _package_modules():
    for directory, _dirs, files in os.walk(PACKAGE_ROOT):
        if "__pycache__" in directory:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(directory, name)


def _imported_modules(tree):
    """Every module named by an import statement in a parsed file."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _imported_names(tree):
    """Every symbol bound by a ``from ... import ...`` statement."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


class TestArchitectureIsolation(unittest.TestCase):
    def test_package_contains_expected_modules(self):
        names = {os.path.basename(path) for path in _package_modules()}
        for expected in (
            "base.py",
            "annotation.py",
            "result.py",
            "evaluation.py",
            "labels.py",
            "tfidf_logistic_multiclass.py",
            "__init__.py",
        ):
            self.assertIn(expected, names)

    def test_package_imports_nothing_outside_itself(self):
        """
        No first-party coupling to preprocessing, topics, issues, sentiment, or
        the database.  Third-party libraries (pydantic, sklearn, numpy) and the
        standard library are fine; only ``app.*`` imports are constrained.
        """
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            first_party = {
                module
                for module in _imported_modules(tree)
                if module == "app" or module.startswith("app.")
            }
            for module in first_party:
                self.assertTrue(
                    module.startswith("app.services.nlp.response_classification"),
                    f"{os.path.basename(path)} imports {module}",
                )

    def test_preprocessing_is_not_imported(self):
        """
        The classifier must not import or re-run preprocessing.

        Checked against the parsed import statements rather than a raw text
        search: the modules legitimately *name* the preprocessor in their
        docstrings to state that ``text`` must already be ``normalized_text``,
        and a substring search would flag that documentation as a violation.
        """
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            imported = _imported_modules(tree)
            for module in imported:
                self.assertNotIn("preprocessing", module, os.path.basename(path))
            imported_names = _imported_names(tree)
            for name in imported_names:
                self.assertNotIn("Preprocessor", name, os.path.basename(path))
            # The text-level check that matters: no reference to the module path.
            source = open(path, "r", encoding="utf-8").read()
            self.assertNotIn("app.services.nlp.preprocessing", source)

    def test_no_preprocessing_or_ingestion_symbols(self):
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in (
                "sqlalchemy",
                "SessionLocal",
                "get_db",
                "app.database",
                "app.models",
                "app.services.ingest",
                "praw",
            ):
                self.assertNotIn(forbidden, source, os.path.basename(path))

    def test_earlier_stages_are_not_coupled_to(self):
        """This stage must not reach into the completed stages."""
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in (
                "app.services.nlp.topic_discovery",
                "app.services.nlp.issue_identification",
                "app.services.nlp.sentiment_emotion",
                "app.services.nlp.language_analysis_service",
            ):
                self.assertNotIn(forbidden, source, os.path.basename(path))

    def test_no_sentiment_inference_from_response_role(self):
        """Response roles are not derived from, or sent to, sentiment."""
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read().lower()
            for marker in ("sentiment_probabilities", "emotion_probabilities", "sentiment="):
                self.assertNotIn(marker, source, os.path.basename(path))

    def test_no_context_features_in_v1(self):
        """
        The optional context fields must be carried, never read.

        The docstring of the featuriser is allowed to name them (that is where
        the design decision is recorded), so only the executable body is
        inspected.
        """
        path = os.path.join(PACKAGE_ROOT, "tfidf_logistic_multiclass.py")
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        bodies = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_featurize_text", "_strip_placeholders")
        ]
        self.assertTrue(bodies, "the featurisation path must exist")
        for function in bodies:
            names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
            attributes = {
                node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
            }
            for forbidden in ("parent_comment_text", "topic_id", "issue_labels"):
                self.assertNotIn(forbidden, names | attributes, function.name)

    def test_context_is_only_echoed_not_used(self):
        """parent_comment_text appears in the package, but never in a feature."""
        source = open(
            os.path.join(PACKAGE_ROOT, "tfidf_logistic_multiclass.py"),
            "r",
            encoding="utf-8",
        ).read()
        self.assertIn("parent_comment_text", source)  # echoed on the prediction
        self.assertIn("uses_parent_comment", source)  # declared unused in config

    def test_existing_stages_still_importable(self):
        """Adding this stage did not break the completed ones."""
        from app.services.nlp.issue_identification import (  # noqa: F401
            IssueModelConfig,
            TfidfLogisticIssueClassifier,
        )
        from app.services.nlp.sentiment_emotion import (  # noqa: F401
            SentimentModelConfig,
            TfidfLogisticSentimentClassifier,
        )
        from app.services.nlp.topic_discovery import (  # noqa: F401
            NMFTopicDiscoverer,
        )

    def test_classifier_implements_the_abc(self):
        self.assertTrue(issubclass(TfidfLogisticResponseClassifier, ResponseClassification))
        classifier = TfidfLogisticResponseClassifier()
        for member in (
            "train",
            "predict",
            "evaluate",
            "name",
            "version",
            "labels",
            "expected_labels",
            "is_trained",
            "threshold",
            "config",
        ):
            self.assertTrue(hasattr(classifier, member), member)

    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            ResponseClassification()

    def test_versioned_identifiers_are_stable(self):
        classifier = TfidfLogisticResponseClassifier()
        self.assertEqual(classifier.name, "tfidf_logistic_multiclass")
        self.assertEqual(classifier.version, "v1")

    def test_sigmoid_is_not_used_for_this_stage(self):
        """Softmax only: the six roles are mutually exclusive."""
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in ("OneVsRestClassifier", "MultiLabelBinarizer", "binarize"):
                self.assertNotIn(forbidden, source, os.path.basename(path))

    def test_input_document_defaults(self):
        document = ResponseDocument(document_id="d")
        self.assertEqual(document.text, "")
        self.assertEqual(document.language, "unknown")
        self.assertEqual(document.script, "unknown")
        self.assertFalse(document.is_code_mixed)
        self.assertEqual(document.detected_languages, [])
        self.assertEqual(document.issue_labels, [])
        self.assertEqual(document.issue_probabilities, {})
        self.assertIsNone(document.parent_comment_text)
        self.assertFalse(document.has_context)


if __name__ == "__main__":
    unittest.main()
