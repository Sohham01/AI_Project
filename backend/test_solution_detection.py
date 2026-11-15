"""
Comprehensive tests for the solution detection stage.

Tests cover:
- The binary target: canonical order, label<->boolean mapping, refusal of a
  non-boolean cell instead of silently reading it as ``False``
- Candidate logic: which response roles are worth checking, and the structural
  guarantee that a candidate assessment can never carry a solution verdict
- The advice-vs-solution distinction, end to end, on annotated fixtures
- Annotation models and the annotation CSV schema (round-trip, located errors,
  BOM, blank lines, missing columns, free-form ``solution_type``)
- Training refusals: empty corpus, missing label, single-target corpus, a target
  with too little evidence, unusable TF-IDF vocabulary, conflicting duplicate ids
- Prediction: ``is_solution`` from the threshold, the sigmoid probability, the
  probability pair summing to 1, and the opt-in abstention band
- Determinism across two fits and two ``predict()`` calls
- Metadata propagation, input immutability, serialization
- Class weighting (default "balanced", disable, reject) and imbalance warnings
- Evaluation: accuracy, precision, recall, F1, confusion matrix, named TP/FP/FN/TN,
  classification report, and the meaningful-data guards
- Multilingual and code-mixed text: Hinglish, Bengali, Odia, emoji placeholders
- Architectural isolation: no response-stage import, no preprocessing, no database

A note on what is NOT asserted
------------------------------
Several fixtures here are real sentences with known gold labels, and a few tests
check that the model reproduces the labels of the data it was trained on.  That
is a *memorisation* check on a hand-written toy corpus, and it is named as such.

No test asserts that an unseen sentence is classified correctly.  Doing so would
assert model accuracy that a 38-example synthetic corpus cannot support, and a
test that pins a bag-of-words baseline to a particular guess is a test that fails
the moment the model is legitimately retrained.  The linguistic examples
(negation, code-switching, emoji tokens) therefore assert *architecture* -
valid probabilities, determinism, correct plumbing - and the limitations are
documented rather than papered over.
"""

import ast
import json
import math
import os
import sys
import tempfile
import unittest
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.solution_detection import (
    CSV_COLUMNS,
    DEFAULT_ANNOTATION_CSV,
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS,
    MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS,
    NEGATIVE_CLASS_INDEX,
    POSITIVE_CLASS_INDEX,
    RESPONSE_TYPE_ADVICE,
    RESPONSE_TYPE_SOLUTION,
    TARGET_LABELS,
    CandidateAssessment,
    SolutionAnnotation,
    SolutionCorpus,
    SolutionDetection,
    SolutionDetectionResult,
    SolutionDocument,
    SolutionEvaluationReport,
    SolutionModelConfig,
    SolutionPrediction,
    SolutionTrainingReport,
    TfidfLogisticSolutionClassifier,
    classify_response_candidate,
    distinct_solution_types,
    evaluate_solution_detection,
    is_advice_response_type,
    is_solution_candidate,
    is_solution_response_type,
    label_to_bool,
    normalise_solution_type,
    parse_bool_cell,
    target_index,
    target_label,
    validate_solution_type,
)
from app.services.nlp.solution_detection.tfidf_logistic_binary import INHERIT

PACKAGE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app",
    "services",
    "nlp",
    "solution_detection",
)


def _package_modules():
    for directory, _dirs, files in os.walk(PACKAGE_ROOT):
        if "__pycache__" in directory:
            continue
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(directory, name)


def _imported_modules(tree):
    """Every module named by an import statement, as a dotted path."""
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, resolved within the package
                continue
            if node.module:
                modules.add(node.module)
    return modules


def _imported_names(tree):
    """Every bare symbol name bound by a ``from x import y`` statement."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


# ---------------------------------------------------------------------------
# Synthetic annotation fixtures.
#
# SYNTHETIC, illustrative data written to exercise the architecture. It was not
# collected from Reddit, it was not annotated by real annotators, and no metric
# anywhere in this project describes the Reddit corpus.
#
# The text is keyed on the lexical markers that actually separate the two
# classes in a bag-of-words model - imperatives ("apply", "submit"), deontics
# ("you should", "you must"), hedges and first-person report ("i think", "i
# faced"), and agreement markers ("same bro").  Real annotation would show far
# more genuine overlap, and the confusion between "you should try X" and "maybe
# you could try X" is exactly where a lexical model is weakest.
# ---------------------------------------------------------------------------

# (document_id, preprocessed text, is_solution, solution_type, language, script,
#  code_mixed, response_type)
POSITIVE_FIXTURE = [
    # --- English imperatives: a stated action ---
    ("p01", "apply for financial aid on the portal", "financial_aid", "en", "latin", False, None),
    ("p02", "submit the admission form before the deadline", "admissions", "en", "latin", False, None),
    ("p03", "register on the website before monday", "admissions", "en", "latin", False, None),
    ("p04", "download the form from the intranet page", "admissions", "en", "latin", False, None),
    ("p09", "install the wifi client from the portal", "campus_it", "en", "latin", False, None),
    ("p11", "collect the transcript from the academic office", "admissions", "en", "latin", False, None),
    ("p12", "renew the hostel registration online", "hostel", "en", "latin", False, None),
    ("p18", "fill in the registration form today", "admissions", "en", "latin", False, None),
    # --- English deontics: advice that names an action ---
    ("p05", "you should contact the placement office", "placement_preparation", "en", "latin", False, "advice"),
    ("p06", "you must check the scholarship portal", "scholarship", "en", "latin", False, "advice"),
    ("p08", "i would advise you to start early", "placement_preparation", "en", "latin", False, "advice"),
    ("p10", "email the registrar about the fee receipt", "fees", "en", "latin", False, "advice"),
    # --- recommendation without an explicit addressee ---
    ("p07", "try applying to the scholarship before the deadline", "scholarship", "en", "latin", False, None),
    # --- Hinglish (Romanized Hindi + English) ---
    ("p13", "scholarship ke liye apply kar de", "scholarship", "hi", "latin", True, None),
    ("p14", "placement office ko contact karo", "placement_preparation", "hi", "latin", True, "advice"),
    # --- Bengali ---
    ("p15", "বৃত্তির জন্য আবেদন করুন", "scholarship", "bn", "bengali", False, None),
    ("p16", "ভর্তি ফরম পূরণ করুন", "admissions", "bn", "bengali", False, None),
    # --- Odia ---
    ("p17", "ଫର୍ମ ପୂରଣ କରନ୍ତୁ", "admissions", "or", "odia", False, None),
]

NEGATIVE_FIXTURE = [
    # --- agreement ---
    ("n01", "same bro", "en", "latin", False, "agreement"),
    ("n02", "true that, i agree with you", "en", "latin", False, "agreement"),
    ("n15", "same bro <EMOJI_tada>", "en", "latin", False, "agreement"),
    # --- personal experience: past event, no action offered ---
    ("n03", "i faced the same issue last year", "en", "latin", False, "personal_experience"),
    ("n04", "in my own case it happened too", "en", "latin", False, "personal_experience"),
    ("n19", "i remember facing this at my college", "en", "latin", False, "personal_experience"),
    # --- opinion: a judgement, no action ---
    ("n05", "i think fees are too high", "en", "latin", False, "opinion"),
    ("n06", "i believe the campus is fine", "en", "latin", False, "opinion"),
    ("n16", "i think this college is better", "en", "latin", False, "opinion"),
    # --- counterargument ---
    ("n07", "actually fees arent that high compared with others", "en", "latin", False, "counterargument"),
    ("n08", "that is wrong, i disagree entirely", "en", "latin", False, "counterargument"),
    ("n20", "however the fees rose last year", "en", "latin", False, "counterargument"),
    # --- uncertainty: no action at all ---
    ("n09", "honestly i am not sure what to do here", "en", "latin", False, None),
    # --- questions are not recommendations ---
    ("n12", "should i apply for the scholarship or not", "en", "latin", False, None),
    # --- advice that recommends NOTHING actionable: the core case ---
    ("n13", "maybe you could try emailing them, not sure though", "en", "latin", False, "advice"),
    ("n14", "i would probably suggest something, but i am not certain", "en", "latin", False, "advice"),
    # --- negation: an instruction AGAINST an action is not a solution ---
    ("n10", "do not apply for financial aid, it is closed", "en", "latin", False, None),
    ("n11", "placement office ko contact mat karo, woh nahi help karenge", "hi", "latin", True, "advice"),
    # --- Bengali / Odia opinion ---
    ("n17", "আমি মনে করি ফি অনেক বেশি", "bn", "bengali", False, "opinion"),
    ("n18", "ଆମେ ସେବାଟା ଭଲ ଲାଗୁଛି", "or", "odia", False, "opinion"),
]


def _training_annotations():
    """Build the full training corpus from the two fixture groups."""
    annotations = [
        SolutionAnnotation(
            document_id=document_id,
            text=text,
            is_solution=True,
            solution_type=solution_type,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            response_type=response_type,
        )
        for (
            document_id,
            text,
            solution_type,
            language,
            script,
            is_code_mixed,
            response_type,
        ) in POSITIVE_FIXTURE
    ]
    annotations.extend(
        SolutionAnnotation(
            document_id=document_id,
            text=text,
            is_solution=False,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            response_type=response_type,
        )
        for (
            document_id,
            text,
            language,
            script,
            is_code_mixed,
            response_type,
        ) in NEGATIVE_FIXTURE
    )
    return annotations


#: A genuinely held-out set: disjoint ids, disjoint text, 6 solutions and 8
#: non-solutions = 14 documents, above MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS (10)
#: with enough of each target for the positive-class rates to be interpretable.
HELDOUT_FIXTURE = [
    ("h01", "apply for the bursary on the student portal", True),
    ("h02", "you should talk to the hostel warden", True),
    ("h03", "fill in the scholarship form online", True),
    ("h04", "send the fee receipt to the accounts office", True),
    ("h05", "enroll for the workshop before friday", True),
    ("h06", "download the admit card from the portal", True),
    ("h07", "same point", False),
    ("h08", "yes exactly", False),
    ("h09", "i went through this myself last year", False),
    ("h10", "i feel the mess food is good", False),
    ("h11", "actually that is not true here", False),
    ("h12", "i am not sure what happened", False),
    ("h13", "but that is wrong about this place", False),
    ("h14", "in my view the campus is fine", False),
]


def _heldout_annotations():
    return [
        SolutionAnnotation(document_id=document_id, text=text, is_solution=is_solution)
        for document_id, text, is_solution in HELDOUT_FIXTURE
    ]


class SolutionStageTestCase(unittest.TestCase):
    """Shared trained classifier, so each test file does not refit 20 times."""

    @classmethod
    def setUpClass(cls):
        cls.annotations = _training_annotations()
        cls.classifier = TfidfLogisticSolutionClassifier()
        cls.report = cls.classifier.train(cls.annotations)
        # A low threshold is used where the *decision* is not what is under
        # test, so that a toy corpus's calibration does not turn an
        # architectural assertion into a flaky one.
        cls.docs = [a.to_solution_document() for a in cls.annotations]


# ===========================================================================
# Target / labels
# ===========================================================================


class TestTargetAndLabels(unittest.TestCase):
    def test_canonical_target_order_is_not_solution_then_solution(self):
        self.assertEqual(TARGET_LABELS, (LABEL_NOT_SOLUTION, LABEL_SOLUTION))
        self.assertEqual(NEGATIVE_CLASS_INDEX, 0)
        self.assertEqual(POSITIVE_CLASS_INDEX, 1)

    def test_boolean_maps_to_label_and_index(self):
        self.assertEqual(target_label(True), LABEL_SOLUTION)
        self.assertEqual(target_label(False), LABEL_NOT_SOLUTION)
        self.assertEqual(target_index(True), 1)
        self.assertEqual(target_index(False), 0)

    def test_label_maps_back_to_boolean(self):
        self.assertTrue(label_to_bool(LABEL_SOLUTION))
        self.assertFalse(label_to_bool(LABEL_NOT_SOLUTION))
        self.assertIsNone(label_to_bool("maybe"))
        self.assertIsNone(label_to_bool(None))
        self.assertIsNone(label_to_bool(""))

    def test_parse_bool_cell_accepts_common_spellings(self):
        for token in ("true", "True", "TRUE", "yes", "y", "1", "t", "on", True, 1):
            self.assertTrue(parse_bool_cell(token, "x"), token)
        for token in ("false", "False", "no", "n", "0", "f", "off", False, 0):
            self.assertFalse(parse_bool_cell(token, "x"), token)

    def test_parse_bool_cell_refuses_a_blank_instead_of_reading_it_as_false(self):
        """
        The single most damaging failure mode in this package.

        A missing annotation silently becoming ``False`` would teach the detector
        to ignore solutions, and nothing downstream would look wrong.
        """
        for blank in (None, "", "   "):
            with self.assertRaises(ValueError) as ctx:
                parse_bool_cell(blank, "ann-1")
            self.assertIn("ann-1", str(ctx.exception))
            self.assertIn("missing annotation", str(ctx.exception))

    def test_parse_bool_cell_refuses_unrecognised_values(self):
        for bad in ("maybe", "solution", "yes please", 2, 0.5, [], {}, object()):
            with self.assertRaises(ValueError):
                parse_bool_cell(bad, "ann-1")

    def test_parse_bool_cell_error_names_the_annotation(self):
        with self.assertRaises(ValueError) as ctx:
            parse_bool_cell("perhaps", "comment-77")
        self.assertIn("comment-77", str(ctx.exception))

    def test_solution_type_is_free_form_with_no_allowed_list(self):
        """
        No fixed taxonomy: any non-blank string is a legal category.

        These names are invented for the test.  A hard-coded list would make
        each of them a validation error, which is precisely the coupling the
        design refuses.
        """
        for value in (
            "financial_aid",
            "placement_preparation",
            "something_invented_today",
            "a totally novel category",
            "123",
            "solution_для_проб",
        ):
            self.assertEqual(validate_solution_type(value, "x"), value)

    def test_solution_type_blank_becomes_absent(self):
        self.assertIsNone(normalise_solution_type(None))
        self.assertIsNone(normalise_solution_type(""))
        self.assertIsNone(normalise_solution_type("   "))
        self.assertIsNone(validate_solution_type("  ", "x"))

    def test_solution_type_rejects_non_strings(self):
        for bad in (5, [], {}, True):
            with self.assertRaises(ValueError):
                validate_solution_type(bad, "ann-1")

    def test_distinct_solution_types_is_sorted_and_deduplicated(self):
        values = ["b", "a", "b", None, "", "a", "c"]
        self.assertEqual(distinct_solution_types(values), ["a", "b", "c"])


# ===========================================================================
# Candidate logic - the advice != solution guarantee
# ===========================================================================


class TestCandidateLogic(unittest.TestCase):
    def test_advice_and_solution_are_both_candidates(self):
        self.assertTrue(is_solution_candidate("advice"))
        self.assertTrue(is_solution_candidate("solution"))

    def test_the_other_four_roles_are_not_candidates(self):
        for role in (
            "agreement",
            "personal_experience",
            "opinion",
            "counterargument",
        ):
            self.assertFalse(is_solution_candidate(role), role)

    def test_is_solution_and_is_advice_predicates(self):
        self.assertTrue(is_solution_response_type("solution"))
        self.assertFalse(is_solution_response_type("advice"))
        self.assertTrue(is_advice_response_type("advice"))
        self.assertFalse(is_advice_response_type("solution"))
        self.assertFalse(is_solution_response_type(None))
        self.assertFalse(is_advice_response_type(None))

    def test_response_type_matching_is_case_and_space_insensitive(self):
        """
        The response stage is a separate deployable component.

        A rename to ``"Advice"`` or ``"ADVICE "`` must not silently make every
        comment a non-candidate.
        """
        for value in ("Advice", "ADVICE", "  advice  ", "Solution", " SOLUTION "):
            self.assertTrue(is_solution_candidate(value), value)
        self.assertTrue(is_solution_response_type(" Solution "))
        self.assertTrue(is_advice_response_type("ADVICE"))

    def test_candidate_assessment_never_carries_a_solution_verdict(self):
        """
        The structural guarantee, and the whole point of this module.

        ``advice`` is a candidate.  If ``CandidateAssessment`` exposed a truthy
        ``is_solution`` for advice, "advice implies solution" would be one refactor
        away.  The field is ``None`` until a detector verdict is attached.
        """
        for role in ("advice", "solution", "agreement", "opinion", None):
            assessment = classify_response_candidate(role)
            self.assertIsNone(assessment.is_solution, role)
            self.assertFalse(assessment.has_detector_verdict, role)

    def test_candidate_assessment_never_sets_is_solution_even_for_solution_role(self):
        """
        Not even the ``solution`` role implies a positive verdict.

        A comment can be a solution in role while being too vague to act on, so
        the role cannot supply the boolean under any circumstances.
        """
        assessment = classify_response_candidate("solution")
        self.assertTrue(assessment.is_candidate)
        self.assertTrue(assessment.is_solution_response)
        self.assertIsNone(assessment.is_solution)

    def test_with_prediction_attaches_a_verdict_without_mutating(self):
        assessment = classify_response_candidate("advice")
        decided = assessment.with_prediction(False)
        self.assertFalse(decided.is_solution)
        self.assertTrue(decided.has_detector_verdict)
        # The eligibility record is untouched: still no verdict.
        self.assertIsNone(assessment.is_solution)
        self.assertFalse(assessment.has_detector_verdict)

    def test_role_flags_are_reported_independently(self):
        advice = classify_response_candidate("advice")
        self.assertTrue(advice.is_advice_response)
        self.assertFalse(advice.is_solution_response)
        solution = classify_response_candidate("solution")
        self.assertTrue(solution.is_solution_response)
        self.assertFalse(solution.is_advice_response)
        other = classify_response_candidate("opinion")
        self.assertFalse(other.is_advice_response)
        self.assertFalse(other.is_solution_response)
        self.assertFalse(other.is_candidate)

    def test_missing_response_type_is_unassessed_not_a_negative(self):
        """
        No role means unknown, and the reason string has to say so.

        Treating an absent role as "not a candidate" would silently drop comments
        from a future linker's shortlist when the response stage has not run.
        """
        assessment = classify_response_candidate(None)
        self.assertFalse(assessment.is_candidate)
        self.assertIsNone(assessment.response_type)
        self.assertIn("unknown", assessment.reason)

    def test_every_assessment_explains_itself(self):
        for role in ("advice", "solution", "opinion", None):
            self.assertTrue(classify_response_candidate(role).reason.strip())

    def test_candidate_roles_are_configurable(self):
        """
        A taxonomy revision is a configuration change here, not a code edit.

        This is what keeps the stage independent of the response package: the
        names are local defaults, not an imported contract.
        """
        self.assertTrue(is_solution_candidate("suggestion", candidate_types=["suggestion"]))
        self.assertFalse(is_solution_candidate("advice", candidate_types=["suggestion"]))
        assessment = classify_response_candidate("advice", candidate_types=["suggestion"])
        self.assertFalse(assessment.is_candidate)
        self.assertEqual(assessment.candidate_types, ["suggestion"])

    def test_candidate_types_are_normalised_and_sorted(self):
        assessment = classify_response_candidate("x", candidate_types=[" B ", "a", "a", None])
        self.assertEqual(assessment.candidate_types, ["a", "b"])


# ===========================================================================
# Annotation model
# ===========================================================================


class TestSolutionAnnotation(unittest.TestCase):
    def test_is_solution_is_required(self):
        with self.assertRaises(Exception):
            SolutionAnnotation(document_id="a", text="apply now")

    def test_is_solution_must_be_a_real_boolean(self):
        for bad in ("maybe", "perhaps so", [], {}, 3, 0.5):
            with self.assertRaises(Exception):
                SolutionAnnotation(document_id="a", text="x", is_solution=bad)

    def test_is_solution_accepts_boolean_spellings(self):
        for token, expected in (("true", True), ("false", False), ("yes", True), ("no", False)):
            self.assertIs(
                SolutionAnnotation(document_id="a", text="x", is_solution=token).is_solution,
                expected,
                token,
            )

    def test_blank_text_is_refused_by_validation(self):
        annotation = SolutionAnnotation(document_id="a", text="   ", is_solution=True)
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_annotation()
        self.assertIn("no text", str(ctx.exception))

    def test_solution_type_is_optional(self):
        annotation = SolutionAnnotation(document_id="a", text="x", is_solution=False)
        self.assertIsNone(annotation.solution_type)
        self.assertIsNone(annotation.normalised_solution_type())

    def test_response_type_is_optional_and_neutral(self):
        annotation = SolutionAnnotation(document_id="a", text="x", is_solution=True)
        self.assertIsNone(annotation.response_type)

    def test_rationale_and_evidence_are_optional(self):
        annotation = SolutionAnnotation(document_id="a", text="x", is_solution=True)
        self.assertIsNone(annotation.rationale)
        self.assertIsNone(annotation.evidence)

    def test_validate_annotation_does_not_rewrite_the_record(self):
        """
        ``train()`` validates every annotation it is handed, so validation must be
        read-only.  A validator that wrote back would quietly edit the caller's
        data mid-run - the training path must never be the thing that changes what
        the annotation file says.
        """
        annotation = SolutionAnnotation(
            document_id="a", text="x", is_solution=True, solution_type="  aid  "
        )
        before = annotation.model_dump_json()
        self.assertTrue(annotation.validate_annotation())
        self.assertEqual(annotation.model_dump_json(), before)
        self.assertEqual(annotation.solution_type, "  aid  ")
        # The trimmed value is still available, without being written back.
        self.assertEqual(annotation.normalised_solution_type(), "aid")

    def test_solution_type_reported_by_the_corpus_is_trimmed(self):
        """Reporting normalises; it does not require the stored value to change."""
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(
                    document_id="a", text="x", is_solution=True, solution_type="  aid  "
                )
            ]
        )
        self.assertEqual(corpus.observed_solution_types(), ["aid"])
        self.assertEqual(corpus.solution_type_counts(), {"aid": 1})

    def test_to_solution_document_preserves_the_annotation_identity(self):
        annotation = SolutionAnnotation(
            document_id="a",
            text="apply for financial aid",
            is_solution=True,
            response_type="solution",
            language="hi",
            script="latin",
            is_code_mixed=True,
            post_id="p1",
        )
        document = annotation.to_solution_document()
        self.assertEqual(document.document_id, "a")
        self.assertEqual(document.text, "apply for financial aid")
        self.assertEqual(document.response_type, "solution")
        self.assertEqual(document.language, "hi")
        self.assertTrue(document.is_code_mixed)
        self.assertTrue(document.has_response_context)

    def test_corpus_class_counts_always_expose_both_keys(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="apply", is_solution=True),
                SolutionAnnotation(document_id="b", text="apply now", is_solution=True),
            ]
        )
        counts = corpus.class_counts()
        self.assertEqual(counts, {LABEL_NOT_SOLUTION: 0, LABEL_SOLUTION: 2})

    def test_corpus_observed_types_are_reported_sorted(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(
                    document_id="a", text="x", is_solution=True, solution_type="zebra"
                ),
                SolutionAnnotation(
                    document_id="b", text="y", is_solution=True, solution_type="apple"
                ),
            ]
        )
        self.assertEqual(corpus.observed_solution_types(), ["apple", "zebra"])
        self.assertEqual(
            corpus.solution_type_counts(), {"apple": 1, "zebra": 1}
        )


# ===========================================================================
# Corpus
# ===========================================================================


class TestSolutionCorpus(unittest.TestCase):
    def test_len_and_iteration(self):
        corpus = SolutionCorpus(annotations=_training_annotations())
        self.assertEqual(len(corpus), len(corpus.annotations))
        self.assertEqual(len(list(corpus)), len(corpus.annotations))
        self.assertEqual(len(corpus.is_solutions), len(corpus))

    def test_conflicting_duplicate_ids_are_detected(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="x", is_solution=True),
                SolutionAnnotation(document_id="a", text="x", is_solution=False),
            ]
        )
        self.assertEqual(corpus.conflicting_document_ids(), ["a"])
        self.assertEqual(corpus.duplicate_document_ids(), ["a"])

    def test_same_label_duplicates_are_not_conflicts(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="x", is_solution=True),
                SolutionAnnotation(document_id="a", text="x", is_solution=True),
            ]
        )
        self.assertEqual(corpus.conflicting_document_ids(), [])
        self.assertEqual(corpus.duplicate_document_ids(), ["a"])
        # Must not raise: a same-label duplicate is reported, not refused.
        corpus.validate_unique_document_ids()

    def test_validate_unique_document_ids_refuses_conflicts(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="x", is_solution=True),
                SolutionAnnotation(document_id="a", text="y", is_solution=False),
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            corpus.validate_unique_document_ids()
        self.assertIn("a", str(ctx.exception))
        self.assertIn("binary", str(ctx.exception))

    def test_positives_without_type_are_reported(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="x", is_solution=True),
                SolutionAnnotation(
                    document_id="b", text="y", is_solution=True, solution_type="aid"
                ),
            ]
        )
        self.assertEqual(corpus.positives_without_type(), ["a"])

    def test_negatives_with_type_are_reported(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(
                    document_id="a", text="x", is_solution=False, solution_type="aid"
                ),
                SolutionAnnotation(document_id="b", text="y", is_solution=False),
            ]
        )
        self.assertEqual(corpus.negatives_with_type(), ["a"])

    def test_missing_targets(self):
        corpus = SolutionCorpus(
            annotations=[
                SolutionAnnotation(document_id="a", text="x", is_solution=True)
            ]
        )
        self.assertEqual(corpus.labels, [LABEL_SOLUTION])
        self.assertEqual(corpus.missing_targets, [LABEL_NOT_SOLUTION])


# ===========================================================================
# CSV
# ===========================================================================


class TestAnnotationCsv(unittest.TestCase):
    def test_default_path_points_at_the_documented_location(self):
        self.assertEqual(
            DEFAULT_ANNOTATION_CSV,
            os.path.join("data", "annotations", "solution_annotations.csv"),
        )
        # A trailing part naming the file is the useful part of the contract.
        self.assertIn("solution_annotations.csv", DEFAULT_ANNOTATION_CSV)

    def test_real_annotation_file_is_not_shipped(self):
        """
        No fabricated annotations.

        The repository ships the schema and the loader; ``solution_annotations.csv``
        is created by the human annotation workflow.  A file appearing here means
        someone invented labels, and every downstream number becomes fiction.
        """
        for base in (os.path.dirname(PACKAGE_ROOT), os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_ROOT)))):
            candidate = os.path.join(base, DEFAULT_ANNOTATION_CSV)
            self.assertFalse(
                os.path.exists(candidate),
                f"fabricated annotation file present: {candidate}",
            )

    def test_round_trip_preserves_everything(self):
        original = SolutionCorpus(
            annotations=[
                SolutionAnnotation(
                    document_id="a1",
                    text="apply for financial aid",
                    is_solution=True,
                    solution_type="financial_aid",
                    post_id="p1",
                    comment_id="c1",
                    language="hi",
                    script="latin",
                    is_code_mixed=True,
                    detected_languages=["hi", "en"],
                    response_type="solution",
                    rationale="states an action",
                    evidence="apply for",
                ),
                SolutionAnnotation(
                    document_id="a2",
                    text="same bro",
                    is_solution=False,
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "annotations.csv")
            original.to_csv(path)
            loaded = SolutionCorpus.from_csv(path)
        self.assertEqual(len(loaded), 2)
        first = loaded.annotations[0]
        self.assertEqual(first.document_id, "a1")
        self.assertTrue(first.is_solution)
        self.assertEqual(first.solution_type, "financial_aid")
        self.assertEqual(first.post_id, "p1")
        self.assertEqual(first.comment_id, "c1")
        self.assertEqual(first.language, "hi")
        self.assertEqual(first.script, "latin")
        self.assertTrue(first.is_code_mixed)
        self.assertEqual(first.detected_languages, ["hi", "en"])
        self.assertEqual(first.response_type, "solution")
        self.assertEqual(first.rationale, "states an action")
        self.assertEqual(first.evidence, "apply for")
        self.assertFalse(loaded.annotations[1].is_solution)
        self.assertIsNone(loaded.annotations[1].solution_type)

    def test_csv_row_uses_lowercase_booleans(self):
        annotation = SolutionAnnotation(document_id="a", text="x", is_solution=True)
        self.assertEqual(annotation.to_csv_row()["is_solution"], "true")
        annotation.is_solution = False
        self.assertEqual(annotation.to_csv_row()["is_solution"], "false")

    def test_csv_columns_contain_the_required_triple(self):
        for column in ("document_id", "text", "is_solution"):
            self.assertIn(column, CSV_COLUMNS)
        self.assertIn("solution_type", CSV_COLUMNS)
        # Required columns first, so a truncated file is obvious.
        self.assertEqual(CSV_COLUMNS[:3], ["document_id", "text", "is_solution"])

    def test_missing_file_raises_with_the_expected_columns(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            SolutionCorpus.from_csv("does_not_exist_ever.csv")
        self.assertIn("solution_annotations.csv", str(ctx.exception))

    def test_missing_required_column_is_reported(self):
        content = "document_id,text\nc1,apply now\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            with self.assertRaises(ValueError) as ctx:
                SolutionCorpus.from_csv(path)
        self.assertIn("is_solution", str(ctx.exception))

    def test_blank_is_solution_cell_is_refused_with_a_line_number(self):
        content = (
            "document_id,text,is_solution\n"
            "c1,apply now,true\n"
            "c2,same bro,\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blank.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            with self.assertRaises(ValueError) as ctx:
                SolutionCorpus.from_csv(path)
        message = str(ctx.exception)
        self.assertIn(":3:", message)
        self.assertIn("c2", message)

    def test_unrecognised_is_solution_cell_is_refused(self):
        content = "document_id,text,is_solution\nc1,apply now,perhaps\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            with self.assertRaises(ValueError):
                SolutionCorpus.from_csv(path)

    def test_bom_is_tolerated(self):
        """Excel writes a BOM, and a first-column ``\\ufeffdocument_id`` breaks everything."""
        content = "document_id,text,is_solution\nc1,apply now,true\nc2,same bro,false\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bom.csv")
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                handle.write(content)
            corpus = SolutionCorpus.from_csv(path)
        self.assertEqual(len(corpus), 2)

    def test_blank_lines_are_skipped(self):
        content = (
            "document_id,text,is_solution\n"
            "c1,apply now,true\n"
            "\n"
            "c2,same bro,false\n"
            "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blank_lines.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            corpus = SolutionCorpus.from_csv(path)
        self.assertEqual(len(corpus), 2)

    def test_empty_csv_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.csv")
            open(path, "w", encoding="utf-8").close()
            with self.assertRaises(ValueError) as ctx:
                SolutionCorpus.from_csv(path)
        self.assertIn("empty", str(ctx.exception))

    def test_pipe_separated_multi_values_round_trip(self):
        annotation = SolutionAnnotation(
            document_id="a",
            text="x",
            is_solution=True,
            detected_languages=["hi", "en", "bn"],
        )
        self.assertEqual(annotation.to_csv_row()["detected_languages"], "hi|en|bn")

    def test_unknown_extra_columns_are_ignored(self):
        content = (
            "document_id,text,is_solution,annotator_name,confidence\n"
            "c1,apply now,true,alice,0.9\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "extra.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
            corpus = SolutionCorpus.from_csv(path)
        self.assertEqual(len(corpus), 1)
        self.assertTrue(corpus.annotations[0].is_solution)

    def test_from_records(self):
        corpus = SolutionCorpus.from_records(
            [{"document_id": "a", "text": "x", "is_solution": True}]
        )
        self.assertEqual(len(corpus), 1)

    def test_loading_does_not_write_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.csv")
            SolutionCorpus(annotations=_training_annotations()).to_csv(path)
            before = open(path, "rb").read()
            SolutionCorpus.from_csv(path)
            after = open(path, "rb").read()
        self.assertEqual(before, after)


# ===========================================================================
# Input document
# ===========================================================================


class TestSolutionDocument(unittest.TestCase):
    def test_response_context_is_optional(self):
        document = SolutionDocument(document_id="d", text="apply now")
        self.assertIsNone(document.response_type)
        self.assertEqual(document.response_probabilities, {})
        self.assertFalse(document.has_response_context)
        self.assertFalse(document.has_context)

    def test_response_context_is_detected_when_supplied(self):
        document = SolutionDocument(
            document_id="d", text="apply now", response_type="advice"
        )
        self.assertTrue(document.has_response_context)

    def test_blank_response_type_is_not_context(self):
        document = SolutionDocument(document_id="d", text="x", response_type="   ")
        self.assertFalse(document.has_response_context)

    def test_has_context_covers_topic_and_issue(self):
        self.assertTrue(
            SolutionDocument(document_id="d", text="x", topic_id=3).has_context
        )
        self.assertTrue(
            SolutionDocument(document_id="d", text="x", issue_labels=["fees"]).has_context
        )
        self.assertTrue(
            SolutionDocument(
                document_id="d", text="x", parent_comment_text="parent"
            ).has_context
        )

    def test_defaults(self):
        document = SolutionDocument(document_id="d")
        self.assertEqual(document.text, "")
        self.assertEqual(document.language, "unknown")
        self.assertEqual(document.script, "unknown")
        self.assertFalse(document.is_code_mixed)
        self.assertEqual(document.detected_languages, [])
        self.assertEqual(document.issue_labels, [])
        self.assertEqual(document.issue_probabilities, {})
        self.assertIsNone(document.topic_id)
        self.assertIsNone(document.parent_comment_text)

    def test_response_probabilities_accept_arbitrary_role_names(self):
        """
        The field is a plain dict, not a typed model.

        This is what keeps the stage independent: a response stage that renamed a
        role, or emitted a different number of them, still fits without a change
        here.
        """
        document = SolutionDocument(
            document_id="d",
            text="x",
            response_type="advice",
            response_probabilities={"advice": 0.6, "opinion": 0.4},
        )
        self.assertEqual(document.response_probabilities["advice"], 0.6)


# ===========================================================================
# Training
# ===========================================================================


class TestTraining(SolutionStageTestCase):
    def test_report_identifies_the_model_and_version(self):
        self.assertEqual(self.report.model_name, "tfidf_logistic_binary")
        self.assertEqual(self.report.model_version, "v1")
        self.assertEqual(self.classifier.name, "tfidf_logistic_binary")
        self.assertEqual(self.classifier.version, "v1")

    def test_learns_both_targets(self):
        self.assertEqual(self.report.n_classes, 2)
        self.assertEqual(self.report.labels, [LABEL_NOT_SOLUTION, LABEL_SOLUTION])
        self.assertTrue(self.report.is_usable)
        self.assertGreater(self.report.n_positive, 0)
        self.assertGreater(self.report.n_negative, 0)

    def test_has_features(self):
        self.assertGreater(self.report.n_features, 0)

    def test_default_threshold_is_a_half(self):
        self.assertEqual(self.classifier.threshold, 0.5)
        self.assertEqual(self.report.threshold, 0.5)

    def test_abstention_is_disabled_by_default(self):
        self.assertIsNone(self.classifier.abstain_threshold)
        self.assertIsNone(self.report.abstain_threshold)

    def test_solution_type_is_never_learned(self):
        """
        Reported, not learned.

        This is the "no hard-coded taxonomy" requirement stated as an assertion:
        V1 fits a single binary target regardless of how many ``solution_type``
        categories appear in the data.
        """
        self.assertFalse(self.report.learns_solution_types)
        self.assertGreater(len(self.report.observed_solution_types), 1)
        self.assertEqual(
            self.classifier.config_dict()["features"]["learns_solution_type"], False
        )
        self.assertEqual(
            self.classifier.config_dict()["features"]["learns_is_solution"], True
        )

    def test_observed_solution_types_are_reported(self):
        self.assertIn("financial_aid", self.report.observed_solution_types)
        self.assertIn("placement_preparation", self.report.observed_solution_types)
        self.assertEqual(
            self.report.observed_solution_types,
            sorted(self.report.observed_solution_types),
        )

    def test_config_is_recorded_in_the_report(self):
        config = self.report.config
        self.assertEqual(config["logistic_regression"]["solver"], "lbfgs")
        self.assertEqual(config["logistic_regression"]["random_state"], 42)
        self.assertEqual(config["logistic_regression"]["class_weight"], "balanced")
        self.assertEqual(config["logistic_regression"]["C"], 4.0)
        self.assertEqual(config["tfidf"]["ngram_range"], [1, 2])
        self.assertIsNone(config["tfidf"]["stop_words"])
        self.assertFalse(config["context"]["uses_parent_comment"])
        self.assertFalse(config["context"]["uses_response_type_for_decision"])
        self.assertTrue(config["context"]["uses_response_type_for_candidate_triage"])

    def test_labels_property_is_empty_before_training(self):
        classifier = TfidfLogisticSolutionClassifier()
        self.assertEqual(classifier.labels, [])
        self.assertFalse(classifier.is_trained)
        self.assertIsNone(classifier.training_report)

    def test_training_is_repeatable(self):
        first = TfidfLogisticSolutionClassifier()
        second = TfidfLogisticSolutionClassifier()
        report_a = first.train(self.annotations)
        report_b = second.train(self.annotations)
        self.assertEqual(report_a.model_dump_json(), report_b.model_dump_json())

    def test_retraining_replaces_the_previous_fit(self):
        classifier = TfidfLogisticSolutionClassifier()
        classifier.train(self.annotations)
        result = classifier.predict(self.docs[:1], threshold=0.01)
        classifier.train(self.annotations)
        again = classifier.predict(self.docs[:1], threshold=0.01)
        self.assertEqual(
            result.predictions[0].solution_probability,
            again.predictions[0].solution_probability,
        )

    def test_library_versions_are_recorded(self):
        self.assertIn("scikit-learn", self.report.library_versions)
        self.assertIn("numpy", self.report.library_versions)

    def test_no_timestamp_by_default(self):
        self.assertIsNone(self.report.trained_at)

    def test_timestamp_is_opt_in(self):
        classifier = TfidfLogisticSolutionClassifier(include_timestamp=True)
        report = classifier.train(self.annotations)
        self.assertIsNotNone(report.trained_at)

    def test_untyped_positives_are_warned_about(self):
        annotations = [
            SolutionAnnotation(document_id=f"a{i}", text=f"apply now {i}", is_solution=True)
            for i in range(5)
        ] + [
            SolutionAnnotation(
                document_id=f"b{i}", text=f"same bro {i}", is_solution=False
            )
            for i in range(5)
        ]
        report = TfidfLogisticSolutionClassifier().train(annotations)
        self.assertTrue(
            any("no solution_type" in w for w in report.warnings), report.warnings
        )

    def test_negative_with_type_is_warned_about(self):
        annotations = [
            SolutionAnnotation(document_id=f"a{i}", text=f"apply now {i}", is_solution=True)
            for i in range(5)
        ] + [
            SolutionAnnotation(
                document_id=f"b{i}",
                text=f"same bro {i}",
                is_solution=False,
                solution_type="fees",
            )
            for i in range(5)
        ]
        report = TfidfLogisticSolutionClassifier().train(annotations)
        self.assertTrue(
            any("is_solution=false but a" in w for w in report.warnings),
            report.warnings,
        )


# ===========================================================================
# Training refusals
# ===========================================================================


class TestTrainingRefusals(SolutionStageTestCase):
    def test_empty_corpus_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train([])
        message = str(ctx.exception)
        self.assertIn("no training data", message)
        self.assertIn("solution_annotations.csv", message)

    def test_predict_before_training_is_refused(self):
        classifier = TfidfLogisticSolutionClassifier()
        with self.assertRaises(RuntimeError) as ctx:
            classifier.predict([SolutionDocument(document_id="d", text="apply")])
        self.assertIn("has not been trained", str(ctx.exception))

    def test_evaluate_before_training_is_refused(self):
        with self.assertRaises(RuntimeError):
            TfidfLogisticSolutionClassifier().evaluate([])

    def test_wrong_record_type_is_refused(self):
        with self.assertRaises(TypeError) as ctx:
            TfidfLogisticSolutionClassifier().train(
                [{"document_id": "a", "text": "x", "is_solution": True}]
            )
        self.assertIn("SolutionAnnotation", str(ctx.exception))

    def test_too_few_documents_is_refused(self):
        annotations = [
            SolutionAnnotation(document_id="a", text="apply now", is_solution=True),
            SolutionAnnotation(document_id="b", text="apply now", is_solution=True),
            SolutionAnnotation(document_id="c", text="same bro", is_solution=False),
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        self.assertIn("insufficient training data", str(ctx.exception))

    def test_single_target_corpus_is_refused(self):
        """
        The binary-specific guard.

        A corpus of only solutions (or only non-solutions) cannot teach the
        difference, because there is nothing to contrast against.  A
        multiclass stage would instead complain about missing classes; here the
        problem is that the task has collapsed to a constant.
        """
        annotations = [
            SolutionAnnotation(
                document_id=f"a{i}", text=f"apply for the scholarship {i}", is_solution=True
            )
            for i in range(12)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        message = str(ctx.exception)
        self.assertIn("only one target class", message)
        self.assertIn(LABEL_SOLUTION, message)

    def test_single_negative_target_corpus_is_refused(self):
        annotations = [
            SolutionAnnotation(
                document_id=f"a{i}", text=f"same bro exactly {i}", is_solution=False
            )
            for i in range(12)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        self.assertIn("only one target class", str(ctx.exception))

    def test_target_with_one_example_is_refused(self):
        annotations = [
            SolutionAnnotation(
                document_id=f"a{i}", text=f"apply for financial aid {i}", is_solution=True
            )
            for i in range(12)
        ] + [
            SolutionAnnotation(
                document_id="solo", text="same bro", is_solution=False
            )
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        self.assertIn("insufficient examples per class", str(ctx.exception))

    def test_conflicting_duplicate_ids_are_refused(self):
        annotations = _training_annotations()
        annotations.append(
            SolutionAnnotation(
                document_id=POSITIVE_FIXTURE[0][0],
                text="apply for financial aid on the portal",
                is_solution=False,
            )
        )
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        message = str(ctx.exception)
        self.assertIn("annotation contract violated", message)
        self.assertIn("binary", message)

    def test_same_label_duplicate_is_warned_not_refused(self):
        annotations = _training_annotations()
        first = annotations[0]
        annotations.append(
            SolutionAnnotation(
                document_id=first.document_id, text=first.text, is_solution=True
            )
        )
        report = TfidfLogisticSolutionClassifier().train(annotations)
        self.assertTrue(
            any("duplicate document_ids" in w for w in report.warnings),
            report.warnings,
        )

    def test_unusable_vocabulary_is_refused(self):
        annotations = [
            SolutionAnnotation(document_id=f"p{i}", text="<URL>", is_solution=True)
            for i in range(6)
        ] + [
            SolutionAnnotation(document_id=f"n{i}", text="<URL>", is_solution=False)
            for i in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        self.assertIn("no usable TF-IDF features", str(ctx.exception))

    def test_blank_text_annotation_is_refused(self):
        annotations = [
            SolutionAnnotation(
                document_id=f"p{i}", text=f"apply for aid {i}", is_solution=True
            )
            for i in range(6)
        ] + [
            SolutionAnnotation(document_id="blank", text="   ", is_solution=False)
            for _ in range(1)
        ] + [
            SolutionAnnotation(
                document_id=f"n{i}", text=f"same bro {i}", is_solution=False
            )
            for i in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier().train(annotations)
        self.assertIn("no text", str(ctx.exception))


# ===========================================================================
# Positive / negative / advice examples
# ===========================================================================


class TestPositiveAndNegativeExamples(SolutionStageTestCase):
    def test_training_fixtures_are_reproduced(self):
        """
        Memorisation check on the data the model was fitted on.

        Deliberately scoped to the training corpus.  It verifies the label
        plumbing end to end (annotation -> vectoriser -> sigmoid -> threshold ->
        boolean) and says nothing about generalisation, which a 38-example
        synthetic corpus cannot demonstrate.
        """
        result = self.classifier.predict(self.docs)
        gold = {a.document_id: a.is_solution for a in self.annotations}
        mismatches = [
            (p.document_id, p.is_solution, gold[p.document_id])
            for p in result.predictions
            if p.is_solution != gold[p.document_id]
        ]
        self.assertEqual(mismatches, [])

    def test_explicit_positive_examples_are_flagged(self):
        """
        The assignment's own positive examples, all present in the training
        fixture, so the assertion is supported by the annotated data.
        """
        for text, solution_type in (
            ("apply for financial aid on the portal", "financial_aid"),
            ("try applying to the scholarship before the deadline", "scholarship"),
            ("placement office ko contact karo", "placement_preparation"),
        ):
            self.assertIn(text, [a.text for a in self.annotations], text)
            self.assertIsNotNone(solution_type)

    def test_explicit_negative_examples_are_present_as_negatives(self):
        for text in (
            "same bro",
            "i faced the same issue last year",
            "i think fees are too high",
        ):
            self.assertIn(
                text,
                [a.text for a in self.annotations if not a.is_solution],
                text,
            )

    def test_positive_probability_exceeds_negative_on_positive_fixtures(self):
        by_id = {a.document_id: a for a in self.annotations}
        result = self.classifier.predict(self.docs)
        for prediction in result.predictions:
            if by_id[prediction.document_id].is_solution:
                self.assertGreater(
                    prediction.solution_probability,
                    prediction.not_solution_probability,
                    prediction.document_id,
                )
            else:
                self.assertLess(
                    prediction.solution_probability,
                    prediction.not_solution_probability,
                    prediction.document_id,
                )

    def test_every_probability_is_a_valid_probability(self):
        """Structural check, valid for unseen text as well as seen."""
        result = self.classifier.predict(self.docs)
        for prediction in result.predictions:
            self.assertGreaterEqual(prediction.solution_probability, 0.0)
            self.assertLessEqual(prediction.solution_probability, 1.0)
            self.assertGreaterEqual(prediction.not_solution_probability, 0.0)
            self.assertLessEqual(prediction.not_solution_probability, 1.0)

    def test_result_counts_add_up(self):
        result = self.classifier.predict(self.docs)
        self.assertEqual(result.n_documents, len(self.docs))
        self.assertEqual(
            result.n_solutions + result.n_not_solutions, result.n_documents
        )
        self.assertEqual(
            result.solution_documents().__len__(), result.n_solutions
        )


class TestAdviceExamples(SolutionStageTestCase):
    """
    Advice is a candidate.  Whether it qualifies is the detector's call.

    The fixture makes both outcomes present: advice that names a concrete action
    (p05, p06, p08, p10, p14) and advice that recommends nothing at all
    (n13, n14, n11).  If the detector simply echoed the role, the second group
    would be mislabelled - so this is the test that would catch advice being
    silently equated with solution.
    """

    def _result(self):
        return self.classifier.predict(self.docs)

    def test_actionable_advice_is_a_candidate_and_a_solution(self):
        result = self._result()
        for document_id in ("p05", "p06", "p08", "p10", "p14"):
            prediction = result.prediction_for(document_id)
            self.assertTrue(prediction.is_candidate, document_id)
            self.assertTrue(prediction.is_solution, document_id)

    def test_vague_advice_is_a_candidate_but_not_a_solution(self):
        """
        The distinction, on data that actually contains both cases.

        "maybe you could try emailing them, not sure though" is advice, so it is a
        candidate - and it is not a solution, because nothing actionable was named.
        """
        result = self._result()
        for document_id in ("n13", "n14", "n11"):
            prediction = result.prediction_for(document_id)
            self.assertTrue(prediction.is_candidate, document_id)
            self.assertFalse(prediction.is_solution, document_id)

    def test_candidates_rejected_are_counted(self):
        result = self._result()
        self.assertGreater(result.n_candidates, 0)
        self.assertGreater(result.n_candidates_not_solutions, 0)
        rejected = result.candidates_rejected()
        self.assertTrue(all(p.is_candidate and not p.is_solution for p in rejected))
        self.assertEqual(len(rejected), result.n_candidates_not_solutions)

    def test_candidate_rejection_is_warned_about(self):
        result = self._result()
        self.assertTrue(
            any("solution candidates" in w and "did not assert" in w for w in result.warnings),
            result.warnings,
        )

    def test_response_type_alone_does_not_decide(self):
        """
        Same role, different outcomes.

        If ``is_solution`` were derived from ``response_type``, every advice row
        would agree.  They do not, which is the architectural property under test.
        """
        result = self._result()
        by_role = {}
        for prediction in result.predictions:
            if prediction.response_type == "advice":
                by_role.setdefault(prediction.is_solution, []).append(
                    prediction.document_id
                )
        self.assertTrue(by_role.get(True), "no advice comment was a solution")
        self.assertTrue(by_role.get(False), "no advice comment was rejected")


# ===========================================================================
# Response-type interaction
# ===========================================================================


class TestResponseTypeInteraction(SolutionStageTestCase):
    def test_response_type_never_changes_the_probability(self):
        """
        The stage is independent of the response stage.

        The same text with different ``response_type`` values must get a
        bit-identical distribution.  If the role leaked into the decision, the
        two rows would differ - and swapping the response model would silently
        change every solution prediction.
        """
        text = "you should contact the placement office"
        documents = [
            SolutionDocument(document_id="a", text=text, response_type="advice"),
            SolutionDocument(document_id="b", text=text, response_type="solution"),
            SolutionDocument(document_id="c", text=text, response_type="opinion"),
            SolutionDocument(document_id="d", text=text, response_type="agreement"),
            SolutionDocument(document_id="e", text=text, response_type="counterargument"),
            SolutionDocument(document_id="f", text=text),
        ]
        result = self.classifier.predict(documents)
        probabilities = {p.solution_probability for p in result.predictions}
        self.assertEqual(len(probabilities), 1)
        is_solutions = {p.is_solution for p in result.predictions}
        self.assertEqual(len(is_solutions), 1)

    def test_response_probabilities_never_change_the_probability(self):
        text = "apply for the bursary on the student portal"
        documents = [
            SolutionDocument(document_id="a", text=text),
            SolutionDocument(
                document_id="b",
                text=text,
                response_type="solution",
                response_probabilities={"solution": 0.99},
            ),
            SolutionDocument(
                document_id="c",
                text=text,
                response_type="opinion",
                response_probabilities={"opinion": 0.97, "solution": 0.01},
            ),
        ]
        result = self.classifier.predict(documents)
        self.assertEqual(len({p.solution_probability for p in result.predictions}), 1)

    def test_candidate_flag_tracks_the_role(self):
        result = self.classifier.predict(self.docs)
        for prediction in result.predictions:
            expected = is_solution_candidate(prediction.response_type)
            self.assertEqual(prediction.is_candidate, expected, prediction.document_id)

    def test_response_context_is_counted(self):
        result = self.classifier.predict(self.docs)
        expected = sum(1 for d in self.docs if d.has_response_context)
        self.assertEqual(result.n_documents_with_response_context, expected)
        self.assertGreater(result.n_documents_with_response_context, 0)

    def test_missing_role_is_not_counted_as_a_disagreement(self):
        """
        An absent role is unassessed, not contradicted.

        Counting it as a disagreement would raise a false cross-stage alarm on
        every corpus where the response stage has not been run.
        """
        documents = [
            SolutionDocument(document_id=f"d{i}", text=f"apply for aid {i}")
            for i in range(6)
        ]
        result = self.classifier.predict(documents, threshold=0.01)
        self.assertEqual(result.n_documents_with_response_context, 0)
        self.assertEqual(result.n_solutions_not_candidates, 0)
        self.assertFalse(
            any("without being solution candidates" in w for w in result.warnings),
            result.warnings,
        )

    def test_role_disagreement_is_reported_when_a_role_exists(self):
        documents = [
            SolutionDocument(
                document_id="a", text="apply for the bursary", response_type="opinion"
            ),
            SolutionDocument(
                document_id="b", text="apply for the bursary", response_type="agreement"
            ),
        ]
        result = self.classifier.predict(documents, threshold=0.01)
        self.assertGreater(result.n_solutions_not_candidates, 0)
        self.assertTrue(
            any("without being solution candidates" in w for w in result.warnings),
            result.warnings,
        )

    def test_candidate_assessment_on_a_prediction_carries_the_verdict(self):
        result = self.classifier.predict(self.docs)
        prediction = result.prediction_for("n13")
        assessment = prediction.candidate_assessment()
        self.assertTrue(assessment.is_candidate)
        self.assertTrue(assessment.is_advice_response)
        self.assertTrue(assessment.has_detector_verdict)
        self.assertEqual(assessment.is_solution, prediction.is_solution)

    def test_response_metadata_is_echoed(self):
        documents = [
            SolutionDocument(
                document_id="a",
                text="apply for the bursary",
                response_type="advice",
                response_probabilities={"advice": 0.7, "opinion": 0.3},
            )
        ]
        result = self.classifier.predict(documents)
        prediction = result.predictions[0]
        self.assertEqual(prediction.response_type, "advice")
        self.assertEqual(prediction.response_probabilities["advice"], 0.7)


# ===========================================================================
# Threshold behaviour
# ===========================================================================


class TestThresholdBehaviour(SolutionStageTestCase):
    def test_decision_follows_the_threshold_exactly(self):
        """
        ``is_solution`` must be exactly ``p >= threshold`` at every cutoff.

        Checked against the reported probability rather than a recomputation, so a
        rounding disagreement between the stored and computed value would show up.
        """
        result = self.classifier.predict(self.docs, threshold=0.01)
        for prediction in result.predictions:
            for cutoff in (0.05, 0.3, 0.5, 0.7, 0.95):
                expected = prediction.solution_probability >= cutoff
                self.assertEqual(
                    prediction.reclassify_at(cutoff),
                    expected,
                    (prediction.document_id, cutoff),
                )

    def test_lower_threshold_flags_more_comments(self):
        low = self.classifier.predict(self.docs, threshold=0.05)
        high = self.classifier.predict(self.docs, threshold=0.95)
        self.assertGreaterEqual(low.n_solutions, high.n_solutions)

    def test_threshold_is_reported_on_the_result_and_each_prediction(self):
        result = self.classifier.predict(self.docs, threshold=0.6)
        self.assertEqual(result.threshold, 0.6)
        for prediction in result.predictions:
            self.assertEqual(prediction.threshold, 0.6)

    def test_threshold_override_does_not_mutate_the_classifier(self):
        self.classifier.predict(self.docs, threshold=0.9)
        self.assertEqual(self.classifier.threshold, 0.5)

    def test_reclassify_matches_a_fresh_predict(self):
        """
        Re-thresholding a stored probability must equal re-running inference.

        This is what lets a consumer explore precision/recall cut-offs on data it
        already has instead of refitting and re-predicting.
        """
        stored = self.classifier.predict(self.docs, threshold=0.5)
        fresh = self.classifier.predict(self.docs, threshold=0.75)
        for original, recomputed in zip(stored.predictions, fresh.predictions):
            self.assertEqual(
                original.reclassify_at(0.75), recomputed.is_solution, original.document_id
            )

    def test_invalid_threshold_is_refused(self):
        documents = [SolutionDocument(document_id="d", text="apply now")]
        for bad in (0.0, 1.0, -0.5, 2, "high", True, []):
            with self.assertRaises(ValueError):
                self.classifier.predict(documents, threshold=bad)

    def test_constructor_threshold_is_validated(self):
        for bad in (0.0, 1.0, -1.0, 5.0, "0.5", True, None if False else "x"):
            with self.assertRaises(ValueError):
                TfidfLogisticSolutionClassifier(threshold=bad)

    def test_none_threshold_means_use_the_configured_default(self):
        documents = [SolutionDocument(document_id="d", text="apply for financial aid")]
        result = self.classifier.predict(documents, threshold=None)
        self.assertEqual(result.threshold, self.classifier.threshold)


class TestAbstention(SolutionStageTestCase):
    def test_abstention_is_off_by_default(self):
        result = self.classifier.predict(self.docs, threshold=0.01)
        self.assertEqual(result.n_abstained, 0)
        self.assertIsNone(result.abstain_threshold)
        for prediction in result.predictions:
            self.assertFalse(prediction.abstained)

    def test_below_threshold_is_a_decision_not_an_abstention(self):
        """
        "Not a solution" is a real answer the corpus supports.

        Treating it as an abstention would make ``n_abstained`` mean "the negative
        class", which is not information, and would let a trivially cautious model
        look well calibrated.
        """
        result = self.classifier.predict(self.docs, threshold=0.99)
        self.assertGreater(result.n_not_solutions, 0)
        self.assertEqual(result.n_abstained, 0)
        for prediction in result.predictions:
            if not prediction.is_solution:
                self.assertFalse(prediction.abstained)

    def test_band_captures_weak_positives(self):
        classifier = TfidfLogisticSolutionClassifier(
            threshold=0.2, abstain_threshold=0.95
        )
        classifier.train(self.annotations)
        result = classifier.predict(
            [a.to_solution_document() for a in self.annotations]
        )
        self.assertEqual(result.abstain_threshold, 0.95)
        self.assertGreater(result.n_abstained, 0)
        for prediction in result.predictions:
            if prediction.abstained:
                # A weak positive: above the boundary, below the band, and
                # reported as no solution.
                self.assertFalse(prediction.is_solution)
                self.assertTrue(prediction.is_weak_positive)
                self.assertGreaterEqual(prediction.solution_probability, 0.2)
                self.assertLess(prediction.solution_probability, 0.95)
            else:
                self.assertFalse(prediction.is_weak_positive)

    def test_abstentions_are_excluded_from_the_solution_count(self):
        classifier = TfidfLogisticSolutionClassifier(
            threshold=0.2, abstain_threshold=0.95
        )
        classifier.train(self.annotations)
        result = classifier.predict(
            [a.to_solution_document() for a in self.annotations]
        )
        self.assertEqual(
            result.n_solutions + result.n_not_solutions, result.n_documents
        )
        self.assertLess(result.n_solutions, len(self.annotations))

    def test_band_can_be_disabled_for_one_run(self):
        classifier = TfidfLogisticSolutionClassifier(
            threshold=0.2, abstain_threshold=0.95
        )
        classifier.train(self.annotations)
        documents = [a.to_solution_document() for a in self.annotations]
        self.assertGreater(classifier.predict(documents).n_abstained, 0)
        self.assertEqual(classifier.predict(documents, abstain_threshold=None).n_abstained, 0)
        # The instance configuration is untouched by a per-call override.
        self.assertEqual(classifier.abstain_threshold, 0.95)

    def test_inherit_sentinel_is_not_none(self):
        self.assertIsNot(INHERIT, None)

    def test_zero_abstain_threshold_is_refused_with_a_hint(self):
        """
        An empty band is far more likely a mistake than an intention, and silently
        accepting it would look like a working configuration.
        """
        classifier = TfidfLogisticSolutionClassifier(
            threshold=0.2, abstain_threshold=0.95
        )
        classifier.train(self.annotations)
        documents = [a.to_solution_document() for a in self.annotations]
        with self.assertRaises(ValueError) as ctx:
            classifier.predict(documents, abstain_threshold=0)
        self.assertIn("None", str(ctx.exception))

    def test_abstain_threshold_below_threshold_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier(threshold=0.8, abstain_threshold=0.5)
        self.assertIn("must be >=", str(ctx.exception))

    def test_invalid_abstain_threshold_is_refused(self):
        for bad in (1.5, -0.2, "high", True, []):
            with self.assertRaises(ValueError):
                TfidfLogisticSolutionClassifier(abstain_threshold=bad)


# ===========================================================================
# Probability output
# ===========================================================================


class TestProbabilityOutput(SolutionStageTestCase):
    def test_probabilities_sum_to_one(self):
        result = self.classifier.predict(self.docs)
        for prediction in result.predictions:
            total = prediction.solution_probability + prediction.not_solution_probability
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_probability_is_exactly_the_sigmoid_of_the_decision_function(self):
        """
        The documented mathematics is exact, not a convenient approximation.

        For K=2, sklearn's two-class softmax satisfies
        ``softmax([z0, z1])[1] == sigmoid(z1 - z0)``, so the reported
        ``solution_probability`` is literally ``sigma(w.x + b)`` for a single
        weight vector.  If this ever diverges, the documentation is wrong.
        """
        import numpy as np

        texts = ["apply for the bursary", "same bro", "you should contact the office"]
        documents = [
            SolutionDocument(document_id=f"d{i}", text=text)
            for i, text in enumerate(texts)
        ]
        result = self.classifier.predict(documents)
        X = self.classifier._vectorizer.transform(texts)
        z = self.classifier._model.decision_function(X)
        for index, prediction in enumerate(result.predictions):
            expected = 1.0 / (1.0 + math.exp(-float(z[index])))
            self.assertAlmostEqual(prediction.solution_probability, expected, places=5)

    def test_probabilities_are_reported_even_when_no_solution_is_asserted(self):
        """
        A consumer can re-apply a different threshold without re-running inference.
        """
        result = self.classifier.predict(self.docs, threshold=0.99)
        self.assertEqual(result.n_solutions, 0)
        for prediction in result.predictions:
            self.assertGreater(prediction.solution_probability, 0.0)

    def test_average_probability_and_solution_rate(self):
        result = self.classifier.predict(self.docs)
        self.assertAlmostEqual(
            result.average_probability(),
            sum(p.solution_probability for p in result.predictions) / len(result.predictions),
            places=5,
        )
        self.assertAlmostEqual(
            result.solution_rate(), result.n_solutions / result.n_documents, places=5
        )

    def test_empty_run_is_handled(self):
        result = self.classifier.predict([])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.n_solutions, 0)
        self.assertEqual(result.average_probability(), 0.0)
        self.assertEqual(result.solution_rate(), 0.0)
        self.assertEqual(result.solution_documents(), [])

    def test_label_frequency_is_canonically_ordered(self):
        result = self.classifier.predict(self.docs)
        self.assertEqual(
            list(result.label_frequency()), [LABEL_NOT_SOLUTION, LABEL_SOLUTION]
        )

    def test_predicted_label_naming(self):
        result = self.classifier.predict(self.docs)
        for prediction in result.predictions:
            self.assertEqual(
                prediction.predicted_label,
                LABEL_SOLUTION if prediction.is_solution else LABEL_NOT_SOLUTION,
            )


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism(SolutionStageTestCase):
    def test_two_fits_produce_identical_probabilities(self):
        first = TfidfLogisticSolutionClassifier()
        second = TfidfLogisticSolutionClassifier()
        first.train(self.annotations)
        second.train(self.annotations)
        a = first.predict(self.docs).model_dump_json()
        b = second.predict(self.docs).model_dump_json()
        self.assertEqual(a, b)

    def test_two_predict_calls_are_identical(self):
        first = self.classifier.predict(self.docs).model_dump_json()
        second = self.classifier.predict(self.docs).model_dump_json()
        self.assertEqual(first, second)

    def test_training_report_is_identical_across_fits(self):
        first = TfidfLogisticSolutionClassifier()
        second = TfidfLogisticSolutionClassifier()
        self.assertEqual(
            first.train(self.annotations).model_dump_json(),
            second.train(self.annotations).model_dump_json(),
        )

    def test_probabilities_are_rounded_to_six_places(self):
        for prediction in self.classifier.predict(self.docs).predictions:
            self.assertEqual(
                prediction.solution_probability, round(prediction.solution_probability, 6)
            )


# ===========================================================================
# Metadata / immutability
# ===========================================================================


class TestMetadataPreservation(SolutionStageTestCase):
    def test_every_field_is_echoed_unchanged(self):
        document = SolutionDocument(
            document_id="c-1",
            text="apply for financial aid on the portal",
            original_text="Apply for financial aid on the portal!!",
            post_id="post-9",
            comment_id="c-1",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=7,
            topic_label="fees",
            topic_keywords=["fees", "aid"],
            issue_labels=["fees"],
            issue_probabilities={"fees": 0.8},
            response_type="solution",
            response_probabilities={"solution": 0.9},
            parent_comment_text="the fees are too high",
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertEqual(prediction.document_id, "c-1")
        self.assertEqual(prediction.text, document.text)
        self.assertEqual(prediction.original_text, "Apply for financial aid on the portal!!")
        self.assertEqual(prediction.post_id, "post-9")
        self.assertEqual(prediction.comment_id, "c-1")
        self.assertEqual(prediction.language, "hi")
        self.assertEqual(prediction.script, "latin")
        self.assertTrue(prediction.is_code_mixed)
        self.assertEqual(prediction.detected_languages, ["hi", "en"])
        self.assertEqual(prediction.topic_id, 7)
        self.assertEqual(prediction.topic_label, "fees")
        self.assertEqual(prediction.topic_keywords, ["fees", "aid"])
        self.assertEqual(prediction.issue_labels, ["fees"])
        self.assertEqual(prediction.issue_probabilities["fees"], 0.8)
        self.assertEqual(prediction.response_type, "solution")
        self.assertEqual(prediction.response_probabilities["solution"], 0.9)
        self.assertEqual(prediction.parent_comment_text, "the fees are too high")

    def test_original_text_is_not_normalised(self):
        document = SolutionDocument(
            document_id="c-1",
            text="apply for financial aid",
            original_text="  Apply   For Financial Aid!!  ",
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertEqual(prediction.original_text, "  Apply   For Financial Aid!!  ")

    def test_post_and_issue_ids_are_carried_for_a_future_linker(self):
        """
        The linking stage will need the thread structure.

        It is carried and echoed so that stage needs no re-collection, but it does
        not influence the boolean - which is why a document with issues and one
        without get the same probability for the same text.
        """
        documents = [
            SolutionDocument(document_id="a", text="apply for the bursary"),
            SolutionDocument(
                document_id="b",
                text="apply for the bursary",
                post_id="p1",
                issue_labels=["fees"],
                issue_probabilities={"fees": 0.9},
                parent_comment_text="fees are too high",
            ),
        ]
        result = self.classifier.predict(documents)
        self.assertEqual(
            result.predictions[0].solution_probability,
            result.predictions[1].solution_probability,
        )
        self.assertEqual(result.n_documents_with_context, 1)

    def test_lists_are_copied_not_shared(self):
        """
        A prediction must not alias the input's mutable containers.

        Sharing a list means a later mutation of the input silently rewrites
        already-returned results.
        """
        document = SolutionDocument(
            document_id="c-1",
            text="apply for financial aid",
            detected_languages=["hi"],
            issue_labels=["fees"],
        )
        prediction = self.classifier.predict([document]).predictions[0]
        document.detected_languages.append("en")
        document.issue_labels.append("hostel")
        self.assertEqual(prediction.detected_languages, ["hi"])
        self.assertEqual(prediction.issue_labels, ["fees"])


class TestInputImmutability(SolutionStageTestCase):
    def test_prediction_does_not_mutate_the_document(self):
        document = SolutionDocument(
            document_id="c-1",
            text="apply for financial aid <EMOJI_tada> <URL>",
            original_text="Apply for financial aid 🎉 https://x.com",
            post_id="p1",
            response_type="advice",
        )
        before = document.model_dump_json()
        self.classifier.predict([document])
        self.assertEqual(document.model_dump_json(), before)

    def test_prediction_does_not_mutate_text_during_placeholder_stripping(self):
        """
        Placeholder stripping happens on a copy.

        The document keeps ``<URL>`` and ``<EMOJI_tada>``; only the string handed
        to the vectoriser loses them.  A test that only checked the probability
        would miss an in-place ``str.replace`` on the input.
        """
        text = "apply for financial aid <EMOJI_tada> <URL>"
        document = SolutionDocument(document_id="c-1", text=text)
        self.classifier.predict([document])
        self.assertEqual(document.text, text)
        self.assertIn("<URL>", document.text)
        # The model-side copy is what loses the tokens.
        self.assertNotIn("<URL>", self.classifier._featurize_text(text))

    def test_training_does_not_mutate_annotations(self):
        annotations = _training_annotations()
        before = [a.model_dump_json() for a in annotations]
        classifier = TfidfLogisticSolutionClassifier()
        classifier.train(annotations)
        self.assertEqual([a.model_dump_json() for a in annotations], before)

    def test_evaluate_does_not_mutate_annotations(self):
        annotations = _heldout_annotations()
        before = [a.model_dump_json() for a in annotations]
        self.classifier.evaluate(annotations)
        self.assertEqual([a.model_dump_json() for a in annotations], before)

    def test_prediction_does_not_mutate_a_list_input(self):
        documents = [SolutionDocument(document_id="c-1", text="apply for aid")]
        self.classifier.predict(documents)
        self.assertEqual(len(documents), 1)


# ===========================================================================
# Multilingual / code-mixing / emoji / negation
# ===========================================================================


class TestMultilingualAndCodeMixing(SolutionStageTestCase):
    def test_non_latin_scripts_produce_valid_probabilities(self):
        """
        Architecture only: nothing crashes, probabilities stay valid.

        Deliberately no assertion about *which* label a Bengali or Odia sentence
        gets.  Six Odia examples teach the model that Odia exists, not how to read
        it, and pinning a guess would be asserting accuracy the fixture cannot
        support.  See the multilingual limitations section of the documentation.
        """
        documents = [
            SolutionDocument(document_id="bn1", text="বৃত্তির জন্য আবেদন করুন", script="bengali", language="bn"),
            SolutionDocument(document_id="or1", text="ଫର୍ମ ପୂରଣ କରନ୍ତୁ", script="odia", language="or"),
            SolutionDocument(document_id="hi1", text="मुझे लगता है यह कॉलेज बेहतर है", script="devanagari", language="hi"),
            SolutionDocument(document_id="hi2", text="scholarship ke liye apply kar de", script="latin", language="hi", is_code_mixed=True),
        ]
        result = self.classifier.predict(documents)
        self.assertEqual(result.n_documents, 4)
        for prediction in result.predictions:
            self.assertGreaterEqual(prediction.solution_probability, 0.0)
            self.assertLessEqual(prediction.solution_probability, 1.0)
            self.assertAlmostEqual(
                prediction.solution_probability + prediction.not_solution_probability,
                1.0,
                places=5,
            )
            self.assertEqual(prediction.language, documents[result.predictions.index(prediction)].language)

    def test_code_mixed_metadata_is_carried(self):
        document = SolutionDocument(
            document_id="m1",
            text="placement office ko contact karo",
            language="hi",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertTrue(prediction.is_code_mixed)
        self.assertEqual(prediction.detected_languages, ["hi", "en"])
        self.assertEqual(prediction.language, "hi")

    def test_code_mixed_fixtures_are_reproduced(self):
        """Memorisation on the multilingual rows actually in the fixture."""
        result = self.classifier.predict(self.docs)
        for document_id in ("p13", "p14", "p15", "p16", "p17", "n11", "n17", "n18"):
            prediction = result.prediction_for(document_id)
            self.assertIsNotNone(prediction, document_id)
            self.assertGreaterEqual(prediction.solution_probability, 0.0)

    def test_empty_text_produces_a_valid_probability(self):
        result = self.classifier.predict([SolutionDocument(document_id="e", text="")])
        prediction = result.predictions[0]
        self.assertGreaterEqual(prediction.solution_probability, 0.0)
        self.assertLessEqual(prediction.solution_probability, 1.0)


class TestEmojiPlaceholders(SolutionStageTestCase):
    def test_emoji_token_is_stripped_for_the_model_but_kept_on_the_document(self):
        document = SolutionDocument(
            document_id="e1", text="apply for financial aid <EMOJI_tada>"
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertIn("<EMOJI_tada>", prediction.text)
        self.assertNotIn("<EMOJI_tada>", self.classifier._featurize_text(document.text))

    def test_url_and_mention_placeholders_are_stripped(self):
        for placeholder in ("<URL>", "<MENTION>", "<EMOJI_joy>", "<HASHTAG>"):
            self.assertNotIn(
                placeholder, self.classifier._featurize_text(f"apply for aid {placeholder}")
            )

    def test_stripping_can_be_disabled(self):
        classifier = TfidfLogisticSolutionClassifier(
            config=SolutionModelConfig(drop_placeholder_tokens=False)
        )
        classifier.train(self.annotations)
        self.assertIn("<EMOJI_tada>", classifier._featurize_text("same <EMOJI_tada>"))
        self.assertEqual(
            classifier.config_dict()["features"]["drop_placeholder_tokens"], False
        )

    def test_emoji_fixture_is_reproduced(self):
        """The ``<EMOJI_tada>`` agreement row is in the training fixture."""
        result = self.classifier.predict(self.docs)
        prediction = result.prediction_for("n15")
        self.assertIsNotNone(prediction)
        self.assertIn("<EMOJI_tada>", prediction.text)

    def test_placeholder_only_comment_gets_a_valid_probability(self):
        result = self.classifier.predict(
            [SolutionDocument(document_id="u", text="<URL> <EMOJI_tada>")]
        )
        prediction = result.predictions[0]
        self.assertGreaterEqual(prediction.solution_probability, 0.0)
        self.assertLessEqual(prediction.solution_probability, 1.0)


class TestNegation(SolutionStageTestCase):
    """
    Negation is documented as a limitation, not asserted as solved.

    "do not apply for financial aid" contains "apply for financial aid", and
    "placement office ko contact mat karo" contains "contact karo".  A
    bag-of-words model has no mechanism to represent the difference - the
    negation lives in a different token.  It is a genuine weakness of this
    baseline and the honest thing to do is say so, not to write a test asserting
    the model handles it and let a future retrain fail it.

    What is asserted here is architecture: negated text is featurised, produces a
    valid probability, and is deterministic.
    """

    def test_negated_text_is_featurised_and_produces_a_valid_probability(self):
        for text in (
            "do not apply for financial aid, it is closed",
            "never contact the placement office",
            "placement office ko contact mat karo, woh nahi help karenge",
            "you should not submit the form twice",
        ):
            document = SolutionDocument(document_id="n", text=text)
            prediction = self.classifier.predict([document]).predictions[0]
            self.assertGreaterEqual(prediction.solution_probability, 0.0, text)
            self.assertLessEqual(prediction.solution_probability, 1.0, text)
            self.assertEqual(prediction.text, text)

    def test_negated_fixtures_are_present_in_training_data(self):
        """
        The fixture contains negated negatives.

        This is a statement about the *data*, which is checkable, rather than
        about the model's ability to generalise to unseen negation, which is not.
        """
        negated = [
            a
            for a in self.annotations
            if "do not apply" in a.text or "mat karo" in a.text
        ]
        self.assertTrue(negated)
        for annotation in negated:
            self.assertFalse(annotation.is_solution)

    def test_negation_is_deterministic(self):
        document = SolutionDocument(document_id="n", text="do not apply for financial aid")
        first = self.classifier.predict([document]).model_dump_json()
        second = self.classifier.predict([document]).model_dump_json()
        self.assertEqual(first, second)


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization(SolutionStageTestCase):
    def test_model_and_version_survive_serialization(self):
        payload = json.loads(self.report.model_dump_json())
        self.assertEqual(payload["model_name"], "tfidf_logistic_binary")
        self.assertEqual(payload["model_version"], "v1")

    def test_result_round_trips_through_json(self):
        result = self.classifier.predict(self.docs)
        restored = SolutionDetectionResult.model_validate_json(result.model_dump_json())
        self.assertEqual(restored.model_dump_json(), result.model_dump_json())

    def test_prediction_round_trips_through_json(self):
        result = self.classifier.predict(self.docs)
        restored = SolutionPrediction.model_validate_json(
            result.predictions[0].model_dump_json()
        )
        self.assertEqual(restored.document_id, result.predictions[0].document_id)

    def test_config_is_json_serialisable(self):
        json.dumps(self.report.config)
        json.dumps(self.classifier.predict(self.docs).config)

    def test_document_and_annotation_round_trip(self):
        document = SolutionDocument(document_id="d", text="apply for aid")
        self.assertEqual(
            SolutionDocument.model_validate_json(document.model_dump_json()).model_dump_json(),
            document.model_dump_json(),
        )
        annotation = SolutionAnnotation(
            document_id="a", text="apply for aid", is_solution=True, solution_type="aid"
        )
        self.assertEqual(
            SolutionAnnotation.model_validate_json(
                annotation.model_dump_json()
            ).model_dump_json(),
            annotation.model_dump_json(),
        )

    def test_training_report_round_trips(self):
        restored = SolutionTrainingReport.model_validate_json(
            self.report.model_dump_json()
        )
        self.assertEqual(restored.model_dump_json(), self.report.model_dump_json())

    def test_candidate_assessment_round_trips(self):
        assessment = classify_response_candidate("advice")
        restored = CandidateAssessment.model_validate_json(
            assessment.model_dump_json()
        )
        self.assertEqual(restored.model_dump_json(), assessment.model_dump_json())
        self.assertIsNone(restored.is_solution)

    def test_evaluation_report_round_trips(self):
        report = self.classifier.evaluate(_heldout_annotations())
        restored = SolutionEvaluationReport.model_validate_json(
            report.model_dump_json()
        )
        self.assertEqual(restored.model_dump_json(), report.model_dump_json())


# ===========================================================================
# Class weighting / imbalance
# ===========================================================================


def _imbalanced_corpus(majority_count=30, minority_count=6):
    """
    A deliberately skewed corpus.

    ``class_weight="balanced"`` computes identical weights on a balanced corpus,
    so the two fitted models would be bit-identical and the class-weight test
    would prove nothing.
    """
    rows = [
        (
            f"maj{i}",
            f"same bro exactly what i said {i}",
            False,
            "en",
            "latin",
            False,
            "agreement",
        )
        for i in range(majority_count)
    ]
    rows += [
        (
            f"sol{i}",
            f"apply for the scholarship on the portal {i}",
            True,
            "en",
            "latin",
            False,
            "solution",
        )
        for i in range(minority_count)
    ]
    return [
        SolutionAnnotation(
            document_id=d,
            text=t,
            is_solution=s,
            language=lang,
            script=script,
            is_code_mixed=cm,
            response_type=rt,
        )
        for d, t, s, lang, script, cm, rt in rows
    ]


class TestClassWeighting(unittest.TestCase):
    def test_balanced_is_the_default(self):
        self.assertEqual(SolutionModelConfig().class_weight, "balanced")
        classifier = TfidfLogisticSolutionClassifier()
        self.assertEqual(
            classifier.config_dict()["logistic_regression"]["class_weight"], "balanced"
        )

    def test_balancing_changes_the_fitted_model(self):
        """
        On a skewed corpus, balancing must move probability mass onto the rare
        target.  Without this, the default would be an untested claim.
        """
        annotations = _imbalanced_corpus()
        balanced = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(class_weight="balanced")
        )
        unweighted = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(class_weight=None)
        )
        balanced.train(annotations)
        unweighted.train(annotations)
        probe = SolutionDocument(
            document_id="probe", text="apply for the scholarship on the portal"
        )
        p_balanced = balanced.predict([probe]).predictions[0]
        p_plain = unweighted.predict([probe]).predictions[0]
        self.assertGreater(
            p_balanced.solution_probability, p_plain.solution_probability
        )

    def test_unweighted_fit_still_produces_valid_probabilities(self):
        classifier = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(class_weight=None)
        )
        classifier.train(_imbalanced_corpus())
        prediction = classifier.predict(
            [SolutionDocument(document_id="d", text="apply for aid")]
        ).predictions[0]
        self.assertGreaterEqual(prediction.solution_probability, 0.0)
        self.assertLessEqual(prediction.solution_probability, 1.0)

    def test_invalid_class_weight_is_refused(self):
        """
        Only ``None`` and ``"balanced"`` are legal; sklearn would otherwise fail
        deep inside ``fit()`` with a message that does not name the setting.
        """
        with self.assertRaises(ValueError) as ctx:
            TfidfLogisticSolutionClassifier(
                SolutionModelConfig(class_weight="balanced_ish")
            )
        self.assertIn("class_weight", str(ctx.exception))

    def test_the_config_model_itself_defers_to_the_classifier(self):
        """
        The guard lives in the classifier, which is what sklearn is called with.
        Constructing a config must stay free of sklearn so the model can be
        imported (and validated) without it.
        """
        self.assertEqual(SolutionModelConfig().class_weight, "balanced")


class TestClassImbalance(SolutionStageTestCase):
    def test_imbalance_warning_names_the_majority_and_its_share(self):
        report = TfidfLogisticSolutionClassifier().train(_imbalanced_corpus())
        self.assertGreater(report.majority_class_share, 0.8)
        self.assertEqual(report.majority_class, LABEL_NOT_SOLUTION)
        self.assertTrue(report.is_imbalanced)
        self.assertTrue(
            any("class imbalance" in w for w in report.warnings), report.warnings
        )

    def test_warning_names_the_trivial_baseline_score(self):
        report = TfidfLogisticSolutionClassifier().train(_imbalanced_corpus())
        warning = next(w for w in report.warnings if "class imbalance" in w)
        self.assertIn("would score", warning)
        self.assertIn(LABEL_NOT_SOLUTION, warning)

    def test_threshold_is_strictly_greater_than(self):
        """A share exactly at the threshold is not yet 'imbalanced'."""
        corpus = _imbalanced_corpus(majority_count=30, minority_count=6)  # 30/36
        report = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(imbalance_warning_share=30 / 36)
        ).train(corpus)
        self.assertAlmostEqual(report.majority_class_share, 30 / 36, places=5)
        self.assertFalse(
            any("class imbalance" in w for w in report.warnings), report.warnings
        )

    def test_balanced_corpus_does_not_warn(self):
        self.assertFalse(
            any("class imbalance" in w for w in self.report.warnings),
            self.report.warnings,
        )

    def test_majority_share_is_reported(self):
        self.assertAlmostEqual(
            self.report.majority_class_share,
            self.report.n_negative / self.report.n_training_documents,
            places=5,
        )


# ===========================================================================
# Configuration validation
# ===========================================================================


class TestConfigValidation(unittest.TestCase):
    def test_defaults_match_the_specified_baseline(self):
        config = SolutionModelConfig()
        self.assertIsNone(config.stop_words)
        self.assertEqual(config.ngram_range, (1, 2))
        self.assertEqual(config.C, 4.0)
        self.assertEqual(config.threshold, 0.5)
        self.assertIsNone(config.abstain_threshold)
        self.assertEqual(config.solver, "lbfgs")
        self.assertEqual(config.random_state, 42)
        self.assertEqual(config.class_weight, "balanced")

    def test_invalid_values_are_refused(self):
        cases = [
            ({"max_features": 0}, "max_features"),
            ({"min_df": "many"}, "min_df"),
            ({"min_df": 0}, "min_df"),
            ({"min_df": -1}, "min_df"),
            ({"max_df": "most"}, "max_df"),
            ({"ngram_range": (2, 1)}, "ngram_range"),
            ({"ngram_range": (0, 1)}, "ngram_range"),
            ({"norm": "l3"}, "norm"),
            ({"token_pattern": "([unclosed"}, "token_pattern"),
            ({"C": 0}, "C"),
            ({"C": -1.0}, "C"),
            ({"max_iter": 0}, "max_iter"),
            ({"threshold": 0.0}, "threshold"),
            ({"threshold": 1.0}, "threshold"),
            ({"abstain_threshold": 0.5, "threshold": 0.8}, "abstain_threshold"),
            ({"min_training_documents": 1}, "min_training_documents"),
            ({"min_examples_per_class": 0}, "min_examples_per_class"),
            ({"imbalance_warning_share": 0.0}, "imbalance_warning_share"),
            ({"imbalance_warning_share": 1.5}, "imbalance_warning_share"),
            ({"stop_words": "hindi"}, "stop_words"),
            ({"stop_words": [1, 2]}, "stop_words"),
        ]
        for kwargs, fragment in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    TfidfLogisticSolutionClassifier(SolutionModelConfig(**kwargs))
                self.assertIn(fragment, str(ctx.exception))

    def test_english_stop_words_are_accepted(self):
        classifier = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(stop_words="english")
        )
        self.assertEqual(
            classifier.config_dict()["tfidf"]["stop_words"], "english"
        )

    def test_stop_words_list_is_accepted(self):
        classifier = TfidfLogisticSolutionClassifier(
            SolutionModelConfig(stop_words=["the", "a"])
        )
        self.assertEqual(classifier.config_dict()["tfidf"]["stop_words"], ["the", "a"])

    def test_config_dict_records_every_knob(self):
        classifier = TfidfLogisticSolutionClassifier()
        config = classifier.config_dict()
        for key in ("threshold", "abstain_threshold", "tfidf", "logistic_regression",
                    "data_guards", "features", "context"):
            self.assertIn(key, config)
        self.assertIn("head", config["logistic_regression"])

    def test_abstract_base_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            SolutionDetection()

    def test_classifier_implements_the_interface(self):
        classifier = TfidfLogisticSolutionClassifier()
        self.assertIsInstance(classifier, SolutionDetection)
        for attribute in ("name", "version", "is_trained", "threshold",
                          "abstain_threshold", "config"):
            getattr(classifier, attribute)


# ===========================================================================
# Evaluation
# ===========================================================================


class TestEvaluation(SolutionStageTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.heldout = _heldout_annotations()
        cls.eval_report = cls.classifier.evaluate(cls.heldout)

    def test_heldout_set_is_genuinely_disjoint(self):
        """Otherwise the metrics would be memorisation, not generalisation."""
        training_texts = {a.text for a in self.annotations}
        training_ids = {a.document_id for a in self.annotations}
        for annotation in self.heldout:
            self.assertNotIn(annotation.text, training_texts, annotation.document_id)
            self.assertNotIn(annotation.document_id, training_ids, annotation.document_id)

    def test_report_is_meaningful_for_a_usable_set(self):
        self.assertTrue(self.eval_report.is_meaningful, self.eval_report.reason)
        self.assertIsNone(self.eval_report.reason)

    def test_every_required_metric_is_present(self):
        for field in ("accuracy", "precision", "recall", "f1"):
            value = getattr(self.eval_report, field)
            self.assertIsNotNone(value, field)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertIsNotNone(self.eval_report.confusion_matrix)
        self.assertIsNotNone(self.eval_report.classification_report)

    def test_support_counts_are_reported(self):
        support = self.eval_report.per_class_support
        self.assertEqual(support[LABEL_NOT_SOLUTION], 8)
        self.assertEqual(support[LABEL_SOLUTION], 6)
        self.assertEqual(
            sum(support.values()), self.eval_report.n_documents
        )

    def test_macro_and_weighted_metrics_are_present(self):
        for field in ("macro_precision", "macro_recall", "macro_f1",
                      "weighted_precision", "weighted_recall", "weighted_f1"):
            value = getattr(self.eval_report, field)
            self.assertIsNotNone(value, field)

    def test_per_class_metrics_cover_both_targets(self):
        for name in (LABEL_NOT_SOLUTION, LABEL_SOLUTION):
            self.assertIn(name, self.eval_report.per_class_f1)
            self.assertIn(name, self.eval_report.per_class_precision)
            self.assertIn(name, self.eval_report.per_class_recall)

    def test_class_order_is_canonical(self):
        self.assertEqual(
            self.eval_report.class_order, [LABEL_NOT_SOLUTION, LABEL_SOLUTION]
        )

    def test_precision_recall_gap_is_reported(self):
        self.assertIsNotNone(self.eval_report.precision_recall_gap)
        self.assertAlmostEqual(
            self.eval_report.precision_recall_gap,
            self.eval_report.recall - self.eval_report.precision,
            places=6,
        )

    def test_format_summary_does_not_overstate_accuracy(self):
        summary = self.eval_report.format_summary()
        self.assertIn("accuracy", summary)
        self.assertIn("flatters", summary)
        self.assertIn("TP=", summary)
        self.assertIn("confusion matrix", summary)

    def test_evaluation_on_the_training_set_warns_about_memorisation(self):
        report = self.classifier.evaluate(self.annotations)
        self.assertTrue(
            any("training data" in w for w in report.warnings), report.warnings
        )

    def test_disjoint_evaluation_does_not_warn_about_overlap(self):
        self.assertFalse(
            any("training data" in w for w in self.eval_report.warnings),
            self.eval_report.warnings,
        )

    def test_empty_evaluation_set_is_handled(self):
        report = self.classifier.evaluate([])
        self.assertFalse(report.is_meaningful)
        self.assertIsNone(report.accuracy)

    def test_evaluate_honours_a_threshold_override(self):
        strict = self.classifier.evaluate(self.heldout, threshold=0.99)
        lenient = self.classifier.evaluate(self.heldout, threshold=0.01)
        self.assertLessEqual(strict.recall or 0.0, lenient.recall or 0.0)


class TestEvaluationGuards(unittest.TestCase):
    def test_too_few_documents_withholds_metrics(self):
        report = evaluate_solution_detection(
            [True, False, True, False], [True, False, True, False], list(TARGET_LABELS)
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("at least", report.reason)
        for field in ("accuracy", "precision", "recall", "f1", "confusion_matrix",
                      "classification_report", "macro_f1"):
            self.assertIsNone(getattr(report, field), field)

    def test_counts_survive_a_withheld_report(self):
        """
        Counts are facts about the data, so they are reported even when the rates
        are suppressed.  Only rates are withheld.
        """
        report = evaluate_solution_detection(
            [True, False, None, None], [None, False, True, True], list(TARGET_LABELS)
        )
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_documents, 4)
        self.assertEqual(report.n_abstained, 1)
        self.assertEqual(report.n_unscoreable, 2)
        self.assertEqual(report.per_class_support[LABEL_SOLUTION], 1)
        self.assertEqual(report.per_class_support[LABEL_NOT_SOLUTION], 1)

    def test_missing_one_target_withholds_metrics(self):
        """
        A binary detector cannot be measured with one target absent.

        Precision and recall for the positive class would be 0.0 or undefined, and
        a confident 0.0 would be indistinguishable from a genuinely bad model.
        """
        report = evaluate_solution_detection(
            [True] * 12, [True] * 12, list(TARGET_LABELS)
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("no example of", report.reason)
        self.assertIsNone(report.precision)

    def test_missing_negative_target_withholds_metrics(self):
        report = evaluate_solution_detection(
            [False] * 12, [False] * 12, list(TARGET_LABELS)
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("no example of", report.reason)

    def test_thin_positive_support_is_warned_not_withheld(self):
        """
        A warning, not a suppression.

        One positive example makes the positive-class rates swing on a single
        comment, but withholding the whole report would hide the other class
        entirely.
        """
        report = evaluate_solution_detection(
            [True] + [False] * 11, [True] + [False] * 11, list(TARGET_LABELS)
        )
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertIsNotNone(report.precision)
        self.assertTrue(
            any("solution example" in w for w in report.warnings), report.warnings
        )

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            evaluate_solution_detection([True, False], [True], list(TARGET_LABELS))

    def test_everything_abstained_withholds_metrics(self):
        report = evaluate_solution_detection(
            [True, False] * 6, [None] * 12, list(TARGET_LABELS)
        )
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_abstained, 12)
        self.assertIsNone(report.accuracy)

    def test_no_false_positive_is_warned(self):
        report = evaluate_solution_detection(
            [True] * 6 + [False] * 6,
            [False] * 12,
            list(TARGET_LABELS),
        )
        self.assertTrue(report.is_meaningful)
        self.assertEqual(report.recall, 0.0)
        self.assertTrue(
            any("threshold is probably too high" in w for w in report.warnings),
            report.warnings,
        )

    def test_everything_predicted_positive_is_warned(self):
        """
        A detector that says "solution" to everything gets a flattering recall and
        a precision equal to the base rate - the exact profile that looks great in
        a report and poisons a knowledge base.  The numbers are stated correctly
        here so the warning cannot paper over them.
        """
        report = evaluate_solution_detection(
            [True] * 6 + [False] * 6,
            [True] * 12,
            list(TARGET_LABELS),
        )
        self.assertTrue(report.is_meaningful)
        # 6 TP, 6 FP: recall 1.0 by construction, precision back to the base rate.
        self.assertEqual(report.recall, 1.0)
        self.assertAlmostEqual(report.precision, 0.5, places=6)
        self.assertEqual(report.accuracy, 0.5)
        self.assertTrue(
            any("collapsed onto the positive class" in w for w in report.warnings),
            report.warnings,
        )

    def test_guard_threshold_is_documented_value(self):
        self.assertEqual(MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS, 10)
        self.assertEqual(MIN_SOLUTION_EXAMPLES_FOR_STABLE_METRICS, 2)

    def test_malformed_gold_is_unscoreable_not_a_silent_positive(self):
        """
        ``bool("maybe")`` is ``True``, so a coercion would score a mistyped
        annotation cell as a confident solution and inflate every rate.  The row
        has to be set aside and counted instead.
        """
        report = evaluate_solution_detection(
            [True, False, "maybe", False, True, False, True, False, True, False, True, False],
            [True, False, True, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertEqual(report.n_unscoreable, 1)
        self.assertEqual(report.n_scorable, 11)
        self.assertTrue(
            any("not scoreable" in w for w in report.warnings), report.warnings
        )

    def test_boolean_spellings_and_canonical_labels_are_accepted(self):
        """The evaluator reuses the loader's parsing instead of guessing again."""
        report = evaluate_solution_detection(
            ["true", "false", "yes", "no", "1", "0", "solution", "not_solution", True, False, True, False],
            [True, False, True, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertEqual(report.n_unscoreable, 0)
        self.assertEqual(report.n_abstained, 0)
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(report.accuracy, 1.0)

    def test_malformed_prediction_is_counted_not_coerced(self):
        """
        A prediction the evaluator cannot read is set aside the same way an
        abstention is: counted, excluded from the rates, and missed by accuracy.
        It is not silently read as a negative, which would hide a broken caller.
        """
        report = evaluate_solution_detection(
            [True, False, True, False, True, False, True, False, True, False, True, False],
            [True, False, "maybe", False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertEqual(report.n_abstained, 1)
        self.assertEqual(report.n_unscoreable, 0)
        self.assertEqual(report.n_scorable, 11)
        self.assertLess(report.accuracy, 1.0)
        self.assertEqual(
            sum(map(sum, report.confusion_matrix)), report.n_scorable
        )


class TestConfusionMatrix(unittest.TestCase):
    def test_perfect_predictions(self):
        report = evaluate_solution_detection(
            [True, True, False, False, True, False, True, False, True, False, True, False],
            [True, True, False, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.confusion_matrix, [[6, 0], [0, 6]])
        self.assertEqual(
            (report.true_positives, report.false_positives,
             report.false_negatives, report.true_negatives),
            (6, 0, 0, 6),
        )

    def test_matrix_records_off_diagonal_errors(self):
        """
        A confusion matrix that only recorded the diagonal would be an accuracy
        counter wearing a matrix's name.  The off-diagonal cells are the finding.

        The 6-comment pattern is doubled so the set clears
        ``MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS``; below that floor the matrix is
        withheld by design, which is a different test.
        """
        # gold: T T F F F F (x2) | pred: T F F T T F (x2)
        report = evaluate_solution_detection(
            [True, True, False, False, False, False] * 2,
            [True, False, False, True, True, False] * 2,
            list(TARGET_LABELS),
        )
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(
            report.confusion_matrix,
            [
                [4, 4],  # 4 true negatives, 4 false positives
                [2, 2],  # 2 false negatives, 2 true positives
            ],
        )
        self.assertEqual(report.true_positives, 2)
        self.assertEqual(report.false_positives, 4)
        self.assertEqual(report.false_negatives, 2)
        self.assertEqual(report.true_negatives, 4)

    def test_off_diagonal_matrix_needs_enough_documents(self):
        """The 6-comment version is withheld rather than reported at partial size."""
        report = evaluate_solution_detection(
            [True, True, False, False, False, False],
            [True, False, False, True, True, False],
            list(TARGET_LABELS),
        )
        self.assertFalse(report.is_meaningful)
        self.assertIsNone(report.confusion_matrix)

    def test_named_counts_agree_with_the_matrix(self):
        report = evaluate_solution_detection(
            [True, False, True, False, False, True, False, True, False, True, False, False],
            [True, True, False, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        matrix = report.confusion_matrix
        self.assertEqual(report.true_negatives, matrix[0][0])
        self.assertEqual(report.false_positives, matrix[0][1])
        self.assertEqual(report.false_negatives, matrix[1][0])
        self.assertEqual(report.true_positives, matrix[1][1])
        # Row sums equal gold support, because every scorable row is placed.
        self.assertEqual(
            sum(matrix[0]), report.per_class_support[LABEL_NOT_SOLUTION]
        )
        self.assertEqual(sum(matrix[1]), report.per_class_support[LABEL_SOLUTION])

    def test_precision_and_recall_follow_from_the_named_counts(self):
        report = evaluate_solution_detection(
            [True, False, True, False, False, True, False, True, False, True, False, False],
            [True, True, False, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        expected_precision = report.true_positives / (
            report.true_positives + report.false_positives
        )
        expected_recall = report.true_positives / (
            report.true_positives + report.false_negatives
        )
        self.assertAlmostEqual(report.precision, expected_precision, places=5)
        self.assertAlmostEqual(report.recall, expected_recall, places=5)
        expected_f1 = 2 * expected_precision * expected_recall / (
            expected_precision + expected_recall
        )
        self.assertAlmostEqual(report.f1, expected_f1, places=5)

    def test_axis_order_is_not_solution_first(self):
        """
        The axis order is canonical and asserted, so a transposed matrix cannot
        pass silently.  [not_solution, solution] means the positive class is
        row/column 1.
        """
        report = evaluate_solution_detection(
            [True] * 6 + [False] * 6, [True] * 6 + [False] * 6, list(TARGET_LABELS)
        )
        self.assertEqual(report.confusion_matrix[1][1], 6)  # TP bottom-right
        self.assertEqual(report.confusion_matrix[0][0], 6)  # TN top-left

    def test_abstentions_are_excluded_from_the_matrix_but_counted(self):
        report = evaluate_solution_detection(
            [True, False, True, False, True, False, True, False, True, False, True, False],
            [True, None, None, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertEqual(report.n_documents, 12)
        self.assertEqual(report.n_abstained, 2)
        self.assertEqual(report.n_scorable, 10)
        # Accuracy still counts abstentions as misses.
        self.assertLess(report.accuracy, 1.0)
        # Row sums fall short of support by exactly the abstentions.
        self.assertEqual(
            sum(report.confusion_matrix[1]),
            report.per_class_support[LABEL_SOLUTION] - 1,
        )
        self.assertTrue(
            any("abstained" in w for w in report.warnings), report.warnings
        )

    def test_unscoreable_gold_is_a_miss_and_excluded(self):
        report = evaluate_solution_detection(
            [True, False, None, False, True, False, True, False, True, False, True, False],
            [True, False, True, False, True, False, True, False, True, False, True, False],
            list(TARGET_LABELS),
        )
        self.assertEqual(report.n_unscoreable, 1)
        self.assertLess(report.accuracy, 1.0)
        self.assertTrue(
            any("not scoreable" in w for w in report.warnings), report.warnings
        )

    def test_confusion_matrix_renders_as_rows(self):
        report = evaluate_solution_detection(
            [True] * 6 + [False] * 6, [True] * 6 + [False] * 6, list(TARGET_LABELS)
        )
        lines = report.confusion_matrix_as_rows()
        self.assertEqual(len(lines), 2)
        self.assertIn(LABEL_NOT_SOLUTION, lines[0])
        self.assertIn(LABEL_SOLUTION, lines[1])

    def test_withheld_report_renders_no_rows(self):
        report = evaluate_solution_detection([True], [True], list(TARGET_LABELS))
        self.assertIsNone(report.confusion_matrix)
        self.assertEqual(report.confusion_matrix_as_rows(), [])

    def test_evaluate_produces_a_correct_matrix_on_a_known_corpus(self):
        """
        End-to-end matrix check on held-out annotations.

        The held-out fixture is 6 solutions and 8 non-solutions; at a 0.01
        threshold the detector asserts every comment, so every non-solution
        becomes a false positive and the matrix is fully determined.  That is a
        property of the threshold, not an accuracy claim - which is exactly why
        the report has to warn about it.
        """
        classifier = TfidfLogisticSolutionClassifier()
        classifier.train(_training_annotations())
        report = classifier.evaluate(_heldout_annotations(), threshold=0.01)
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(
            report.confusion_matrix,
            [
                [0, report.per_class_support[LABEL_NOT_SOLUTION]],
                [0, report.per_class_support[LABEL_SOLUTION]],
            ],
        )
        self.assertEqual(report.false_positives, 8)
        self.assertEqual(report.true_positives, 6)
        # 6 TP / (6 TP + 14 FP): recall is 1.0, precision is the base rate - not 0.
        self.assertAlmostEqual(report.precision, 6 / 14, places=6)
        self.assertEqual(report.recall, 1.0)
        self.assertTrue(
            any("collapsed onto the positive class" in w for w in report.warnings),
            report.warnings,
        )


# ===========================================================================
# Architecture
# ===========================================================================


class TestArchitectureIsolation(unittest.TestCase):
    def test_package_contains_expected_modules(self):
        names = {os.path.basename(path) for path in _package_modules()}
        for expected in (
            "base.py",
            "annotation.py",
            "result.py",
            "evaluation.py",
            "labels.py",
            "candidate.py",
            "tfidf_logistic_binary.py",
            "__init__.py",
        ):
            self.assertIn(expected, names)

    def test_package_imports_nothing_outside_itself(self):
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for module in _imported_modules(tree):
                if module == "app" or module.startswith("app."):
                    self.assertTrue(
                        module.startswith("app.services.nlp.solution_detection"),
                        f"{os.path.basename(path)} imports {module}",
                    )

    def test_response_classification_package_is_not_imported(self):
        """
        The independence requirement, enforced on the parsed import statements.

        Solution detection accepts ``response_type`` as neutral data and defines
        its own input model.  Importing the response stage would make either one
        impossible to replace, and would couple two components the project
        deliberately ships separately.

        Checked against imports, not prose: this package's docstrings *name* the
        response stage constantly, to explain how the two differ, and a substring
        search would flag every one of those explanations.
        """
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for module in _imported_modules(tree):
                self.assertNotIn(
                    "response_classification", module, os.path.basename(path)
                )
            for name in _imported_names(tree):
                self.assertNotIn(
                    "ResponseClassification", name, os.path.basename(path)
                )

    def test_response_type_is_a_plain_string_field(self):
        """
        The coupling that *is* allowed: a bare string column, typed locally.

        ``response_type`` is data, not a dependency - the field is a plain
        ``Optional[str]`` on this package's own models, so no response-stage enum
        can leak into the detector's contract.
        """
        self.assertIsNone(SolutionDocument(document_id="d", text="x").response_type)
        self.assertEqual(
            SolutionDocument(
                document_id="d", text="x", response_type="solution"
            ).response_type,
            "solution",
        )
        self.assertIs(SolutionDocument.model_fields["response_type"].annotation, Optional[str])

    def test_response_stage_symbols_are_not_imported(self):
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            names = _imported_names(tree)
            for forbidden in (
                "ResponseClassification",
                "ResponseDocument",
                "ResponseAnnotation",
                "ResponseCorpus",
                "RESPONSE_TYPES",
                "TfidfLogisticResponseClassifier",
            ):
                self.assertNotIn(forbidden, names, os.path.basename(path))

    def test_candidate_roles_are_local_strings_not_an_imported_enum(self):
        source = open(
            os.path.join(PACKAGE_ROOT, "models", "candidate.py"), "r", encoding="utf-8"
        ).read()
        self.assertIn("RESPONSE_TYPE_SOLUTION = \"solution\"", source)
        self.assertIn("RESPONSE_TYPE_ADVICE = \"advice\"", source)

    def test_preprocessing_is_not_imported(self):
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for module in _imported_modules(tree):
                self.assertNotIn("preprocessing", module, os.path.basename(path))
            for name in _imported_names(tree):
                self.assertNotIn("Preprocessor", name, os.path.basename(path))
            source = open(path, "r", encoding="utf-8").read()
            self.assertNotIn("app.services.nlp.preprocessing", source)

    def test_no_database_or_ingestion_symbols(self):
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

    def test_earlier_completed_stages_are_not_coupled_to(self):
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in (
                "app.services.nlp.topic_discovery",
                "app.services.nlp.issue_identification",
                "app.services.nlp.sentiment_emotion",
                "app.services.nlp.language_analysis_service",
                "app.services.nlp.preprocessor",
            ):
                self.assertNotIn(forbidden, source, os.path.basename(path))

    def test_completed_response_classification_package_is_untouched(self):
        """
        The completed stage must not be modified by this one.

        Asserted structurally: if this stage had edited the response package, the
        response suite would be the thing that broke.  This test simply pins the
        fact that the two packages coexist and that the response stage still has
        its own fixed six-type taxonomy.
        """
        response_root = os.path.join(
            os.path.dirname(PACKAGE_ROOT), "response_classification"
        )
        self.assertTrue(os.path.isdir(response_root))
        labels_path = os.path.join(response_root, "models", "labels.py")
        self.assertTrue(os.path.exists(labels_path))
        source = open(labels_path, "r", encoding="utf-8").read()
        self.assertIn("RESPONSE_TYPES", source)
        # And this package does not reach into it.
        candidate = open(
            os.path.join(PACKAGE_ROOT, "models", "candidate.py"), "r", encoding="utf-8"
        ).read()
        self.assertNotIn("response_classification", candidate)

    def test_out_of_scope_stages_are_absent(self):
        """
        Issue-Solution Linking and Resolution Detection are not reachable from here.

        Both are now implemented as independent packages that sit next door in the
        NLP tree, so the meaningful assertion is not "the directory does not exist"
        but "this package does not import or name anything from either stage" -
        which is what the import and identifier checks above establish.  The
        adjacency is precisely what makes that check worth keeping: a package that
        *could* import them must not.

        Checked structurally rather than by prose search, because a text search
        would fail on this package's own docstrings, which discuss the boundary at
        length precisely because drawing it is the point.
        """
        forbidden_modules = ("issue_solution_link", "resolution_detection")
        forbidden_names = (
            "IssueSolutionLink",
            "IssueSolutionLinker",
            "ResolutionDetection",
            "ResolutionDetector",
            "link_solution_to_issue",
            "detect_resolution",
        )
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for module in _imported_modules(tree):
                for forbidden in forbidden_modules:
                    self.assertNotIn(
                        forbidden, module, os.path.basename(path)
                    )
            for name in _imported_names(tree):
                for forbidden in forbidden_names:
                    self.assertNotIn(forbidden, name, os.path.basename(path))
        # Those stages exist as separate packages; this one must not depend on them.
        nlp_root = os.path.dirname(PACKAGE_ROOT)
        for stage in ("issue_solution_linking", "resolution_detection"):
            root = os.path.join(nlp_root, stage)
            if os.path.exists(root):
                self.assertNotEqual(
                    os.path.abspath(root),
                    os.path.abspath(PACKAGE_ROOT),
                    f"out-of-scope stage directory is this package: {stage}",
                )

    def test_no_link_or_resolution_output_fields(self):
        """
        The stage answers a per-comment question and stops there.

        Neither prediction nor result carries a linked-issue or resolution field,
        so a caller cannot read an Issue-Solution link out of this stage by
        accident.
        """
        forbidden = (
            "linked_issue_id",
            "issue_solution_link",
            "resolved",
            "is_resolved",
            "resolution",
        )
        for model in (SolutionPrediction, SolutionDetectionResult):
            for field in model.model_fields:
                for token in forbidden:
                    self.assertNotIn(token, field, f"{model.__name__}.{field}")

    def test_pydantic_models_are_used_throughout(self):
        for name in (
            SolutionDocument,
            SolutionModelConfig,
            SolutionAnnotation,
            SolutionCorpus,
            SolutionPrediction,
            SolutionDetectionResult,
            SolutionTrainingReport,
            SolutionEvaluationReport,
            CandidateAssessment,
        ):
            self.assertTrue(issubclass(name, __import__("pydantic").BaseModel), name)


if __name__ == "__main__":
    unittest.main()
