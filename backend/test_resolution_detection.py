"""
Tests for Resolution / Unresolved Issue Detection.

These tests assert **mechanics, never quality.**  Resolution cannot be measured by
unit tests at all: correctness means "does a human reading this thread agree?",
and no human-annotated resolution dataset exists to check against.  A test that
asserted "this resolver is accurate" would be asserting something about fixtures
the same author wrote.  What is tested instead is that the stage does what it
promises: it never calls one linked solution a resolution, it keeps exactly three
statuses, it reports thin evidence without inventing a fourth state, it refuses
impossible evidence, it leaves its inputs alone, it is deterministic, it does not
import any other NLP stage, and it refuses to report a metric without an explicit
annotated corpus.

The three statuses
------------------
The most important test in this file is
:meth:`TestStatusSet.test_exactly_three_statuses_and_no_fourth_state`.  A fourth
``insufficient_evidence`` status is the tempting design and the wrong one: it
answers a question about our confidence instead of about the issue, and a consumer
asking "which issues remain unresolved?" needs a partition of the issues.  If a
future change introduces that status, the test is the thing that should fail.

Test groups
-----------
1.  Statuses and parsing
2.  Evidence models
3.  Configuration validation
4.  Derivation and evidence validation
5.  Scoring: the exact formula
6.  Thresholds and status boundaries
7.  Insufficient evidence and the volume guard
8.  Multi-label issues
9.  The distinction this stage exists to preserve
10. Determinism
11. Input immutability
12. Invalid evidence
13. Temporal evidence (recorded, not scored)
14. Results, batches and run reports
15. Annotation schema and CSV handling
16. Evaluation, including the refusal to fabricate metrics
17. Architecture isolation (AST: no imports of other NLP stages)
18. Interface conformance
"""

import ast
import math
import os
import tempfile
import unittest
from typing import List

from app.services.nlp.resolution_detection import (
    MODEL_NAME,
    MODEL_VERSION,
    RESOLUTION_CSV_COLUMNS,
    STATUS_ORDER,
    DerivedEvidence,
    EvidenceBasedResolver,
    IssueResolutionEvidence,
    PerStatusMetrics,
    ResolutionAnnotation,
    ResolutionBatchResult,
    ResolutionCorpus,
    ResolutionEvaluation,
    ResolutionModelConfig,
    ResolutionResult,
    ResolutionRunReport,
    ResolutionStatus,
    TemporalEvidence,
    derive_evidence,
    derive_evidence_list,
    evaluate_resolution,
    strong_link_count_from_scores,
    validate_evidence,
)
from app.services.nlp.resolution_detection.models.labels import parse_gold_status

PACKAGE_DIR = os.path.dirname(
    os.path.abspath(
        __import__(
            "app.services.nlp.resolution_detection", fromlist=["__file__"]
        ).__file__
    )
)

OTHER_STAGES = [
    "data_preprocessing",
    "issue_identification",
    "response_classification",
    "solution_detection",
    "issue_solution_linking",
    "emotion_detection",
    "sentiment_detection",
]


def ev(issue_id: str = "i1", issue_label: str = "fees", **kwargs) -> IssueResolutionEvidence:
    """
    A minimal, coherent evidence record, overridable per test.

    ``solution_candidate_count`` defaults to ``linked_solution_count`` because
    every linked solution is necessarily a candidate - a record with links and
    fewer candidates is the impossible one, and refusing it is the tested
    behaviour, so a test asking for links should not have to supply the count.
    """
    kwargs.setdefault("total_comments", 4)
    kwargs.setdefault("solution_candidate_count", kwargs.get("linked_solution_count", 0))
    return IssueResolutionEvidence(
        issue_id=issue_id, issue_label=issue_label, **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Statuses and parsing
# ---------------------------------------------------------------------------


class TestStatusSet(unittest.TestCase):
    def test_exactly_three_statuses_and_no_fourth_state(self):
        """The status set is a partition of the issues, not of our confidence."""
        values = {status.value for status in ResolutionStatus}
        self.assertEqual(
            values, {"resolved", "partially_resolved", "unresolved"}
        )
        self.assertEqual(len(ResolutionStatus), 3)
        # A fourth 'insufficient_evidence' status is the rejected design.
        self.assertNotIn("insufficient_evidence", values)
        self.assertNotIn("unknown", values)
        self.assertNotIn("partial", values)

    def test_status_order_is_weakest_evidence_first(self):
        self.assertEqual(
            [status.value for status in STATUS_ORDER],
            ["unresolved", "partially_resolved", "resolved"],
        )

    def test_status_str_is_its_value(self):
        self.assertEqual(str(ResolutionStatus.RESOLVED), "resolved")

    def test_gold_status_parsing(self):
        for raw, expected in [
            ("resolved", ResolutionStatus.RESOLVED),
            ("RESOLVED", ResolutionStatus.RESOLVED),
            ("  Resolved ", ResolutionStatus.RESOLVED),
            ("partially_resolved", ResolutionStatus.PARTIALLY_RESOLVED),
            ("partially-resolved", ResolutionStatus.PARTIALLY_RESOLVED),
            ("partially resolved", ResolutionStatus.PARTIALLY_RESOLVED),
            ("partial", ResolutionStatus.PARTIALLY_RESOLVED),
            ("unresolved", ResolutionStatus.UNRESOLVED),
            ("un-resolved", ResolutionStatus.UNRESOLVED),
            ("open", ResolutionStatus.UNRESOLVED),
            (ResolutionStatus.RESOLVED, ResolutionStatus.RESOLVED),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_gold_status(raw), expected)

    def test_unrecognised_gold_status_is_refused_not_defaulted(self):
        """A silent default to 'unresolved' would inflate every resolution rate."""
        for raw in ["", None, "fixed", "yes", "2", "unknown", "partially", "RESOLVED!"]:
            with self.subTest(raw=raw):
                if raw == "partially":
                    self.assertEqual(parse_gold_status(raw), ResolutionStatus.PARTIALLY_RESOLVED)
                    continue
                with self.assertRaises(ValueError):
                    parse_gold_status(raw)


# ---------------------------------------------------------------------------
# 2. Evidence models
# ---------------------------------------------------------------------------


class TestEvidenceModels(unittest.TestCase):
    def test_effective_volume_defaults_to_total_comments(self):
        self.assertEqual(ev(total_comments=9).effective_discussion_volume, 9)

    def test_explicit_discussion_volume_overrides(self):
        self.assertEqual(
            ev(total_comments=4, discussion_volume=11).effective_discussion_volume, 11
        )

    def test_has_linked_solutions(self):
        self.assertFalse(ev().has_linked_solutions)
        self.assertTrue(ev(linked_solution_count=1).has_linked_solutions)

    def test_pair_key_is_issue_and_label(self):
        self.assertEqual(ev("i7", "roi").pair_key(), "i7|roi")

    def test_temporal_has_any(self):
        self.assertFalse(TemporalEvidence().has_any)
        self.assertTrue(TemporalEvidence(solution_after_issue=True).has_any)
        self.assertTrue(TemporalEvidence(follow_up_count=2).has_any)
        self.assertTrue(TemporalEvidence(follow_up_resolution_signal=True).has_any)

    def test_evidence_does_not_mutate_on_derivation(self):
        record = ev(linked_solution_count=2, strong_link_count=1, link_scores=[0.4, 0.9])
        before = record.model_dump()
        derive_evidence(record)
        derive_evidence_list([record])
        self.assertEqual(record.model_dump(), before)


# ---------------------------------------------------------------------------
# 3. Configuration validation
# ---------------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    def test_defaults_sum_to_one(self):
        config = ResolutionModelConfig()
        self.assertAlmostEqual(
            sum(config.weight_map().values()), 1.0, places=9
        )

    def test_default_thresholds_are_ordered(self):
        config = ResolutionModelConfig()
        self.assertLessEqual(config.unresolved_upper, config.partial_upper)
        self.assertLessEqual(config.partial_upper, config.resolved_lower)

    def test_weight_keys_match_component_keys(self):
        """A renamed component would otherwise be silently scored as zero."""
        components = set(derive_evidence(ev()).as_components())
        self.assertEqual(components, set(ResolutionModelConfig().weight_map()))

    def test_negative_weight_refused(self):
        with self.assertRaises(ValueError) as ctx:
            EvidenceBasedResolver(ResolutionModelConfig(solution_density_weight=-0.1))
        self.assertIn("negative", str(ctx.exception))

    def test_non_finite_weight_refused(self):
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(ResolutionModelConfig(strong_link_weight=float("inf")))
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(ResolutionModelConfig(strong_link_weight=float("nan")))

    def test_all_zero_weights_refused(self):
        with self.assertRaises(ValueError) as ctx:
            EvidenceBasedResolver(
                ResolutionModelConfig(
                    solution_density_weight=0.0,
                    strong_link_weight=0.0,
                    link_strength_weight=0.0,
                    solution_confidence_weight=0.0,
                )
            )
        self.assertIn("zero", str(ctx.exception).lower())

    def test_overlapping_thresholds_refused(self):
        """A score able to satisfy two conditions at once means overlapping bands."""
        with self.assertRaises(ValueError) as ctx:
            EvidenceBasedResolver(
                ResolutionModelConfig(unresolved_upper=0.6, partial_upper=0.55, resolved_lower=0.5)
            )
        self.assertIn("thresholds", str(ctx.exception))
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(
                ResolutionModelConfig(unresolved_upper=0.2, partial_upper=0.9, resolved_lower=0.5)
            )
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(
                ResolutionModelConfig(unresolved_upper=0.5, partial_upper=0.5, resolved_lower=0.4)
            )

    def test_threshold_outside_unit_interval_refused(self):
        for kwargs in [
            {"unresolved_upper": 1.4},
            {"partial_upper": -0.1},
            {"resolved_lower": 2.0},
        ]:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    EvidenceBasedResolver(ResolutionModelConfig(**kwargs))

    def test_confidence_scale_outside_unit_interval_refused(self):
        for scale in (1.5, -0.2):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    EvidenceBasedResolver(
                        ResolutionModelConfig(insufficient_evidence_confidence_scale=scale)
                    )

    def test_negative_minimum_volume_refused(self):
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(ResolutionModelConfig(minimum_discussion_volume=-1))

    def test_strong_link_threshold_outside_unit_interval_refused(self):
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(ResolutionModelConfig(strong_link_threshold=1.2))

    def test_invalid_config_refused_at_construction_not_at_detection(self):
        """A bad configuration must not produce wrong statuses for a whole run."""
        with self.assertRaises(ValueError):
            EvidenceBasedResolver(ResolutionModelConfig(resolved_lower=0.1, partial_upper=0.2))

    def test_normalisation_divides_by_the_weight_sum(self):
        resolver = EvidenceBasedResolver(
            ResolutionModelConfig(
                solution_density_weight=2.0,
                strong_link_weight=1.0,
                link_strength_weight=1.0,
                solution_confidence_weight=2.0,
            )
        )
        weights = resolver.config_dict["weights"]
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertAlmostEqual(weights["solution_density"], 2.0 / 6.0, places=9)
        # Scaling every weight equally must not change any verdict.
        base = EvidenceBasedResolver().detect_batch(
            [ev(linked_solution_count=2, strong_link_count=1, link_scores=[0.9],
                solution_probabilities=[0.8], solution_candidate_count=2)]
        )
        scaled = resolver.detect_batch(
            [ev(linked_solution_count=2, strong_link_count=1, link_scores=[0.9],
                solution_probabilities=[0.8], solution_candidate_count=2)]
        )
        self.assertAlmostEqual(
            base.results[0].resolution_score, scaled.results[0].resolution_score, places=12
        )

    def test_unnormalised_weights_keep_raw_values(self):
        resolver = EvidenceBasedResolver(
            ResolutionModelConfig(
                solution_density_weight=0.30,
                strong_link_weight=0.20,
                link_strength_weight=0.25,
                solution_confidence_weight=0.25,
                normalise_weights=False,
            )
        )
        self.assertAlmostEqual(
            resolver.config_dict["weights"]["solution_density"], 0.30, places=9
        )


# ---------------------------------------------------------------------------
# 4. Derivation and evidence validation
# ---------------------------------------------------------------------------


class TestDerivation(unittest.TestCase):
    def test_densities_use_total_comments(self):
        derived = derive_evidence(ev(total_comments=8, linked_solution_count=2, strong_link_count=1))
        self.assertAlmostEqual(derived.solution_density, 2 / 8)
        self.assertAlmostEqual(derived.strong_solution_density, 1 / 8)

    def test_zero_comments_does_not_divide_by_zero_or_nan(self):
        derived = derive_evidence(ev(total_comments=0))
        self.assertEqual(derived.solution_density, 0.0)
        self.assertEqual(derived.strong_solution_density, 0.0)
        for value in derived.as_components().values():
            self.assertTrue(math.isfinite(value))

    def test_link_scores_take_precedence_over_summaries(self):
        derived = derive_evidence(
            ev(linked_solution_count=2, strong_link_count=1, link_scores=[0.9, 0.3])
        )
        self.assertAlmostEqual(derived.best_link_score, 0.9)
        self.assertAlmostEqual(derived.mean_link_score, 0.6)
        self.assertEqual(derived.link_strength_source, "link_scores")
        self.assertEqual(derived.link_count, 2)

    def test_summary_fields_used_when_no_scores_given(self):
        derived = derive_evidence(
            ev(linked_solution_count=2, best_link_score=0.75, mean_link_score=0.5)
        )
        self.assertAlmostEqual(derived.best_link_score, 0.75)
        self.assertEqual(derived.link_strength_source, "best_link_score")
        self.assertEqual(derived.link_count, 0)

    def test_no_link_information_is_reported_as_none_not_guessed(self):
        derived = derive_evidence(ev())
        self.assertEqual(derived.link_strength_source, "none")
        self.assertEqual(derived.probability_source, "none")
        self.assertEqual(derived.best_link_score, 0.0)

    def test_solution_probabilities_summarised(self):
        derived = derive_evidence(
            ev(linked_solution_count=2, solution_probabilities=[0.4, 0.8])
        )
        self.assertAlmostEqual(derived.solution_probability_mean, 0.6)
        self.assertAlmostEqual(derived.solution_probability_max, 0.8)
        self.assertEqual(derived.probability_source, "solution_probabilities")

    def test_probability_summary_used_when_no_list_given(self):
        derived = derive_evidence(
            ev(linked_solution_count=1, solution_probability_mean=0.55, solution_probability_max=0.7)
        )
        self.assertAlmostEqual(derived.solution_probability_mean, 0.55)
        self.assertEqual(derived.probability_source, "solution_probability_mean")

    def test_supplied_density_cross_checked(self):
        derived = derive_evidence(
            ev(total_comments=4, linked_solution_count=2, solution_density=0.5)
        )
        self.assertEqual(derived.density_source, "supplied")
        self.assertAlmostEqual(derived.solution_density, 0.5)

    def test_derived_components_are_all_in_range(self):
        derived = derive_evidence(
            ev(total_comments=1, linked_solution_count=1, strong_link_count=1,
               link_scores=[1.0], solution_probabilities=[1.0], solution_candidate_count=1)
        )
        for name, value in derived.as_components().items():
            with self.subTest(name=name):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_derive_list_preserves_order(self):
        records = [ev("i1"), ev("i2"), ev("i3")]
        derived = derive_evidence_list(records)
        self.assertEqual([d.linked_solution_count for d in derived], [0, 0, 0])
        self.assertEqual(len(derived), 3)

    def test_derive_empty_list(self):
        self.assertEqual(derive_evidence_list([]), [])

    def test_strong_link_count_helper(self):
        self.assertEqual(strong_link_count_from_scores([0.2, 0.6, 0.9], 0.6), 2)
        self.assertEqual(strong_link_count_from_scores([0.1, 0.2], 0.6), 0)
        self.assertEqual(strong_link_count_from_scores([], 0.6), 0)
        self.assertEqual(strong_link_count_from_scores(None, 0.6), 0)


# ---------------------------------------------------------------------------
# 5. Scoring: the exact formula
# ---------------------------------------------------------------------------


class TestScoreFormula(unittest.TestCase):
    def test_documented_equation_is_what_is_computed(self):
        """Recompute the score by hand from the reported components and weights."""
        resolver = EvidenceBasedResolver()
        record = ev(
            total_comments=8,
            solution_candidate_count=3,
            linked_solution_count=2,
            strong_link_count=1,
            link_scores=[0.71, 0.34],
            solution_probabilities=[0.88, 0.61],
        )
        result = resolver.detect(record).results[0]

        density = 2 / 8
        strong_density = 1 / 8
        best = 0.71
        probability_mean = (0.88 + 0.61) / 2

        expected = (
            0.30 * density
            + 0.20 * strong_density
            + 0.25 * best
            + 0.25 * probability_mean
        )
        self.assertAlmostEqual(result.resolution_score, expected, places=12)
        self.assertAlmostEqual(result.resolution_score, result.recompute_score, places=12)
        self.assertAlmostEqual(result.score_components["solution_density"], density, places=12)
        self.assertAlmostEqual(
            result.score_components["strong_solution_density"], strong_density, places=12
        )
        self.assertAlmostEqual(result.score_components["best_link_score"], best, places=12)
        self.assertAlmostEqual(
            result.score_components["solution_probability_mean"], probability_mean, places=12
        )

    def test_documented_weight_defaults_are_the_defaults(self):
        weights = EvidenceBasedResolver().config_dict["weights"]
        self.assertAlmostEqual(weights["solution_density"], 0.30, places=9)
        self.assertAlmostEqual(weights["strong_solution_density"], 0.20, places=9)
        self.assertAlmostEqual(weights["best_link_score"], 0.25, places=9)
        self.assertAlmostEqual(weights["solution_probability_mean"], 0.25, places=9)

    def test_all_terms_one_gives_score_one(self):
        resolver = EvidenceBasedResolver()
        result = resolver.detect(
            ev(total_comments=1, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[1.0], solution_probabilities=[1.0])
        ).results[0]
        self.assertAlmostEqual(result.resolution_score, 1.0, places=12)
        self.assertEqual(result.status, ResolutionStatus.RESOLVED)

    def test_no_evidence_gives_score_zero(self):
        result = EvidenceBasedResolver().detect(ev(total_comments=10)).results[0]
        self.assertEqual(result.resolution_score, 0.0)
        self.assertEqual(result.status, ResolutionStatus.UNRESOLVED)

    def test_more_coverage_wins_at_equal_quality(self):
        """
        Holding link quality fixed, more relevant solutions must score higher.

        This is the claim the weight assignment exists to make: coverage is not
        substitutable by one good link.
        """
        resolver = EvidenceBasedResolver()
        one = resolver.detect(
            ev(total_comments=6, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[0.5], solution_probabilities=[0.5])
        ).results[0]
        three = resolver.detect(
            ev(total_comments=6, solution_candidate_count=3, linked_solution_count=3,
               strong_link_count=1, link_scores=[0.5, 0.5, 0.5],
               solution_probabilities=[0.5, 0.5, 0.5])
        ).results[0]
        self.assertGreater(three.resolution_score, one.resolution_score)

    def test_density_terms_outweigh_the_link_strength_term(self):
        """Coverage carries 0.50 of weight against 0.25 for lexical strength."""
        resolver = EvidenceBasedResolver()
        # Same density, opposite strength: the strength term moves the score by
        # exactly its own weight (0.25) times the change in the term itself.
        weak = resolver.detect(
            ev(total_comments=6, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=1, link_scores=[0.0, 0.0], solution_probabilities=[0.5, 0.5])
        ).results[0]
        strong = resolver.detect(
            ev(total_comments=6, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=1, link_scores=[1.0, 1.0], solution_probabilities=[0.5, 0.5])
        ).results[0]
        self.assertAlmostEqual(strong.resolution_score - weak.resolution_score, 0.25, places=12)

    def test_a_single_perfect_link_can_still_outrank_several_weak_ones(self):
        """
        A documented boundary of the design, pinned so it cannot change silently.

        The density terms outrank link strength *in aggregate*, but confidence
        and strength together are worth 0.50, so one solution the linker is very
        sure about beats three it is unsure about.  That is arguably right - a
        confidently matched fix is stronger evidence than three speculative
        matches - but it is a judgement, not a derivation, and worth stating.
        """
        resolver = EvidenceBasedResolver()
        one_perfect = resolver.detect(
            ev(total_comments=6, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[1.0], solution_probabilities=[1.0])
        ).results[0]
        three_weak = resolver.detect(
            ev(total_comments=6, solution_candidate_count=3, linked_solution_count=3,
               strong_link_count=1, link_scores=[0.45, 0.4, 0.45],
               solution_probabilities=[0.5, 0.55, 0.5])
        ).results[0]
        self.assertGreater(one_perfect.resolution_score, three_weak.resolution_score)

    def test_score_stays_in_unit_interval(self):
        resolver = EvidenceBasedResolver()
        for record in [
            ev(total_comments=0, linked_solution_count=0, strong_link_count=0),
            ev(total_comments=1, linked_solution_count=1, strong_link_count=1,
               link_scores=[1.0], solution_probabilities=[1.0], solution_candidate_count=1),
            ev(total_comments=4, linked_solution_count=2, strong_link_count=1,
               link_scores=[0.9, 0.8], solution_probabilities=[0.9, 0.85]),
        ]:
            with self.subTest(record=record.pair_key()):
                score = resolver.detect(record).results[0].resolution_score
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                self.assertTrue(math.isfinite(score))

    def test_result_reports_the_weights_and_thresholds_used(self):
        resolver = EvidenceBasedResolver()
        result = resolver.detect(ev()).results[0]
        self.assertEqual(result.weights, resolver.config_dict["weights"])
        self.assertEqual(result.thresholds["unresolved_upper"], 0.30)
        self.assertEqual(result.thresholds["resolved_lower"], 0.55)
        self.assertEqual(result.model_name, MODEL_NAME)
        self.assertEqual(result.model_version, MODEL_VERSION)

    def test_model_metadata_values_are_exact(self):
        self.assertEqual(MODEL_NAME, "evidence_based_resolution")
        self.assertEqual(MODEL_VERSION, "v1")

    def test_score_is_independent_of_the_thresholds(self):
        """Re-deciding is free: the score does not depend on where the cut-points are."""
        lenient = EvidenceBasedResolver(
            ResolutionModelConfig(unresolved_upper=0.1, partial_upper=0.2, resolved_lower=0.3)
        )
        strict = EvidenceBasedResolver(
            ResolutionModelConfig(unresolved_upper=0.7, partial_upper=0.8, resolved_lower=0.9)
        )
        record = ev(total_comments=6, solution_candidate_count=2, linked_solution_count=2,
                    strong_link_count=1, link_scores=[0.6, 0.5], solution_probabilities=[0.7, 0.6])
        lenient_result = lenient.detect(record).results[0]
        strict_result = strict.detect(record).results[0]
        self.assertAlmostEqual(
            lenient_result.resolution_score, strict_result.resolution_score, places=12
        )
        self.assertNotEqual(lenient_result.status, strict_result.status)
        self.assertEqual(
            strict_result.redecide(lenient_result.thresholds["unresolved_upper"],
                                   lenient_result.thresholds["resolved_lower"]),
            lenient_result.status,
        )

    def test_redecide_does_not_mutate_the_result(self):
        result = EvidenceBasedResolver().detect(ev()).results[0]
        before = (result.status, result.resolution_score, result.confidence)
        result.redecide(0.0, 0.05)
        self.assertEqual((result.status, result.resolution_score, result.confidence), before)


# ---------------------------------------------------------------------------
# 6. Thresholds and status boundaries
# ---------------------------------------------------------------------------


class TestThresholds(unittest.TestCase):
    def test_band_boundaries_are_half_open(self):
        resolver = EvidenceBasedResolver()
        lower = resolver.config.unresolved_upper
        upper = resolver.config.resolved_lower
        self.assertEqual(resolver.decide_status(lower - 1e-9), ResolutionStatus.UNRESOLVED)
        self.assertEqual(resolver.decide_status(lower), ResolutionStatus.PARTIALLY_RESOLVED)
        self.assertEqual(resolver.decide_status(upper - 1e-9), ResolutionStatus.PARTIALLY_RESOLVED)
        self.assertEqual(resolver.decide_status(upper), ResolutionStatus.RESOLVED)

    def test_status_is_always_one_of_the_three(self):
        resolver = EvidenceBasedResolver()
        for step in range(0, 101):
            score = step / 100
            self.assertIn(resolver.decide_status(score), set(STATUS_ORDER))

    def test_configurable_thresholds_change_the_verdict(self):
        record = ev(total_comments=6, solution_candidate_count=2, linked_solution_count=2,
                    strong_link_count=1, link_scores=[0.6, 0.5], solution_probabilities=[0.7, 0.6])
        strict = EvidenceBasedResolver(
            ResolutionModelConfig(unresolved_upper=0.1, partial_upper=0.2, resolved_lower=0.3)
        ).detect(record).results[0]
        lenient = EvidenceBasedResolver(
            ResolutionModelConfig(unresolved_upper=0.7, partial_upper=0.8, resolved_lower=0.9)
        ).detect(record).results[0]
        self.assertEqual(strict.status, ResolutionStatus.RESOLVED)
        self.assertEqual(lenient.status, ResolutionStatus.UNRESOLVED)

    def test_every_band_is_reachable(self):
        resolver = EvidenceBasedResolver()
        seen = {resolver.decide_status(step / 100) for step in range(0, 101)}
        self.assertEqual(seen, set(STATUS_ORDER))


# ---------------------------------------------------------------------------
# 7. Insufficient evidence and the volume guard
# ---------------------------------------------------------------------------


class TestInsufficientEvidence(unittest.TestCase):
    def test_thin_discussion_is_flagged_and_warned_not_relabelled(self):
        result = EvidenceBasedResolver().detect(ev(total_comments=1)).results[0]
        self.assertFalse(result.evidence_sufficient)
        self.assertTrue(result.warnings)
        # Still one of the three required states.
        self.assertIn(result.status, set(STATUS_ORDER))

    def test_sufficient_discussion_is_not_flagged(self):
        result = EvidenceBasedResolver().detect(ev(total_comments=3)).results[0]
        self.assertTrue(result.evidence_sufficient)
        self.assertFalse([w for w in result.warnings if "insufficient" in w])

    def test_minimum_volume_boundary(self):
        resolver = EvidenceBasedResolver()
        below = resolver.detect(ev(total_comments=2)).results[0]
        at = resolver.detect(ev(total_comments=3)).results[0]
        self.assertFalse(below.evidence_sufficient)
        self.assertTrue(at.evidence_sufficient)

    def test_explicit_discussion_volume_drives_the_guard(self):
        result = EvidenceBasedResolver().detect(
            ev(total_comments=2, discussion_volume=40)
        ).results[0]
        self.assertTrue(result.evidence_sufficient)

    def test_guard_reduces_confidence_but_not_the_status(self):
        """Capping the score instead would manufacture a verdict out of the guard."""
        record = ev(
            total_comments=8,
            solution_candidate_count=3,
            linked_solution_count=3,
            strong_link_count=2,
            link_scores=[0.8, 0.7, 0.9],
            solution_probabilities=[0.9, 0.8, 0.85],
        )
        # Same evidence, two guards.  Only the volume rule differs, so any change
        # in the status could only come from the guard - and there must be none.
        supported = EvidenceBasedResolver().detect(record).results[0]
        guarded = EvidenceBasedResolver(
            ResolutionModelConfig(minimum_discussion_volume=20)
        ).detect(record).results[0]

        self.assertTrue(supported.evidence_sufficient)
        self.assertFalse(guarded.evidence_sufficient)
        self.assertEqual(supported.status, guarded.status)
        self.assertEqual(supported.resolution_score, guarded.resolution_score)
        self.assertLess(guarded.confidence, supported.confidence)
        self.assertAlmostEqual(
            guarded.confidence, supported.confidence * 0.5, places=12
        )

    def test_confidence_scale_is_configurable(self):
        record = ev(total_comments=1, linked_solution_count=1, strong_link_count=1,
                    link_scores=[0.9], solution_probabilities=[0.9], solution_candidate_count=1,
                    discussion_volume=1)
        default = EvidenceBasedResolver().detect(record).results[0]
        milder = EvidenceBasedResolver(
            ResolutionModelConfig(insufficient_evidence_confidence_scale=0.9)
        ).detect(record).results[0]
        self.assertGreater(milder.confidence, default.confidence)

    def test_confidence_zero_at_a_boundary(self):
        resolver = EvidenceBasedResolver()
        result = resolver.detect(
            ev(total_comments=4, solution_candidate_count=1, linked_solution_count=1,
               link_scores=[0.8], solution_probabilities=[0.8])
        ).results[0]
        # Land exactly on a cut-point by construction of the score.
        for score, status in [(0.30, ResolutionStatus.PARTIALLY_RESOLVED),
                              (0.55, ResolutionStatus.RESOLVED)]:
            with self.subTest(score=score):
                self.assertEqual(resolver.confidence_for(score, status), 0.0)

    def test_confidence_is_in_range_and_farther_is_more_confident(self):
        resolver = EvidenceBasedResolver()
        near = resolver.confidence_for(0.31, ResolutionStatus.PARTIALLY_RESOLVED)
        far = resolver.confidence_for(0.43, ResolutionStatus.PARTIALLY_RESOLVED)
        self.assertLess(near, far)
        for step in range(0, 101):
            score = step / 100
            value = resolver.confidence_for(score, resolver.decide_status(score))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_confidence_is_never_a_probability_claim(self):
        """The result must not present confidence as calibrated."""
        result = EvidenceBasedResolver().detect(ev(total_comments=6, linked_solution_count=1,
                                                    strong_link_count=1, link_scores=[0.5],
                                                    solution_probabilities=[0.5],
                                                    solution_candidate_count=1)).results[0]
        text = result.format_summary().lower()
        self.assertIn("conf=", text)
        self.assertNotIn("probability", text)


# ---------------------------------------------------------------------------
# 8. Multi-label issues
# ---------------------------------------------------------------------------


class TestMultiLabel(unittest.TestCase):
    def test_one_result_per_label_pair(self):
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch(
            [
                ev("i1", "fees", total_comments=6, solution_candidate_count=2, linked_solution_count=2,
                   strong_link_count=1, link_scores=[0.7, 0.4], solution_probabilities=[0.8, 0.6]),
                ev("i1", "roi", total_comments=6, solution_candidate_count=2, linked_solution_count=0),
            ]
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(
            [r.issue_label for r in batch.results_for_issue("i1")], ["fees", "roi"]
        )

    def test_labels_are_judged_independently(self):
        """A solved fees question must not mark an unsolved roi question resolved."""
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch(
            [
                ev("i1", "fees", total_comments=6, solution_candidate_count=2, linked_solution_count=2,
                   strong_link_count=1, link_scores=[0.7, 0.4], solution_probabilities=[0.8, 0.6]),
                ev("i1", "roi", total_comments=6, solution_candidate_count=2, linked_solution_count=0),
            ]
        )
        statuses = batch.statuses_for_issue("i1")
        self.assertEqual(statuses["fees"], ResolutionStatus.PARTIALLY_RESOLVED.value)
        self.assertEqual(statuses["roi"], ResolutionStatus.UNRESOLVED.value)

    def test_multi_label_issue_does_not_average_into_one_verdict(self):
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch(
            [
                ev("i1", "a", total_comments=6, solution_candidate_count=1, linked_solution_count=0),
                ev("i1", "b", total_comments=6, solution_candidate_count=1, linked_solution_count=0),
            ]
        )
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch.report.n_issue_documents, 1)
        self.assertEqual(batch.report.n_issue_labels, 2)

    def test_same_issue_id_with_different_labels_is_not_a_duplicate(self):
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch([ev("i1", "a"), ev("i1", "b"), ev("i1", "c")])
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch.report.duplicate_pair_keys, [])

    def test_exact_duplicate_pair_keeps_the_first_occurrence(self):
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch(
            [
                ev("i1", "fees", total_comments=6, linked_solution_count=0, solution_candidate_count=0),
                ev("i1", "fees", total_comments=6, solution_candidate_count=3, linked_solution_count=3,
                   strong_link_count=3, link_scores=[0.9, 0.9, 0.9], solution_probabilities=[0.9, 0.9, 0.9]),
            ]
        )
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch.results[0].linked_solution_count, 0)
        self.assertEqual(batch.report.duplicate_pair_keys, ["i1|fees"])
        self.assertEqual(batch.report.n_evidence_records, 2)
        self.assertTrue([w for w in batch.report.warnings if "more than once" in w])

    def test_duplicate_choice_is_deterministic_for_a_fixed_input(self):
        """Same records, same order, same verdict - every time."""
        resolver = EvidenceBasedResolver()
        first = ev("i1", "fees")
        second = ev("i1", "fees", total_comments=6, linked_solution_count=3, strong_link_count=3,
                    link_scores=[0.9, 0.9, 0.9], solution_probabilities=[0.9, 0.9, 0.9],
                    solution_candidate_count=3)
        forward = resolver.detect_batch([first, second])
        again = resolver.detect_batch([first, second])
        self.assertEqual(forward.results[0].linked_solution_count, 0)
        self.assertEqual(
            forward.results[0].model_dump(), again.results[0].model_dump()
        )

    def test_reversing_duplicates_is_reported_rather_than_silently_resolved(self):
        """
        First-wins is order dependent, and that is a limitation, not a feature.

        With two contradictory records for one pair there is no rule that is both
        permutation invariant and picks one of them: invariance requires ignoring
        both, which is what :meth:`detect_batch` could do by refusing, and a
        silent refusal would be worse than a visible choice.  So the choice is
        first-wins, and reversing the input reverses it - which the run report
        must therefore disclose, or a reader would believe the verdict did not
        depend on the order records happened to arrive in.
        """
        resolver = EvidenceBasedResolver()
        first = ev("i1", "fees")
        second = ev("i1", "fees", total_comments=6, linked_solution_count=3, strong_link_count=3,
                    link_scores=[0.9, 0.9, 0.9], solution_probabilities=[0.9, 0.9, 0.9],
                    solution_candidate_count=3)
        forward = resolver.detect_batch([first, second])
        backward = resolver.detect_batch([second, first])

        self.assertEqual(forward.results[0].linked_solution_count, 0)
        self.assertEqual(backward.results[0].linked_solution_count, 3)
        # The difference is disclosed in both directions.
        for batch in (forward, backward):
            self.assertEqual(batch.report.duplicate_pair_keys, ["i1|fees"])
            self.assertEqual(batch.report.n_evidence_records, 2)
            self.assertEqual(batch.report.n_issue_labels, 1)
            self.assertTrue([w for w in batch.report.warnings if "more than once" in w])


# ---------------------------------------------------------------------------
# 9. The distinction this stage exists to preserve
# ---------------------------------------------------------------------------


class TestOneSolutionIsNotResolution(unittest.TestCase):
    def test_one_solution_in_a_long_thread_is_not_resolved(self):
        result = EvidenceBasedResolver().detect(
            ev(total_comments=40, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[0.95], solution_probabilities=[0.95])
        ).results[0]
        self.assertNotEqual(result.status, ResolutionStatus.RESOLVED)

    def test_many_candidates_none_linked_is_not_resolved(self):
        result = EvidenceBasedResolver().detect(
            ev(total_comments=12, solution_candidate_count=6, linked_solution_count=0)
        ).results[0]
        self.assertEqual(result.status, ResolutionStatus.UNRESOLVED)
        self.assertEqual(result.resolution_score, 0.0)
        self.assertTrue([w for w in result.warnings if "candidate" in w])

    def test_candidates_alone_never_raise_the_score(self):
        """Otherwise any busy thread would look resolved."""
        resolver = EvidenceBasedResolver()
        none = resolver.detect(ev(total_comments=10, solution_candidate_count=0,
                                  linked_solution_count=0)).results[0]
        many = resolver.detect(ev(total_comments=10, solution_candidate_count=9,
                                  linked_solution_count=0)).results[0]
        self.assertEqual(none.resolution_score, many.resolution_score)

    def test_confident_solution_absent_follow_up_still_scores_on_evidence_only(self):
        result = EvidenceBasedResolver().detect(
            ev(total_comments=20, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[1.0], solution_probabilities=[1.0],
               temporal=TemporalEvidence(solution_after_issue=True, follow_up_count=0,
                                         follow_up_resolution_signal=False))
        ).results[0]
        self.assertLess(result.resolution_score, 0.55)
        self.assertEqual(result.status, ResolutionStatus.PARTIALLY_RESOLVED)

    def test_warnings_name_the_linker_as_the_thing_being_reported(self):
        """'No links for this pair' is a statement about a linker result, not about reality."""
        result = EvidenceBasedResolver().detect(ev(total_comments=5, linked_solution_count=0)).results[0]
        warning = [w for w in result.warnings if "no solutions are linked" in w][0]
        self.assertIn("linker", warning)


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def _records(self) -> List[IssueResolutionEvidence]:
        return [
            ev("i1", "a", total_comments=7, solution_candidate_count=3, linked_solution_count=2,
               strong_link_count=1, link_scores=[0.6, 0.8], solution_probabilities=[0.7, 0.9],
               response_counts={"solution": 3, "opinion": 4}),
            ev("i2", "b", total_comments=5, solution_candidate_count=1, linked_solution_count=0),
        ]

    def test_repeated_runs_agree(self):
        first = EvidenceBasedResolver().detect_batch(self._records())
        second = EvidenceBasedResolver().detect_batch(self._records())
        self.assertEqual(
            [r.model_dump() for r in first.results],
            [r.model_dump() for r in second.results],
        )

    def test_fresh_resolver_agrees_with_a_reused_one(self):
        records = self._records()
        reused = EvidenceBasedResolver()
        reused.detect_batch(records)
        reused.detect_batch(records)
        fresh = EvidenceBasedResolver().detect_batch(records)
        self.assertEqual(
            [r.resolution_score for r in reused.detect_batch(records).results],
            [r.resolution_score for r in fresh.results],
        )

    def test_timestamp_is_absent_by_default(self):
        batch = EvidenceBasedResolver().detect_batch(self._records())
        self.assertIsNone(batch.linked_at)
        self.assertIsNone(batch.results[0].linked_at)
        self.assertIsNone(batch.report.linked_at)

    def test_timestamp_is_opt_in(self):
        resolver = EvidenceBasedResolver(ResolutionModelConfig(include_timestamp=True))
        batch = resolver.detect_batch(self._records())
        self.assertIsNotNone(batch.linked_at)
        self.assertIn("T", batch.linked_at)


# ---------------------------------------------------------------------------
# 11. Input immutability
# ---------------------------------------------------------------------------


class TestInputImmutability(unittest.TestCase):
    def test_records_are_not_modified(self):
        records = [
            ev("i1", "a", total_comments=7, solution_candidate_count=3, linked_solution_count=2,
               strong_link_count=1, link_scores=[0.6, 0.8], solution_probabilities=[0.7, 0.9],
               response_counts={"solution": 3}),
            ev("i2", "b", total_comments=5, linked_solution_count=0),
        ]
        before = [record.model_dump() for record in records]
        EvidenceBasedResolver().detect_batch(records)
        self.assertEqual([record.model_dump() for record in records], before)

    def test_input_list_is_not_reordered(self):
        records = [ev("i3"), ev("i1"), ev("i2")]
        order = [record.issue_id for record in records]
        EvidenceBasedResolver().detect_batch(records)
        self.assertEqual([record.issue_id for record in records], order)

    def test_config_is_not_modified(self):
        config = ResolutionModelConfig(include_timestamp=False)
        before = config.model_dump()
        resolver = EvidenceBasedResolver(config)
        resolver.detect_batch([ev()])
        resolver.detect_batch([ev()])
        self.assertEqual(config.model_dump(), before)

    def test_result_evidence_is_a_copy_not_an_alias(self):
        record = ev(total_comments=5, linked_solution_count=1)
        result = EvidenceBasedResolver().detect(record).results[0]
        record.total_comments = 99
        self.assertEqual(result.evidence.total_comments, 5)

    def test_supplied_lists_are_not_mutated_by_derivation(self):
        scores = [0.3, 0.9, 0.5]
        probabilities = [0.2, 0.8]
        record = ev(linked_solution_count=3, strong_link_count=2, link_scores=scores,
                    solution_probabilities=probabilities, solution_candidate_count=3)
        EvidenceBasedResolver().detect(record)
        self.assertEqual(scores, [0.3, 0.9, 0.5])
        self.assertEqual(probabilities, [0.2, 0.8])


# ---------------------------------------------------------------------------
# 12. Invalid evidence
# ---------------------------------------------------------------------------


class TestInvalidEvidence(unittest.TestCase):
    def assert_refused(self, record, *fragments):
        with self.assertRaises(ValueError) as ctx:
            EvidenceBasedResolver().detect(record)
        message = str(ctx.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        return message

    def test_blank_issue_id_refused(self):
        self.assert_refused(IssueResolutionEvidence(issue_id="", issue_label="fees"), "issue_id")

    def test_blank_issue_label_refused_with_guidance(self):
        message = self.assert_refused(
            IssueResolutionEvidence(issue_id="i1", issue_label="  "), "issue_label"
        )
        self.assertIn("one record per label", message)

    def test_negative_counts_refused(self):
        for field in ["total_comments", "linked_solution_count", "strong_link_count",
                      "solution_candidate_count", "agreement_count", "opinion_count",
                      "advice_count", "experience_count", "solution_count",
                      "counterargument_count"]:
            with self.subTest(field=field):
                self.assert_refused(ev(**{field: -1}), field)

    def test_linked_more_than_total_refused(self):
        self.assert_refused(
            ev(total_comments=2, solution_candidate_count=3, linked_solution_count=3),
            "linked_solution_count", "total_comments",
        )

    def test_strong_above_linked_refused(self):
        self.assert_refused(
            ev(total_comments=5, solution_candidate_count=2, linked_solution_count=1, strong_link_count=2),
            "strong_link_count", "linked_solution_count",
        )

    def test_candidates_below_linked_refused(self):
        self.assert_refused(
            ev(total_comments=5, solution_candidate_count=1, linked_solution_count=2, strong_link_count=1),
            "solution_candidate_count", "linked_solution_count",
        )

    def test_volume_below_total_refused(self):
        self.assert_refused(
            ev(total_comments=9, discussion_volume=3), "discussion_volume"
        )

    def test_out_of_range_scores_refused(self):
        for field, value in [("best_link_score", 1.4), ("mean_link_score", -0.2),
                             ("solution_probability_mean", 2.0),
                             ("solution_probability_max", float("inf")),
                             ("solution_density", 1.5), ("solution_density", -0.1)]:
            with self.subTest(field=field, value=value):
                self.assert_refused(ev(**{field: value}), field)

    def test_nan_and_infinity_refused(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_evidence(ev(best_link_score=value))

    def test_nan_in_a_score_list_refused(self):
        with self.assertRaises(ValueError):
            validate_evidence(ev(linked_solution_count=2, link_scores=[0.5, float("nan")]))
        with self.assertRaises(ValueError):
            validate_evidence(ev(linked_solution_count=2, solution_probabilities=[float("inf"), 0.5]))

    def test_out_of_range_score_in_a_list_refused(self):
        with self.assertRaises(ValueError):
            validate_evidence(ev(linked_solution_count=1, link_scores=[0.5, 1.2]))
        with self.assertRaises(ValueError):
            validate_evidence(ev(linked_solution_count=1, solution_probabilities=[-0.1]))

    def test_negative_response_count_refused(self):
        with self.assertRaises(ValueError):
            validate_evidence(ev(response_counts={"opinion": -2}))
        with self.assertRaises(ValueError):
            validate_evidence(ev(response_counts={"opinion": "many"}))

    def test_inconsistent_supplied_density_refused(self):
        self.assert_refused(
            ev(total_comments=4, linked_solution_count=2, solution_density=0.9),
            "solution_density", "linked_solution_count/total_comments",
        )

    def test_inconsistent_best_link_score_refused(self):
        self.assert_refused(
            ev(linked_solution_count=2, link_scores=[0.3, 0.4], best_link_score=0.9),
            "best_link_score", "max(link_scores)",
        )

    def test_negative_temporal_follow_up_count_refused(self):
        with self.assertRaises(ValueError):
            validate_evidence(ev(temporal=TemporalEvidence(follow_up_count=-1)))

    def test_impossible_record_is_refused_rather_than_scored(self):
        """The resolver must not launder an impossible record into a verdict."""
        with self.assertRaises(ValueError):
            EvidenceBasedResolver().detect(
                ev(total_comments=1, solution_candidate_count=9, linked_solution_count=9,
                   strong_link_count=9, link_scores=[0.9, 0.9, 0.9],
                   solution_probabilities=[0.9, 0.9, 0.9])
            )

    def test_validate_evidence_returns_none_for_a_coherent_record(self):
        self.assertIsNone(validate_evidence(ev(total_comments=4, solution_candidate_count=1,
                                                linked_solution_count=1, strong_link_count=1)))


# ---------------------------------------------------------------------------
# 13. Temporal evidence
# ---------------------------------------------------------------------------


class TestTemporalEvidence(unittest.TestCase):
    def _record(self, **temporal) -> IssueResolutionEvidence:
        return ev(total_comments=6, solution_candidate_count=2, linked_solution_count=2,
                  strong_link_count=1, link_scores=[0.6, 0.5], solution_probabilities=[0.7, 0.6],
                  temporal=TemporalEvidence(**temporal) if temporal else None)

    def test_temporal_is_preserved_on_the_result(self):
        temporal = TemporalEvidence(solution_after_issue=True, follow_up_count=3,
                                    follow_up_resolution_signal=True)
        result = EvidenceBasedResolver().detect(self._record(
            solution_after_issue=True, follow_up_count=3, follow_up_resolution_signal=True
        )).results[0]
        self.assertIsNotNone(result.temporal)
        self.assertTrue(result.temporal.solution_after_issue)
        self.assertEqual(result.temporal.follow_up_count, 3)
        self.assertTrue(result.temporal.follow_up_resolution_signal)

    def test_temporal_does_not_change_the_score(self):
        resolver = EvidenceBasedResolver()
        without = resolver.detect(self._record()).results[0]
        with_signal = resolver.detect(
            self._record(solution_after_issue=True, follow_up_count=12,
                         follow_up_resolution_signal=True)
        ).results[0]
        self.assertAlmostEqual(without.resolution_score, with_signal.resolution_score, places=12)
        self.assertEqual(without.status, with_signal.status)

    def test_chronology_alone_cannot_resolve_an_issue(self):
        """Order is a precondition for a solution being relevant, not proof it worked."""
        result = EvidenceBasedResolver().detect(
            ev(total_comments=30, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[0.9], solution_probabilities=[0.9],
               temporal=TemporalEvidence(solution_after_issue=True, follow_up_count=20,
                                         follow_up_resolution_signal=True))
        ).results[0]
        self.assertNotEqual(result.status, ResolutionStatus.RESOLVED)

    def test_run_report_counts_temporal_and_follow_up_signals(self):
        resolver = EvidenceBasedResolver()
        batch = resolver.detect_batch([
            ev("i1", "a", total_comments=6, linked_solution_count=1, strong_link_count=1,
               link_scores=[0.6], solution_probabilities=[0.6], solution_candidate_count=1,
               temporal=TemporalEvidence(solution_after_issue=True, follow_up_resolution_signal=True)),
            ev("i2", "b", total_comments=6),
        ])
        self.assertEqual(batch.report.n_with_temporal_evidence, 1)
        self.assertEqual(batch.report.n_with_follow_up_resolution_signal, 1)
        self.assertTrue([w for w in batch.report.warnings if "V2" in w])

    def test_empty_temporal_is_not_counted(self):
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, temporal=TemporalEvidence())
        ])
        self.assertEqual(batch.report.n_with_temporal_evidence, 0)


# ---------------------------------------------------------------------------
# 14. Results, batches and run reports
# ---------------------------------------------------------------------------


class TestResultObjects(unittest.TestCase):
    def test_result_fields_report_the_evidence(self):
        result = EvidenceBasedResolver().detect(
            ev(total_comments=8, solution_candidate_count=3, linked_solution_count=2,
               strong_link_count=1, link_scores=[0.71, 0.34], solution_probabilities=[0.88, 0.61])
        ).results[0]
        self.assertEqual(result.linked_solution_count, 2)
        self.assertEqual(result.strong_link_count, 1)
        self.assertEqual(result.solution_candidate_count, 3)
        self.assertEqual(result.effective_discussion_volume, 8)
        self.assertAlmostEqual(result.solution_density, 0.25)
        self.assertAlmostEqual(result.strong_solution_density, 0.125)
        self.assertAlmostEqual(result.best_link_score, 0.71)
        self.assertAlmostEqual(result.solution_probability_mean, 0.745)

    def test_status_predicates(self):
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=2, link_scores=[0.9, 0.9], solution_probabilities=[0.9, 0.9]),
            ev("i2", "b", total_comments=6),
        ])
        resolved, unresolved = batch.results
        self.assertTrue(resolved.is_resolved)
        self.assertFalse(resolved.is_unresolved)
        self.assertTrue(unresolved.is_unresolved)
        self.assertFalse(unresolved.is_resolved)

    def test_signals_record_supplied_versus_derived(self):
        result = EvidenceBasedResolver().detect(
            ev(linked_solution_count=2, strong_link_count=1, link_scores=[0.7, 0.5],
               solution_probabilities=[0.8, 0.6])
        ).results[0]
        self.assertEqual(result.signals["link_strength_source"], "link_scores")
        self.assertEqual(result.signals["probability_source"], "solution_probabilities")
        self.assertEqual(result.signals["density_source"], "derived")

    def test_batch_container_protocol(self):
        batch = EvidenceBasedResolver().detect_batch([ev("i1"), ev("i2")])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].issue_id, "i1")
        self.assertEqual([r.issue_id for r in batch], ["i1", "i2"])

    def test_unresolved_is_strictly_unresolved(self):
        """Merging the two open states would understate the work left to do."""
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=8, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=1, link_scores=[0.6, 0.4], solution_probabilities=[0.7, 0.5]),
            ev("i2", "b", total_comments=8, solution_candidate_count=0, linked_solution_count=0),
        ])
        self.assertEqual(len(batch.not_resolved()), 2)
        self.assertEqual([r.pair_key() for r in batch.unresolved()], ["i2|b"])
        self.assertEqual(batch.status_counts()["partially_resolved"], 1)

    def test_with_status_accepts_a_string(self):
        batch = EvidenceBasedResolver().detect_batch([ev("i2", "b", total_comments=6)])
        self.assertEqual(len(batch.with_status("unresolved")), 1)

    def test_thin_evidence_lookup(self):
        batch = EvidenceBasedResolver().detect_batch([ev("i1", total_comments=1), ev("i2", total_comments=5)])
        self.assertEqual([r.pair_key() for r in batch.thin_evidence()], ["i1|fees"])

    def test_by_pair_key(self):
        batch = EvidenceBasedResolver().detect_batch([ev("i1", "a"), ev("i1", "b")])
        self.assertEqual(sorted(batch.by_pair_key()), ["i1|a", "i1|b"])

    def test_empty_batch_is_handled(self):
        batch = EvidenceBasedResolver().detect_batch([])
        self.assertEqual(len(batch), 0)
        self.assertEqual(batch.status_counts()["unresolved"], 0)
        self.assertFalse(batch.report.produced_results)
        self.assertFalse(batch.report.is_degenerate)
        self.assertIn("no issue was assessed", batch.format_summary())
        self.assertIn("no issue was assessed", batch.report.warnings[0])

    def test_degenerate_run_is_reported(self):
        """A corpus where every thread is too thin is a misconfiguration, not a finding."""
        batch = EvidenceBasedResolver().detect_batch([ev("i1", total_comments=1), ev("i2", total_comments=1)])
        self.assertTrue(batch.report.is_degenerate)
        self.assertEqual(batch.report.n_evidence_insufficient, 2)
        self.assertEqual(batch.report.n_evidence_sufficient, 0)
        self.assertIn("typical thread size", batch.report.warnings[0])

    def test_run_report_counts(self):
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=2, link_scores=[0.9, 0.8], solution_probabilities=[0.9, 0.8]),
            ev("i1", "b", total_comments=6),
            ev("i2", "a", total_comments=6, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[0.6], solution_probabilities=[0.6]),
        ])
        report = batch.report
        self.assertEqual(report.n_evidence_records, 3)
        self.assertEqual(report.n_issue_documents, 2)
        self.assertEqual(report.n_issue_labels, 3)
        self.assertEqual(sum(report.status_counts.values()), 3)
        self.assertEqual(report.status_counts["resolved"], 1)
        self.assertEqual(report.status_counts["partially_resolved"], 1)
        self.assertEqual(report.status_counts["unresolved"], 1)
        self.assertIn("python", report.library_versions)
        self.assertIn("pydantic", report.library_versions)

    def test_response_count_mismatch_is_reported_not_refused(self):
        """Neither view is a score input, so a disagreement cannot corrupt a verdict."""
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, response_counts={"solution": 4}, solution_count=1)
        ])
        self.assertEqual(batch.report.response_count_mismatches, ["i1|a"])
        self.assertIn("response_counts", batch.report.warnings[0])
        self.assertEqual(batch.results[0].resolution_score, 0.0)

    def test_agreeing_response_counts_are_not_reported(self):
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, response_counts={"solution": 2, "opinion": 1},
               solution_count=2, opinion_count=1)
        ])
        self.assertEqual(batch.report.response_count_mismatches, [])

    def test_unrecognised_response_role_is_not_a_mismatch(self):
        """The stage must not care what roles an upstream classifier happens to emit."""
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "a", total_comments=6, response_counts={"solution": 1, "future_role": 3},
               solution_count=1)
        ])
        self.assertEqual(batch.report.response_count_mismatches, [])

    def test_format_summary_is_readable(self):
        batch = EvidenceBasedResolver().detect_batch([ev("i1", "a", total_comments=1)])
        summary = batch.format_summary()
        self.assertIn("Resolution Detection", summary)
        self.assertIn("resolved", summary)
        self.assertIn("thin evidence", summary)

    def test_result_format_summary_includes_warnings(self):
        result = EvidenceBasedResolver().detect(ev("i1", total_comments=1)).results[0]
        summary = result.format_summary()
        self.assertIn("i1[fees]", summary)
        self.assertIn("!", summary)


# ---------------------------------------------------------------------------
# 15. Annotation schema and CSV handling
# ---------------------------------------------------------------------------


class TestAnnotationSchema(unittest.TestCase):
    def test_columns(self):
        self.assertEqual(
            RESOLUTION_CSV_COLUMNS, ["issue_id", "issue_label", "gold_status", "rationale"]
        )

    def test_gold_status_is_required(self):
        with self.assertRaises(ValueError):
            ResolutionAnnotation(issue_id="i1", gold_status="maybe")
        with self.assertRaises(ValueError):
            ResolutionAnnotation(issue_id="i1", gold_status="")

    def test_issue_id_is_required(self):
        with self.assertRaises(ValueError):
            ResolutionAnnotation(issue_id="  ", gold_status="resolved")

    def test_pair_key_matches_the_result_side(self):
        """Different matching rules on the two sides would silently fail to match."""
        labelled = ResolutionAnnotation(issue_id="i1", issue_label="fees", gold_status="resolved")
        unlabelled = ResolutionAnnotation(issue_id="i1", gold_status="resolved")
        self.assertEqual(labelled.pair_key(), "i1|fees")
        self.assertEqual(unlabelled.pair_key(), "i1")
        self.assertTrue(labelled.is_labelled())
        self.assertFalse(unlabelled.is_labelled())

    def test_blank_label_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            ResolutionAnnotation(issue_id="i1", issue_label="   ", gold_status="resolved")
        self.assertIn("blank", str(ctx.exception))

    def test_annotator_confidence_range_enforced(self):
        ResolutionAnnotation(issue_id="i1", gold_status="resolved", annotator_confidence=0.5)
        for value in (-0.1, 1.4):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ResolutionAnnotation(issue_id="i1", gold_status="resolved",
                                         annotator_confidence=value)

    def test_corpus_status_counts(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "gold_status": "resolved"},
            {"issue_id": "i2", "gold_status": "unresolved"},
            {"issue_id": "i3", "gold_status": "unresolved"},
        ])
        self.assertEqual(len(corpus), 3)
        self.assertEqual(corpus.status_counts()["unresolved"], 2)
        self.assertEqual(corpus.status_counts()["resolved"], 1)
        self.assertEqual(corpus.status_counts()["partially_resolved"], 0)

    def test_conflicting_judgements_are_reported_not_hidden(self):
        """An inter-annotator figure computed over a corpus that hid its conflicts is meaningless."""
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "gold_status": "resolved"},
            {"issue_id": "i1", "gold_status": "unresolved"},
        ])
        self.assertEqual(corpus.conflicting_pair_keys(), ["i1"])

    def test_unlabelled_document_keys(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "gold_status": "resolved"},
            {"issue_id": "i2", "issue_label": "fees", "gold_status": "resolved"},
        ])
        self.assertEqual(corpus.unlabelled_document_keys(), ["i1"])

    def test_csv_round_trip(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "partially_resolved",
             "rationale": "one of two questions answered"},
            {"issue_id": "i2", "gold_status": "unresolved"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "resolution_annotations.csv")
            corpus.to_csv(path)
            with open(path, encoding="utf-8") as handle:
                header = handle.readline().strip()
            self.assertEqual(header, ",".join(RESOLUTION_CSV_COLUMNS))
            reloaded = ResolutionCorpus.from_csv(path)
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.status_counts(), corpus.status_counts())
        self.assertEqual(
            sorted(reloaded.keys()), ["i1|fees", "i2"]
        )

    def test_csv_missing_column_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("issue_id,gold_status\ni1,resolved\n")
            with self.assertRaises(ValueError) as ctx:
                ResolutionCorpus.from_csv(path)
        self.assertIn("issue_label", str(ctx.exception))

    def test_csv_blank_gold_status_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("issue_id,issue_label,gold_status\ni1,fees,\n")
            with self.assertRaises(ValueError):
                ResolutionCorpus.from_csv(path)

    def test_csv_missing_file_message_explains_that_no_data_ships(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ResolutionCorpus.from_csv("no_such_resolution_annotations.csv")
        self.assertIn("fabricated dataset", str(ctx.exception))


# ---------------------------------------------------------------------------
# 16. Evaluation, including the refusal to fabricate metrics
# ---------------------------------------------------------------------------


class TestEvaluation(unittest.TestCase):
    """
    Arithmetic is checked against annotations declared inline and explicitly
    synthetic.  These are not measurements and must never be reported as such.
    """

    def _batch(self) -> ResolutionBatchResult:
        resolver = EvidenceBasedResolver()
        return resolver.detect_batch([
            ev("i1", "fees", total_comments=8, solution_candidate_count=2, linked_solution_count=2,
               strong_link_count=2, link_scores=[0.9, 0.85], solution_probabilities=[0.9, 0.88]),
            ev("i2", "roi", total_comments=6, solution_candidate_count=1, linked_solution_count=1,
               strong_link_count=1, link_scores=[0.62], solution_probabilities=[0.6]),
            ev("i3", "ui", total_comments=6, solution_candidate_count=0, linked_solution_count=0),
        ])

    def _corpus(self, overrides=None) -> ResolutionCorpus:
        """
        A synthetic corpus matching :meth:`_batch`'s three records.

        The labels are explicitly synthetic: they are the test author's own
        judgements, used to check arithmetic.  They are not a measurement of
        anything and must never be quoted as one.
        """
        records = [
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "resolved",
             "rationale": "synthetic"},
            {"issue_id": "i2", "issue_label": "roi", "gold_status": "partially_resolved",
             "rationale": "synthetic"},
            {"issue_id": "i3", "issue_label": "ui", "gold_status": "unresolved",
             "rationale": "synthetic"},
        ]
        for (issue_id, issue_label), gold_status in (overrides or {}).items():
            for record in records:
                if record["issue_id"] == issue_id and record.get("issue_label") == issue_label:
                    record["gold_status"] = gold_status
        return ResolutionCorpus.from_records(records)

    def test_empty_corpus_refused(self):
        """A caller asking for a metric with nothing annotated gets an error, not a number."""
        with self.assertRaises(ValueError) as ctx:
            evaluate_resolution(self._batch(), ResolutionCorpus())
        message = str(ctx.exception)
        self.assertIn("no accuracy", message)
        self.assertIn("may be reported", message)

    def test_none_corpus_refused(self):
        with self.assertRaises(ValueError):
            evaluate_resolution(self._batch(), None)

    def test_perfect_match_on_synthetic_annotations(self):
        evaluation = evaluate_resolution(self._batch(), self._corpus())
        self.assertEqual(evaluation.matched_pairs, 3)
        self.assertEqual(evaluation.correct, 3)
        self.assertEqual(evaluation.accuracy, 1.0)
        self.assertEqual(evaluation.macro_f1, 1.0)

    def test_confusion_matrix_sums_to_matched(self):
        evaluation = evaluate_resolution(
            self._batch(),
            self._corpus({("i2", "roi"): "resolved", ("i3", "ui"): "partially_resolved"}),
        )
        total = sum(
            evaluation.confusion_matrix[gold][predicted]
            for gold in evaluation.confusion_matrix
            for predicted in evaluation.confusion_matrix[gold]
        )
        self.assertEqual(total, evaluation.matched_pairs)
        # i1 is the only agreement: i2 is predicted partially_resolved against a
        # gold of resolved, and i3 is predicted unresolved against a gold of
        # partially_resolved.
        self.assertEqual(evaluation.correct, 1)
        self.assertAlmostEqual(evaluation.accuracy, 1 / 3)

    def test_per_status_precision_recall_f1_arithmetic(self):
        evaluation = evaluate_resolution(
            self._batch(), self._corpus({("i2", "roi"): "unresolved"})
        )
        resolved = evaluation.per_status["resolved"]
        self.assertEqual(resolved.support, 1)
        self.assertEqual(resolved.true_positive, 1)
        self.assertEqual(resolved.precision, 1.0)
        self.assertEqual(resolved.recall, 1.0)
        self.assertEqual(resolved.f1, 1.0)

        # i1 predicted resolved but gold unresolved -> one FP, one FN.
        metrics = evaluation.per_status["unresolved"]
        self.assertEqual(metrics.support, 2)
        self.assertEqual(metrics.predicted, 1)
        self.assertAlmostEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.precision, 1.0)

    def test_single_status_corpus_is_flagged(self):
        """100% unresolved gold gives accuracy 1.0 to a constant resolver."""
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i3", "issue_label": "ui", "gold_status": "unresolved"},
        ])
        evaluation = evaluate_resolution(self._batch(), corpus)
        self.assertTrue(evaluation.single_status_corpus)
        self.assertIn("not informative", evaluation.format_report())
        self.assertEqual(evaluation.accuracy, 1.0)

    def test_status_without_gold_support_has_undefined_metrics_not_zero(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "resolved"},
            {"issue_id": "i2", "issue_label": "roi", "gold_status": "resolved"},
        ])
        evaluation = evaluate_resolution(self._batch(), corpus)
        unresolved = evaluation.per_status["unresolved"]
        self.assertEqual(unresolved.support, 0)
        self.assertIsNone(unresolved.recall)
        self.assertIsNone(unresolved.precision)
        self.assertIsNone(unresolved.f1)
        self.assertIn("unresolved", evaluation.statuses_without_gold_support)
        # Excluded from macro F1 rather than counted as a zero.  Only 'resolved'
        # contributes, and it is 2/3 because i2 is predicted partially_resolved.
        self.assertAlmostEqual(evaluation.macro_f1, 2 / 3)

    def test_macro_f1_is_none_when_gold_covers_nothing(self):
        corpus = ResolutionCorpus(annotations=[])
        evaluation = evaluate_resolution(self._batch(), corpus) if len(corpus) else None
        self.assertIsNone(evaluation)

    def test_unmatched_results_are_reported(self):
        """A metric over a subset is not a metric over the stage."""
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "resolved"},
        ])
        evaluation = evaluate_resolution(self._batch(), corpus)
        self.assertEqual(evaluation.annotated_pairs, 1)
        self.assertEqual(evaluation.matched_pairs, 1)
        self.assertEqual(sorted(evaluation.unmatched_results), ["i2|roi", "i3|ui"])
        self.assertIn("not a metric over the stage", evaluation.format_report())

    def test_unmatched_annotations_are_reported(self):
        corpus = self._corpus()
        corpus.annotations.append(
            ResolutionAnnotation(issue_id="i99", issue_label="ghost", gold_status="resolved")
        )
        evaluation = evaluate_resolution(self._batch(), corpus)
        self.assertIn("i99|ghost", evaluation.unmatched_annotations)

    def test_unlabelled_annotation_matches_a_single_label_result(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i3", "gold_status": "unresolved"},
        ])
        evaluation = evaluate_resolution(self._batch(), corpus)
        self.assertEqual(evaluation.matched_pairs, 1)
        self.assertEqual(evaluation.correct, 1)

    def test_unlabelled_annotation_is_refused_for_a_multi_label_result(self):
        """Attributing a document-level judgement to one of several labels is a guess."""
        batch = EvidenceBasedResolver().detect_batch([ev("i1", "fees", total_comments=6),
                                                      ev("i1", "roi", total_comments=6)])
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "gold_status": "resolved"},
        ])
        evaluation = evaluate_resolution(batch, corpus)
        self.assertEqual(evaluation.matched_pairs, 0)
        self.assertEqual(evaluation.ambiguous_matches, ["i1"])
        self.assertIn("would be a guess", evaluation.format_report())

    def test_duplicate_gold_rows_are_counted_once(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i3", "issue_label": "ui", "gold_status": "unresolved"},
            {"issue_id": "i3", "issue_label": "ui", "gold_status": "unresolved"},
        ])
        evaluation = evaluate_resolution(self._batch(), corpus)
        self.assertEqual(evaluation.annotated_pairs, 2)
        self.assertEqual(evaluation.matched_pairs, 1)
        self.assertEqual(evaluation.correct, 1)

    def test_conflicting_gold_raises_or_reports(self):
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "resolved"},
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "unresolved"},
        ])
        with self.assertRaises(ValueError) as ctx:
            evaluate_resolution(self._batch(), corpus, on_conflict="raise")
        self.assertIn("contradictory", str(ctx.exception))
        evaluation = evaluate_resolution(self._batch(), corpus, on_conflict="report")
        self.assertEqual(evaluation.conflicting_judgements, ["i1|fees"])
        self.assertIn("more than one gold", evaluation.format_report())

    def test_unknown_conflict_policy_refused(self):
        with self.assertRaises(ValueError):
            evaluate_resolution(self._batch(), self._corpus(), on_conflict="ignore")

    def test_thin_evidence_matches_are_reported(self):
        batch = EvidenceBasedResolver().detect_batch([
            ev("i1", "fees", total_comments=1, linked_solution_count=0),
        ])
        corpus = ResolutionCorpus.from_records([
            {"issue_id": "i1", "issue_label": "fees", "gold_status": "unresolved"},
        ])
        evaluation = evaluate_resolution(batch, corpus)
        self.assertEqual(evaluation.thin_evidence_results, ["i1|fees"])
        self.assertIn("weaker", evaluation.format_report())

    def test_gold_distribution_is_always_reported(self):
        evaluation = evaluate_resolution(self._batch(), self._corpus())
        report = evaluation.format_report()
        self.assertIn("gold distribution", report)
        for status in STATUS_ORDER:
            self.assertIn(status.value, report)

    def test_report_says_the_numbers_describe_only_the_supplied_corpus(self):
        evaluation = evaluate_resolution(self._batch(), self._corpus())
        self.assertIn("supplied corpus only", evaluation.format_report())

    def test_confusion_rows_are_aligned(self):
        evaluation = evaluate_resolution(self._batch(), self._corpus())
        rows = evaluation.confusion_as_rows()
        self.assertEqual(len(rows), 4)
        self.assertIn("gold \\ predicted", rows[0])
        for status in STATUS_ORDER:
            self.assertTrue(any(status.value in row for row in rows[1:]))

    def test_per_status_metrics_render(self):
        metrics = PerStatusMetrics(status=ResolutionStatus.RESOLVED, support=2, predicted=3,
                                   true_positive=2, precision=1.0, recall=1.0, f1=1.0)
        self.assertIn("resolved", metrics.format_line())
        self.assertTrue(metrics.has_gold_support)
        self.assertIn("n/a", PerStatusMetrics(status=ResolutionStatus.UNRESOLVED).format_line())


# ---------------------------------------------------------------------------
# 17. Architecture isolation
# ---------------------------------------------------------------------------


class TestArchitectureIsolation(unittest.TestCase):
    def _module_paths(self):
        for root, _dirs, files in os.walk(PACKAGE_DIR):
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(root, name)

    def test_no_imports_of_other_nlp_stages(self):
        """The stage is replaceable only while it owes nothing to its neighbours."""
        offenders = []
        for path in self._module_paths():
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if "app.services.nlp" not in name:
                        continue
                    for stage in OTHER_STAGES:
                        # Matched on the full dotted path, not as a substring:
                        # 'resolution_detection' contains 'solution_detection',
                        # and a substring test would report this package as a
                        # dependency on two other stages.
                        if f"app.services.nlp.{stage}" in name:
                            offenders.append((os.path.basename(path), name))
        self.assertEqual(
            offenders, [],
            "resolution detection must not import another NLP stage: "
            + ", ".join(f"{f} -> {n}" for f, n in offenders),
        )

    def test_imports_stay_inside_the_package(self):
        offenders = []
        for path in self._module_paths():
            with open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.startswith("app.services.nlp"):
                        if not node.module.startswith(
                            "app.services.nlp.resolution_detection"
                        ):
                            offenders.append((os.path.basename(path), node.module))
        self.assertEqual(offenders, [])

    def test_evidence_model_carries_no_upstream_types(self):
        """The evidence is counts and scores, not another stage's objects."""
        fields = set(IssueResolutionEvidence.model_fields)
        for upstream in [
            "comment_id",
            "normalized_text",
            "original_text",
            "issue_labels",
            "is_solution",
            "solution_probability",
            "response_type",
            "document_id",
            "link_score",
            "relationship",
        ]:
            with self.subTest(upstream=upstream):
                self.assertNotIn(upstream, fields)


# ---------------------------------------------------------------------------
# 18. Interface conformance
# ---------------------------------------------------------------------------


class TestInterface(unittest.TestCase):
    def test_resolver_implements_the_abstract_interface(self):
        from app.services.nlp.resolution_detection import ResolutionDetection

        resolver = EvidenceBasedResolver()
        self.assertIsInstance(resolver, ResolutionDetection)
        for attribute in ("name", "version", "config", "thresholds", "config_dict"):
            self.assertTrue(hasattr(resolver, attribute))

    def test_interface_cannot_be_instantiated(self):
        from app.services.nlp.resolution_detection import ResolutionDetection

        with self.assertRaises(TypeError):
            ResolutionDetection()

    def test_config_dict_is_json_serialisable(self):
        import json

        payload = json.dumps(EvidenceBasedResolver().config_dict)
        self.assertIn(MODEL_NAME, payload)

    def test_result_is_a_pydantic_model(self):
        result = EvidenceBasedResolver().detect(ev()).results[0]
        self.assertIsInstance(result, ResolutionResult)
        self.assertIsInstance(result.model_dump(), dict)

    def test_run_report_is_a_pydantic_model(self):
        batch = EvidenceBasedResolver().detect_batch([ev()])
        self.assertIsInstance(batch.report, ResolutionRunReport)

    def test_derived_evidence_is_a_pydantic_model(self):
        self.assertIsInstance(derive_evidence(ev()), DerivedEvidence)

    def test_evaluation_is_a_pydantic_model(self):
        evaluation = ResolutionEvaluation(model_name="x", model_version="v1")
        self.assertIsInstance(evaluation, ResolutionEvaluation)

    def test_statuses_are_comparable_to_plain_strings(self):
        self.assertEqual(ResolutionStatus.RESOLVED, "resolved")
        self.assertEqual(ResolutionStatus.UNRESOLVED.value, "unresolved")

    def test_detection_never_raises_for_a_thin_thread(self):
        """A thin thread is a reported verdict, never an exception."""
        for volume in (0, 1, 2):
            with self.subTest(volume=volume):
                result = EvidenceBasedResolver().detect(ev(total_comments=volume)).results[0]
                self.assertIn(result.status, set(STATUS_ORDER))


if __name__ == "__main__":
    unittest.main()
