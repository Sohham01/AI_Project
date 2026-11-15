"""
Tests for Issue-Solution Linking.

These tests assert **mechanics**, never quality.  The linker's precision and
recall cannot be tested here, because no human-annotated link dataset exists to
test them against - a test that asserted "this linker is accurate" would be
asserting something about fixtures the same author wrote.  What is tested
instead is that the stage does the things it promises: it refuses to score
cross-thread pairs by default, it keeps the issue's actual text rather than its
label, it preserves every score, it is deterministic, it does not mutate its
inputs, and it refuses to report a metric without an explicit annotated corpus.

Test groups
-----------
1.  Evidence models
2.  Configuration validation
3.  Candidate generation (thread scoping, candidacy, caps, labels)
4.  Scoring (formula, thresholds, role evidence, zero vectors)
5.  Multi-label and many-to-many behaviour
6.  Multilingual and code-mixed text
7.  Placeholder handling and non-mutation
8.  Determinism
9.  Results, summaries and lookups
10. Annotation schema and CSV handling
11. Evaluation, including the refusal to fabricate metrics
12. Run reports and empty-input edge cases
13. Architecture isolation (AST: no imports of other NLP stages)
14. The cross-product failure this stage exists to prevent
"""

import ast
import math
import os
import tempfile
import unittest
from typing import List

from app.services.nlp.issue_solution_linking import (
    DEFAULT_LINK_CSV,
    LINK_CSV_COLUMNS,
    REASON_BELOW_CANDIDATE_THRESHOLD,
    REASON_CAPPED,
    REASON_CROSS_POST,
    REASON_CROSS_THREAD_DISABLED,
    REASON_ISSUE_HAS_NO_LABEL,
    REASON_PARENT_CHILD,
    REASON_SAME_POST,
    REASON_UNRESOLVED_THREAD,
    CandidateDecision,
    IssueEvidence,
    LinkAnnotation,
    LinkModelConfig,
    LinkPrediction,
    LinkCorpus,
    LinkingResult,
    LinkingRunReport,
    SolutionEvidence,
    TfidfCosineIssueSolutionLinker,
    evaluate_linking,
    generate_candidates,
    is_solution_candidate,
    normalised_labels,
    parse_linked_cell,
    thread_signals,
    threshold_sweep,
)
from app.services.nlp.issue_solution_linking.models.labels import (
    normalise_labels,
    parse_bool_cell,
    parse_float_cell,
    strip_placeholder_tokens,
)

PACKAGE_DIR = os.path.dirname(
    os.path.abspath(
        __import__(
            "app.services.nlp.issue_solution_linking", fromlist=["__file__"]
        ).__file__
    )
)


def issue(document_id: str, text: str, labels: List[str], post_id: str = "p1", **kwargs) -> IssueEvidence:
    return IssueEvidence(
        document_id=document_id,
        post_id=post_id,
        comment_id=f"{document_id}-c",
        normalized_text=text,
        original_text=kwargs.pop("original_text", text),
        issue_labels=labels,
        **kwargs,
    )


def solution(
    document_id: str,
    text: str,
    post_id: str = "p1",
    probability: float = 0.9,
    is_solution: bool = True,
    **kwargs,
) -> SolutionEvidence:
    return SolutionEvidence(
        document_id=document_id,
        post_id=post_id,
        comment_id=f"{document_id}-c",
        normalized_text=text,
        original_text=kwargs.pop("original_text", text),
        is_solution=is_solution,
        solution_probability=probability,
        response_type=kwargs.pop("response_type", "solution"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Evidence models
# ---------------------------------------------------------------------------


class TestEvidenceModels(unittest.TestCase):
    def test_issue_evidence_carries_every_required_field(self):
        evidence = IssueEvidence(
            document_id="d1",
            post_id="p1",
            comment_id="c1",
            original_text="Fees are too high",
            normalized_text="fees are too high",
            issue_labels=["fees", "hostel"],
            issue_probabilities={"fees": 0.91, "hostel": 0.44},
            topic_id=3,
            language="en",
            script="Latin",
            is_code_mixed=False,
            detected_languages=["en"],
        )
        for field in (
            "document_id",
            "post_id",
            "comment_id",
            "original_text",
            "normalized_text",
            "issue_labels",
            "issue_probabilities",
            "topic_id",
            "language",
            "script",
            "is_code_mixed",
            "detected_languages",
        ):
            self.assertTrue(hasattr(evidence, field), f"missing field {field}")
        self.assertTrue(evidence.has_thread)
        self.assertTrue(evidence.has_labels)
        self.assertEqual(evidence.comparable_text, "fees are too high")

    def test_solution_evidence_carries_every_required_field(self):
        evidence = SolutionEvidence(
            document_id="d2",
            post_id="p1",
            comment_id="c2",
            original_text="Ask the warden",
            normalized_text="ask the warden",
            is_solution=True,
            solution_probability=0.88,
            response_type="advice",
            response_probabilities={"advice": 0.7},
            parent_comment_id="c1",
            parent_comment_text="parent text",
            language="en",
            script="Latin",
            is_code_mixed=True,
            detected_languages=["en"],
        )
        for field in (
            "document_id",
            "post_id",
            "comment_id",
            "original_text",
            "normalized_text",
            "is_solution",
            "solution_probability",
            "response_type",
            "response_probabilities",
            "parent_comment_id",
            "parent_comment_text",
            "language",
            "script",
            "is_code_mixed",
            "detected_languages",
        ):
            self.assertTrue(hasattr(evidence, field), f"missing field {field}")

    def test_evidence_defaults_are_neutral_not_optimistic(self):
        issue_evidence = IssueEvidence(document_id="d")
        self.assertIsNone(issue_evidence.post_id)
        self.assertEqual(issue_evidence.issue_labels, [])
        self.assertEqual(issue_evidence.normalized_text, "")
        self.assertFalse(issue_evidence.has_thread)
        self.assertFalse(issue_evidence.has_labels)

        solution_evidence = SolutionEvidence(document_id="d")
        self.assertFalse(solution_evidence.is_solution)
        self.assertEqual(solution_evidence.solution_probability, 0.0)
        self.assertIsNone(solution_evidence.response_type)

    def test_has_thread_requires_a_non_blank_post_id(self):
        self.assertFalse(IssueEvidence(document_id="d", post_id="   ").has_thread)
        self.assertFalse(IssueEvidence(document_id="d", post_id=None).has_thread)
        self.assertTrue(IssueEvidence(document_id="d", post_id="p1").has_thread)

    def test_comparable_text_prefers_normalized_then_original(self):
        both = SolutionEvidence(
            document_id="d", normalized_text="normalized words", original_text="original words"
        )
        self.assertEqual(both.comparable_text, "normalized words")
        only_original = SolutionEvidence(document_id="d", original_text="original words")
        self.assertEqual(only_original.comparable_text, "original words")
        self.assertEqual(SolutionEvidence(document_id="d").comparable_text, "")


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_documented_defaults(self):
        config = LinkModelConfig()
        self.assertEqual(config.semantic_similarity_weight, 0.50)
        self.assertEqual(config.solution_probability_weight, 0.30)
        self.assertEqual(config.thread_relevance_weight, 0.15)
        self.assertEqual(config.response_role_weight, 0.05)
        self.assertAlmostEqual(
            config.semantic_similarity_weight
            + config.solution_probability_weight
            + config.thread_relevance_weight
            + config.response_role_weight,
            1.0,
        )
        self.assertEqual(config.link_threshold, 0.5)
        self.assertEqual(config.candidate_solution_threshold, 0.5)
        self.assertTrue(config.require_same_post)
        self.assertFalse(config.allow_cross_post)
        self.assertTrue(config.use_parent_context)
        self.assertIsNone(config.max_candidates_per_issue)
        self.assertIsNone(config.stop_words)
        self.assertEqual(config.ngram_range, (1, 1))
        self.assertEqual(config.norm, "l2")
        self.assertTrue(config.use_idf)
        self.assertTrue(config.smooth_idf)
        self.assertFalse(config.include_issue_label_in_text)
        self.assertFalse(config.include_timestamp)
        self.assertEqual(config.default_role_score, 0.5)
        self.assertEqual(config.role_scores["solution"], 1.0)
        self.assertEqual(config.role_scores["advice"], 0.8)
        self.assertEqual(config.cross_post_thread_relevance, 0.0)
        self.assertEqual(config.parent_child_thread_relevance, 1.0)

    def test_token_pattern_keeps_every_script(self):
        self.assertEqual(LinkModelConfig().token_pattern, r"(?u)\b\w+\b")

    def test_linker_accepts_config_or_overrides(self):
        linker = TfidfCosineIssueSolutionLinker(link_threshold=0.7)
        self.assertEqual(linker.link_threshold, 0.7)
        self.assertEqual(linker.config.link_threshold, 0.7)

        base = LinkModelConfig(link_threshold=0.4)
        merged = TfidfCosineIssueSolutionLinker(base, link_threshold=0.9)
        self.assertEqual(merged.link_threshold, 0.9)
        self.assertEqual(base.link_threshold, 0.4, "caller's config must not be mutated")

    def test_rejects_non_coercible_threshold(self):
        # Pydantic coerces "0.5" and True to numbers on a config field, which is
        # fine; what must not pass is a value that is not a number at all.
        for bad in ("high", [], {}, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TfidfCosineIssueSolutionLinker(link_threshold=bad)

    def test_rejects_out_of_range_thresholds(self):
        for bad in (-0.1, 1.5, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TfidfCosineIssueSolutionLinker(link_threshold=bad)

    def test_rejects_weights_that_sum_to_zero(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(
                semantic_similarity_weight=0.0,
                solution_probability_weight=0.0,
                thread_relevance_weight=0.0,
                response_role_weight=0.0,
            )
        self.assertIn("sum", str(ctx.exception).lower())

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(semantic_similarity_weight=-1.0)
        self.assertIn("non-negative", str(ctx.exception))

    def test_rejects_out_of_range_candidate_threshold(self):
        with self.assertRaises(ValueError):
            TfidfCosineIssueSolutionLinker(candidate_solution_threshold=1.5)

    def test_rejects_zero_max_candidates(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(max_candidates_per_issue=0)
        self.assertIn("max_candidates_per_issue", str(ctx.exception))

    def test_rejects_impossible_min_df(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(min_df=0)
        self.assertIn("min_df", str(ctx.exception))

    def test_rejects_impossible_max_df(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(max_df=1.5)
        self.assertIn("max_df", str(ctx.exception))

    def test_rejects_min_df_above_max_df(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(min_df=0.6, max_df=0.3)
        self.assertIn("min_df", str(ctx.exception))

    def test_rejects_invalid_ngram_range(self):
        for bad in ((0, 2), (2, 1), (3, 1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as ctx:
                    TfidfCosineIssueSolutionLinker(ngram_range=bad)
                self.assertIn("ngram_range", str(ctx.exception))

    def test_accepts_a_trigram_only_range(self):
        self.assertEqual(TfidfCosineIssueSolutionLinker(ngram_range=(3, 3)).config.ngram_range, (3, 3))

    def test_rejects_out_of_range_role_score(self):
        with self.assertRaises(ValueError) as ctx:
            TfidfCosineIssueSolutionLinker(role_scores={"solution": 1.5})
        self.assertIn("role_scores", str(ctx.exception))

    def test_config_dict_is_json_serialisable_and_complete(self):
        import json

        snapshot = TfidfCosineIssueSolutionLinker().config_dict
        json.dumps(snapshot)  # must not raise
        self.assertEqual(snapshot["_model_name"], "tfidf_cosine_issue_solution")
        self.assertEqual(snapshot["_model_version"], "v1")
        self.assertIn("_confidence_definition", snapshot)
        for key in (
            "semantic_similarity_weight",
            "solution_probability_weight",
            "thread_relevance_weight",
            "response_role_weight",
            "link_threshold",
            "candidate_solution_threshold",
            "role_scores",
            "ngram_range",
            "token_pattern",
            "stop_words",
        ):
            self.assertIn(key, snapshot, f"config snapshot missing {key}")

    def test_model_identity_is_exact(self):
        linker = TfidfCosineIssueSolutionLinker()
        self.assertEqual(linker.name, "tfidf_cosine_issue_solution")
        self.assertEqual(linker.version, "v1")


# ---------------------------------------------------------------------------
# 3. Candidate generation
# ---------------------------------------------------------------------------


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.config = LinkModelConfig()
        self.issue = issue("i1", "the fees are far too high", ["fees"], post_id="p1")
        self.same_post = solution("s1", "ask the warden to lower the fees", post_id="p1")
        self.other_post = solution("s2", "ask the bursar about refunds", post_id="p2")

    def test_is_solution_flag_alone_makes_a_candidate(self):
        accepted, basis = is_solution_candidate(
            SolutionEvidence(document_id="d", is_solution=True, solution_probability=0.01), 0.5
        )
        self.assertTrue(accepted)
        self.assertEqual(basis, "is_solution")

    def test_probability_alone_can_make_a_candidate(self):
        accepted, basis = is_solution_candidate(
            SolutionEvidence(document_id="d", is_solution=False, solution_probability=0.6), 0.5
        )
        self.assertTrue(accepted)
        self.assertEqual(basis, "probability_threshold")

    def test_neither_flag_nor_probability_is_not_a_candidate(self):
        accepted, basis = is_solution_candidate(
            SolutionEvidence(document_id="d", is_solution=False, solution_probability=0.49), 0.5
        )
        self.assertFalse(accepted)
        self.assertEqual(basis, "")

    def test_candidate_threshold_is_configurable(self):
        strict = SolutionEvidence(document_id="d", is_solution=False, solution_probability=0.6)
        self.assertFalse(is_solution_candidate(strict, 0.8)[0])
        self.assertTrue(is_solution_candidate(strict, 0.5)[0])

    def test_same_post_is_accepted(self):
        accepted, rejections = generate_candidates(
            [self.issue], [self.same_post], self.config
        )
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0].same_post)
        self.assertFalse(accepted[0].parent_child)
        self.assertEqual(accepted[0].thread_relevance, 1.0)
        self.assertEqual(accepted[0].reason, REASON_SAME_POST)
        self.assertEqual(rejections.get(REASON_CROSS_THREAD_DISABLED, 0), 0)

    def test_cross_post_is_rejected_by_default(self):
        accepted, rejections = generate_candidates(
            [self.issue], [self.other_post], self.config
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejections[REASON_CROSS_THREAD_DISABLED], 1)

    def test_cross_post_can_be_enabled(self):
        config = LinkModelConfig(allow_cross_post=True)
        accepted, rejections = generate_candidates([self.issue], [self.other_post], config)
        self.assertEqual(len(accepted), 1)
        self.assertFalse(accepted[0].same_post)
        self.assertEqual(accepted[0].reason, REASON_CROSS_POST)
        self.assertEqual(accepted[0].thread_relevance, 0.0)

    def test_parent_reply_is_the_strongest_thread_signal(self):
        reply = solution(
            "s3", "ask the warden", post_id="p1", parent_comment_id="i1-c"
        )
        accepted, _ = generate_candidates([self.issue], [reply], self.config)
        self.assertTrue(accepted[0].parent_child)
        self.assertTrue(accepted[0].same_post)
        self.assertEqual(accepted[0].reason, REASON_PARENT_CHILD)
        self.assertEqual(accepted[0].thread_relevance, 1.0)
        self.assertEqual(accepted[0].relation, REASON_PARENT_CHILD)

    def test_parent_context_can_be_disabled(self):
        config = LinkModelConfig(use_parent_context=False)
        reply = solution("s3", "ask the warden", post_id="p1", parent_comment_id="i1-c")
        accepted, _ = generate_candidates([self.issue], [reply], config)
        self.assertFalse(accepted[0].parent_child)
        self.assertEqual(accepted[0].reason, REASON_SAME_POST)

    def test_unrelated_parent_id_is_not_a_thread_signal(self):
        reply = solution("s3", "ask the warden", post_id="p1", parent_comment_id="someone-else")
        accepted, _ = generate_candidates([self.issue], [reply], self.config)
        self.assertFalse(accepted[0].parent_child)

    def test_missing_post_ids_do_not_fabricate_same_post(self):
        orphan_issue = IssueEvidence(document_id="i9", issue_labels=["fees"])
        orphan_solution = SolutionEvidence(document_id="s9", is_solution=True)
        _, rejections = generate_candidates([orphan_issue], [orphan_solution], self.config)
        self.assertEqual(rejections[REASON_UNRESOLVED_THREAD], 1)
        self.assertEqual(rejections.get(REASON_CROSS_THREAD_DISABLED), None)

    def test_missing_post_ids_can_be_admitted_as_cross_post(self):
        orphan_issue = IssueEvidence(document_id="i9", issue_labels=["fees"])
        orphan_solution = SolutionEvidence(document_id="s9", is_solution=True)
        config = LinkModelConfig(allow_cross_post=True)
        accepted, _ = generate_candidates([orphan_issue], [orphan_solution], config)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].reason, REASON_UNRESOLVED_THREAD)

    def test_issue_without_labels_is_skipped_with_a_reason(self):
        unlabelled = issue("i2", "some text", [], post_id="p1")
        accepted, rejections = generate_candidates([unlabelled], [self.same_post], self.config)
        self.assertEqual(accepted, [])
        self.assertEqual(rejections[REASON_ISSUE_HAS_NO_LABEL], 1)

    def test_blank_labels_count_as_no_labels(self):
        blank = IssueEvidence(document_id="i3", post_id="p1", issue_labels=["  ", ""])
        _, rejections = generate_candidates([blank], [self.same_post], self.config)
        self.assertEqual(rejections[REASON_ISSUE_HAS_NO_LABEL], 1)

    def test_non_solution_is_counted_as_rejected(self):
        chatter = solution("s4", "same bro", post_id="p1", probability=0.05, is_solution=False)
        accepted, rejections = generate_candidates([self.issue], [chatter], self.config)
        self.assertEqual(accepted, [])
        self.assertEqual(rejections[REASON_BELOW_CANDIDATE_THRESHOLD], 1)

    def test_labels_are_expanded_into_separate_candidates(self):
        multi = issue("i5", "fees are high and the hostel is bad", ["fees", "hostel"], post_id="p1")
        accepted, _ = generate_candidates([multi], [self.same_post], self.config)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(
            sorted(c.issue_label for c in accepted), ["fees", "hostel"]
        )
        self.assertEqual(
            {c.solution_document_id for c in accepted}, {"s1"}, "text must be shared"
        )

    def test_duplicate_labels_collapse(self):
        dupes = issue("i6", "fees", ["fees", " fees ", "fees"], post_id="p1")
        self.assertEqual(normalised_labels(dupes), ["fees"])
        accepted, _ = generate_candidates([dupes], [self.same_post], self.config)
        self.assertEqual(len(accepted), 1)

    def test_candidate_cap_is_deterministic_and_counted(self):
        config = LinkModelConfig(max_candidates_per_issue=2)
        solutions = [
            solution(f"s{n}", f"lower the fees {n}", post_id="p1") for n in range(5)
        ]
        first, rejections = generate_candidates([self.issue], solutions, config)
        second, _ = generate_candidates([self.issue], solutions, config)
        kept = sorted(c.solution_document_id for c in first)
        self.assertEqual(kept, ["s0", "s1"])
        self.assertEqual([c.solution_document_id for c in first], [c.solution_document_id for c in second])
        self.assertEqual(rejections[REASON_CAPPED], 3)

    def test_candidate_cap_applies_per_label(self):
        multi = issue("i7", "fees and hostel", ["fees", "hostel"], post_id="p1")
        config = LinkModelConfig(max_candidates_per_issue=1)
        solutions = [solution(f"s{n}", "text", post_id="p1") for n in range(3)]
        accepted, _ = generate_candidates([multi], solutions, config)
        self.assertEqual(len(accepted), 2, "one per label")

    def test_candidate_order_is_input_order_and_label_sorted(self):
        multi = issue("i8", "fees and hostel", ["hostel", "fees"], post_id="p1")
        solutions = [
            solution("s1", "first", post_id="p1"),
            solution("s2", "second", post_id="p1"),
        ]
        accepted, _ = generate_candidates([multi], solutions, self.config)
        self.assertEqual(
            [(c.issue_label, c.solution_document_id) for c in accepted],
            [("fees", "s1"), ("fees", "s2"), ("hostel", "s1"), ("hostel", "s2")],
        )

    def test_rejection_reasons_are_stable_across_runs(self):
        solutions = [
            self.same_post,
            self.other_post,
            solution("s5", "chatter", post_id="p1", probability=0.01, is_solution=False),
        ]
        _, first = generate_candidates([self.issue], solutions, self.config)
        _, second = generate_candidates([self.issue], solutions, self.config)
        self.assertEqual(first, second)

    def test_empty_inputs_produce_no_candidates(self):
        accepted, _ = generate_candidates([], [], self.config)
        self.assertEqual(accepted, [])

    def test_every_rejection_carries_a_reason(self):
        decisions = [
            CandidateDecision(
                issue_document_id="i", issue_label="fees", solution_document_id="s"
            )
        ]
        self.assertFalse(decisions[0].accepted)
        self.assertEqual(decisions[0].relation, "none")

    def test_thread_signals_reports_parent_child_for_cross_post_reply(self):
        config = LinkModelConfig(allow_cross_post=True)
        reply = solution("s3", "text", post_id="p2", parent_comment_id="i1-c")
        same_post, parent_child, relevance, reason = thread_signals(
            self.issue, reply, config
        )
        self.assertFalse(same_post)
        self.assertTrue(parent_child)
        self.assertEqual(relevance, 0.0)
        self.assertEqual(reason, REASON_CROSS_POST)


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.issue = issue("i1", "the hostel mess charges 9000 a month which is too high", ["fees"])
        self.solution = solution(
            "s1",
            "ask the warden to lower the mess charges, they can reduce the monthly fee",
        )
        self.linker = TfidfCosineIssueSolutionLinker()

    def test_every_required_field_is_reported(self):
        result = self.linker.link([self.issue], [self.solution])
        link = result.links[0]
        for field in (
            "issue_document_id",
            "issue_label",
            "solution_document_id",
            "similarity_score",
            "solution_probability",
            "thread_relevance",
            "link_score",
            "linked",
            "confidence",
            "model_name",
            "model_version",
        ):
            self.assertTrue(hasattr(link, field), f"LinkPrediction missing {field}")
        self.assertTrue(link.signals, "signals/evidence must not be empty")

    def test_model_name_and_version_on_every_link_including_rejected(self):
        weak = solution("s2", "ok", probability=0.55, is_solution=True)
        result = self.linker.link([self.issue], [weak], link_threshold=0.99)
        self.assertTrue(result.links)
        for link in result.links:
            self.assertEqual(link.model_name, "tfidf_cosine_issue_solution")
            self.assertEqual(link.model_version, "v1")
        self.assertTrue(any(not link.linked for link in result.links))

    def test_scores_are_preserved_for_unasserted_links(self):
        result = self.linker.link([self.issue], [self.solution], link_threshold=0.99)
        link = result.links[0]
        self.assertFalse(link.linked)
        self.assertGreater(link.similarity_score, 0.0)
        self.assertGreater(link.link_score, 0.0)
        self.assertAlmostEqual(link.link_threshold, 0.99)

    def test_score_matches_the_documented_formula(self):
        result = self.linker.link([self.issue], [self.solution])
        link = result.links[0]
        expected = (
            0.50 * link.score_components["semantic_similarity"]
            + 0.30 * link.score_components["solution_probability"]
            + 0.15 * link.score_components["thread_relevance"]
            + 0.05 * link.score_components["response_role"]
        )
        self.assertAlmostEqual(link.link_score, expected, places=5)

    def test_weights_are_normalised_to_a_0_1_scale(self):
        linker = TfidfCosineIssueSolutionLinker(
            semantic_similarity_weight=2.0,
            solution_probability_weight=1.0,
            thread_relevance_weight=1.0,
            response_role_weight=1.0,
        )
        result = linker.link([self.issue], [self.solution])
        self.assertLessEqual(result.links[0].link_score, 1.0)

    def test_unnormalised_weights_scale_the_score(self):
        result = TfidfCosineIssueSolutionLinker(normalise_weights=False).link(
            [self.issue], [self.solution]
        )
        self.assertGreater(result.links[0].link_score, 0.5)

    def test_semantic_similarity_is_never_nan_and_in_range(self):
        result = TfidfCosineIssueSolutionLinker().link(
            [self.issue, IssueEvidence(document_id="i2", post_id="p1", issue_labels=["x"])],
            [self.solution, SolutionEvidence(document_id="s2", post_id="p1", is_solution=True)],
        )
        for link in result.links:
            self.assertFalse(math.isnan(link.similarity_score))
            self.assertFalse(math.isnan(link.link_score))
            self.assertFalse(math.isnan(link.confidence))
            self.assertGreaterEqual(link.similarity_score, 0.0)
            self.assertLessEqual(link.similarity_score, 1.0)
            self.assertGreaterEqual(link.link_score, 0.0)
            self.assertLessEqual(link.link_score, 1.0)

    def test_related_text_scores_higher_than_unrelated_text(self):
        related = solution("s1", "lower the hostel mess charges for the monthly fee", post_id="p1")
        unrelated = solution(
            "s2", "try a different recipe with garlic and onions tonight", post_id="p1"
        )
        result = self.linker.link([self.issue], [related, unrelated])
        scores = {link.solution_document_id: link.similarity_score for link in result.links}
        self.assertGreater(scores["s1"], scores["s2"])

    def test_zero_vector_issue_scores_zero_and_is_flagged(self):
        empty_issue = IssueEvidence(document_id="i9", post_id="p1", issue_labels=["fees"])
        result = self.linker.link([empty_issue], [self.solution])
        link = result.links[0]
        self.assertEqual(link.similarity_score, 0.0)
        self.assertTrue(link.signals["issue_zero_vector"])
        self.assertIn("zero_vector_warning", link.signals)
        self.assertEqual(result.n_zero_vector_issues, 1)
        self.assertTrue(any("no usable terms" in w for w in result.warnings))

    def test_zero_vector_solution_scores_zero_and_is_flagged(self):
        empty_solution = SolutionEvidence(document_id="s9", post_id="p1", is_solution=True)
        result = self.linker.link([self.issue], [empty_solution])
        link = result.links[0]
        self.assertEqual(link.similarity_score, 0.0)
        self.assertTrue(link.signals["solution_zero_vector"])
        self.assertEqual(result.n_zero_vector_solutions, 1)

    def test_empty_vocabulary_does_not_crash_and_warns(self):
        punct_issue = issue("i1", "!!! ... ???", ["fees"])
        punct_solution = solution("s1", "!!! ??? ...", post_id="p1")
        result = self.linker.link([punct_issue], [punct_solution])
        self.assertEqual(result.n_features, 0)
        for link in result.links:
            self.assertEqual(link.similarity_score, 0.0)
        self.assertTrue(any("empty vocabulary" in w for w in result.warnings))

    def test_high_threshold_suppresses_links_low_threshold_asserts(self):
        args = ([self.issue], [self.solution])
        self.assertGreater(TfidfCosineIssueSolutionLinker().link(*args).n_linked, 0)
        self.assertEqual(
            TfidfCosineIssueSolutionLinker().link(*args, link_threshold=1.0).n_linked, 0
        )

    def test_linked_is_exactly_the_threshold_comparison(self):
        args = ([self.issue], [self.solution])
        result = TfidfCosineIssueSolutionLinker().link(*args)
        for link in result.links:
            self.assertEqual(link.linked, link.link_score >= link.link_threshold)

    def test_invalid_runtime_threshold_raises(self):
        for bad in ("0.5", -0.2, 1.1, [1], "high"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.linker.link([self.issue], [self.solution], link_threshold=bad)

    def test_runtime_threshold_of_none_means_use_the_configured_one(self):
        result = self.linker.link([self.issue], [self.solution], link_threshold=None)
        self.assertEqual(result.link_threshold, self.linker.config.link_threshold)

    def test_confidence_is_documented_as_not_a_probability(self):
        note = self.linker.config_dict["_confidence_definition"]
        self.assertIn("NOT a calibrated probability", note)
        result = self.linker.link([self.issue], [self.solution])
        link = result.links[0]
        expected = max(0.0, min(1.0, 1.0 - 2.0 * abs(link.link_score - link.link_threshold)))
        self.assertAlmostEqual(link.confidence, expected, places=5)

    def test_confidence_is_lowest_exactly_at_the_threshold(self):
        boundary = self.linker.link([self.issue], [self.solution])
        link = boundary.links[0]
        self.assertAlmostEqual(
            link.confidence, 1.0 - 2.0 * abs(link.link_score - link.link_threshold), places=5
        )

    def test_role_score_adjusts_weight_without_gating(self):
        advice = solution(
            "s1", "ask the warden about the mess charges", post_id="p1", response_type="advice"
        )
        direct = solution(
            "s2", "ask the warden about the mess charges", post_id="p1", response_type="solution"
        )
        result = self.linker.link([self.issue], [advice, direct])
        by_id = {link.solution_document_id: link for link in result.links}
        self.assertEqual(by_id["s1"].response_role_score, 0.8)
        self.assertEqual(by_id["s2"].response_role_score, 1.0)
        self.assertGreater(by_id["s2"].link_score, by_id["s1"].link_score)

    def test_advice_is_candidate_but_never_promoted_to_is_solution(self):
        """A role may carry weight; it may not rewrite the upstream boolean."""
        advice = solution(
            "s1", "ask the warden", post_id="p1", response_type="advice", is_solution=False,
            probability=0.6,
        )
        self.assertFalse(advice.is_solution, "the linker must not rewrite the flag")
        result = self.linker.link([self.issue], [advice])
        self.assertEqual(result.n_scored_links, 1)
        self.assertEqual(result.links[0].solution_document_id, "s1")
        self.assertEqual(result.links[0].response_role_score, 0.8)
        self.assertFalse(advice.is_solution, "still false after linking")

    def test_low_probability_advice_is_not_a_candidate_at_the_default_threshold(self):
        weak = solution(
            "s1", "ask the warden", post_id="p1", response_type="advice", is_solution=False,
            probability=0.4,
        )
        result = self.linker.link([self.issue], [weak])
        self.assertEqual(result.n_scored_links, 0)
        self.assertEqual(
            result.candidate_rejections[REASON_BELOW_CANDIDATE_THRESHOLD], 1
        )

    def test_unknown_role_gets_the_neutral_default(self):
        odd = solution("s1", "ask the warden", post_id="p1", response_type="mumble")
        result = self.linker.link([self.issue], [odd])
        self.assertEqual(result.links[0].response_role_score, 0.5)

    def test_missing_role_gets_the_neutral_default(self):
        none = SolutionEvidence(
            document_id="s1", post_id="p1", normalized_text="ask the warden", is_solution=True
        )
        result = self.linker.link([self.issue], [none])
        self.assertEqual(result.links[0].response_role_score, 0.5)

    def test_signal_only_link_still_records_weak_semantics(self):
        strong_detector = solution(
            "s1", "unrelated cooking recipe", post_id="p1", probability=1.0
        )
        result = self.linker.link([self.issue], [strong_detector])
        link = result.links[0]
        self.assertEqual(link.similarity_score, 0.0)
        self.assertFalse(link.is_strong_candidate)

    def test_include_issue_label_in_text_is_off_by_default(self):
        result = self.linker.link([self.issue], [self.solution])
        self.assertNotIn("fees", result.links[0].signals["issue_text_used"])

    def test_label_in_text_changes_the_similarity(self):
        off = self.linker.link([self.issue], [self.solution]).links[0].similarity_score
        on = (
            TfidfCosineIssueSolutionLinker(include_issue_label_in_text=True)
            .link([self.issue], [self.solution])
            .links[0]
            .similarity_score
        )
        self.assertNotAlmostEqual(off, on, places=6)

    def test_issue_text_not_the_label_drives_similarity(self):
        """A label shared with a solution but absent from the issue text."""
        issue_evidence = issue(
            "i1", "the monthly charge is far beyond what I can afford", ["financial_aid"]
        )
        shares_label_only = solution(
            "s1", "financial_aid is worth applying for at the office", post_id="p1"
        )
        shares_text = solution(
            "s2", "the monthly charge can be reduced by applying for aid", post_id="p1"
        )
        result = self.linker.link([issue_evidence], [shares_label_only, shares_text])
        by_id = {link.solution_document_id: link for link in result.links}
        self.assertGreater(
            by_id["s2"].similarity_score,
            by_id["s1"].similarity_score,
            "text overlap must matter more than a shared label token",
        )

    def test_max_features_caps_the_vocabulary(self):
        result = TfidfCosineIssueSolutionLinker(max_features=5).link(
            [self.issue], [self.solution]
        )
        self.assertLessEqual(result.n_features, 5)

    def test_stop_words_stay_off_by_default(self):
        self.assertIsNone(self.linker.config.stop_words)

    def test_n_features_is_reported(self):
        result = self.linker.link([self.issue], [self.solution])
        self.assertGreater(result.n_features, 0)


# ---------------------------------------------------------------------------
# 5. Multi-label and many-to-many
# ---------------------------------------------------------------------------


class TestMultiLabelAndGraph(unittest.TestCase):
    def setUp(self):
        self.linker = TfidfCosineIssueSolutionLinker()

    def test_one_issue_to_many_solutions(self):
        evidence = issue("i1", "the fees are far too high for me", ["fees"])
        solutions = [
            solution("s1", "ask the warden to reduce the mess fee", post_id="p1"),
            solution("s2", "apply for the scholarship to offset the fee", post_id="p1"),
        ]
        result = self.linker.link([evidence], solutions)
        self.assertEqual(len(result.solutions_for("i1")), 2)
        self.assertEqual(result.n_linked_solutions, 2)

    def test_many_issues_to_one_solution(self):
        solutions = [solution("s1", "lower the fee by asking the warden", post_id="p1")]
        issues = [
            issue("i1", "the hostel fee is too high", ["fees"], post_id="p1"),
            issue("i2", "the hostel is badly maintained", ["hostel"], post_id="p1"),
        ]
        result = self.linker.link(issues, solutions)
        self.assertEqual(result.issues_for("s1"), ["i1", "i2"])

    def test_multi_label_issue_keeps_every_label_separate(self):
        evidence = issue(
            "i1", "the hostel fee is high and the food quality is poor", ["fees", "food"]
        )
        solutions = [solution("s1", "ask the warden about the fee", post_id="p1")]
        result = self.linker.link([evidence], solutions)
        self.assertEqual(len(result.links), 2)
        self.assertEqual(sorted(l.issue_label for l in result.links), ["fees", "food"])
        self.assertEqual(result.links_for_issue("i1"), result.links)
        self.assertEqual(len(result.solutions_for("i1", "fees")), 1)
        self.assertEqual(len(result.solutions_for("i1", "food")), 1)

    def test_a_link_covers_exactly_one_label(self):
        evidence = issue(
            "i1", "the hostel fee is high and the food quality is poor", ["fees", "food"]
        )
        solutions = [solution("s1", "ask the warden about the fee", post_id="p1")]
        result = self.linker.link([evidence], solutions)
        for link in result.links:
            self.assertIn(link.issue_label, {"fees", "food"})

    def test_labels_by_document_is_reported(self):
        evidence = issue("i1", "fee and food", ["food", "fees"])
        result = self.linker.link([evidence], [])
        self.assertEqual(result.issue_labels_by_document, {"i1": ["fees", "food"]})
        self.assertEqual(result.n_issue_labels, 2)

    def test_linked_pairs_are_minimal_triples(self):
        evidence = issue("i1", "the fee is high", ["fees"])
        solutions = [solution("s1", "ask the warden", post_id="p1")]
        result = self.linker.link([evidence], solutions)
        for pair in result.linked_pairs():
            self.assertEqual(set(pair), {"issue_document_id", "issue_label", "solution_document_id"})


# ---------------------------------------------------------------------------
# 6. Multilingual
# ---------------------------------------------------------------------------


class TestMultilingual(unittest.TestCase):
    def setUp(self):
        self.linker = TfidfCosineIssueSolutionLinker()

    def _similarity(self, issue_text, solution_text, **issue_kwargs):
        evidence = issue("i1", issue_text, ["fees"], **issue_kwargs)
        candidate = solution("s1", solution_text, post_id="p1")
        result = self.linker.link([evidence], [candidate])
        return result.links[0].similarity_score

    def test_english_related_beats_english_unrelated(self):
        related = self._similarity(
            "the hostel mess charges 9000 a month which is too high",
            "ask the warden to reduce the mess charges each month",
        )
        unrelated = self._similarity(
            "the hostel mess charges 9000 a month which is too high",
            "here is a recipe for garlic bread",
        )
        self.assertGreater(related, unrelated)

    def test_bengali_text_is_not_stripped_to_nothing(self):
        similarity = self._similarity(
            "হোস্টেলের মেসের খরচ অনেক বেশি",
            "ওয়ার্ডেনের সাথে কথা বলে খরচ কমান",
            language="bn",
            script="Bengali",
        )
        self.assertGreater(similarity, 0.0)
        self.assertGreater(self.linker.link(
            [issue("i1", "হোস্টেলের মেসের খরচ অনেক বেশি", ["fees"], language="bn", script="Bengali")],
            [solution("s1", "ওয়ার্ডেনের সাথে কথা বলে খরচ কমান", post_id="p1")],
        ).n_features, 0)

    def test_odia_text_is_not_stripped_to_nothing(self):
        result = self.linker.link(
            [issue("i1", "ହୋଷ୍ଟେଲ ଖର୍ଚ ବହୁତ ଅଧିକ", ["fees"], language="or", script="Oriya")],
            [solution("s1", "ୱାର୍ଡନଙ୍କ ସହ କଥା କରି ଖର୍ଚ କମାନ୍ତୁ", post_id="p1")],
        )
        self.assertGreater(result.n_features, 0)
        self.assertGreater(result.links[0].similarity_score, 0.0)

    def test_romanized_hindi_and_hinglish_share_terms(self):
        related = self._similarity(
            "hostel ka mess charge bahut zyada hai paise nahi hain",
            "warden se baat karke mess charge kam karwao",
        )
        unrelated = self._similarity(
            "hostel ka mess charge bahut zyada hai paise nahi hain",
            "yesterday i watched a very long film about space travel",
        )
        self.assertGreater(related, unrelated)

    def test_code_mixed_text_is_scored_not_stripped(self):
        evidence = issue(
            "i1",
            "the wifi drops every 5 min, tried reset router already",
            ["network"],
            language="hinglish",
            script="Latin",
            is_code_mixed=True,
        )
        candidate = solution(
            "s1",
            "run `ipconfig /flushdns` and reconnect to the access point",
            post_id="p1",
            language="en",
            is_code_mixed=True,
        )
        result = self.linker.link([evidence], [candidate])
        self.assertGreater(result.links[0].similarity_score, 0.0)
        self.assertTrue(result.links[0].is_code_mixed)
        self.assertTrue(result.links[0].signals["is_code_mixed_issue"])
        self.assertTrue(result.links[0].signals["is_code_mixed_solution"])

    def test_emoji_only_text_does_not_crash(self):
        result = self.linker.link(
            [issue("i1", "\U0001f600\U0001f601\U0001f602", ["fees"])],
            [solution("s1", "\U0001f680\U0001f929", post_id="p1")],
        )
        for link in result.links:
            self.assertEqual(link.similarity_score, 0.0)
            self.assertFalse(math.isnan(link.link_score))

    def test_language_metadata_is_echoed_not_used_for_branching(self):
        result = self.linker.link(
            [issue("i1", "fees high", ["fees"], language="bn", script="Bengali")],
            [solution("s1", "ask the warden", post_id="p1", language="en", script="Latin")],
        )
        link = result.links[0]
        self.assertEqual(link.language, "en")
        self.assertEqual(link.script, "Latin")

    def test_same_text_in_two_scripts_yields_high_similarity(self):
        """No translation, no transliteration: identical text matches itself."""
        evidence = issue("i1", "mess charge bahut zyada hai", ["fees"], language="hinglish")
        candidate = solution("s1", "mess charge bahut zyada hai", post_id="p1", language="hinglish")
        result = self.linker.link([evidence], [candidate])
        self.assertGreater(result.links[0].similarity_score, 0.9)


# ---------------------------------------------------------------------------
# 7. Placeholders and non-mutation
# ---------------------------------------------------------------------------


class TestPlaceholdersAndNonMutation(unittest.TestCase):
    def test_strip_placeholder_tokens_removes_all_known_shapes(self):
        cleaned = strip_placeholder_tokens("see <URL> and <MENTION> and <EMOJI_SAD> ok")
        self.assertNotIn("<", cleaned)
        self.assertIn("ok", cleaned)

    def test_strip_placeholder_tokens_leaves_real_angle_brackets_alone(self):
        self.assertIn("a < b", strip_placeholder_tokens("a < b"))

    def test_strip_placeholder_tokens_handles_empty(self):
        self.assertEqual(strip_placeholder_tokens(""), "")
        self.assertEqual(strip_placeholder_tokens(None), "")

    def test_placeholder_only_text_becomes_a_zero_vector(self):
        evidence = IssueEvidence(
            document_id="i1", post_id="p1", issue_labels=["fees"],
            normalized_text="<URL> <MENTION> <EMOJI_SAD>",
        )
        result = TfidfCosineIssueSolutionLinker().link(
            [evidence], [solution("s1", "ask the warden", post_id="p1")]
        )
        self.assertEqual(result.links[0].similarity_score, 0.0)
        self.assertTrue(result.links[0].signals["issue_zero_vector"])

    def test_placeholder_stripping_does_not_mutate_the_evidence(self):
        evidence = IssueEvidence(
            document_id="i1", post_id="p1", issue_labels=["fees"],
            normalized_text="the fee at <URL> is too high",
            original_text="the fee at <URL> is too high",
        )
        candidate = SolutionEvidence(
            document_id="s1", post_id="p1", normalized_text="ask <MENTION> about <URL>",
            original_text="ask <MENTION> about <URL>",
        )
        before = evidence.model_dump()
        before_solution = candidate.model_dump()
        TfidfCosineIssueSolutionLinker().link([evidence], [candidate])
        self.assertEqual(evidence.model_dump(), before)
        self.assertEqual(candidate.model_dump(), before_solution)
        self.assertIn("<URL>", evidence.normalized_text)
        self.assertIn("<MENTION>", candidate.normalized_text)

    def test_placeholder_stripping_is_visible_in_the_audited_text(self):
        evidence = issue("i1", "the fee at <URL> is high", ["fees"])
        result = TfidfCosineIssueSolutionLinker().link(
            [evidence], [solution("s1", "ask the warden", post_id="p1")]
        )
        self.assertNotIn("<URL>", result.links[0].signals["issue_text_used"])
        self.assertIn("<URL>", result.links[0].issue_text, "echoed text stays intact")

    def test_placeholder_stripping_can_be_disabled(self):
        evidence = issue("i1", "the fee at <URL> is high", ["fees"])
        result = TfidfCosineIssueSolutionLinker(drop_placeholder_tokens=False).link(
            [evidence], [solution("s1", "ask the warden", post_id="p1")]
        )
        self.assertIn("<URL>", result.links[0].signals["issue_text_used"])

    def test_link_does_not_mutate_any_evidence_object(self):
        issues = [issue("i1", "fees high", ["fees"]), issue("i2", "food bad", ["food"])]
        solutions = [solution("s1", "ask the warden", post_id="p1")]
        snapshot = [i.model_dump() for i in issues] + [s.model_dump() for s in solutions]
        linker = TfidfCosineIssueSolutionLinker()
        linker.link(issues, solutions)
        linker.link(issues, solutions)
        after = [i.model_dump() for i in issues] + [s.model_dump() for s in solutions]
        self.assertEqual(snapshot, after)

    def test_timestamp_is_absent_by_default_and_present_on_request(self):
        args = ([issue("i1", "fees high", ["fees"])], [solution("s1", "ask", post_id="p1")])
        self.assertIsNone(TfidfCosineIssueSolutionLinker().link(*args).linked_at)
        stamped = TfidfCosineIssueSolutionLinker(include_timestamp=True).link(*args)
        self.assertIsNotNone(stamped.linked_at)
        self.assertIn("T", stamped.linked_at)


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):
    def _corpus(self):
        issues = [
            issue("i1", "the hostel fee is too high", ["fees", "hostel"]),
            issue("i2", "the wifi keeps dropping", ["network"]),
            issue("i3", "the food is not edible", ["food"]),
        ]
        solutions = [
            solution("s1", "ask the warden to lower the hostel fee", post_id="p1"),
            solution("s2", "restart the router for the wifi", post_id="p1"),
            solution("s3", "use a different mess provider", post_id="p2"),
            solution("s4", "nice post", post_id="p1", probability=0.05, is_solution=False),
        ]
        return issues, solutions

    def test_repeated_runs_are_byte_identical(self):
        linker = TfidfCosineIssueSolutionLinker()
        first = linker.link(*self._corpus())
        second = linker.link(*self._corpus())
        self.assertEqual(first.model_dump(), second.model_dump())

    def test_input_order_does_not_change_the_output(self):
        linker = TfidfCosineIssueSolutionLinker()
        issues, solutions = self._corpus()
        forward = linker.link(issues, solutions)
        backward = linker.link(list(reversed(issues)), list(reversed(solutions)))
        key = lambda r: sorted(  # noqa: E731
            (l.issue_document_id, l.issue_label, l.solution_document_id, l.link_score, l.linked)
            for l in r.links
        )
        self.assertEqual(key(forward), key(backward))

    def test_link_order_is_total_and_stable(self):
        linker = TfidfCosineIssueSolutionLinker()
        result = linker.link(*self._corpus())
        tuples = [
            (l.issue_document_id, l.issue_label, l.solution_document_id) for l in result.links
        ]
        self.assertEqual(len(tuples), len(set(tuples)), "no duplicate triples")
        self.assertEqual(tuples, linker.link(*self._corpus()).links and tuples)

    def test_asserted_links_sort_before_unasserted(self):
        result = TfidfCosineIssueSolutionLinker().link(*self._corpus())
        flags = [link.linked for link in result.links]
        self.assertEqual(flags, sorted(flags, reverse=True))

    def test_equal_scores_break_ties_on_ids(self):
        issues = [issue("i2", "same words here", ["fees"]), issue("i1", "same words here", ["fees"])]
        solutions = [solution("s2", "same words here", post_id="p1")]
        result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
        ids = [link.issue_document_id for link in result.links]
        self.assertEqual(ids, sorted(ids))


# ---------------------------------------------------------------------------
# 9. Result behaviour
# ---------------------------------------------------------------------------


class TestResultBehaviour(unittest.TestCase):
    def setUp(self):
        self.evidence = issue("i1", "the hostel fee is too high", ["fees"])
        self.solutions = [
            solution("s1", "ask the warden to lower the fee", post_id="p1"),
            solution("s2", "apply for financial aid", post_id="p1"),
        ]
        self.result = TfidfCosineIssueSolutionLinker().link([self.evidence], self.solutions)

    def test_result_counts_are_consistent(self):
        result = self.result
        self.assertEqual(result.n_issues, 1)
        self.assertEqual(result.n_solutions, 2)
        self.assertEqual(result.n_scored_links, len(result.links))
        self.assertEqual(result.n_linked, len([l for l in result.links if l.linked]))
        self.assertEqual(
            result.n_linked_issues, len({l.issue_document_id for l in result.links if l.linked})
        )

    def test_model_identity_on_the_result(self):
        self.assertEqual(self.result.model_name, "tfidf_cosine_issue_solution")
        self.assertEqual(self.result.model_version, "v1")
        self.assertIn("scikit-learn", self.result.library_versions)

    def test_lookups(self):
        self.assertTrue(self.result.links_for_issue("i1"))
        self.assertEqual(self.result.links_for_issue("nope"), [])
        self.assertTrue(self.result.links_for_solution("s1"))
        self.assertEqual(self.result.labels_for("i1"), ["fees"])
        self.assertEqual(self.result.solutions_for("nope"), [])

    def test_links_at_threshold_preserves_scores(self):
        high = self.result.links_at_threshold(0.99)
        self.assertTrue(all(not link.linked for link in high))
        for original, derived in zip(self.result.links, high):
            self.assertEqual(original.link_score, derived.link_score)
            self.assertEqual(original.similarity_score, derived.similarity_score)
            self.assertEqual(derived.link_threshold, 0.99)

    def test_links_at_threshold_does_not_mutate_the_result(self):
        before = self.result.model_dump()
        self.result.links_at_threshold(0.99)
        self.result.links_at_threshold(0.1)
        self.assertEqual(self.result.model_dump(), before)

    def test_links_at_threshold_rejects_invalid_thresholds(self):
        for bad in (-0.1, 1.5, "0.5", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.result.links_at_threshold(bad)

    def test_summary_lists_asserted_links(self):
        summary = self.result.format_summary()
        self.assertIn("Issue-Solution Linking", summary)
        self.assertIn("asserted links", summary)
        self.assertIn("fees", summary)

    def test_summary_preserved_but_does_not_overstate(self):
        low = TfidfCosineIssueSolutionLinker().link([self.evidence], self.solutions)
        high = TfidfCosineIssueSolutionLinker().link(
            [self.evidence], self.solutions, link_threshold=0.99
        )
        self.assertIn("not asserted", high.format_summary())
        self.assertIn("scores preserved", high.format_summary())
        self.assertNotIn("no candidate was scored", low.format_summary())

    def test_result_is_json_serialisable(self):
        import json

        json.dumps(self.result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# 10. Annotation schema
# ---------------------------------------------------------------------------


class TestAnnotationSchema(unittest.TestCase):
    def test_canonical_columns(self):
        self.assertEqual(
            LINK_CSV_COLUMNS,
            [
                "issue_document_id",
                "issue_label",
                "solution_document_id",
                "linked",
                "rationale",
            ],
        )
        self.assertIn("issue_solution_links.csv", DEFAULT_LINK_CSV)

    def test_annotation_validation_is_read_only(self):
        """Matching the solution stage: validation never rewrites the record."""
        annotation = LinkAnnotation(
            issue_document_id="i1", issue_label=" fees ", solution_document_id="s1", linked=True
        )
        before = annotation.model_dump()
        self.assertTrue(annotation.validate_annotation())
        self.assertEqual(annotation.model_dump(), before, "validation must not write back")

    def test_blank_label_is_refused_by_validate_annotation(self):
        with self.assertRaises(ValueError) as ctx:
            LinkAnnotation(
                issue_document_id="i1", issue_label="   ", solution_document_id="s1", linked=True
            ).validate_annotation()
        self.assertIn("issue_label", str(ctx.exception))

    def test_blank_label_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            LinkAnnotation(
                issue_document_id="i1", issue_label="   ", solution_document_id="s1", linked=True
            ).validate_annotation()
        self.assertIn("issue_label", str(ctx.exception))

    def test_blank_linked_cell_is_refused_not_defaulted(self):
        for blank in ("", "   ", None):
            with self.subTest(blank=blank):
                with self.assertRaises(ValueError) as ctx:
                    parse_linked_cell(blank, "i1|fees|s1")
                self.assertIn("no value", str(ctx.exception))

    def test_unrecognised_linked_value_is_refused(self):
        with self.assertRaises(ValueError):
            parse_linked_cell("maybe", "i1|fees|s1")

    def test_linked_accepts_documented_spellings(self):
        for truthy in ("true", "TRUE", "1", "t", "yes", "Y", "on", True):
            with self.subTest(value=truthy):
                self.assertTrue(parse_linked_cell(truthy))
        for falsy in ("false", "FALSE", "0", "f", "no", "N", "off", False):
            with self.subTest(value=falsy):
                self.assertFalse(parse_linked_cell(falsy))

    def test_parse_bool_cell_is_shared_between_stages(self):
        self.assertTrue(parse_bool_cell("yes"))
        with self.assertRaises(ValueError):
            parse_bool_cell("perhaps")

    def test_parse_float_cell_validates(self):
        self.assertAlmostEqual(parse_float_cell("0.5"), 0.5)
        for bad in ("", None, "abc"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_float_cell(bad)
        with self.assertRaises(ValueError):
            parse_float_cell(1.5, minimum=0.0, maximum=1.0)

    def test_pair_key_identity(self):
        annotation = LinkAnnotation(
            issue_document_id="i1", issue_label="fees", solution_document_id="s1", linked=True
        )
        self.assertEqual(annotation.pair_key(), "i1|fees|s1")

    def test_corpus_splits_positives_and_negatives(self):
        corpus = LinkCorpus.from_records(
            [
                {"issue_document_id": "i1", "issue_label": "fees", "solution_document_id": "s1", "linked": True},
                {"issue_document_id": "i1", "issue_label": "food", "solution_document_id": "s1", "linked": False},
            ]
        )
        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus.positive_keys(), ["i1|fees|s1"])
        self.assertEqual(corpus.negative_keys(), ["i1|food|s1"])
        self.assertEqual(corpus.conflicting_pair_keys(), [])

    def test_conflicting_judgements_are_reported_not_hidden(self):
        corpus = LinkCorpus.from_records(
            [
                {"issue_document_id": "i1", "issue_label": "fees", "solution_document_id": "s1", "linked": True},
                {"issue_document_id": "i1", "issue_label": "fees", "solution_document_id": "s1", "linked": False},
            ]
        )
        self.assertEqual(corpus.conflicting_pair_keys(), ["i1|fees|s1"])

    def test_csv_round_trip(self):
        corpus = LinkCorpus.from_records(
            [
                {
                    "issue_document_id": "i1",
                    "issue_label": "fees",
                    "solution_document_id": "s1",
                    "linked": True,
                    "rationale": "same thread, direct reply",
                },
                {
                    "issue_document_id": "i1",
                    "issue_label": "food",
                    "solution_document_id": "s1",
                    "linked": False,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "links.csv")
            corpus.to_csv(path)
            reloaded = LinkCorpus.from_csv(path)
        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded.positive_keys(), corpus.positive_keys())
        self.assertEqual(reloaded.negative_keys(), corpus.negative_keys())

    def test_csv_missing_columns_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "links.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("issue_document_id,solution_document_id,linked\ni1,s1,true\n")
            with self.assertRaises(ValueError) as ctx:
                LinkCorpus.from_csv(path)
        self.assertIn("issue_label", str(ctx.exception))

    def test_csv_blank_judgement_is_refused_with_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "links.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "issue_document_id,issue_label,solution_document_id,linked\ni1,fees,s1,\n"
                )
            with self.assertRaises(ValueError) as ctx:
                LinkCorpus.from_csv(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_missing_csv_explains_that_no_dataset_ships(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            LinkCorpus.from_csv("data/annotations/issue_solution_links.csv")
        message = str(ctx.exception)
        self.assertIn("never a fabricated dataset", message)

    def test_no_fabricated_dataset_is_shipped(self):
        """The schema exists; the data must not."""
        self.assertFalse(
            os.path.exists(
                os.path.join(
                    os.path.dirname(PACKAGE_DIR), "..", "..", "..", "data",
                    "annotations", "issue_solution_links.csv",
                )
            ),
            "a link-annotation dataset must not be fabricated",
        )

    def test_normalise_labels_helper(self):
        self.assertEqual(normalise_labels([" b ", "a", "b", ""]), ["a", "b"])
        self.assertEqual(normalise_labels(None), [])


# ---------------------------------------------------------------------------
# 11. Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.evidence = issue("i1", "the hostel fee is too high", ["fees"])
        self.solutions = [
            solution("s1", "ask the warden to lower the fee", post_id="p1"),
            solution("s2", "try this garlic bread recipe", post_id="p1"),
        ]
        self.result = TfidfCosineIssueSolutionLinker().link([self.evidence], self.solutions)

    def _corpus(self, **pairs):
        """Build a corpus from (solution_document_id, linked) keyword pairs."""
        return LinkCorpus.from_records(
            [
                {
                    "issue_document_id": "i1",
                    "issue_label": "fees",
                    "solution_document_id": solution_id,
                    "linked": linked,
                }
                for solution_id, linked in pairs.items()
            ]
        )

    def test_refuses_to_evaluate_without_annotations(self):
        with self.assertRaises(ValueError) as ctx:
            evaluate_linking(self.result, LinkCorpus())
        self.assertIn("no link precision/recall", str(ctx.exception))

    def test_refuses_a_none_corpus(self):
        with self.assertRaises(ValueError):
            evaluate_linking(self.result, None)

    def test_threshold_sweep_also_refuses_an_empty_corpus(self):
        with self.assertRaises(ValueError):
            threshold_sweep(self.result, LinkCorpus(), [0.3, 0.5])

    def test_confusion_matrix_on_known_synthetic_judgements(self):
        """Arithmetic test with labels declared inline - NOT a quality claim."""
        corpus = self._corpus(s1=True, s2=False)
        evaluation = evaluate_linking(self.result, corpus)
        linked = {l.solution_document_id for l in self.result.links if l.linked}
        expected_tp = 1 if "s1" in linked else 0
        expected_fp = 1 if "s2" in linked else 0
        self.assertEqual(evaluation.true_positive, expected_tp)
        self.assertEqual(evaluation.false_positive, expected_fp)
        self.assertEqual(evaluation.false_negative, 1 - expected_tp)
        self.assertEqual(evaluation.annotated_pairs, 2)
        self.assertEqual(evaluation.scored_annotated_pairs, 2)

    def test_precision_and_recall_arithmetic(self):
        corpus = self._corpus(s1=True, s2=True)
        evaluation = evaluate_linking(self.result, corpus)
        tp, fp, fn = evaluation.true_positive, evaluation.false_positive, evaluation.false_negative
        if tp + fp:
            self.assertAlmostEqual(evaluation.precision, tp / (tp + fp))
        if tp + fn:
            self.assertAlmostEqual(evaluation.recall, tp / (tp + fn))
        if evaluation.precision and evaluation.recall:
            self.assertAlmostEqual(
                evaluation.f1,
                2 * evaluation.precision * evaluation.recall
                / (evaluation.precision + evaluation.recall),
            )

    def test_undefined_ratios_are_none_not_zero(self):
        corpus = self._corpus(s1=False)
        evaluation = evaluate_linking(self.result, corpus)
        self.assertTrue(
            evaluation.precision is None or evaluation.precision == 0.0,
            "precision with no asserted links must be None or 0, never invented",
        )
        corpus_all_positive = self._corpus(
            s1=True, s2=True
        )
        all_positive = evaluate_linking(self.result, corpus_all_positive)
        self.assertIsNone(all_positive.specificity, "no negatives means no specificity")

    def test_degenerate_corpus_is_flagged(self):
        corpus = self._corpus(
            s1=False, s2=False
        )
        self.assertTrue(evaluate_linking(self.result, corpus).is_degenerate_corpus)

    def test_annotated_pair_never_scored_is_reported_not_hidden(self):
        corpus = self._corpus(s1=True, **{"s-elsewhere": True})
        evaluation = evaluate_linking(self.result, corpus)
        self.assertIn("i1|fees|s-elsewhere", evaluation.unscored_annotated_pairs)
        self.assertIn("i1|fees|s-elsewhere", evaluation.uncovered_positives)
        self.assertIn("never scored", evaluation.format_report())

    def test_conflicting_annotations_can_be_refused_or_reported(self):
        corpus = self._corpus(s1=True)
        corpus = LinkCorpus(
            annotations=corpus.annotations
            + [
                LinkAnnotation(
                    issue_document_id="i1", issue_label="fees",
                    solution_document_id="s1", linked=False,
                )
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            evaluate_linking(self.result, corpus, on_conflict="raise")
        self.assertIn("contradictory", str(ctx.exception))
        reported = evaluate_linking(self.result, corpus, on_conflict="report")
        self.assertEqual(reported.conflicting_judgements, ["i1|fees|s1"])
        self.assertIn("annotated both", reported.format_report())

    def test_rejects_unknown_conflict_policy(self):
        corpus = self._corpus(s1=True)
        with self.assertRaises(ValueError):
            evaluate_linking(self.result, corpus, on_conflict="ignore")

    def test_report_states_the_no_dataset_provenance(self):
        corpus = self._corpus(
            s1=True, s2=False
        )
        report = evaluate_linking(self.result, corpus).format_report()
        self.assertIn("no such dataset ships", report.lower())

    def test_threshold_sweep_reuses_scores(self):
        corpus = self._corpus(
            s1=True, s2=True
        )
        evaluations = threshold_sweep(self.result, corpus, [0.1, 0.5, 0.9])
        self.assertEqual([e.link_threshold for e in evaluations], [0.1, 0.5, 0.9])
        self.assertLessEqual(
            sum(e.false_positive for e in evaluations),
            sum(e.false_positive for e in evaluations[:1]),
        )
        self.assertEqual(self.result.links[0].link_threshold, 0.5, "sweep must not mutate")


# ---------------------------------------------------------------------------
# 12. Run reports and empty inputs
# ---------------------------------------------------------------------------


class TestRunReportAndEdgeCases(unittest.TestCase):
    def test_run_report_from_inputs(self):
        linker = TfidfCosineIssueSolutionLinker()
        report = linker.run_report(
            [issue("i1", "the fee is too high", ["fees"])],
            [solution("s1", "ask the warden", post_id="p1")],
        )
        self.assertIsInstance(report, LinkingRunReport)
        self.assertEqual(report.n_issue_documents, 1)
        self.assertEqual(report.n_links_asserted, report.n_candidates_scored)
        self.assertGreater(report.n_features, 0)
        self.assertEqual(report.model_name, "tfidf_cosine_issue_solution")

    def test_run_report_from_an_existing_result(self):
        linker = TfidfCosineIssueSolutionLinker()
        args = ([issue("i1", "the fee is too high", ["fees"])], [solution("s1", "ask", post_id="p1")])
        result = linker.link(*args)
        self.assertEqual(linker.run_report(*args, result=result).n_candidates_scored, result.n_scored_links)

    def test_run_report_flags_a_degenerate_run(self):
        linker = TfidfCosineIssueSolutionLinker()
        report = linker.run_report(
            [issue("i1", "the fee is too high", ["fees"])],
            [solution("s1", "nice post", post_id="p9", probability=0.01, is_solution=False)],
        )
        self.assertEqual(report.n_candidates_scored, 0)
        self.assertTrue(report.is_degenerate)
        self.assertFalse(report.produced_links)
        self.assertTrue(any("no candidate" in w for w in report.warnings))

    def test_empty_inputs_produce_an_empty_but_valid_result(self):
        result = TfidfCosineIssueSolutionLinker().link([], [])
        self.assertEqual(result.links, [])
        self.assertEqual(result.n_linked, 0)
        self.assertEqual(result.n_issues, 0)
        self.assertIn("no candidate", result.format_summary())

    def test_none_inputs_are_tolerated(self):
        result = TfidfCosineIssueSolutionLinker().link(None, None)
        self.assertEqual(result.n_issues, 0)
        self.assertEqual(result.n_solutions, 0)

    def test_issues_without_solutions(self):
        result = TfidfCosineIssueSolutionLinker().link([issue("i1", "fees", ["fees"])], [])
        self.assertEqual(result.n_scored_links, 0)
        self.assertEqual(result.n_solutions, 0)

    def test_solutions_without_issues(self):
        result = TfidfCosineIssueSolutionLinker().link([], [solution("s1", "ask", post_id="p1")])
        self.assertEqual(result.n_scored_links, 0)

    def test_every_rejection_reason_is_reachable(self):
        reasons = {
            REASON_BELOW_CANDIDATE_THRESHOLD,
            REASON_CROSS_THREAD_DISABLED,
            REASON_ISSUE_HAS_NO_LABEL,
            REASON_UNRESOLVED_THREAD,
            REASON_CAPPED,
        }
        config = LinkModelConfig(max_candidates_per_issue=1)
        issues = [
            issue("i1", "fees", ["fees"], post_id="p1"),
            issue("i2", "food", [], post_id="p1"),
            IssueEvidence(document_id="i3", issue_labels=["food"]),
        ]
        solutions = [
            solution("s1", "a", post_id="p1"),
            solution("s2", "b", post_id="p1"),
            solution("s3", "c", post_id="p2"),
            solution("s4", "d", post_id="p1", probability=0.01, is_solution=False),
        ]
        _, rejections = generate_candidates(issues, solutions, config)
        self.assertTrue(
            reasons.issubset(set(rejections)), f"unreached reasons: {reasons - set(rejections)}"
        )

    def test_rejection_counts_are_deterministic(self):
        config = LinkModelConfig()
        issues = [issue("i1", "fees", ["fees"], post_id="p1")]
        solutions = [
            solution("s1", "a", post_id="p1"),
            solution("s2", "b", post_id="p2"),
            solution("s3", "c", post_id="p1", probability=0.01, is_solution=False),
        ]
        first = generate_candidates(issues, solutions, config)[1]
        second = generate_candidates(issues, solutions, config)[1]
        self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# 13. Architecture isolation
# ---------------------------------------------------------------------------


class TestArchitectureIsolation(unittest.TestCase):
    def _python_files(self):
        found = []
        for root, _dirs, files in os.walk(PACKAGE_DIR):
            for name in files:
                if name.endswith(".py"):
                    found.append(os.path.join(root, name))
        return found

    def _imported_modules(self):
        modules = set()
        for path in self._python_files():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        modules.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:  # relative import within this package
                        continue
                    if node.module:
                        modules.add(node.module)
        return modules

    def test_does_not_import_other_nlp_stages(self):
        forbidden = (
            "solution_detection",
            "response_classification",
            "preprocessing",
            "issue_identification",
            "topic_discovery",
        )
        imported = self._imported_modules()
        for module in imported:
            for banned in forbidden:
                self.assertNotIn(
                    banned, module, f"this stage must not import {banned} (found {module})"
                )

    def test_imports_nothing_from_another_nlp_stage_by_path(self):
        for module in self._imported_modules():
            self.assertFalse(
                module.startswith("app.services.nlp.")
                and not module.startswith("app.services.nlp.issue_solution_linking"),
                f"cross-stage import detected: {module}",
            )

    def test_does_not_import_or_write_a_database(self):
        banned = ("sqlalchemy", "psycopg2", "pymysql", "sqlite3", "psycopg")
        for module in self._imported_modules():
            for name in banned:
                self.assertNotIn(name, module, f"database import detected: {module}")

    def test_does_not_import_requests_or_an_http_client(self):
        for module in self._imported_modules():
            self.assertNotIn("requests", module)
            self.assertNotIn("httpx", module)
            self.assertNotIn("urllib", module)

    def test_package_has_no_network_side_effects_at_import(self):
        import app.services.nlp.issue_solution_linking as package

        self.assertTrue(hasattr(package, "TfidfCosineIssueSolutionLinker"))
        self.assertTrue(hasattr(package, "IssueEvidence"))
        self.assertTrue(hasattr(package, "SolutionEvidence"))

    def test_stage_cannot_answer_resolution_questions(self):
        """No resolution field, no unresolved verdict, no generated text."""
        for path in self._python_files():
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            for banned in (
                "def detect_resolution",
                "resolution_detector",
                "generate_solution",
                "generate_answer",
            ):
                self.assertNotIn(banned, source, f"{banned} must not exist in this stage")

    def test_no_resolution_or_unresolved_field_on_the_result(self):
        result = TfidfCosineIssueSolutionLinker().link(
            [issue("i1", "fees high", ["fees"])], [solution("s1", "ask", post_id="p1")]
        )
        fields = set(result.model_dump())
        link_fields = set(result.links[0].model_dump())
        for banned in ("is_resolved", "resolution", "unresolved", "generated_solution", "answer"):
            self.assertNotIn(banned, fields, f"result must not carry {banned}")
            self.assertNotIn(banned, link_fields, f"link must not carry {banned}")

    def test_public_api_is_exported(self):
        import app.services.nlp.issue_solution_linking as package

        for name in (
            "IssueEvidence",
            "SolutionEvidence",
            "LinkModelConfig",
            "IssueSolutionLinking",
            "TfidfCosineIssueSolutionLinker",
            "LinkPrediction",
            "LinkingResult",
            "LinkingRunReport",
            "LinkAnnotation",
            "LinkCorpus",
            "evaluate_linking",
            "generate_candidates",
        ):
            self.assertIn(name, package.__all__, f"{name} missing from __all__")
            self.assertTrue(hasattr(package, name))


# ---------------------------------------------------------------------------
# 14. The failure this stage exists to prevent
# ---------------------------------------------------------------------------


class TestNoCrossProduct(unittest.TestCase):
    def test_cross_post_solutions_never_reach_an_issue(self):
        linker = TfidfCosineIssueSolutionLinker()
        issues = [issue("i1", "the fee is too high", ["fees"], post_id="p1")]
        solutions = [
            solution("s1", "lower the fee", post_id="p2"),
            solution("s2", "reduce the fee", post_id="p3"),
            solution("s3", "cut the fee", post_id="p4"),
        ]
        result = linker.link(issues, solutions)
        self.assertEqual(result.n_scored_links, 0)
        self.assertEqual(result.links, [])
        self.assertEqual(result.candidate_rejections[REASON_CROSS_THREAD_DISABLED], 3)

    def test_a_high_confidence_solution_in_another_thread_still_cannot_link(self):
        issues = [issue("i1", "the fee is too high", ["fees"], post_id="p1")]
        elsewhere = solution(
            "s1",
            "the fee is too high",
            post_id="p2",
            probability=1.0,
            response_type="solution",
        )
        result = TfidfCosineIssueSolutionLinker().link(issues, [elsewhere])
        self.assertEqual(result.n_linked, 0)
        self.assertEqual(result.n_scored_links, 0)

    def test_cross_post_must_be_asked_for_explicitly(self):
        issues = [issue("i1", "the fee is too high", ["fees"], post_id="p1")]
        solutions = [solution("s1", "lower the fee", post_id="p2", probability=1.0)]
        enabled = TfidfCosineIssueSolutionLinker(allow_cross_post=True).link(issues, solutions)
        self.assertEqual(enabled.n_scored_links, 1)
        self.assertEqual(enabled.links[0].relation, REASON_CROSS_POST)

    def test_candidate_count_never_exceeds_issues_times_candidate_solutions(self):
        linker = TfidfCosineIssueSolutionLinker()
        issues = [
            issue("i1", "fee too high", ["fees"], post_id="p1"),
            issue("i2", "food is bad", ["food"], post_id="p2"),
            issue("i3", "wifi drops", ["network"], post_id="p3"),
        ]
        solutions = [
            solution("s1", "ask the warden", post_id="p1"),
            solution("s2", "ask the bursar", post_id="p2"),
            solution("s3", "restart the router", post_id="p3"),
        ]
        result = linker.link(issues, solutions)
        self.assertEqual(result.n_scored_links, 3, "one same-thread pair each, not nine")
        self.assertLessEqual(result.n_scored_links, 3 * 3)

    def test_linker_implements_the_abstract_interface(self):
        from app.services.nlp.issue_solution_linking import IssueSolutionLinking

        linker = TfidfCosineIssueSolutionLinker()
        self.assertIsInstance(linker, IssueSolutionLinking)
        for attribute in ("name", "version", "config", "link_threshold", "config_dict"):
            self.assertTrue(hasattr(linker, attribute))


if __name__ == "__main__":
    unittest.main()
