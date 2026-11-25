"""Sprint P0-C Unit Tests — Multi-Annotator Agreement, Blind Sheet Verification & Adjudication."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.annotation_reliability import (
    AnnotationSubmission,
    CANONICAL_RESPONSE_ORDER,
    compute_full_multiannotator_iaa,
    compute_multilabel_iaa,
    compute_single_label_iaa,
    fleiss_kappa,
    krippendorff_alpha_nominal,
    merge_independent_annotations_and_adjudicate,
    verify_blind_sheet,
)


def _reference_2coder_nominal_alpha(a: list[str], b: list[str]) -> float:
    n = len(a)
    do = sum(1.0 for x, y in zip(a, b) if x != y) / float(n)
    counts = Counter(a + b)
    same = sum(c * (c - 1.0) for c in counts.values()) / float(2 * n * (2 * n - 1))
    de = 1.0 - same
    return 1.0 if de <= 1e-12 else 1.0 - do / de


def test_krippendorff_alpha_matches_2coder_and_handles_k_coders() -> None:
    a = ["negative", "neutral", "positive", "negative", "neutral", "positive"]
    b = ["negative", "neutral", "neutral", "negative", "positive", "positive"]
    units = [[x, y] for x, y in zip(a, b)]
    assert abs(krippendorff_alpha_nominal(units) - _reference_2coder_nominal_alpha(a, b)) < 1e-9
    if importlib.util.find_spec("torch") is not None:
        from ml.experiments.v5_optimization.score_human_iaa import krippendorff_alpha_nominal as alpha_2coder

        assert abs(krippendorff_alpha_nominal(units) - alpha_2coder(a, b)) < 1e-9

    # K = 3 raters with unanimous agreement -> Fleiss kappa == 1.0 and alpha == 1.0
    unanimous = [["advice", "advice", "advice"], ["solution", "solution", "solution"], ["opinion", "opinion", "opinion"]]
    assert abs(fleiss_kappa(unanimous, CANONICAL_RESPONSE_ORDER) - 1.0) < 1e-9
    assert abs(krippendorff_alpha_nominal(unanimous) - 1.0) < 1e-9


def test_single_annotator_only_refuses_to_fabricate_iaa() -> None:
    only_one = {
        "ann_1": {
            "d1": AnnotationSubmission("d1", "ann_1", sentiment="positive", response="solution", solution="solution"),
        }
    }
    rep = compute_full_multiannotator_iaa(only_one)
    assert rep["status"] == "SINGLE_ANNOTATOR_ONLY_CANNOT_COMPUTE_IAA"
    assert "cohen_kappa" in rep["pending_metrics"]


def test_multilabel_reports_per_class_warnings_and_preserves_originals_on_adjudicate() -> None:
    # Two annotators agree strongly on 'frustration', disagree heavily on 'hope' (only 2 total positives -> low support + unreliable)
    subs = {"ann_a": {}, "ann_b": {}}
    data_a = [
        ("d1", "negative", {"frustration"}, {"hostel_facilities"}, "opinion", "not_solution"),
        ("d2", "negative", {"frustration", "stress"}, {"hostel_facilities"}, "opinion", "not_solution"),
        ("d3", "neutral", set(), {"placements"}, "advice", "not_solution"),
        ("d4", "positive", {"hope"}, {"financial_concerns"}, "solution", "solution"),
        ("d5", "neutral", {"curiosity"}, {"administrative_communication"}, "opinion", "not_solution"),
        ("d6", "negative", {"frustration"}, {"academic_workload"}, "opinion", "not_solution"),
    ]
    data_b = [
        ("d1", "negative", {"frustration"}, {"hostel_facilities"}, "opinion", "not_solution"),
        ("d2", "negative", {"frustration", "stress"}, {"hostel_facilities"}, "opinion", "not_solution"),
        ("d3", "neutral", set(), {"placements"}, "advice", "not_solution"),
        ("d4", "neutral", {"confidence"}, {"financial_concerns"}, "advice", "not_solution"),
        ("d5", "neutral", {"curiosity"}, {"administrative_communication"}, "opinion", "not_solution"),
        ("d6", "negative", {"frustration"}, {"academic_workload"}, "opinion", "not_solution"),
    ]
    for did, s, e, iss, r, sol in data_a:
        subs["ann_a"][did] = AnnotationSubmission(did, "ann_a", sentiment=s, emotions=e, issues=iss, response=r, solution=sol)
    for did, s, e, iss, r, sol in data_b:
        subs["ann_b"][did] = AnnotationSubmission(did, "ann_b", sentiment=s, emotions=e, issues=iss, response=r, solution=sol)

    emo_iaa = compute_multilabel_iaa(subs, "emotion", ("confidence", "curiosity", "frustration", "gratitude", "hope", "stress"))
    assert emo_iaa["per_label"]["frustration"]["krippendorff_alpha"] > 0.90
    assert "hope" in emo_iaa["unreliable_classes"]
    assert "hope" in emo_iaa["low_support_classes"]
    assert emo_iaa["gate_passed"] is False  # blocked because weak per-label class exists!

    full = compute_full_multiannotator_iaa(subs, round_id="round_test")
    assert full["n_disagreements"] == 1
    assert full["disagreements"][0]["document_id"] == "d4"

    merged = merge_independent_annotations_and_adjudicate(
        subs,
        adjudications_by_doc={
            "d4": {
                "sentiment": "positive",
                "emotions": ["hope"],
                "issues": ["financial_concerns"],
                "response": "solution",
                "solution": "solution",
                "review_status": "labeled",
                "adjudicator_id": "lead_reviewer",
                "notes": "Confirmed actionable email remedy.",
            }
        },
        round_id="round_test",
    )
    by_id = {m["document_id"]: m for m in merged}
    assert by_id["d1"]["resolution_method"] == "unanimous_multi_annotator"
    assert by_id["d4"]["resolution_method"] == "human_adjudicated"
    # Originals remain preserved intact!
    assert by_id["d4"]["original_annotations"]["ann_a"]["solution"] == "solution"
    assert by_id["d4"]["original_annotations"]["ann_b"]["solution"] == "not_solution"


def test_verify_blind_sheet_rejects_pre_filled_or_peer_columns() -> None:
    ok, _ = verify_blind_sheet([{"document_id": "d1", "text": "hello", "sentiment": "", "review_status": ""}])
    assert ok is True
    ok2, errs2 = verify_blind_sheet([{"document_id": "d1", "text": "hello", "sentiment": "positive", "peer_sentiment": "neutral"}])
    assert ok2 is False
    assert any("non_blind_columns_present" in e for e in errs2)
