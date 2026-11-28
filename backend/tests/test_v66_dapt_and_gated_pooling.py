"""V6.6 Unit & Integration Tests — Universal WWM, Gated Reply-Masked Pooling & Nested OOF Stacker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.experiments.v6_programme.v66_architecture import (
    HAS_TORCH,
    fit_predict_nested_oof_stacker,
    group_subword_positions,
)


def test_v66_wwm_wordpiece_and_sentencepiece_grouping() -> None:
    special = {101, 102, 0}
    wp_toks = ["[CLS]", "IIM", "##S", "##am", "##balpur", "wait", "##list", "[SEP]", "[PAD]"]
    wp_ids = [101, 10, 11, 12, 13, 20, 21, 102, 0]
    wp_attn = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    wp_groups, wp_valid = group_subword_positions(wp_toks, wp_ids, wp_attn, special, is_sentencepiece=False)
    assert wp_valid == 6
    assert wp_groups == [[1, 2, 3, 4], [5, 6]]

    sp_special = {0, 1, 2}
    sp_toks = ["<s>", "\u2581wait", "list", "\u2581IIM", "\u2581Cal", "cutta", "!", "</s>", "<pad>"]
    sp_ids = [0, 30, 31, 32, 33, 34, 35, 2, 1]
    sp_attn = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    sp_groups, sp_valid = group_subword_positions(sp_toks, sp_ids, sp_attn, sp_special, is_sentencepiece=True)
    assert sp_valid == 6
    assert sp_groups == [[1, 2], [3], [4, 5], [6]]


def test_v66_nested_oof_stacker_simplex_and_fold_isolation() -> None:
    rng = np.random.default_rng(123)
    n_train, n_test = 70, 20
    labels = ["negative", "neutral", "positive"]
    y_true = np.array([i % 3 for i in range(n_train)], dtype=np.int64)

    # 7 outer folds
    folds = []
    for k in range(7):
        va = list(range(k * 10, (k + 1) * 10))
        tr = [i for i in range(n_train) if i not in set(va)]
        folds.append((tr, va))

    # Candidate A accurate, Candidate B noisy, Candidate C weak
    p_true = np.eye(3, dtype=np.float64)[y_true]
    cand_a = np.clip(0.80 * p_true + 0.08 * rng.uniform(0, 1, size=(n_train, 3)), 1e-4, 1.0)
    cand_a /= cand_a.sum(axis=1, keepdims=True)
    cand_b = np.clip(0.55 * p_true + 0.30 * rng.uniform(0, 1, size=(n_train, 3)), 1e-4, 1.0)
    cand_b /= cand_b.sum(axis=1, keepdims=True)
    cand_c = np.full((n_train, 3), 1.0 / 3.0)

    test_a = np.full((n_test, 3), 1.0 / 3.0)
    res = fit_predict_nested_oof_stacker(
        task="sentiment",
        folds=folds,
        y_true=y_true,
        oof_prob_candidates=[cand_a, cand_b, cand_c],
        test_prob_candidates=[test_a, test_a, test_a],
        labels=labels,
        candidate_names=["m_a", "m_b", "m_c"],
        units=5,
    )
    w_sum = sum(res["mean_weights"].values())
    assert abs(w_sum - 1.0) < 1e-3
    assert all(v >= 0.0 for v in res["mean_weights"].values())
    assert res["mean_weights"]["m_a"] >= res["mean_weights"]["m_c"]
    assert res["nested_oof_probs"].shape == (n_train, 3)
    assert res["stacked_test_probs"].shape == (n_test, 3)
    assert res["oof_eval"]["macro_f1"] > 0.85


@pytest.mark.skipif(not HAS_TORCH, reason="torch optional in lightweight base environment")
def test_v66_gated_reply_context_masking_and_gradient_flow() -> None:
    import torch
    from torch import nn
    from ml.experiments.v6_programme.v66_architecture import GatedReplyContextClassifier

    class _DummyOut:
        def __init__(self, last_hidden_state: torch.Tensor) -> None:
            self.last_hidden_state = last_hidden_state

    class _DummyEncoder(nn.Module):
        def __init__(self, hidden_size: int = 16) -> None:
            super().__init__()
            self.emb = nn.Embedding(64, hidden_size)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> _DummyOut:
            return _DummyOut(self.emb(input_ids))

    clf = GatedReplyContextClassifier(
        encoder=_DummyEncoder(16),
        hidden_size=16,
        num_labels=6,
        pooling_mode="gated_reply",
        sep_token_id=2,
        multi_sample_dropout=1,
        dropout_rate=0.0,
    )
    # Single-SEP layout + Double-SEP (XLM-R) layout
    input_ids = torch.tensor([
        [0, 10, 11, 2, 20, 21, 2, 1],   # reply=[10,11], ctx=[20,21], pad=1
        [0, 12, 2, 2, 22, 23, 24, 2],   # reply=[12], ctx=[22,23,24]
    ])
    attn_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])
    r_mask, c_mask = clf.extract_span_masks(input_ids, attn_mask)
    assert r_mask[0].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert c_mask[0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert r_mask[1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert c_mask[1].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]

    # Zero-initialized gate produces zero gradient leakage check & updates gate_alpha when classifier runs
    assert float(clf.gate_alpha.abs().max().item()) == 0.0
    logits = clf(input_ids, attn_mask)
    assert logits.shape == (2, 6)
    logits.sum().backward()
    assert clf.gate_alpha.grad is not None
    assert float(clf.gate_alpha.grad.abs().sum().item()) > 0.0


def test_v66_artifacts_zero_leakage_when_present() -> None:
    dapt_m = ROOT / "ml" / "artifacts" / "v6_6_experiments" / "dapt_10x" / "v66_strict_dapt_manifest.json"
    if dapt_m.is_file():
        m = json.loads(dapt_m.read_text(encoding="utf-8"))
        assert m["frozen_v1_test_threads"] == 37
        assert m["intersection_train_and_blocked"] == 0
        assert m["intersection_val_and_blocked"] == 0
        assert m["intersection_train_and_val"] == 0

    stk_m = ROOT / "ml" / "artifacts" / "v6_6_experiments" / "stacking_v1" / "v66_ablation_and_stacking.json"
    if stk_m.is_file():
        s = json.loads(stk_m.read_text(encoding="utf-8"))
        assert set(s["locked_oof_champions"].keys()) == {"issue", "sentiment", "emotion", "response", "solution"}
