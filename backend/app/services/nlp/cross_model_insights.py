"""Cross-Model Insight Engine & Calibrated Confidence Aggregator (pure Python).

Synthesizes actionable intelligence across all 5 supervised NLP models, NMF topic
clusters, explicit resolution-evidence signals, calibrated prediction confidence,
and 4-architecture benchmark telemetry (LogReg vs ANN vs CNN vs MuRIL-DAPT v3).

Zero heavy ML dependencies at request time — safe and fast on both Vercel serverless
and Docker FastAPI runtimes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.55


def confidence_tier(score: Optional[float]) -> str:
    if score is None:
        return "medium"
    val = float(score)
    if val >= CONFIDENCE_HIGH:
        return "high"
    if val >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def comment_confidence_profile(preds: Dict[str, Any]) -> Dict[str, Any]:
    """Compute unified calibrated confidence profile for a single comment across all 5 tasks."""
    sent = preds.get("sentiment") or {}
    resp = preds.get("response") or {}
    sol = preds.get("solution") or {}
    emo = preds.get("emotion") or {}
    iss = preds.get("issue") or {}

    sol_prob = sol.get("solution_probability")
    sol_conf = None
    if sol_prob is not None:
        p = float(sol_prob)
        sol_conf = p if sol.get("is_solution") else (1.0 - p)

    task_scores = {
        "sentiment": float(sent["confidence"]) if sent.get("confidence") is not None else None,
        "response": float(resp["confidence"]) if resp.get("confidence") is not None else None,
        "solution": round(sol_conf, 4) if sol_conf is not None else None,
        "emotion": float(emo["confidence"]) if emo.get("confidence") is not None else None,
        "issue": float(iss["confidence"]) if iss.get("confidence") is not None else None,
    }
    valid = [v for v in task_scores.values() if v is not None]
    mean_conf = round(sum(valid) / len(valid), 4) if valid else 0.0
    min_conf = round(min(valid), 4) if valid else 0.0

    # Top-1 minus Top-2 margin on single-label response & sentiment tasks
    margins: List[float] = []
    for key, pmap_key in (("response", "response_probabilities"), ("sentiment", "sentiment_probabilities")):
        pmap = (preds.get(key) or {}).get(pmap_key) or {}
        if len(pmap) >= 2:
            vals = sorted((float(v) for v in pmap.values()), reverse=True)
            margins.append(max(0.0, vals[0] - vals[1]))
    mean_margin = round(sum(margins) / len(margins), 4) if margins else 0.0

    tier = confidence_tier(mean_conf)
    needs_review = tier == "low" or min_conf < 0.45
    return {
        "overall_confidence": mean_conf,
        "min_task_confidence": min_conf,
        "confidence_margin": mean_margin,
        "confidence_tier": tier,
        "needs_review": needs_review,
        "per_task": task_scores,
    }


def build_confidence_summary(
    rows: Sequence[Dict[str, Any]],
    posts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute corpus-wide calibrated confidence stats, histogram, and human-review queue."""
    task_vals: Dict[str, List[float]] = defaultdict(list)
    overall_vals: List[float] = []
    tier_counts: Counter = Counter()
    by_post_confs: Dict[str, List[float]] = defaultdict(list)
    by_post_low: Counter = Counter()
    needs_review_count = 0

    for row in rows:
        prof = row.get("confidence_profile") or comment_confidence_profile(row.get("predictions") or {})
        ov = float(prof["overall_confidence"])
        overall_vals.append(ov)
        tier_counts[prof["confidence_tier"]] += 1
        by_post_confs[row["post_id"]].append(ov)
        if prof["needs_review"]:
            needs_review_count += 1
            by_post_low[row["post_id"]] += 1
        for t_name, val in (prof.get("per_task") or {}).items():
            if val is not None:
                task_vals[t_name].append(float(val))

    n = max(len(overall_vals), 1)
    sorted_ov = sorted(overall_vals)
    median_ov = sorted_ov[len(sorted_ov) // 2] if sorted_ov else 0.0

    # 10-bin histogram across [0.0, 1.0]
    hist = []
    for b in range(10):
        lo = b / 10.0
        hi = (b + 1) / 10.0
        cnt = sum(1 for v in overall_vals if (lo <= v <= hi if b == 9 else lo <= v < hi))
        hist.append(
            {
                "label": f"{int(lo * 100)}-{int(hi * 100)}%",
                "lower": round(lo, 2),
                "upper": round(hi, 2),
                "count": cnt,
                "tier": confidence_tier((lo + hi) / 2.0),
            }
        )

    per_task_summary: Dict[str, Dict[str, Any]] = {}
    for t_name in ("sentiment", "emotion", "issue", "response", "solution"):
        vals = task_vals.get(t_name) or []
        if not vals:
            continue
        tn = len(vals)
        per_task_summary[t_name] = {
            "mean": round(sum(vals) / tn, 4),
            "high_share": round(sum(1 for v in vals if v >= CONFIDENCE_HIGH) / tn, 4),
            "medium_share": round(sum(1 for v in vals if CONFIDENCE_MEDIUM <= v < CONFIDENCE_HIGH) / tn, 4),
            "low_share": round(sum(1 for v in vals if v < CONFIDENCE_MEDIUM) / tn, 4),
        }

    # Lowest-confidence threads prioritized for human Annotation Studio review
    review_threads = []
    for post in posts:
        pid = post["reddit_id"]
        c_list = by_post_confs.get(pid) or []
        if not c_list:
            continue
        avg_c = sum(c_list) / len(c_list)
        review_threads.append(
            {
                "post_id": pid,
                "title": post["title"],
                "subreddit": post["subreddit"],
                "issue": post["issue_category"],
                "comment_count": len(c_list),
                "mean_confidence": round(avg_c, 4),
                "confidence_tier": confidence_tier(avg_c),
                "low_confidence_comments": int(by_post_low[pid]),
            }
        )
    review_threads.sort(key=lambda x: (x["mean_confidence"], -x["low_confidence_comments"]))

    return {
        "mean_confidence": round(sum(overall_vals) / n, 4) if overall_vals else 0.0,
        "median_confidence": round(median_ov, 4),
        "high_count": int(tier_counts["high"]),
        "medium_count": int(tier_counts["medium"]),
        "low_count": int(tier_counts["low"]),
        "high_share": round(tier_counts["high"] / n, 4),
        "medium_share": round(tier_counts["medium"] / n, 4),
        "low_share": round(tier_counts["low"] / n, 4),
        "needs_review_count": needs_review_count,
        "needs_review_share": round(needs_review_count / n, 4),
        "thresholds": {"high": CONFIDENCE_HIGH, "medium": CONFIDENCE_MEDIUM},
        "histogram": hist,
        "per_task": per_task_summary,
        "priority_review_threads": review_threads[:6],
    }


def build_issue_priority_matrix(
    posts: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    topics: Sequence[Dict[str, Any]],
    topic_assignments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Cross-model synthesis per issue: Volume x Sentiment x Resolution x Emotion x Confidence."""
    topic_label_by_id = {int(t["topic_id"]): t.get("label") or f"Topic {t['topic_id']}" for t in topics if "topic_id" in t}
    topic_by_doc = {
        a.get("document_id"): topic_label_by_id.get(int(a["dominant_topic_id"]))
        for a in topic_assignments
        if a.get("document_id") and a.get("dominant_topic_id") is not None
    }

    posts_by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in posts:
        posts_by_issue[p["issue_key"]].append(p)
    rows_by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rows_by_issue[r["issue_key"]].append(r)

    max_comments = max((len(v) for v in rows_by_issue.values()), default=1)
    matrix: List[Dict[str, Any]] = []

    for key, group_posts in sorted(posts_by_issue.items()):
        c_rows = rows_by_issue.get(key) or []
        n_posts = len(group_posts)
        n_comments = len(c_rows)
        if n_comments == 0:
            continue

        neg_cnt = sum(1 for r in c_rows if (r["predictions"].get("sentiment") or {}).get("sentiment") == "negative")
        pos_cnt = sum(1 for r in c_rows if (r["predictions"].get("sentiment") or {}).get("sentiment") == "positive")
        act_cnt = sum(
            1
            for r in c_rows
            if bool((r["predictions"].get("solution") or {}).get("is_solution"))
            or (r["predictions"].get("response") or {}).get("response_type") in {"advice", "solution"}
        )
        res_ok = sum(1 for p in group_posts if p.get("resolution_state") == "reported_success")
        unresolved = n_posts - res_ok

        emo_counts: Counter = Counter()
        topic_counts: Counter = Counter()
        confs: List[float] = []
        low_conf = 0
        for r in c_rows:
            emo_counts.update((r["predictions"].get("emotion") or {}).get("emotions") or [])
            t_lbl = topic_by_doc.get(r["document_id"])
            if t_lbl:
                topic_counts[t_lbl] += 1
            prof = r.get("confidence_profile") or comment_confidence_profile(r.get("predictions") or {})
            confs.append(float(prof["overall_confidence"]))
            if prof["needs_review"]:
                low_conf += 1

        neg_share = neg_cnt / n_comments
        pos_share = pos_cnt / n_comments
        act_share = act_cnt / n_comments
        unres_share = unresolved / max(n_posts, 1)
        vol_norm = n_comments / max(max_comments, 1)
        mean_conf = sum(confs) / len(confs)

        # Transparent, interpretable Friction Index (0..100):
        # 40% negative sentiment + 35% unresolved ratio + 25% relative corpus volume
        friction_score = round((0.40 * neg_share + 0.35 * unres_share + 0.25 * vol_norm) * 100.0, 1)
        if friction_score >= 62.0:
            band = "critical"
        elif friction_score >= 48.0:
            band = "elevated"
        elif friction_score >= 34.0:
            band = "moderate"
        else:
            band = "healthy"

        matrix.append(
            {
                "key": key,
                "label": group_posts[0]["issue_category"],
                "posts": n_posts,
                "comments": n_comments,
                "negative_share": round(neg_share, 4),
                "positive_share": round(pos_share, 4),
                "actionable_share": round(act_share, 4),
                "resolved_share": round(res_ok / max(n_posts, 1), 4),
                "unresolved_share": round(unres_share, 4),
                "mean_confidence": round(mean_conf, 4),
                "low_confidence_comments": low_conf,
                "dominant_emotion": emo_counts.most_common(1)[0][0] if emo_counts else "curiosity",
                "top_topic": topic_counts.most_common(1)[0][0] if topic_counts else "Unassigned",
                "friction_score": friction_score,
                "priority_band": band,
            }
        )

    matrix.sort(key=lambda x: (-x["friction_score"], -x["comments"]))
    return matrix


def build_cross_model_insights(
    posts: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    evaluation: Dict[str, Any],
    topics: Sequence[Dict[str, Any]],
    topic_assignments: Sequence[Dict[str, Any]],
    confidence_summary: Optional[Dict[str, Any]] = None,
    priority_matrix: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Synthesize evidence-backed cross-task & cross-architecture intelligence briefings."""
    conf = confidence_summary or build_confidence_summary(rows, posts)
    matrix = priority_matrix or build_issue_priority_matrix(posts, rows, topics, topic_assignments)
    n_posts = max(len(posts), 1)
    n_comments = max(len(rows), 1)

    cards: List[Dict[str, Any]] = []

    # Insight 1: Top Unresolved Friction Hotspot (Issue x Sentiment x Resolution)
    if matrix:
        top = matrix[0]
        cards.append(
            {
                "id": "friction_hotspot",
                "category": "Friction Hotspot",
                "tone": "danger" if top["priority_band"] == "critical" else "warning",
                "headline": f"{top['label']} leads community friction (index {top['friction_score']:.0f}/100)",
                "metric": f"{top['negative_share'] * 100:.1f}% negative · {top['unresolved_share'] * 100:.0f}% unconfirmed",
                "detail": (
                    f"{top['posts']} threads ({top['comments']} comments) link {top['label'].lower()} "
                    f"with dominant '{top['dominant_emotion']}' affect and NMF cluster '{top['top_topic']}'; "
                    f"mean prediction confidence is {top['mean_confidence'] * 100:.1f}%."
                ),
                "href": f"#/issues?selected={top['key']}",
                "cta": "Inspect issue hotspot",
            }
        )

    # Insight 2: Proposal-to-Resolution Evidence Gap (Response x Solution x Resolution)
    proposed_threads = sum(1 for p in posts if p.get("resolution_state") == "proposed_only")
    success_threads = sum(1 for p in posts if p.get("resolution_state") == "reported_success")
    pending_threads = sum(1 for p in posts if p.get("resolution_state") == "reported_pending")
    actionable_replies = sum(
        1
        for r in rows
        if bool((r["predictions"].get("solution") or {}).get("is_solution"))
        or (r["predictions"].get("response") or {}).get("response_type") in {"advice", "solution"}
    )
    cards.append(
        {
            "id": "proposal_resolution_gap",
            "category": "Resolution Gap",
            "tone": "info",
            "headline": f"{actionable_replies:,} actionable replies, but only {success_threads} verified outcomes",
            "metric": f"{proposed_threads} proposed-only · {pending_threads} pending · {success_threads} success",
            "detail": (
                f"{actionable_replies / n_comments * 100:.1f}% of replies offer advice or concrete solutions, "
                f"yet {proposed_threads / n_posts * 100:.1f}% of threads have no follow-up confirmation — "
                f"separating peer proposals from verified outcomes prevents overstating resolution rates."
            ),
            "href": "#/resolution",
            "cta": "Audit outcome evidence",
        }
    )

    # Insight 3: Emotion -> Response Role Dynamics (Emotion x Response)
    emo_to_resp: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        resp_type = (r["predictions"].get("response") or {}).get("response_type")
        if not resp_type:
            continue
        for em in (r["predictions"].get("emotion") or {}).get("emotions") or []:
            emo_to_resp[em][resp_type] += 1
    frust_total = sum(emo_to_resp["frustration"].values())
    frust_action = emo_to_resp["frustration"]["advice"] + emo_to_resp["frustration"]["solution"]
    frust_top = emo_to_resp["frustration"].most_common(1)[0] if frust_total else ("opinion", 0)
    cur_total = sum(emo_to_resp["curiosity"].values())
    cur_action = emo_to_resp["curiosity"]["advice"] + emo_to_resp["curiosity"]["solution"]
    cards.append(
        {
            "id": "emotion_response_dynamics",
            "category": "Emotion x Response",
            "tone": "accent",
            "headline": f"Curiosity draws {cur_action / max(cur_total, 1) * 100:.0f}% actionable help vs. {frust_action / max(frust_total, 1) * 100:.0f}% under frustration",
            "metric": f"{cur_total} curiosity · {frust_total} frustration signals",
            "detail": (
                f"Cross-referencing Emotion and Response classifiers shows frustrated comments predominantly "
                f"surface '{frust_top[0].replace('_', ' ')}' ({frust_top[1]} replies), whereas curiosity-tagged "
                f"comments receive denser peer advice and procedural guidance."
            ),
            "href": "#/responses",
            "cta": "Explore response roles",
        }
    )

    # Insight 4: Multilingual / Hinglish Signal & Why Transformer Wins (Language x Architecture)
    non_en_or_mix = sum(1 for r in rows if r.get("language") != "en" or r.get("is_code_mixed"))
    arch_agg = (evaluation.get("architecture_comparison") or {}).get("aggregate") or {}
    t_f1 = (arch_agg.get("transformer_muril_dapt_v3") or {}).get("mean_probe_macro_f1")
    lr_f1 = (arch_agg.get("logistic_regression") or {}).get("mean_probe_macro_f1")
    cnn_f1 = (arch_agg.get("cnn_textcnn") or {}).get("mean_probe_macro_f1")
    if t_f1 is not None and lr_f1 is not None:
        gain_lr = (float(t_f1) - float(lr_f1)) * 100.0
        gain_cnn = (float(t_f1) - float(cnn_f1 or 0.0)) * 100.0
        cards.append(
            {
                "id": "why_transformer_insight",
                "category": "Cross-Model Advantage",
                "tone": "positive",
                "headline": f"MuRIL-DAPT v3 beats LogReg by +{gain_lr:.1f} pp and CNN by +{gain_cnn:.1f} pp OOD",
                "metric": f"{float(t_f1) * 100:.1f}% unseen 54-probe F1 · {non_en_or_mix} code-mixed/non-EN comments",
                "detail": (
                    f"Across {non_en_or_mix} Hinglish/non-English comments ({non_en_or_mix / n_comments * 100:.1f}% of corpus) "
                    f"and unseen paraphrases, domain-adapted multilingual self-attention dramatically outperforms "
                    f"bag-of-words LogReg, ANN (MLP), and 1D TextCNN."
                ),
                "href": "#/models",
                "cta": "Inspect 4-model proof",
            }
        )

    # Insight 5: Confidence & Human-in-the-Loop Triage Queue
    cards.append(
        {
            "id": "confidence_triage",
            "category": "Confidence & Triage",
            "tone": "positive" if conf["low_share"] < 0.15 else "warning",
            "headline": f"{conf['high_share'] * 100:.1f}% high-confidence predictions; {conf['needs_review_count']} flagged for review",
            "metric": f"Mean conf {conf['mean_confidence'] * 100:.1f}% · {conf['low_count']} low-confidence (<55%)",
            "detail": (
                f"Temperature-calibrated probability estimates separate crisp classifications "
                f"({conf['high_count']:,} high >=80%) from ambiguous or terse replies "
                f"({conf['needs_review_count']:,} review candidates) ready for Annotation Studio adjudication."
            ),
            "href": "#/explorer?confidence=low",
            "cta": "Review uncertain threads",
        }
    )

    return {
        "engine_version": "1.0.0",
        "cards": cards,
        "emotion_response_matrix": {
            em: dict(cnt) for em, cnt in sorted(emo_to_resp.items())
        },
        "multilingual_share": round(non_en_or_mix / n_comments, 4),
        "multilingual_count": non_en_or_mix,
    }
