"""Read-only local API for the synthetic research-intelligence demo.

Loads only generated files. It never contacts Reddit, a database, or a model
registry outside ml/artifacts/demo_synthetic.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.nlp.response_classification.models.labels import RESPONSE_TYPES

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "demo_synthetic"
ARTIFACT_DIR = (
    ROOT / "ml" / "artifacts" / "demo_synthetic_v2"
    if (ROOT / "ml" / "artifacts" / "demo_synthetic_v2" / "evaluation.json").is_file()
    else ROOT / "ml" / "artifacts" / "demo_synthetic"
)
SENTIMENTS = ["positive", "neutral", "negative"]

router = APIRouter(tags=["synthetic demo"])


class HealthResponse(BaseModel):
    status: str
    synthetic_only: bool
    dataset_version: str
    posts: int
    comments: int
    notice: str


class SummaryResponse(BaseModel):
    synthetic_only: bool
    dataset_version: str
    generated_at: str
    provenance: dict[str, Any]
    posts: int
    comments: int
    issue_count: int
    topic_count: int
    canonical_response_types: list[str]
    community_distribution: list[dict[str, Any]]
    issue_distribution: list[dict[str, Any]]
    issue_trends: list[dict[str, Any]]
    sentiment_distribution: list[dict[str, Any]]
    emotion_distribution: list[dict[str, Any]]
    response_distribution: list[dict[str, Any]]
    resolution_summary: list[dict[str, Any]]
    recent_discussions: list[dict[str, Any]]


class DiscussionListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_more: bool
    synthetic_only: bool


class DiscussionDetailResponse(BaseModel):
    post: dict[str, Any]
    comments: list[dict[str, Any]]
    synthetic_only: bool


class ItemsResponse(BaseModel):
    items: list[dict[str, Any]]
    synthetic_only: bool


class SentimentResponse(BaseModel):
    sentiment: list[dict[str, Any]]
    emotions: list[dict[str, Any]]
    by_issue: list[dict[str, Any]]
    monthly: list[dict[str, Any]]
    synthetic_only: bool


class ModelsResponse(BaseModel):
    synthetic_only: bool
    evaluation_scope: str
    tasks: dict[str, Any]


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Demo artifact missing: {path.name}. Run the documented generation/training commands first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    corpus = _read(DATA_DIR / "corpus.json")
    predictions = _read(ARTIFACT_DIR / "predictions.json")["tasks"]
    evaluation = _read(ARTIFACT_DIR / "evaluation.json")
    topic_result = _read(ARTIFACT_DIR / "topic_discovery.json")["result"]
    posts = corpus["posts"]
    flat: list[dict[str, Any]] = []
    for post in posts:
        comments = post["comments"]
        comment_by_id = {c["reddit_id"]: c for c in comments}
        for comment in comments:
            row = dict(comment)
            row.update({
                "post_id": post["reddit_id"], "post_title": post["title"],
                "post_body": post["body"], "subreddit": post["subreddit"],
                "issue_category": post["issue_category"], "issue_key": post["issue_key"],
                "parent_comment_text": comment_by_id.get(comment.get("parent_reddit_id"), {}).get("body"),
            })
            row["predictions"] = {
                task: predictions.get(task, {}).get(comment["document_id"], {})
                for task in ("sentiment", "emotion", "issue", "response", "solution")
            }
            flat.append(row)
    return {
        "metadata": corpus["metadata"], "posts": posts, "comments": flat,
        "evaluation": evaluation["tasks"], "topics": topic_result.get("topics", []),
        "topic_assignments": topic_result.get("documents", []),
    }


def _counts(items: list[str], order: list[str] | None = None) -> list[dict[str, Any]]:
    counts = Counter(v for v in items if v)
    keys = order or sorted(counts)
    return [{"label": key, "count": counts.get(key, 0)} for key in keys]


def _prediction_distribution(rows: list[dict[str, Any]], task: str, field: str, order: list[str] | None = None) -> list[dict[str, Any]]:
    values: list[str] = []
    for row in rows:
        value = row["predictions"].get(task, {}).get(field)
        if isinstance(value, list):
            values.extend(str(v) for v in value)
        elif isinstance(value, str) and value:
            values.append(value)
    return _counts(values, order)


def _summary_data() -> dict[str, Any]:
    data = _data()
    rows, posts = data["comments"], data["posts"]
    issue_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in posts:
        issue_groups[post["issue_key"]].append(post)
    issues = []
    for key, group in sorted(issue_groups.items(), key=lambda item: item[0]):
        comments = [c for c in rows if c["issue_key"] == key]
        issues.append({
            "key": key, "label": group[0]["issue_category"],
            "posts": len(group), "comments": len(comments),
            "solutions_proposed": sum(c["response_type"] == "solution" for c in comments),
            "advice_comments": sum(c["response_type"] == "advice" for c in comments),
            "actionable_recommendations": sum(c["is_solution"] for c in comments),
            "reported_successes": sum(p["resolution_state"] == "reported_success" for p in group),
            "no_confirmed_outcome": sum(p["resolution_state"] != "reported_success" for p in group),
        })
    dated_posts = [(datetime.fromisoformat(post["created_at"]), post) for post in posts]
    latest_date = max((date for date, _ in dated_posts), default=None)
    recent_counts: Counter = Counter()
    prior_counts: Counter = Counter()
    if latest_date is not None:
        recent_start = latest_date - timedelta(days=90)
        prior_start = recent_start - timedelta(days=90)
        for date, post in dated_posts:
            if recent_start < date <= latest_date:
                recent_counts[post["issue_key"]] += 1
            elif prior_start < date <= recent_start:
                prior_counts[post["issue_key"]] += 1
    issue_trends = [{
        "key": issue["key"], "label": issue["label"],
        "recent_90d": recent_counts[issue["key"]],
        "prior_90d": prior_counts[issue["key"]],
        "change": recent_counts[issue["key"]] - prior_counts[issue["key"]],
        "as_of": latest_date.isoformat() if latest_date else None,
        "basis": "generated_post_timestamps",
    } for issue in issues]
    topics = data["topics"]
    topic_assignment = {item["document_id"]: item for item in data["topic_assignments"]}
    for topic in topics:
        topic_id = topic.get("topic_id")
        assigned = sum(1 for item in topic_assignment.values() if item.get("topic_id") == topic_id)
        topic["document_count"] = assigned or topic.get("document_count", 0)
    resolution_order = ["reported_success", "reported_pending", "proposed_only", "insufficient_evidence"]
    resolution_counts = Counter(p["resolution_state"] for p in posts)
    recent = sorted(posts, key=lambda p: p["created_at"], reverse=True)[:6]
    return {
        "synthetic_only": True, "dataset_version": data["metadata"]["dataset_version"],
        "generated_at": data["metadata"]["generation_timestamp"],
        "provenance": {
            key: data["metadata"].get(key)
            for key in ("dataset_id", "dataset_version", "generation_seed", "generation_timestamp", "source_provider", "source_mode", "synthetic", "authorization_status", "notice")
        },
        "posts": len(posts), "comments": len(rows), "issue_count": len(issues),
        "topic_count": len(topics),
        "canonical_response_types": list(RESPONSE_TYPES),
        "community_distribution": _counts([p["subreddit"] for p in posts]),
        "issue_distribution": issues,
        "issue_trends": issue_trends,
        "sentiment_distribution": _prediction_distribution(rows, "sentiment", "sentiment", SENTIMENTS),
        "emotion_distribution": _prediction_distribution(rows, "emotion", "emotions"),
        "response_distribution": _prediction_distribution(rows, "response", "response_type", RESPONSE_TYPES),
        "resolution_summary": [{"label": state, "count": resolution_counts.get(state, 0)} for state in resolution_order],
        "recent_discussions": [{
            "id": p["reddit_id"], "title": p["title"], "issue": p["issue_category"],
            "created_at": p["created_at"], "comment_count": len(p["comments"]),
            "resolution_state": p["resolution_state"],
        } for p in recent],
    }


@router.get("/health", response_model=HealthResponse)
def demo_health() -> dict[str, Any]:
    meta = _data()["metadata"]
    return {
        "status": "ready", "synthetic_only": True,
        "dataset_version": meta["dataset_version"],
        "posts": meta["post_count"], "comments": meta["comment_count"],
        "notice": meta["notice"],
    }


@router.get("/summary", response_model=SummaryResponse)
def summary() -> dict[str, Any]:
    return _summary_data()


@router.get("/discussions", response_model=DiscussionListResponse)
def discussions(
    q: str = Query(default="", max_length=120),
    community: str = Query(default="", max_length=80),
    issue: str = Query(default="", max_length=80),
    sentiment: str = Query(default="", max_length=24),
    emotion: str = Query(default="", max_length=40),
    response: str = Query(default="", max_length=40),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=12, ge=1, le=100),
) -> dict[str, Any]:
    data = _data()
    if community and community not in {p["subreddit"] for p in data["posts"]}:
        raise HTTPException(status_code=422, detail="Unknown synthetic community filter")
    if issue and issue not in {p["issue_key"] for p in data["posts"]}:
        raise HTTPException(status_code=422, detail="Unknown issue filter")
    if sentiment and sentiment not in SENTIMENTS:
        raise HTTPException(status_code=422, detail="Unknown sentiment filter")
    if response and response not in RESPONSE_TYPES:
        raise HTTPException(status_code=422, detail="Unknown response filter")
    if emotion and emotion not in {e for row in data["comments"] for e in row["emotions"]}:
        raise HTTPException(status_code=422, detail="Unknown emotion filter")
    posts_by_id = {p["reddit_id"]: p for p in data["posts"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["comments"]:
        grouped[row["post_id"]].append(row)
    query = q.casefold().strip()
    result = []
    for post in data["posts"]:
        rows = grouped[post["reddit_id"]]
        if community and post["subreddit"] != community:
            continue
        if issue and post["issue_key"] != issue:
            continue
        if sentiment and not any(r["predictions"]["sentiment"].get("sentiment") == sentiment for r in rows):
            continue
        if emotion and not any(emotion in r["predictions"]["emotion"].get("emotions", []) for r in rows):
            continue
        if response and not any(r["predictions"]["response"].get("response_type") == response for r in rows):
            continue
        if query and not any(query in value.casefold() for value in [post["title"], post["body"], *(r["body"] for r in rows)]):
            continue
        result.append({
            "id": post["reddit_id"], "title": post["title"], "body": post["body"],
            "subreddit": post["subreddit"], "issue": post["issue_category"],
            "issue_key": post["issue_key"], "created_at": post["created_at"],
            "comment_count": len(rows), "resolution_state": post["resolution_state"],
            "response_counts": _counts([r["predictions"]["response"].get("response_type") or "" for r in rows], RESPONSE_TYPES),
        })
    result.sort(key=lambda item: item["created_at"], reverse=True)
    start = (page - 1) * page_size
    page_items = result[start:start + page_size]
    return {"items": page_items, "total": len(result), "page": page,
            "page_size": page_size, "has_more": start + page_size < len(result), "synthetic_only": True}


@router.get("/discussions/{post_id}", response_model=DiscussionDetailResponse)
def discussion_detail(post_id: str) -> dict[str, Any]:
    data = _data()
    post = next((p for p in data["posts"] if p["reddit_id"] == post_id), None)
    if post is None:
        raise HTTPException(status_code=404, detail="Discussion not found")
    by_id = {c["reddit_id"]: c for c in post["comments"]}
    comments = []
    for comment in post["comments"]:
        predictions = {
            task: _data()["comments"][0]["predictions"].get(task, {})
            for task in ()
        }
        flat = next((r for r in data["comments"] if r["reddit_id"] == comment["reddit_id"]), {})
        comments.append({
            **comment, "parent_comment_text": by_id.get(comment.get("parent_reddit_id"), {}).get("body"),
            "predictions": flat.get("predictions", {}),
        })
    return {"post": post, "comments": comments, "synthetic_only": True}


@router.get("/issues", response_model=ItemsResponse)
def issues() -> dict[str, Any]:
    return {"items": _summary_data()["issue_distribution"], "synthetic_only": True}


@router.get("/topics", response_model=ItemsResponse)
def topics() -> dict[str, Any]:
    data = _data()
    rows_by_id = {row["reddit_id"]: row for row in data["comments"]}
    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for document in data["topic_assignments"]:
        topic_id = document.get("dominant_topic_id")
        row = rows_by_id.get(document.get("comment_id"))
        if topic_id is not None and row is not None:
            assigned[int(topic_id)].append({
                "post_id": row["post_id"], "post_title": row["post_title"],
                "text": row["body"], "score": document.get("dominant_topic_score"),
            })
    items = []
    for topic in data["topics"]:
        item = dict(topic)
        examples = sorted(assigned.get(int(topic["topic_id"]), []), key=lambda x: x.get("score") or 0, reverse=True)
        item["document_count"] = len(examples)
        item["representative_examples"] = examples[:2]
        items.append(item)
    return {"items": items, "synthetic_only": True}


@router.get("/sentiment", response_model=SentimentResponse)
def sentiment() -> dict[str, Any]:
    data = _data()
    rows = data["comments"]
    per_issue: dict[str, Counter] = defaultdict(Counter)
    monthly: dict[str, Counter] = defaultdict(Counter)
    emotions: Counter = Counter()
    sentiments: Counter = Counter()
    for row in rows:
        pred = row["predictions"]
        label = pred["sentiment"].get("sentiment")
        if label:
            sentiments[label] += 1
            per_issue[row["issue_key"]][label] += 1
            monthly[row["created_at"][:7]][label] += 1
        emotions.update(pred["emotion"].get("emotions", []))
    return {
        "sentiment": [{"label": key, "count": sentiments.get(key, 0)} for key in SENTIMENTS],
        "emotions": [{"label": key, "count": count} for key, count in sorted(emotions.items())],
        "by_issue": [{"issue": key, "distribution": [{"label": label, "count": counter.get(label, 0)} for label in SENTIMENTS]} for key, counter in sorted(per_issue.items())],
        "monthly": [{"month": month, "distribution": [{"label": label, "count": counter.get(label, 0)} for label in SENTIMENTS]} for month, counter in sorted(monthly.items())],
        "synthetic_only": True,
    }


@router.get("/responses", response_model=ItemsResponse)
def responses() -> dict[str, Any]:
    data = _data()
    rows = data["comments"]
    counts = Counter(r["predictions"]["response"].get("response_type") for r in rows if r["predictions"]["response"].get("response_type"))
    items = []
    for label in RESPONSE_TYPES:
        examples = [r for r in rows if r["predictions"]["response"].get("response_type") == label][:3]
        items.append({
            "label": label, "count": counts.get(label, 0),
            "examples": [{"text": r["body"], "post_title": r["post_title"], "confidence": r["predictions"]["response"].get("confidence"), "post_id": r["post_id"]} for r in examples],
        })
    return {"items": items, "synthetic_only": True}


@router.get("/resolution", response_model=ItemsResponse)
def resolution() -> dict[str, Any]:
    data = _data()
    items = []
    for post in data["posts"]:
        evidence = next((c for c in post["comments"] if c["reddit_id"] == post.get("resolution_evidence_comment_id")), None)
        proposed = [c for c in post["comments"] if c["response_type"] == "solution"]
        advice = [c for c in post["comments"] if c["response_type"] == "advice"]
        actionable = [c for c in post["comments"] if c["is_solution"]]
        items.append({
            "post_id": post["reddit_id"], "title": post["title"],
            "issue": post["issue_category"], "issue_key": post["issue_key"],
            "state": post["resolution_state"],
            "evidence_text": evidence["body"] if evidence else None,
            "evidence_type": evidence.get("resolution_evidence_type") if evidence else None,
            "evidence_comment_id": evidence["reddit_id"] if evidence else None,
            "proposals": [{"text": c["body"], "role": c["response_type"], "comment_id": c["reddit_id"]} for c in proposed],
            "advice": [{"text": c["body"], "comment_id": c["reddit_id"]} for c in advice],
            "actionable_recommendations": [{"text": c["body"], "comment_id": c["reddit_id"]} for c in actionable],
            "solution_model_output": [{
                "comment_id": c["reddit_id"], "is_solution": c["predictions"]["solution"].get("is_solution"),
                "probability": c["predictions"]["solution"].get("solution_probability"),
            } for c in data["comments"] if c["post_id"] == post["reddit_id"] and c["predictions"]["solution"]],
            "synthetic": True,
        })
    return {"items": items, "synthetic_only": True}


@router.get("/models", response_model=ModelsResponse)
def models() -> dict[str, Any]:
    return {"synthetic_only": True,
            "evaluation_scope": "Held-out thread-grouped splits from synthetic scenario-labeled data; not real-world performance.",
            "tasks": _data()["evaluation"]}
