"""Authorized static-export intake, privacy sanitization, quality reporting,
and blank-label 5-task annotation-candidate generator.

Reads operator-supplied, legitimately obtained Reddit export files (never calls
live providers, never scrapes, never mutates raw inputs, never fabricates
missing fields or ground-truth labels).

Supported input formats (file or directory via ``--input``):
  1. **Reddit Thread JSON** (``[post_listing, comment_listing]``)
  2. **Reddit Subreddit / Search Listing JSON** (``{"kind": "Listing", ...}``)
  3. **Arctic Shift / Pushshift JSONL/NDJSON or JSON envelope** (``{"data": [...]}``)
  4. **Pre-structured ``format_version: 1`` static-export JSON or JSONL**

Privacy & provenance guarantees:
  - Strips all author/account metadata (``author``, ``author_fullname``, flairs, etc.).
  - Redacts inline email addresses (``<EMAIL>``), phone numbers (``<PHONE>``),
    and Reddit username mentions ``u/<name>`` / ``/u/<name>`` (``<USER_MENTION>``)
    in sanitized outputs while recording redaction counts (never redacted text).
  - Never silently fabricates missing scores, comment counts, or timestamps:
    absent numeric fields remain omitted so ``static_export.py`` reports
    ``lossy_default_fields`` honestly, and records missing required IDs/text/
    timestamps are rejected and counted by safe diagnostic code.
  - Raw source files are never overwritten; sanitized exports and blank-label
    annotation candidates are written to separate output locations.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.services.annotation.candidates import (
    build_candidate_rows,
    write_candidates_csv,
)
from app.services.annotation.manifest import (
    build_manifest_records,
    validate_candidate_manifest_join,
    validate_candidate_records,
    write_jsonl_manifest,
)
from app.services.ingestion.static_export import (
    StaticExportLimits,
    load_static_export,
)

_ID_CLEAN = re.compile(r"^[a-z0-9]+$")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{4,5}(?!\w)"
)
_USER_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])/?u/[A-Za-z0-9_-]{3,20}\b")
_AUTHOR_META_KEYS = frozenset(
    {
        "author",
        "author_fullname",
        "author_flair_text",
        "author_flair_css_class",
        "author_cakeday",
        "author_premium",
        "author_patreon_flair",
    }
)


def sanitize_pii_text(
    text: Optional[str],
    counters: Counter,
) -> Optional[str]:
    """Deterministically redact inline PII/usernames without logging values."""
    if text is None:
        return None
    cleaned = text

    cleaned, n_email = _EMAIL_RE.subn("<EMAIL>", cleaned)
    if n_email:
        counters["email_redactions"] += n_email

    cleaned, n_mention = _USER_MENTION_RE.subn("<USER_MENTION>", cleaned)
    if n_mention:
        counters["user_mention_redactions"] += n_mention

    # Only redact digit spans with at least 10 digits (e.g. Indian mobile /
    # international phone numbers) so scores, years, CAT percentiles, and fees
    # like "2026" or "12,00,000" are never mistakenly redacted.
    def _phone_sub(match: re.Match[str]) -> str:
        raw_val = match.group(0)
        digits = re.sub(r"\D", "", raw_val)
        if 10 <= len(digits) <= 13 and not raw_val.strip().startswith("20"):
            counters["phone_redactions"] += 1
            return "<PHONE>"
        return raw_val

    cleaned = _PHONE_RE.sub(_phone_sub, cleaned)
    return cleaned


def _clean_id(value: Any, prefix: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().removeprefix(f"{prefix}_").lower()
    if not cleaned or not _ID_CLEAN.fullmatch(cleaned):
        return None
    return cleaned


def _iso_utc(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return None  # Reject ambiguous naive timestamps rather than guessing timezone
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return None


def _count_stripped_author_keys(raw: dict[str, Any], pii_counts: Counter) -> None:
    stripped = sum(
        1 for key in _AUTHOR_META_KEYS if key in raw and raw.get(key) not in (None, "", "[deleted]")
    )
    if stripped:
        pii_counts["author_fields_stripped"] += stripped


def _extract_post_dict(
    raw: dict[str, Any],
    *,
    reject_counts: Counter,
    pii_counts: Counter,
) -> Optional[dict[str, Any]]:
    _count_stripped_author_keys(raw, pii_counts)
    post_id = _clean_id(raw.get("id") or raw.get("post_id"), "t3")
    if not post_id:
        reject_counts["post_missing_or_invalid_id"] += 1
        return None
    raw_title = str(raw.get("title") or "").strip()
    if not raw_title:
        reject_counts["post_missing_title"] += 1
        return None
    subreddit = str(raw.get("subreddit") or "").strip().removeprefix("r/")
    if not subreddit or any(ch.isspace() for ch in subreddit) or "/" in subreddit:
        reject_counts["post_missing_or_invalid_subreddit"] += 1
        return None
    created_at = _iso_utc(
        raw.get("created_utc") if "created_utc" in raw else raw.get("created_at")
    )
    if not created_at:
        reject_counts["post_invalid_or_missing_timestamp"] += 1
        return None

    title = sanitize_pii_text(raw_title, pii_counts) or raw_title
    raw_body = raw.get("selftext") if "selftext" in raw else raw.get("body")
    body = str(raw_body).strip() if isinstance(raw_body, str) else ""
    if body in {"", "[deleted]", "[removed]"}:
        body_val = None
    else:
        body_val = sanitize_pii_text(body, pii_counts)

    permalink = raw.get("permalink")
    if isinstance(permalink, str) and permalink.startswith("/"):
        url = f"https://www.reddit.com{permalink}"
    else:
        raw_url = raw.get("url")
        url = (
            raw_url.strip()
            if isinstance(raw_url, str) and raw_url.startswith(("http://", "https://"))
            else None
        )

    post_obj: dict[str, Any] = {
        "post_id": post_id,
        "subreddit": subreddit,
        "title": title,
        "body": body_val,
        "created_at": created_at,
        "url": url,
        "comments": [],
    }
    # Preserve genuine observed numeric values only; omit missing values so
    # static_export.py flags lossy defaults explicitly instead of fabricating 0.
    raw_score = raw.get("score")
    if isinstance(raw_score, int) and not isinstance(raw_score, bool):
        post_obj["score"] = raw_score
    else:
        reject_counts["post_missing_score_unfabricated"] += 1

    raw_num = raw.get("num_comments")
    if isinstance(raw_num, int) and not isinstance(raw_num, bool) and raw_num >= 0:
        post_obj["num_comments"] = raw_num
    else:
        reject_counts["post_missing_num_comments_unfabricated"] += 1

    return post_obj


def _extract_comment_dict(
    raw: dict[str, Any],
    *,
    reject_counts: Counter,
    pii_counts: Counter,
) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
    _count_stripped_author_keys(raw, pii_counts)
    comment_id = _clean_id(raw.get("id") or raw.get("comment_id"), "t1")
    link_id = _clean_id(raw.get("link_id") or raw.get("post_id"), "t3")
    if not comment_id:
        reject_counts["comment_missing_or_invalid_id"] += 1
        return link_id, None
    raw_body = str(raw.get("body") or "").strip()
    if raw_body in {"[deleted]", "[removed]"}:
        reject_counts["comment_deleted_or_removed"] += 1
        return link_id, None
    if not raw_body:
        reject_counts["comment_blank_body"] += 1
        return link_id, None
    created_at = _iso_utc(
        raw.get("created_utc") if "created_utc" in raw else raw.get("created_at")
    )
    if not created_at:
        reject_counts["comment_invalid_or_missing_timestamp"] += 1
        return link_id, None

    body = sanitize_pii_text(raw_body, pii_counts) or raw_body
    parent_raw = raw.get("parent_id") if "parent_id" in raw else raw.get("parent_comment_id")
    parent_id = None
    if isinstance(parent_raw, str) and parent_raw.strip():
        if parent_raw.startswith("t1_"):
            parent_id = _clean_id(parent_raw, "t1")
        elif not parent_raw.startswith("t3_"):
            parent_id = _clean_id(parent_raw, "t1")
    if parent_id == comment_id:
        reject_counts["comment_self_parent_cleared"] += 1
        parent_id = None

    comment_obj: dict[str, Any] = {
        "comment_id": comment_id,
        "body": body,
        "created_at": created_at,
        "parent_comment_id": parent_id,
    }
    raw_score = raw.get("score")
    if isinstance(raw_score, int) and not isinstance(raw_score, bool):
        comment_obj["score"] = raw_score
    else:
        reject_counts["comment_missing_score_unfabricated"] += 1

    return link_id, comment_obj


def _record_post(
    norm: dict[str, Any],
    posts: Dict[str, dict[str, Any]],
    reject_counts: Counter,
) -> None:
    pid = norm["post_id"]
    if pid in posts:
        existing = {k: v for k, v in posts[pid].items() if k != "comments"}
        candidate = {k: v for k, v in norm.items() if k != "comments"}
        if existing == candidate:
            reject_counts["duplicate_post_id_deduped"] += 1
        else:
            reject_counts["conflicting_post_id_kept_first"] += 1
        return
    posts[pid] = norm


def _record_comment(
    post_id: str,
    cnorm: dict[str, Any],
    comments_by_post: Dict[str, Dict[str, dict[str, Any]]],
    reject_counts: Counter,
) -> None:
    bucket = comments_by_post.setdefault(post_id, {})
    cid = cnorm["comment_id"]
    if cid in bucket:
        if bucket[cid] == cnorm:
            reject_counts["duplicate_comment_id_deduped"] += 1
        else:
            reject_counts["conflicting_comment_id_kept_first"] += 1
        return
    bucket[cid] = cnorm


def _walk_comment_listing(
    children: Sequence[Any],
    post_id: str,
    comments_by_post: Dict[str, Dict[str, dict[str, Any]]],
    reject_counts: Counter,
    pii_counts: Counter,
) -> None:
    stack = list(reversed(children))
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        kind = node.get("kind")
        if kind == "more":
            reject_counts["collapsed_more_comment_branches"] += 1
            continue
        if kind != "t1":
            continue
        data = node.get("data")
        if not isinstance(data, dict):
            reject_counts["comment_malformed_node"] += 1
            continue
        _, normalized = _extract_comment_dict(
            {**data, "link_id": data.get("link_id") or f"t3_{post_id}"},
            reject_counts=reject_counts,
            pii_counts=pii_counts,
        )
        if normalized is not None:
            _record_comment(post_id, normalized, comments_by_post, reject_counts)
        replies = data.get("replies")
        if isinstance(replies, dict):
            sub_children = replies.get("data", {}).get("children")
            if isinstance(sub_children, list):
                stack.extend(reversed(sub_children))


def parse_raw_reddit_files(
    paths: Sequence[Path],
) -> Tuple[List[dict[str, Any]], Dict[str, Any]]:
    """Parse browser thread JSONs, listing JSONs, Arctic Shift JSONLs, and format_version=1 exports."""
    posts: Dict[str, dict[str, Any]] = {}
    comments_by_post: Dict[str, Dict[str, dict[str, Any]]] = {}
    reject_counts: Counter = Counter()
    pii_counts: Counter = Counter()
    files_parsed = 0
    posts_seen = 0
    comments_seen = 0
    sha256 = hashlib.sha256()

    for file_path in sorted(paths):
        suffix = file_path.suffix.lower()
        if suffix not in {".json", ".jsonl", ".ndjson"}:
            continue
        raw_bytes = file_path.read_bytes()
        sha256.update(raw_bytes)
        try:
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            reject_counts["file_invalid_utf8"] += 1
            continue
        files_parsed += 1

        if suffix == ".json":
            try:
                payload = json.loads(text)
            except ValueError:
                reject_counts["file_invalid_json"] += 1
                continue
            # Case 0: Pre-structured format_version == 1 static export JSON
            if (
                isinstance(payload, dict)
                and payload.get("format_version") == 1
                and isinstance(payload.get("posts"), list)
            ):
                for raw_post in payload["posts"]:
                    if not isinstance(raw_post, dict):
                        reject_counts["post_malformed_record"] += 1
                        continue
                    posts_seen += 1
                    norm = _extract_post_dict(
                        raw_post, reject_counts=reject_counts, pii_counts=pii_counts
                    )
                    if norm:
                        _record_post(norm, posts, reject_counts)
                        for raw_comment in raw_post.get("comments", []) or []:
                            if not isinstance(raw_comment, dict):
                                reject_counts["comment_malformed_node"] += 1
                                continue
                            comments_seen += 1
                            _, cnorm = _extract_comment_dict(
                                {**raw_comment, "post_id": norm["post_id"]},
                                reject_counts=reject_counts,
                                pii_counts=pii_counts,
                            )
                            if cnorm:
                                _record_comment(
                                    norm["post_id"], cnorm, comments_by_post, reject_counts
                                )
            # Case 1: Reddit thread JSON [post_listing, comment_listing]
            elif isinstance(payload, list) and len(payload) >= 1 and isinstance(payload[0], dict):
                post_children = payload[0].get("data", {}).get("children", [])
                thread_post_id: Optional[str] = None
                for child in post_children:
                    if isinstance(child, dict) and child.get("kind") == "t3":
                        posts_seen += 1
                        norm = _extract_post_dict(
                            child.get("data", {}),
                            reject_counts=reject_counts,
                            pii_counts=pii_counts,
                        )
                        if norm:
                            thread_post_id = norm["post_id"]
                            _record_post(norm, posts, reject_counts)
                if len(payload) >= 2 and isinstance(payload[1], dict) and thread_post_id:
                    comment_children = payload[1].get("data", {}).get("children", [])
                    _walk_comment_listing(
                        comment_children,
                        thread_post_id,
                        comments_by_post,
                        reject_counts,
                        pii_counts,
                    )
            # Case 2: Reddit Subreddit/Search Listing JSON
            elif isinstance(payload, dict) and payload.get("kind") == "Listing":
                for child in payload.get("data", {}).get("children", []):
                    if not isinstance(child, dict):
                        continue
                    kind = child.get("kind")
                    data = child.get("data", {})
                    if kind == "t3":
                        posts_seen += 1
                        norm = _extract_post_dict(
                            data, reject_counts=reject_counts, pii_counts=pii_counts
                        )
                        if norm:
                            _record_post(norm, posts, reject_counts)
                    elif kind == "t1":
                        comments_seen += 1
                        pid, cnorm = _extract_comment_dict(
                            data, reject_counts=reject_counts, pii_counts=pii_counts
                        )
                        if pid and cnorm:
                            _record_comment(pid, cnorm, comments_by_post, reject_counts)
                        elif cnorm and not pid:
                            reject_counts["comment_missing_post_id"] += 1
            # Case 3: Arctic Shift API JSON envelope {"data": [...]}
            elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
                for row in payload["data"]:
                    if not isinstance(row, dict):
                        continue
                    if "title" in row:
                        posts_seen += 1
                        norm = _extract_post_dict(
                            row, reject_counts=reject_counts, pii_counts=pii_counts
                        )
                        if norm:
                            _record_post(norm, posts, reject_counts)
                    elif "body" in row:
                        comments_seen += 1
                        pid, cnorm = _extract_comment_dict(
                            row, reject_counts=reject_counts, pii_counts=pii_counts
                        )
                        if pid and cnorm:
                            _record_comment(pid, cnorm, comments_by_post, reject_counts)
                        elif cnorm and not pid:
                            reject_counts["comment_missing_post_id"] += 1
            else:
                reject_counts["file_unsupported_json_shape"] += 1
        else:
            # JSONL / NDJSON (Arctic Shift Download Tool, Pushshift, or format_version=1 JSONL)
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except ValueError:
                    reject_counts["jsonl_invalid_line"] += 1
                    continue
                if not isinstance(row, dict):
                    reject_counts["jsonl_non_object_line"] += 1
                    continue
                rec_type = row.get("record_type")
                if rec_type == "metadata":
                    continue
                if rec_type == "post" and isinstance(row.get("post"), dict):
                    posts_seen += 1
                    norm = _extract_post_dict(
                        row["post"], reject_counts=reject_counts, pii_counts=pii_counts
                    )
                    if norm:
                        _record_post(norm, posts, reject_counts)
                elif rec_type == "comment" and isinstance(row.get("comment"), dict):
                    comments_seen += 1
                    pid, cnorm = _extract_comment_dict(
                        {**row["comment"], "post_id": row.get("post_id")},
                        reject_counts=reject_counts,
                        pii_counts=pii_counts,
                    )
                    if pid and cnorm:
                        _record_comment(pid, cnorm, comments_by_post, reject_counts)
                    elif cnorm and not pid:
                        reject_counts["comment_missing_post_id"] += 1
                elif "title" in row and "body" not in row:
                    posts_seen += 1
                    norm = _extract_post_dict(
                        row, reject_counts=reject_counts, pii_counts=pii_counts
                    )
                    if norm:
                        _record_post(norm, posts, reject_counts)
                elif "body" in row and ("link_id" in row or "post_id" in row):
                    comments_seen += 1
                    pid, cnorm = _extract_comment_dict(
                        row, reject_counts=reject_counts, pii_counts=pii_counts
                    )
                    if pid and cnorm:
                        _record_comment(pid, cnorm, comments_by_post, reject_counts)
                    elif cnorm and not pid:
                        reject_counts["comment_missing_post_id"] += 1
                else:
                    reject_counts["jsonl_unrecognized_record"] += 1

    # Attach comments to matching posts and audit parent references
    result_posts: List[dict[str, Any]] = []
    parent_status_counts: Counter = Counter()
    for post_id, post_obj in posts.items():
        attached = list(comments_by_post.pop(post_id, {}).values())
        attached.sort(key=lambda c: (c["created_at"], c["comment_id"]))
        present_ids = {c["comment_id"] for c in attached}
        for c in attached:
            if c["parent_comment_id"] is None:
                parent_status_counts["top_level"] += 1
            elif c["parent_comment_id"] in present_ids:
                parent_status_counts["available"] += 1
            else:
                parent_status_counts["unavailable"] += 1
                reject_counts["unresolved_comment_parent"] += 1
        post_obj["comments"] = attached
        result_posts.append(post_obj)

    orphaned_comments = sum(len(bucket) for bucket in comments_by_post.values())
    if orphaned_comments:
        reject_counts["orphaned_comments_post_missing"] += orphaned_comments

    result_posts.sort(key=lambda p: (p["created_at"], p["post_id"]), reverse=True)
    comments_attached = sum(len(p["comments"]) for p in result_posts)
    stats = {
        "files_parsed": files_parsed,
        "posts_seen": max(posts_seen, len(result_posts)),
        "posts_extracted": len(result_posts),
        "comments_seen": max(comments_seen, comments_attached),
        "comments_attached": comments_attached,
        "orphaned_comments_skipped": orphaned_comments,
        "parent_context_breakdown": {
            "top_level": parent_status_counts["top_level"],
            "available": parent_status_counts["available"],
            "unavailable": parent_status_counts["unavailable"],
        },
        "pii_redactions": {
            "author_fields_stripped": pii_counts["author_fields_stripped"],
            "email_redactions": pii_counts["email_redactions"],
            "phone_redactions": pii_counts["phone_redactions"],
            "user_mention_redactions": pii_counts["user_mention_redactions"],
        },
        "diagnostics_and_rejections": dict(sorted(reject_counts.items())),
        "input_sha256": sha256.hexdigest(),
    }
    return result_posts, stats


def export_blank_annotation_candidates(
    posts: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    source_provider: str = "reddit-static-export",
    source_mode: str = "direct",
) -> Dict[str, Any]:
    """Export sanitized blank-label 5-task annotation candidates + canonical manifest sidecar.

    Outputs under ``out_dir``:
      1. ``real_annotation_candidates.csv`` — unified 5-task review sheet with
         blank labels + explicit ``review_status`` and ``abstain_reason`` columns.
      2. ``response_candidates.csv`` — canonical response-candidate handoff via
         ``app.services.annotation.candidates.write_candidates_csv()``.
      3. ``review_manifest.jsonl`` — canonical v1.0 review manifest sidecar via
         ``app.services.annotation.manifest.write_jsonl_manifest()``, verified
         against ``response_candidates.csv`` with ``validate_candidate_manifest_join()``.

    No model prediction is ever written into any ground-truth label column.
    """
    from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
    from app.services.nlp.language import SimpleLanguageDetector
    from app.services.nlp.language_analysis_service import LanguageAnalysisService
    from app.services.nlp.preprocessing.multilingual_preprocessor import (
        MultilingualTextPreprocessor,
    )
    from app.services.nlp.script_detector import SimpleScriptDetector

    preprocessor = MultilingualTextPreprocessor()
    lang_service = LanguageAnalysisService(
        language_detector=SimpleLanguageDetector(),
        script_detector=SimpleScriptDetector(),
        code_mix_detector=SimpleCodeMixDetector(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    unified_csv_path = out_dir / "real_annotation_candidates.csv"
    response_csv_path = out_dir / "response_candidates.csv"
    manifest_jsonl_path = out_dir / "review_manifest.jsonl"

    unified_fieldnames = [
        "document_id",
        "post_id",
        "comment_id",
        "subreddit",
        "post_title",
        "parent_comment_id",
        "parent_text",
        "parent_context_status",
        "created_at",
        "score",
        "original_text",
        "normalized_text",
        "language",
        "script",
        "is_code_mixed",
        "detected_languages",
        "sentiment",
        "emotions",
        "issues",
        "response_type",
        "is_solution",
        "review_status",
        "abstain_reason",
        "annotator_id",
        "annotator_notes",
    ]

    lang_counts: Counter = Counter()
    script_counts: Counter = Counter()
    code_mix_count = 0
    candidate_inputs: List[Dict[str, Any]] = []
    rows_written = 0

    with unified_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=unified_fieldnames)
        writer.writeheader()
        for post in posts:
            comments_by_id = {c["comment_id"]: c for c in post.get("comments", [])}
            for comment in post.get("comments", []):
                raw_text = comment["body"]
                pre = preprocessor.preprocess(raw_text)
                lang = lang_service.analyze(raw_text)

                lang_counts[lang.language] += 1
                script_counts[lang.script] += 1
                if lang.is_code_mixed:
                    code_mix_count += 1

                parent_id = comment.get("parent_comment_id")
                parent = comments_by_id.get(parent_id) if parent_id else None
                parent_status = (
                    "top_level"
                    if not parent_id
                    else ("available" if parent is not None else "unavailable")
                )
                doc_id = f"comment-{comment['comment_id']}"
                score_val = comment.get("score")

                writer.writerow(
                    {
                        "document_id": doc_id,
                        "post_id": post["post_id"],
                        "comment_id": comment["comment_id"],
                        "subreddit": post["subreddit"],
                        "post_title": post["title"],
                        "parent_comment_id": parent_id or "",
                        "parent_text": parent["body"] if parent is not None else "",
                        "parent_context_status": parent_status,
                        "created_at": comment["created_at"],
                        "score": "" if score_val is None else str(score_val),
                        "original_text": raw_text,
                        "normalized_text": pre.normalized_text,
                        "language": lang.language,
                        "script": lang.script,
                        "is_code_mixed": str(bool(lang.is_code_mixed)).lower(),
                        "detected_languages": "|".join(lang.detected_languages),
                        "sentiment": "",
                        "emotions": "",
                        "issues": "",
                        "response_type": "",
                        "is_solution": "",
                        "review_status": "",
                        "abstain_reason": "",
                        "annotator_id": "",
                        "annotator_notes": "",
                    }
                )
                candidate_inputs.append(
                    {
                        "document_id": doc_id,
                        "comment_id": comment["comment_id"],
                        "text": raw_text,
                        "response_type": "",
                        "post_id": post["post_id"],
                        "parent_comment_id": parent_id or "",
                        "parent_text": parent["body"] if parent is not None else "",
                        "post_title": post["title"],
                        "subreddit": post["subreddit"],
                        "score": score_val,
                        "created_at": comment["created_at"],
                        "source_url": post.get("url") or "",
                        "source_provider": source_provider,
                        "source_mode": source_mode,
                        "parent_context_status": parent_status,
                    }
                )
                rows_written += 1

    # Also build & validate the canonical response_candidates.csv + review_manifest.jsonl
    response_rows = build_candidate_rows(
        candidate_inputs,
        preprocessor=preprocessor,
        language_service=lang_service,
    )
    candidate_report = validate_candidate_records(response_rows)
    manifest_records = build_manifest_records(response_rows)
    join_report = validate_candidate_manifest_join(response_rows, manifest_records)

    if response_csv_path.exists():
        response_csv_path.unlink()
    write_candidates_csv(str(response_csv_path), response_rows)
    if manifest_jsonl_path.exists():
        manifest_jsonl_path.unlink()
    write_jsonl_manifest(str(manifest_jsonl_path), manifest_records)

    print(
        f"Exported {rows_written} sanitized blank-label candidate rows to {unified_csv_path} "
        f"(and manifest sidecar to {manifest_jsonl_path})"
    )
    return {
        "rows_exported": rows_written,
        "unified_candidates_csv": str(unified_csv_path),
        "response_candidates_csv": str(response_csv_path),
        "review_manifest_jsonl": str(manifest_jsonl_path),
        "candidate_validation_valid": candidate_report.valid,
        "manifest_join_valid": join_report.valid,
        "language_distribution": dict(sorted(lang_counts.items())),
        "script_distribution": dict(sorted(script_counts.items())),
        "code_mixed_comments": code_mix_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import_reddit_export",
        description="Sanitize, validate, and convert authorized Reddit exports into RCI static exports and blank-label annotation candidates.",
    )
    parser.add_argument("--input", required=True, help="Input .json/.jsonl file or directory of files (read-only)")
    parser.add_argument("--output", required=True, help="Output sanitized static-export .json path")
    parser.add_argument("--dataset-id", required=True, help="Explicit dataset version ID (e.g. real-pilot-20261119-v1)")
    parser.add_argument("--community", required=True, help="Primary community name (e.g. IIMSambalpur or MBAIndia)")
    parser.add_argument("--source-provider", default="reddit-static-export", help="Provenance provider label")
    parser.add_argument("--source-mode", default="direct", choices=["direct", "context"])
    parser.add_argument("--authorization-status", default="pending-review", help="Metadata authorization_status")
    parser.add_argument("--authorization-reference", default="operator-local-import", help="Metadata authorization_reference")
    parser.add_argument("--export-candidates", default=None, help="Optional directory to write blank-label annotation candidates + manifest")
    parser.add_argument("--quality-report", default=None, help="Optional JSON path to save the dataset quality & provenance report")
    parser.add_argument("--max-posts", type=int, default=200, help="Validation limit for max posts (default 200)")
    parser.add_argument("--max-comments", type=int, default=5000, help="Validation limit for max comments (default 5000)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    in_path = Path(args.input)
    if in_path.is_dir():
        files = [
            p for p in sorted(in_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        ]
    elif in_path.is_file():
        files = [in_path]
    else:
        print(f"Error: input path not found: {in_path}", file=sys.stderr)
        return 2

    if not files:
        print(f"Error: no .json/.jsonl files found in {in_path}", file=sys.stderr)
        return 2

    posts, stats = parse_raw_reddit_files(files)
    if not posts:
        print("Error: no valid Reddit posts found in input files.", file=sys.stderr)
        return 2

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "format_version": 1,
        "metadata": {
            "dataset_id": args.dataset_id,
            "source": {
                "community": args.community,
                "source_reference": f"sha256:{stats['input_sha256']}",
                "source_provider": args.source_provider,
                "source_mode": args.source_mode,
                "collected_at": now_iso,
            },
            "completeness": {
                "comments_complete": False,
            },
            "authorization": {
                "authorization_status": args.authorization_status,
                "authorization_reference": args.authorization_reference,
            },
        },
        "posts": posts,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Immediately validate against the project's canonical static_export contract
    limits = StaticExportLimits(
        max_bytes=max(10 * 1024 * 1024, out_path.stat().st_size + 1024),
        max_posts=max(args.max_posts, len(posts)),
        max_comments_per_post=2000,
        max_comments=max(args.max_comments, stats["comments_attached"] + 1),
    )
    validated = load_static_export(out_path, limits=limits)

    candidate_summary = None
    if args.export_candidates and validated.valid:
        candidate_summary = export_blank_annotation_candidates(
            posts,
            Path(args.export_candidates),
            source_provider=args.source_provider,
            source_mode=args.source_mode,
        )

    quality_report = {
        "dataset_id": args.dataset_id,
        "generated_at": now_iso,
        "provenance": payload["metadata"],
        "import_stats": stats,
        "validation_summary": validated.as_summary(),
        "candidate_export_summary": candidate_summary,
    }

    report_path = (
        Path(args.quality_report)
        if args.quality_report
        else (Path(args.export_candidates) / "dataset_quality_provenance_report.json" if args.export_candidates else None)
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(quality_report, indent=2))
    if not validated.valid:
        print("Warning: static_export validation reported error codes above.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
