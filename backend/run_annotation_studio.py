"""Local-only browser annotation studio (`/annotate`) for RCI real-data pilots.

Reads and atomically updates the human review CSV
(default: ``data/annotations/arctic_shift_pilot_corpus_v1/real_annotation_candidates.csv``),
enforcing all canonical taxonomy rules and readiness gates from
``backend/validate_pilot_annotations.py`` without touching ``/api/demo``,
``data/demo_synthetic/``, ``ml/artifacts/demo_synthetic_v2/``, or
``ml/artifacts/registry.json``.

Usage::

    python backend/run_annotation_studio.py
    # Opens on http://127.0.0.1:8765/annotate
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "backend"
for _p in (str(_PROJECT_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.validate_pilot_annotations import (
    CANONICAL_EMOTIONS,
    CANONICAL_ISSUES,
    CANONICAL_RESPONSES,
    CANONICAL_SENTIMENTS,
    CANONICAL_SOLUTIONS,
    FORBIDDEN_LABEL_TOKENS,
    VALID_ABSTAIN_REASONS,
    VALID_REVIEW_STATUSES,
    validate_and_split_pilot_csv,
)

DEFAULT_CSV = (
    _PROJECT_ROOT
    / "data"
    / "annotations"
    / "arctic_shift_pilot_corpus_v1"
    / "real_annotation_candidates.csv"
)
DEFAULT_STATIC_EXPORT = (
    _PROJECT_ROOT
    / "data"
    / "processed"
    / "arctic_shift_pilot_corpus_v1"
    / "sanitized_static_export.json"
)
DEFAULT_OUT_DIR = _PROJECT_ROOT / "ml" / "artifacts" / "real_arctic_shift_v1"
UI_DIR = _BACKEND_DIR / "annotation_studio_ui"
TOKENS_CSS = _BACKEND_DIR / "app" / "static" / "tokens.css"
BASE_CSS = _BACKEND_DIR / "app" / "static" / "base.css"

_GIF_ONLY_RE = re.compile(r"^\s*!\[gif\]\([^)]+\)\s*$", re.IGNORECASE)


def is_automod_boilerplate(text: str) -> bool:
    low = (text or "").lower()
    return (
        "this action was performed automatically" in low
        or "i am a bot, and this action was performed automatically" in low
        or ("i am a bot" in low and "contact the moderators" in low)
        or ("welcome to r/iimsambalpurug" in low and "bot" in low)
    )


def is_gif_only(text: str) -> bool:
    return bool(_GIF_ONLY_RE.fullmatch((text or "").strip()))


class RowUpdatePayload(BaseModel):
    review_status: str = Field(..., description="labeled | uncertain | abstain | excluded | unreviewed")
    sentiment: str = ""
    emotions: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    response_type: str = ""
    is_solution: str = ""
    abstain_reason: str = ""
    annotator_id: str = ""
    annotator_notes: str = ""


class BulkExcludePayload(BaseModel):
    annotator_id: str = "local_annotator"


class TrainRequestPayload(BaseModel):
    include_muril: bool = False
    muril_epochs: int = Field(default=6, ge=1, le=12)


class PilotAnnotationStore:
    """Thread-safe, crash-safe CSV store for unified 5-task pilot annotations."""

    def __init__(
        self,
        csv_path: Path,
        *,
        min_comments: int = 60,
        min_threads: int = 10,
        min_per_class: int = 3,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Annotation candidates CSV not found: {self.csv_path}")
        self.min_comments = min_comments
        self.min_threads = min_threads
        self.min_per_class = min_per_class
        self._lock = threading.RLock()
        self._backup_created = False
        self._fieldnames: List[str] = []
        self._rows: List[Dict[str, str]] = []
        self._load_locked()

    def _load_locked(self) -> None:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self._fieldnames = list(reader.fieldnames or [])
            self._rows = [dict(r) for r in reader]

    def _save_locked(self) -> None:
        if not self._backup_created:
            bak_path = self.csv_path.with_suffix(".csv.bak")
            if not bak_path.exists():
                shutil.copy2(self.csv_path, bak_path)
            self._backup_created = True

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".annotate_tmp_",
            suffix=".csv",
            dir=str(self.csv_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)
            os.replace(tmp_path, self.csv_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _compute_thread_balanced_order(self) -> Dict[str, int]:
        """Round-robin human comments across threads so labeling covers >=10 threads quickly.

        AutoMod / GIF-only noise rows are placed at the end of the queue so human
        annotators immediately see substantive student discussions.
        """
        human_buckets: Dict[str, List[str]] = defaultdict(list)
        noise_ids: List[str] = []
        thread_order: List[str] = []
        seen_threads: set[str] = set()

        for row in self._rows:
            doc_id = (row.get("document_id") or "").strip()
            text = row.get("original_text") or ""
            if is_automod_boilerplate(text) or is_gif_only(text):
                noise_ids.append(doc_id)
                continue
            pid = (row.get("post_id") or "").strip() or doc_id
            if pid not in seen_threads:
                seen_threads.add(pid)
                thread_order.append(pid)
            human_buckets[pid].append(doc_id)

        # Interleave subreddits across thread_order for community diversity too
        sub_threads: Dict[str, List[str]] = defaultdict(list)
        post_to_sub = {
            (r.get("post_id") or "").strip(): (r.get("subreddit") or "").strip()
            for r in self._rows
        }
        for pid in thread_order:
            sub_threads[post_to_sub.get(pid, "")].append(pid)

        interleaved_threads: List[str] = []
        sub_keys = sorted(sub_threads.keys())
        while any(sub_threads[k] for k in sub_keys):
            for k in sub_keys:
                if sub_threads[k]:
                    interleaved_threads.append(sub_threads[k].pop(0))

        ordered_ids: List[str] = []
        while any(human_buckets[pid] for pid in interleaved_threads):
            for pid in interleaved_threads:
                if human_buckets[pid]:
                    ordered_ids.append(human_buckets[pid].pop(0))

        ordered_ids.extend(noise_ids)
        return {doc_id: idx for idx, doc_id in enumerate(ordered_ids)}

    def _serialize_row(self, row: Dict[str, str], balanced_rank: int) -> Dict[str, Any]:
        orig = row.get("original_text") or ""
        norm = row.get("normalized_text") or ""
        emotions = [
            e.strip().lower()
            for e in (row.get("emotions") or "").split("|")
            if e.strip()
        ]
        issues = [
            i.strip().lower()
            for i in (row.get("issues") or "").split("|")
            if i.strip()
        ]
        status = (row.get("review_status") or "").strip().lower() or "unreviewed"
        return {
            "document_id": row.get("document_id", ""),
            "post_id": row.get("post_id", ""),
            "comment_id": row.get("comment_id", ""),
            "subreddit": row.get("subreddit", ""),
            "post_title": row.get("post_title", ""),
            "parent_comment_id": row.get("parent_comment_id", ""),
            "parent_text": row.get("parent_text", ""),
            "parent_context_status": row.get("parent_context_status", "top_level"),
            "created_at": row.get("created_at", ""),
            "score": row.get("score", ""),
            "original_text": orig,
            "normalized_text": norm,
            "language": row.get("language", "unknown"),
            "script": row.get("script", "unknown"),
            "is_code_mixed": (row.get("is_code_mixed") or "false").lower() == "true",
            "detected_languages": row.get("detected_languages", ""),
            "sentiment": (row.get("sentiment") or "").strip().lower(),
            "emotions": emotions,
            "issues": issues,
            "response_type": (row.get("response_type") or "").strip().lower(),
            "is_solution": (row.get("is_solution") or "").strip().lower(),
            "review_status": status,
            "abstain_reason": (row.get("abstain_reason") or "").strip().lower(),
            "annotator_id": row.get("annotator_id", ""),
            "annotator_notes": row.get("annotator_notes", ""),
            "is_automod": is_automod_boilerplate(orig),
            "is_gif_only": is_gif_only(orig),
            "balanced_rank": balanced_rank,
        }

    def validation_summary(self) -> Dict[str, Any]:
        with self._lock:
            return validate_and_split_pilot_csv(
                self.csv_path,
                out_dir=None,
                min_labeled_comments=self.min_comments,
                min_labeled_threads=self.min_threads,
                min_examples_per_class=self.min_per_class,
            )

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            ranks = self._compute_thread_balanced_order()
            serialized = [
                self._serialize_row(r, ranks.get((r.get("document_id") or "").strip(), idx))
                for idx, r in enumerate(self._rows)
            ]
            unreviewed_noise = sum(
                1
                for r in serialized
                if r["review_status"] == "unreviewed" and (r["is_automod"] or r["is_gif_only"])
            )
            subreddits = sorted({r["subreddit"] for r in serialized if r["subreddit"]})
            validation = validate_and_split_pilot_csv(
                self.csv_path,
                out_dir=None,
                min_labeled_comments=self.min_comments,
                min_labeled_threads=self.min_threads,
                min_examples_per_class=self.min_per_class,
            )
            return {
                "csv_path": str(self.csv_path),
                "total_rows": len(serialized),
                "unreviewed_bot_or_gif_noise": unreviewed_noise,
                "subreddits": subreddits,
                "thresholds": {
                    "min_comments": self.min_comments,
                    "min_threads": self.min_threads,
                    "min_per_class": self.min_per_class,
                },
                "taxonomy": {
                    "sentiments": list(CANONICAL_SENTIMENTS),
                    "emotions": list(CANONICAL_EMOTIONS),
                    "issues": list(CANONICAL_ISSUES),
                    "responses": list(CANONICAL_RESPONSES),
                    "solutions": list(CANONICAL_SOLUTIONS),
                    "abstain_reasons": sorted(VALID_ABSTAIN_REASONS),
                },
                "validation": validation,
                "rows": serialized,
            }

    def update_row(self, document_id: str, payload: RowUpdatePayload) -> Dict[str, Any]:
        status = (payload.review_status or "").strip().lower()
        if status not in VALID_REVIEW_STATUSES or status == "":
            raise HTTPException(
                status_code=422,
                detail=f"Invalid review_status '{status}'. Expected labeled, uncertain, abstain, excluded, or unreviewed.",
            )

        with self._lock:
            target_idx = next(
                (i for i, r in enumerate(self._rows) if (r.get("document_id") or "").strip() == document_id),
                None,
            )
            if target_idx is None:
                raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

            row = self._rows[target_idx]
            annotator_id = (payload.annotator_id or "").strip()
            notes = (payload.annotator_notes or "").strip()

            if status == "unreviewed":
                row["sentiment"] = ""
                row["emotions"] = ""
                row["issues"] = ""
                row["response_type"] = ""
                row["is_solution"] = ""
                row["review_status"] = "unreviewed"
                row["abstain_reason"] = ""
                row["annotator_id"] = annotator_id
                row["annotator_notes"] = notes
            elif status in {"uncertain", "abstain", "excluded"}:
                reason = (payload.abstain_reason or "").strip().lower()
                if reason not in VALID_ABSTAIN_REASONS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Status '{status}' requires a valid abstain_reason from: {sorted(VALID_ABSTAIN_REASONS)}",
                    )
                # Explicit scientific boundary: never leave or coerce training labels on uncertain/abstained items.
                row["sentiment"] = ""
                row["emotions"] = ""
                row["issues"] = ""
                row["response_type"] = ""
                row["is_solution"] = ""
                row["review_status"] = status
                row["abstain_reason"] = reason
                row["annotator_id"] = annotator_id
                row["annotator_notes"] = notes
            else:
                # status == "labeled"
                sent = (payload.sentiment or "").strip().lower()
                resp = (payload.response_type or "").strip().lower()
                sol = (payload.is_solution or "").strip().lower()
                if sol == "true":
                    sol = "solution"
                elif sol == "false":
                    sol = "not_solution"

                if sent in FORBIDDEN_LABEL_TOKENS or sent not in CANONICAL_SENTIMENTS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid sentiment '{sent}'. Must be one of {list(CANONICAL_SENTIMENTS)}.",
                    )
                if resp in FORBIDDEN_LABEL_TOKENS or resp not in CANONICAL_RESPONSES:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid response_type '{resp}'. Must be one of {list(CANONICAL_RESPONSES)}.",
                    )
                if sol in FORBIDDEN_LABEL_TOKENS or sol not in CANONICAL_SOLUTIONS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Invalid is_solution '{sol}'. Must be one of {list(CANONICAL_SOLUTIONS)}.",
                    )

                cleaned_emotions: List[str] = []
                for e in payload.emotions:
                    ev = (e or "").strip().lower()
                    if not ev:
                        continue
                    if ev in FORBIDDEN_LABEL_TOKENS or ev not in CANONICAL_EMOTIONS:
                        raise HTTPException(status_code=422, detail=f"Invalid emotion '{ev}'.")
                    if ev not in cleaned_emotions:
                        cleaned_emotions.append(ev)

                cleaned_issues: List[str] = []
                for iss in payload.issues:
                    iv = (iss or "").strip().lower()
                    if not iv:
                        continue
                    if iv in FORBIDDEN_LABEL_TOKENS or iv not in CANONICAL_ISSUES:
                        raise HTTPException(status_code=422, detail=f"Invalid issue '{iv}'.")
                    if iv not in cleaned_issues:
                        cleaned_issues.append(iv)

                row["sentiment"] = sent
                row["emotions"] = "|".join(sorted(cleaned_emotions))
                row["issues"] = "|".join(sorted(cleaned_issues))
                row["response_type"] = resp
                row["is_solution"] = sol
                row["review_status"] = "labeled"
                row["abstain_reason"] = ""
                row["annotator_id"] = annotator_id
                row["annotator_notes"] = notes

            self._save_locked()
            ranks = self._compute_thread_balanced_order()
            updated_serialized = self._serialize_row(
                row, ranks.get(document_id, target_idx)
            )
            validation = validate_and_split_pilot_csv(
                self.csv_path,
                out_dir=None,
                min_labeled_comments=self.min_comments,
                min_labeled_threads=self.min_threads,
                min_examples_per_class=self.min_per_class,
            )
            return {
                "saved": True,
                "row": updated_serialized,
                "validation": validation,
            }

    def exclude_unreviewed_bot_noise(self, annotator_id: str) -> Dict[str, Any]:
        with self._lock:
            excluded_ids: List[str] = []
            for row in self._rows:
                status = (row.get("review_status") or "").strip().lower() or "unreviewed"
                if status != "unreviewed":
                    continue
                orig = row.get("original_text") or ""
                is_bot = is_automod_boilerplate(orig)
                is_gif = is_gif_only(orig)
                if not (is_bot or is_gif):
                    continue
                row["sentiment"] = ""
                row["emotions"] = ""
                row["issues"] = ""
                row["response_type"] = ""
                row["is_solution"] = ""
                row["review_status"] = "excluded"
                row["abstain_reason"] = "off_topic_noise"
                row["annotator_id"] = (annotator_id or "local_annotator").strip()
                row["annotator_notes"] = (
                    "Auto-excluded deterministic AutoModerator boilerplate"
                    if is_bot
                    else "Auto-excluded deterministic GIF-only embed"
                )
                excluded_ids.append((row.get("document_id") or "").strip())

            if excluded_ids:
                self._save_locked()

            validation = validate_and_split_pilot_csv(
                self.csv_path,
                out_dir=None,
                min_labeled_comments=self.min_comments,
                min_labeled_threads=self.min_threads,
                min_examples_per_class=self.min_per_class,
            )
            return {
                "excluded_count": len(excluded_ids),
                "excluded_ids": excluded_ids,
                "validation": validation,
            }


class LocalTrainingRunner:
    """Runs ``train_real_pilot_models.py --mode supervised`` in a background thread."""

    def __init__(
        self,
        *,
        csv_path: Path,
        static_export_path: Path,
        out_dir: Path,
        min_comments: int,
        min_threads: int,
        min_per_class: int,
    ) -> None:
        self.csv_path = csv_path
        self.static_export_path = static_export_path
        self.out_dir = out_dir
        self.min_comments = min_comments
        self.min_threads = min_threads
        self.min_per_class = min_per_class
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "status": "idle",
            "started_at": None,
            "finished_at": None,
            "include_muril": False,
            "return_code": None,
            "error": None,
            "summary": None,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, *, include_muril: bool, muril_epochs: int) -> Dict[str, Any]:
        with self._lock:
            if self._state["status"] == "running":
                raise HTTPException(status_code=409, detail="A training job is already running.")
            self._state = {
                "status": "running",
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_at": None,
                "include_muril": include_muril,
                "muril_epochs": muril_epochs,
                "return_code": None,
                "error": None,
                "summary": None,
            }

        cmd = [
            sys.executable,
            str(_BACKEND_DIR / "train_real_pilot_models.py"),
            "--static-export",
            str(self.static_export_path),
            "--annotations",
            str(self.csv_path),
            "--out-dir",
            str(self.out_dir),
            "--mode",
            "supervised",
            "--min-comments",
            str(self.min_comments),
            "--min-threads",
            str(self.min_threads),
            "--min-per-class",
            str(self.min_per_class),
        ]
        if include_muril:
            cmd.extend(["--include-muril", "--muril-epochs", str(muril_epochs)])

        def _worker() -> None:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(_PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                summary_path = self.out_dir / "training_summary.json"
                parsed = None
                if summary_path.exists():
                    try:
                        parsed = json.loads(summary_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        parsed = None
                with self._lock:
                    self._state["finished_at"] = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    self._state["return_code"] = proc.returncode
                    self._state["summary"] = parsed
                    if proc.returncode == 0:
                        self._state["status"] = "completed"
                    else:
                        self._state["status"] = "failed"
                        self._state["error"] = (proc.stderr or proc.stdout or "")[-800:]
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._state["finished_at"] = datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )
                    self._state["status"] = "failed"
                    self._state["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return self.status()


def create_annotation_studio_app(
    *,
    csv_path: Path = DEFAULT_CSV,
    static_export_path: Path = DEFAULT_STATIC_EXPORT,
    out_dir: Path = DEFAULT_OUT_DIR,
    min_comments: int = 60,
    min_threads: int = 10,
    min_per_class: int = 3,
) -> FastAPI:
    """Create the isolated local-only FastAPI app for human pilot annotation."""
    store = PilotAnnotationStore(
        csv_path,
        min_comments=min_comments,
        min_threads=min_threads,
        min_per_class=min_per_class,
    )
    trainer = LocalTrainingRunner(
        csv_path=csv_path,
        static_export_path=static_export_path,
        out_dir=out_dir,
        min_comments=min_comments,
        min_threads=min_threads,
        min_per_class=min_per_class,
    )

    app = FastAPI(
        title="RCI Local Annotation Studio",
        description="Local-only keyboard-first human annotation workbench for real Reddit pilots.",
        docs_url=None,
        redoc_url=None,
    )

    if UI_DIR.exists():
        app.mount("/annotate-assets", StaticFiles(directory=str(UI_DIR)), name="annotate-assets")

    @app.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/annotate")

    @app.get("/annotate", response_class=HTMLResponse)
    def serve_annotate_page() -> FileResponse:
        index_html = UI_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(status_code=500, detail="Annotation UI assets not found.")
        return FileResponse(str(index_html), media_type="text/html; charset=utf-8")

    @app.get("/annotate-tokens.css", include_in_schema=False)
    def serve_tokens_css() -> FileResponse:
        if not TOKENS_CSS.exists():
            raise HTTPException(status_code=404, detail="tokens.css not found.")
        return FileResponse(str(TOKENS_CSS), media_type="text/css; charset=utf-8")

    @app.get("/annotate-base.css", include_in_schema=False)
    def serve_base_css() -> FileResponse:
        if not BASE_CSS.exists():
            raise HTTPException(status_code=404, detail="base.css not found.")
        return FileResponse(str(BASE_CSS), media_type="text/css; charset=utf-8")

    @app.get("/api/annotate/state")
    def get_annotate_state() -> Dict[str, Any]:
        state = store.get_state()
        state["train_job"] = trainer.status()
        state["train_command"] = (
            f"python backend/train_real_pilot_models.py --mode supervised --include-muril "
            f"--annotations {csv_path} --out-dir {out_dir}"
        )
        return state

    @app.post("/api/annotate/row/{document_id}")
    def save_annotate_row(document_id: str, payload: RowUpdatePayload) -> Dict[str, Any]:
        return store.update_row(document_id, payload)

    @app.post("/api/annotate/exclude-bot-noise")
    def exclude_bot_noise(payload: BulkExcludePayload) -> Dict[str, Any]:
        return store.exclude_unreviewed_bot_noise(payload.annotator_id)

    @app.post("/api/annotate/train")
    def trigger_training(payload: TrainRequestPayload) -> Dict[str, Any]:
        validation = store.validation_summary()
        if not validation["ready_for_training"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Dataset has not met human-annotation readiness thresholds yet.",
                    "readiness_blockers": validation["readiness_blockers"],
                },
            )
        return trainer.start(
            include_muril=payload.include_muril,
            muril_epochs=payload.muril_epochs,
        )

    @app.get("/api/annotate/train/status")
    def get_training_status() -> Dict[str, Any]:
        return trainer.status()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help="Path to unified 5-task human review CSV",
    )
    parser.add_argument(
        "--static-export",
        default=str(DEFAULT_STATIC_EXPORT),
        help="Validated static export JSON path (for training provenance)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Isolated gitignored output directory for trained real-pilot models",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1 localhost-only)")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765)")
    parser.add_argument("--min-comments", type=int, default=60)
    parser.add_argument("--min-threads", type=int, default=10)
    parser.add_argument("--min-per-class", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "Warning: Binding annotation studio outside localhost is not recommended "
            "because real Reddit pilot CSVs are private local research files.",
            file=sys.stderr,
        )
    app = create_annotation_studio_app(
        csv_path=Path(args.csv),
        static_export_path=Path(args.static_export),
        out_dir=Path(args.out_dir),
        min_comments=args.min_comments,
        min_threads=args.min_threads,
        min_per_class=args.min_per_class,
    )
    print(f"\nRCI Local Annotation Studio running at: http://{args.host}:{args.port}/annotate")
    print(f"CSV file: {args.csv}")
    print("Press Ctrl+C to stop.\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
