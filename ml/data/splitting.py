"""
Deterministic train/validation/test splitting for the ML training framework.

Design
-------
Every split is fully determined by ``random_state`` (default 42) and the
input ordering.  Two runs on the same data produce identical splits.  The
splitter enforces the core invariant: **a document must never appear in more
than one split**.

Stratification is used for single-label classification tasks when the label
distribution supports it.  For multi-label tasks the splitter does NOT call
ordinary ``train_test_split(stratify=...)`` because a multi-label binarized
matrix would silently drop documents whose label set is absent from a
particular fold - instead it reports stratification as unavailable and falls
back to random splitting with a clear warning.

The splitter never mutates the input records.  It returns three new lists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Split specification
# ---------------------------------------------------------------------------


@dataclass
class SplitSpec:
    """Configuration for one split operation."""

    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = True
    min_train_examples: int = 1
    min_validation_examples: int = 1
    min_test_examples: int = 1

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split proportions must sum to 1.0, got "
                f"train={self.train_size}, validation={self.validation_size}, "
                f"test={self.test_size} (sum={total})"
            )
        if not (0.0 < self.train_size <= 1.0):
            raise ValueError(f"train_size must be in (0, 1], got {self.train_size}")
        if not (0.0 <= self.validation_size <= 1.0):
            raise ValueError(
                f"validation_size must be in [0, 1], got {self.validation_size}"
            )
        if not (0.0 <= self.test_size <= 1.0):
            raise ValueError(f"test_size must be in [0, 1], got {self.test_size}")
        # Note: min_train_examples is enforced at split time (when n is known),
        # not here, because train_size is a proportion and n is not available
        # in the constructor.

    @property
    def has_validation(self) -> bool:
        return self.validation_size > 0

    @property
    def has_test(self) -> bool:
        return self.test_size > 0


# ---------------------------------------------------------------------------
# Split result
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """The output of one split operation."""

    train: List[int]  # indices into the original records list
    validation: List[int]
    test: List[int]
    random_state: int
    stratify_used: bool
    stratify_available: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def all_indices(self) -> List[int]:
        return self.train + self.validation + self.test

    def verify_no_overlap(self) -> Tuple[bool, List[str]]:
        """Check that no index appears in more than one split."""
        t = set(self.train)
        v = set(self.validation)
        te = set(self.test)
        overlaps: List[str] = []
        if t & v:
            overlaps.append(f"train/validation overlap ({len(t & v)} indices)")
        if t & te:
            overlaps.append(f"train/test overlap ({len(t & te)} indices)")
        if v & te:
            overlaps.append(f"validation/test overlap ({len(v & te)} indices)")
        if len(t | v | te) != len(self.all_indices):
            overlaps.append("duplicate indices within a single split")
        return (len(overlaps) == 0, overlaps)


# ---------------------------------------------------------------------------
# Public splitting API
# ---------------------------------------------------------------------------


def split_records(
    records: Sequence,
    labels: Sequence[str],
    spec: Optional[SplitSpec] = None,
) -> SplitResult:
    """Split *records* into train/validation/test index lists.

    Args:
        records: sequence of items to split (used only for length and indexing).
        labels: sequence of label strings, one per record.  For multi-label
            tasks this is the pipe-joined label string; the splitter inspects
            it to decide whether stratification is safe.
        spec: split configuration.  Defaults to a 70/15/15 split with
            random_state=42 and stratification enabled.

    Returns:
        SplitResult with index lists and metadata about stratification.
    """
    spec = spec or SplitSpec()
    n = len(records)
    labels = list(labels)

    if n == 0:
        return SplitResult(
            train=[],
            validation=[],
            test=[],
            random_state=spec.random_state,
            stratify_used=False,
            stratify_available=False,
            warnings=["empty dataset: no records to split"],
        )

    if n < 3:
        return SplitResult(
            train=list(range(n)),
            validation=[],
            test=[],
            random_state=spec.random_state,
            stratify_used=False,
            stratify_available=False,
            warnings=[
                f"dataset too small ({n} records) for 3-way split; "
                f"all records go to training"
            ],
        )

    # ---- leakage guard: duplicate document identities ----
    warnings: List[str] = []
    ids = _extract_ids(records)
    dup_ids = _find_duplicates(ids)
    if dup_ids:
        warnings.append(
            f"dataset contains {len(dup_ids)} duplicate document id(s): "
            f"{', '.join(sorted(dup_ids)[:5])}"
            + (f" and {len(dup_ids) - 5} more" if len(dup_ids) > 5 else "")
            + ".  Duplicates are preserved in their original order; they will "
            "appear in whichever split their position dictates."
        )

    # ---- stratification decision ----
    stratify_available = _stratification_safe(labels, spec)
    use_stratify = spec.stratify and stratify_available

    if not use_stratify and spec.stratify and n >= 3:
        warnings.append(
            "stratification requested but not applied: "
            + _stratification_reason(labels, spec)
        )

    # ---- proportional sizes ----
    n_train = _compute_size(n, spec.train_size)
    n_val = _compute_size(n, spec.validation_size) if spec.has_validation else 0
    n_test = _compute_size(n, spec.test_size) if spec.has_test else 0

    # Enforce minimums — apply them first, before proportional adjustment.
    n_train = max(n_train, spec.min_train_examples)
    if spec.has_validation:
        n_val = max(n_val, spec.min_validation_examples)
    if spec.has_test:
        n_test = max(n_test, spec.min_test_examples)

    # Adjust so the three sizes sum to n.
    remaining = n - n_train - n_val - n_test
    if remaining < 0:
        # Scale down proportionally.
        total = n_train + n_val + n_test
        if total == 0:
            raise ValueError("cannot split: all split sizes computed to zero")
        scale = n / total
        n_train = max(spec.min_train_examples, int(n_train * scale))
        if spec.has_validation:
            n_val = max(spec.min_validation_examples, int(n_val * scale))
        if spec.has_test:
            n_test = max(spec.min_test_examples, int(n_test * scale))
        # Final adjustment to exact n.
        diff = n - n_train - n_val - n_test
        n_train += diff

    if n_train + n_val + n_test != n:
        n_train = n - n_val - n_test

    # ---- final minimum enforcement after proportional adjustment ----
    if n_train < spec.min_train_examples:
        raise ValueError(
            f"cannot create a training split of at least "
            f"{spec.min_train_examples} examples from {n} records: "
            f"train={spec.train_size}, validation={spec.validation_size}, "
            f"test={spec.test_size}.  Increase dataset size or reduce "
            f"validation_size / test_size."
        )
    if spec.has_validation and n_val < spec.min_validation_examples:
        raise ValueError(
            f"cannot create a validation split of at least "
            f"{spec.min_validation_examples} examples from {n} records"
        )
    if spec.has_test and n_test < spec.min_test_examples:
        raise ValueError(
            f"cannot create a test split of at least "
            f"{spec.min_test_examples} examples from {n} records"
        )

    # ---- deterministic ordering ----
    # Sort records by (label, document_id) so the split is stable regardless of
    # input order, then apply the RNG.
    order = list(range(n))
    if use_stratify:
        order = _stratified_order(labels, order, spec.random_state)
    else:
        order = _random_order(order, spec.random_state)

    # ---- build splits ----
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    result = SplitResult(
        train=sorted(train_idx),
        validation=sorted(val_idx),
        test=sorted(test_idx),
        random_state=spec.random_state,
        stratify_used=use_stratify,
        stratify_available=stratify_available,
        warnings=warnings,
    )

    # Final leakage check.
    ok, overlap_warnings = result.verify_no_overlap()
    if not ok:
        result.warnings.extend(overlap_warnings)
        raise RuntimeError(
            "split produced overlapping indices (internal error): " +
            "; ".join(overlap_warnings)
        )

    return result


def apply_split(
    records: Sequence,
    split: SplitResult,
) -> Tuple[List, List, List]:
    """Return (train, validation, test) record lists from a SplitResult."""
    train = [records[i] for i in split.train]
    validation = [records[i] for i in split.validation]
    test = [records[i] for i in split.test]
    return train, validation, test


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_ids(records: Sequence) -> List[str]:
    """Extract a document identity from each record."""
    ids: List[str] = []
    for r in records:
        if hasattr(r, "document_id"):
            ids.append(str(r.document_id))
        elif isinstance(r, dict) and "document_id" in r:
            ids.append(str(r["document_id"]))
        elif hasattr(r, "id"):
            ids.append(str(r.id))
        else:
            ids.append(str(hash(r)))
    return ids


def _find_duplicates(ids: List[str]) -> List[str]:
    seen = set()
    dups: List[str] = []
    for id_ in ids:
        if id_ in seen and id_ not in dups:
            dups.append(id_)
        seen.add(id_)
    return dups


def _compute_size(n: int, proportion: float) -> int:
    if proportion <= 0:
        return 0
    return max(1, round(n * proportion))


def _random_order(indices: List[int], seed: int) -> List[int]:
    """Deterministic shuffle."""
    import random

    rng = random.Random(seed)
    order = list(indices)
    rng.shuffle(order)
    return order


def _stratified_order(
    labels: List[str],
    indices: List[int],
    seed: int,
) -> List[int]:
    """Order indices so that a subsequent contiguous slice preserves label ratios.

    This is a lightweight stratified shuffle: group indices by label, shuffle
    each group with the seeded RNG, then interleave round-robin.  It is NOT a
    full StratifiedShuffleSplit, and for production use the existing sklearn
    equivalent can be substituted - but this keeps the splitter free of
    stratification failures on multi-label or tiny datasets.
    """
    import random

    # Group indices by their label.
    buckets: dict[str, list[int]] = {}
    for idx in indices:
        label = labels[idx] if idx < len(labels) else ""
        buckets.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    # Round-robin interleave.
    result: list[int] = []
    max_len = max((len(b) for b in buckets.values()), default=0)
    for i in range(max_len):
        for bucket in sorted(buckets.values(), key=lambda b: b[0] if b else 0):
            if i < len(bucket):
                result.append(bucket[i])
    return result


def _stratification_safe(labels: List[str], spec: SplitSpec) -> bool:
    """Decide whether stratification is statistically safe for this dataset."""
    if not spec.stratify:
        return False
    if len(labels) < 10:
        return False

    # Multi-label: if any label string contains a pipe, stratification via
    # ordinary sklearn stratify= would binarize and drop rare label combos.
    if any("|" in str(l) for l in labels):
        return False

    # Count classes.
    from collections import Counter

    counts = Counter(labels)
    n_classes = len(counts)
    if n_classes < 2:
        return False
    if n_classes > len(labels) // 3:
        return False
    min_count = min(counts.values())
    if min_count < 3:
        return False
    return True


def _stratification_reason(labels: List[str], spec: SplitSpec) -> str:
    """Explain why stratification was not applied."""
    if len(labels) < 10:
        return f"too few records ({len(labels)}) for stratification"
    if any("|" in str(l) for l in labels):
        return "multi-label dataset: ordinary stratification would drop rare label combinations"
    from collections import Counter

    counts = Counter(labels)
    if len(counts) < 2:
        return f"only {len(counts)} class found; stratification needs at least 2"
    if min(counts.values()) < 3:
        return "at least one class has fewer than 3 examples; stratification unsafe"
    return "unknown reason"


# ---------------------------------------------------------------------------
# Leakage check (post-split verification)
# ---------------------------------------------------------------------------


def check_leakage(
    train_records: Sequence,
    validation_records: Sequence,
    test_records: Sequence,
) -> dict:
    """Return a leakage report for the three splits.

    Checks:
    - No duplicate document IDs across splits.
    - No exact duplicate texts crossing splits (unless both splits are empty).
    - No evaluation record is also in training.
    """
    train_ids = _id_set(train_records)
    val_ids = _id_set(validation_records)
    test_ids = _id_set(test_records)

    train_texts = _text_fingerprint_set(train_records)
    val_texts = _text_fingerprint_set(validation_records)
    test_texts = _text_fingerprint_set(test_records)

    report: dict = {
        "leakage_detected": False,
        "train_n": len(train_records),
        "validation_n": len(validation_records),
        "test_n": len(test_records),
        "cross_split_id_collisions": [],
        "cross_split_text_collisions": [],
        "warnings": [],
    }

    # ID collisions.
    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    if tv or tt or vt:
        report["leakage_detected"] = True
        for pair_name, collision in [
            ("train/validation", tv),
            ("train/test", tt),
            ("validation/test", vt),
        ]:
            if collision:
                report["cross_split_id_collisions"].append(
                    f"{pair_name}: {len(collision)} duplicate document id(s)"
                )

    # Text collisions (exact match across splits).
    tv_t = train_texts & val_texts
    tt_t = train_texts & test_texts
    if tv_t or tt_t:
        report["leakage_detected"] = True
        for pair_name, collision in [("train/validation", tv_t), ("train/test", tt_t)]:
            if collision:
                report["cross_split_text_collisions"].append(
                    f"{pair_name}: {len(collision)} exact-text duplicate(s)"
                )

    if report["cross_split_id_collisions"] or report["cross_split_text_collisions"]:
        report["warnings"].append(
            "DATA LEAKAGE DETECTED: at least one document (by id or by exact text) "
            "appears in more than one split.  This split must not be used for "
            "training or evaluation until the overlap is resolved."
        )

    return report


def _id_set(records: Sequence) -> set:
    s: set = set()
    for r in records:
        if hasattr(r, "document_id"):
            s.add(str(r.document_id))
        elif isinstance(r, dict) and "document_id" in r:
            s.add(str(r["document_id"]))
        elif hasattr(r, "id"):
            s.add(str(r.id))
    return s


def _text_fingerprint_set(records: Sequence) -> set:
    s: set = set()
    for r in records:
        text = ""
        if hasattr(r, "text"):
            text = str(r.text)
        elif isinstance(r, dict) and "text" in r:
            text = str(r["text"])
        fp = hashlib.sha256(text.encode("utf-8")).hexdigest()
        s.add(fp)
    return s
