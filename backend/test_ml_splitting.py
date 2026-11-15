"""
Tests for ML data splitting utilities.

Focuses on deterministic splitting, leakage detection, stratification
behaviour, and edge cases.
"""

from __future__ import annotations

import unittest

from ml.data.dataset import (
    TrainingRecord,
    make_synthetic_records,
    split_label,
)
from ml.data.splitting import (
    SplitResult,
    SplitSpec,
    apply_split,
    check_leakage,
    extract_groups,
    split_records,
)


class TestSplittingDeterminism(unittest.TestCase):
    """Splitting is fully deterministic given the same seed."""

    def test_same_seed_same_split(self):
        records = make_synthetic_records(n=50, label_options=["x", "y", "z"])
        labels = [r.label for r in records]
        s1 = split_records(records, labels, SplitSpec(random_state=42))
        s2 = split_records(records, labels, SplitSpec(random_state=42))
        self.assertEqual(s1.train, s2.train)
        self.assertEqual(s1.validation, s2.validation)
        self.assertEqual(s1.test, s2.test)

    def test_different_seed_different_split(self):
        records = make_synthetic_records(n=50, label_options=["x", "y"])
        labels = [r.label for r in records]
        s1 = split_records(records, labels, SplitSpec(random_state=1))
        s2 = split_records(records, labels, SplitSpec(random_state=99))
        self.assertNotEqual(s1.train, s2.train)


class TestNoLeakage(unittest.TestCase):
    """A document must never appear in more than one split."""

    def test_train_val_test_disjoint(self):
        records = make_synthetic_records(n=90, label_options=["a", "b", "c"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        train, val, test = apply_split(records, split)
        ids_train = {r.document_id for r in train}
        ids_val = {r.document_id for r in val}
        ids_test = {r.document_id for r in test}
        self.assertEqual(ids_train & ids_val, set())
        self.assertEqual(ids_train & ids_test, set())
        self.assertEqual(ids_val & ids_test, set())

    def test_verify_no_overlap_succeeds(self):
        records = make_synthetic_records(n=60, label_options=["p", "q"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        ok, msg = split.verify_no_overlap()
        self.assertTrue(ok, msg)

    def test_leakage_report_clean(self):
        # Use enough label options to avoid text collisions in the
        # small template pool used by make_synthetic_records.
        records = make_synthetic_records(n=60, label_options=["x", "y", "z"])
        labels = [r.label for r in records]
        split = split_records(records, labels)
        train, val, test = apply_split(records, split)
        report = check_leakage(train, val, test)
        self.assertFalse(report["leakage_detected"])


class TestStratification(unittest.TestCase):
    """Stratification rules for single vs multi-label."""

    def test_stratification_used_for_single_label(self):
        records = make_synthetic_records(n=200, label_options=["a", "b", "c"])
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(stratify=True))
        self.assertTrue(split.stratify_used)

    def test_stratification_not_used_multilabel(self):
        records = make_synthetic_records(
            n=200, label_options=["a", "b", "c"], single_label=False
        )
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(stratify=True))
        self.assertFalse(split.stratify_used)
        self.assertTrue(any("multi-label" in w for w in split.warnings))

    def test_stratification_not_used_tiny_dataset(self):
        records = make_synthetic_records(n=5, label_options=["a", "b"])
        labels = [r.label for r in records]
        split = split_records(records, labels, SplitSpec(stratify=True))
        self.assertFalse(split.stratify_used)


class TestEdgeCases(unittest.TestCase):
    """Edge cases for the splitter."""

    def test_empty_dataset(self):
        split = split_records([], [])
        self.assertEqual(split.train, [])
        self.assertEqual(split.validation, [])
        self.assertEqual(split.test, [])

    def test_single_record(self):
        records = [TrainingRecord("d1", "x", "a")]
        split = split_records(records, ["a"])
        self.assertEqual(len(split.train), 1)
        self.assertEqual(split.validation, [])
        self.assertEqual(split.test, [])

    def test_impossible_proportions(self):
        with self.assertRaises(ValueError):
            SplitSpec(train_size=0.5, validation_size=0.3, test_size=0.3)

    def test_zero_validation_and_test(self):
        records = make_synthetic_records(n=50, label_options=["a", "b"])
        labels = [r.label for r in records]
        split = split_records(
            records,
            labels,
            SplitSpec(train_size=1.0, validation_size=0.0, test_size=0.0),
        )
        self.assertEqual(len(split.train), 50)
        self.assertEqual(split.validation, [])
        self.assertEqual(split.test, [])


class TestGroupAwareSplit(unittest.TestCase):
    """Whole groups (e.g. a post thread) must stay in a single split."""

    @staticmethod
    def _records(n: int = 100, n_groups: int = 10):
        return [
            TrainingRecord(
                document_id=f"d{i}",
                text=f"text {i}",
                label="a" if i % 2 == 0 else "b",
                post_id=f"p{i % n_groups}",
            )
            for i in range(n)
        ]

    def test_groups_never_span_splits(self):
        records = self._records()
        split = split_records(
            records,
            [r.label for r in records],
            SplitSpec(random_state=42),
            groups=extract_groups(records, "post_id"),
        )
        by_split = {
            "train": {records[i].post_id for i in split.train},
            "validation": {records[i].post_id for i in split.validation},
            "test": {records[i].post_id for i in split.test},
        }
        self.assertEqual(by_split["train"] & by_split["validation"], set())
        self.assertEqual(by_split["train"] & by_split["test"], set())
        self.assertEqual(by_split["validation"] & by_split["test"], set())
        total = len(split.train) + len(split.validation) + len(split.test)
        self.assertEqual(total, len(records))
        self.assertFalse(split.stratify_used)

    def test_group_split_is_deterministic(self):
        records = self._records()
        groups = extract_groups(records, "post_id")
        labels = [r.label for r in records]
        s1 = split_records(records, labels, SplitSpec(random_state=7), groups=groups)
        s2 = split_records(records, labels, SplitSpec(random_state=7), groups=groups)
        self.assertEqual(s1.train, s2.train)
        self.assertEqual(s1.validation, s2.validation)
        self.assertEqual(s1.test, s2.test)

    def test_missing_group_value_falls_back_to_document(self):
        records = [
            TrainingRecord(f"d{i}", "t", "a", post_id=None) for i in range(30)
        ]
        groups = extract_groups(records, "post_id")
        self.assertEqual(len(groups), 30)
        self.assertEqual(len(set(groups)), 30)

    def test_groups_length_mismatch_raises(self):
        records = self._records(n=10)
        with self.assertRaises(ValueError):
            split_records(
                records,
                [r.label for r in records],
                SplitSpec(),
                groups=["only-one"],
            )

    def test_large_groups_still_populate_requested_held_out_splits(self):
        records = self._records(n=15, n_groups=3)
        split = split_records(
            records,
            [r.label for r in records],
            SplitSpec(random_state=42),
            groups=extract_groups(records, "post_id"),
        )
        self.assertGreaterEqual(len(split.train), 1)
        self.assertGreaterEqual(len(split.validation), 1)
        self.assertGreaterEqual(len(split.test), 1)
        by_split = {
            "train": {records[i].post_id for i in split.train},
            "validation": {records[i].post_id for i in split.validation},
            "test": {records[i].post_id for i in split.test},
        }
        self.assertEqual(by_split["train"] & by_split["validation"], set())
        self.assertEqual(by_split["train"] & by_split["test"], set())
        self.assertEqual(by_split["validation"] & by_split["test"], set())

    def test_too_few_groups_for_requested_splits_raises(self):
        records = self._records(n=10, n_groups=2)
        with self.assertRaises(ValueError):
            split_records(
                records,
                [r.label for r in records],
                SplitSpec(random_state=42),
                groups=extract_groups(records, "post_id"),
            )


if __name__ == "__main__":
    unittest.main()
