import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reproduction.data.pretraining_pool import (
    assign_pool_splits,
    complexity_bucket,
    distribution,
    select_synth_candidates,
    stable_priority,
)


class SelectionTests(unittest.TestCase):
    def test_stable_priority_depends_on_seed_namespace_and_key(self):
        value = stable_priority(42, "pool", "a")
        self.assertEqual(value, stable_priority(42, "pool", "a"))
        self.assertNotEqual(value, stable_priority(43, "pool", "a"))
        self.assertNotEqual(value, stable_priority(42, "other", "a"))

    def test_synth_selection_is_seeded_uniform_hash_not_first_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            records = [
                {"imgid": i, "split": "train" if i < 8 else "val"} for i in range(10)
            ]
            source.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            selected, counts = select_synth_candidates(source, 3, seed=42)
            expected = sorted(
                range(8),
                key=lambda i: stable_priority(42, "synthtabnet-selection", f"train:{i}"),
            )[:3]
            self.assertEqual([r["source_row_index"] for r in selected], expected)
            self.assertEqual(counts, {"train": 8, "val": 2})

    def test_source_stratified_split_has_exact_counts_and_is_reproducible(self):
        def records():
            return [
                {"sample": SimpleNamespace(source=source, sample_id=f"{source}-{i}")}
                for source in ("synthtabnet", "fintabnet")
                for i in range(20)
            ]
        first, second = records(), records()
        assign_pool_splits(first, 42, 0.10)
        assign_pool_splits(second, 42, 0.10)
        self.assertEqual([r["pool_split"] for r in first], [r["pool_split"] for r in second])
        for source in ("synthtabnet", "fintabnet"):
            self.assertEqual(
                sum(r["pool_split"] == "validation" for r in first if r["sample"].source == source),
                2,
            )


class StatisticsTests(unittest.TestCase):
    def test_complexity_buckets_are_mutually_exclusive(self):
        self.assertEqual(complexity_bucket(3, 3, 9, 0), "simple")
        self.assertEqual(complexity_bucket(3, 3, 8, 1), "merged")
        self.assertEqual(complexity_bucket(20, 3, 60, 0), "complex")
        self.assertEqual(complexity_bucket(3, 3, 8, 5), "complex")

    def test_distribution_uses_nearest_rank_percentiles(self):
        result = distribution(range(1, 101))
        self.assertEqual(result["p50"], 50)
        self.assertEqual(result["p95"], 95)
        self.assertEqual(result["max"], 100)


if __name__ == "__main__":
    unittest.main()
