import unittest
from collections import Counter
from pathlib import Path

from reproduction.data.build_pubtables_high_quality_pool import _portable_path
from reproduction.data.pubtables_quality import (
    assign_usage,
    complexity_bucket,
    proportional_quotas,
    quality_reasons,
    select_stratified,
)
from reproduction.data.schema import NormalizedTableSample
from reproduction.structure.parser import parse_html_structure


def sample(index: int, html: str = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"):
    parsed = parse_html_structure(html)
    return NormalizedTableSample(
        sample_id=f"pubtables-{index:016x}",
        source="pubtables",
        split="train",
        image_path=f"images/{index}.jpg",
        html=html,
        metadata={
            "source_key": f"shard:{index}",
            "rows": parsed["row_count"],
            "cols": parsed["col_count"],
            "cell_annotations": [{} for _ in parsed["cells"]],
        },
    )


class QualityRuleTests(unittest.TestCase):
    def test_custom_output_path_outside_project_remains_serializable(self):
        self.assertEqual(_portable_path(Path("/tmp/phase9-output.jsonl")), "/tmp/phase9-output.jsonl")

    def test_quality_accepts_consistent_two_by_two_table(self):
        record = sample(1)
        self.assertEqual(quality_reasons(record, parse_html_structure(record.html)), [])

    def test_quality_rejects_single_row_and_metadata_conflict(self):
        record = sample(2, "<table><tr><td>a</td><td>b</td></tr></table>")
        metadata = dict(record.metadata)
        metadata["rows"] = 2
        conflicting = NormalizedTableSample(
            sample_id=record.sample_id, source=record.source, split=record.split,
            image_path=record.image_path, html=record.html, metadata=metadata,
        )
        self.assertEqual(
            quality_reasons(conflicting, parse_html_structure(conflicting.html)),
            ["too_few_rows", "metadata_row_mismatch"],
        )

    def test_complexity_buckets_are_exclusive(self):
        self.assertEqual(complexity_bucket(4, 4, 16, 0), "simple")
        self.assertEqual(complexity_bucket(4, 4, 15, 1), "merged")
        self.assertEqual(complexity_bucket(20, 4, 80, 0), "complex")


class StratificationTests(unittest.TestCase):
    def records(self):
        rows = []
        index = 0
        for bucket, count in (("simple", 5), ("merged", 3), ("complex", 2)):
            for _ in range(count):
                rows.append({"sample": sample(index), "complexity": bucket})
                index += 1
        return rows

    def test_largest_remainder_has_exact_total(self):
        self.assertEqual(
            proportional_quotas({"simple": 5, "merged": 3, "complex": 2}, 7),
            {"simple": 4, "merged": 2, "complex": 1},
        )

    def test_selection_and_usage_are_deterministic_and_disjoint(self):
        first = select_stratified(self.records(), 10, 42, "test")
        second = select_stratified(list(reversed(self.records())), 10, 42, "test")
        assign_usage(first, 2, 42)
        assign_usage(second, 2, 42)
        self.assertEqual(
            [(row["sample"].sample_id, row["usage"]) for row in first],
            [(row["sample"].sample_id, row["usage"]) for row in second],
        )
        self.assertEqual(Counter(row["usage"] for row in first), {"sft": 8, "rl": 2})
        sft = {row["sample"].sample_id for row in first if row["usage"] == "sft"}
        rl = {row["sample"].sample_id for row in first if row["usage"] == "rl"}
        self.assertTrue(sft.isdisjoint(rl))


if __name__ == "__main__":
    unittest.main()
