import unittest
from collections import Counter

from reproduction.data.build_budgeted_pretraining_subset import (
    DEFAULT_QUOTAS,
    quota_totals,
    select_with_quotas,
)
from reproduction.data.pretraining_pool import stable_priority
from reproduction.data.schema import NormalizedTableSample


def sample(index: int, source: str = "synthtabnet") -> NormalizedTableSample:
    sample_id = f"{source}-{index:016x}"
    return NormalizedTableSample(
        sample_id=sample_id,
        source=source,
        split="train",
        image_path=f"images/{sample_id}.png",
        html="<table><tr><td>x</td></tr></table>",
        metadata={"source_key": sample_id},
    )


class BudgetQuotaTests(unittest.TestCase):
    def test_default_budget_is_320_train_and_40_validation_bases(self):
        self.assertEqual(quota_totals(DEFAULT_QUOTAS), {"train": 320, "validation": 40})
        self.assertEqual(
            sum(
                count
                for (split, source, _), count in DEFAULT_QUOTAS.items()
                if split == "validation" and source == "fintabnet"
            ),
            16,
        )

    def test_selection_skips_rejected_candidates_and_is_deterministic(self):
        key = ("train", "synthtabnet", "complex")
        records = [sample(index) for index in range(5)]
        rejected_id = min(
            records,
            key=lambda record: stable_priority(
                42, "budgeted-subset:" + ":".join(key), record.sample_id
            ),
        ).sample_id

        def evaluate(record):
            if record.sample_id == rejected_id:
                return None, {"sample_id": record.sample_id, "reason": "over_budget"}
            return {"sample": record}, None

        first, failures = select_with_quotas({key: records}, {key: 2}, 42, evaluate)
        second, _ = select_with_quotas({key: list(reversed(records))}, {key: 2}, 42, evaluate)
        self.assertEqual(
            [item["sample"].sample_id for item in first],
            [item["sample"].sample_id for item in second],
        )
        self.assertEqual(Counter(row["reason"] for row in failures), {"over_budget": 1})

    def test_selection_fails_when_a_stratum_cannot_fill(self):
        key = ("train", "fintabnet", "simple")
        with self.assertRaisesRegex(ValueError, "only 1/2"):
            select_with_quotas(
                {key: [sample(1, source="fintabnet")]},
                {key: 2},
                42,
                lambda record: ({"sample": record}, None),
            )


if __name__ == "__main__":
    unittest.main()
