import unittest

from reproduction.reward.anchor_cache import select_anchor_audit_records, selection_manifest


def record(index, complexity, tokens=500):
    return {
        "sample_id": f"pubtables-{index:016x}",
        "image_path": f"images/{index}.png",
        "html": f"<table><tr><td>{index}</td></tr></table>",
        "metadata": {"phase9": {
            "complexity": complexity, "row_count": 3, "col_count": 3,
            "merge_count": 0, "stage2_target_token_count": tokens,
        }},
    }


class AnchorAuditSelectionTests(unittest.TestCase):
    def test_selection_covers_each_complexity_and_is_deterministic(self):
        records = [
            record(index, complexity)
            for index, complexity in enumerate(["simple"] * 4 + ["merged"] * 4 + ["complex"] * 4)
        ]
        ids = {row["sample_id"] for row in records}
        first = select_anchor_audit_records(records, ids, per_complexity=2)
        second = select_anchor_audit_records(list(reversed(records)), ids, per_complexity=2)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual(
            [row["metadata"]["phase9"]["complexity"] for row in first],
            ["simple", "simple", "merged", "merged", "complex", "complex"],
        )

    def test_selection_excludes_non_rl_and_overlong_records(self):
        records = [record(i, "simple", 2000 if i == 0 else 500) for i in range(4)]
        ids = {records[0]["sample_id"], records[1]["sample_id"], records[2]["sample_id"]}
        with self.assertRaisesRegex(ValueError, "only 2/3"):
            select_anchor_audit_records(records, ids, per_complexity=3)

    def test_manifest_records_reference_hash_and_limits(self):
        row = record(1, "simple")
        manifest = selection_manifest([row], 42, 1024)[0]
        self.assertEqual(manifest["sample_id"], row["sample_id"])
        self.assertEqual(len(manifest["reference_html_sha256"]), 64)
        self.assertEqual(manifest["selection_target_token_limit"], 1024)


if __name__ == "__main__":
    unittest.main()
