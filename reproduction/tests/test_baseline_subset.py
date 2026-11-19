import hashlib
import json
import unittest

from reproduction.common.ids import make_sample_id
from reproduction.common.paths import project_path
from reproduction.data.build_baseline_subset import PRIMARY_QUOTAS, analyze_html


class BaselineSubsetAnalysisTests(unittest.TestCase):
    def test_analyze_html_counts_logical_spans(self):
        html = (
            '<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
            '<tr><td>C</td><td>D</td></tr></table>'
        )
        stats = analyze_html(html)
        self.assertEqual(stats["row_count"], 2)
        self.assertEqual(stats["logical_column_count"], 3)
        self.assertEqual(stats["rowspan_cell_count"], 1)
        self.assertEqual(stats["colspan_cell_count"], 1)

    def test_analyze_html_language_and_formula_labels(self):
        stats = analyze_html("<table><tr><td>面积 Area $x^2$</td></tr></table>")
        self.assertIn("mixed_zh_en", stats["strata"])
        self.assertIn("formula", stats["strata"])


class CommittedBaselineSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subset_path = project_path("reproduction/data/splits/baseline_test.jsonl")
        cls.summary_path = project_path("reproduction/data/splits/baseline_test_summary.json")
        with cls.subset_path.open(encoding="utf-8") as subset_file:
            cls.records = [json.loads(line) for line in subset_file if line.strip()]
        with cls.summary_path.open(encoding="utf-8") as summary_file:
            cls.summary = json.load(summary_file)

    def test_subset_has_100_unique_samples_and_images(self):
        self.assertEqual(len(self.records), 100)
        self.assertEqual(len({row["sample_id"] for row in self.records}), 100)
        self.assertEqual(len({row["source_image_path"] for row in self.records}), 100)

    def test_primary_quotas_are_exact(self):
        counts = {}
        for row in self.records:
            primary = row["primary_stratum"]
            counts[primary] = counts.get(primary, 0) + 1
        self.assertEqual(counts, PRIMARY_QUOTAS)

    def test_ids_and_paths_are_reproducible(self):
        for row in self.records:
            expected_id = make_sample_id(
                "tableverse",
                row["source_index"],
                row["source_image_path"],
                row["reference_html"],
            )
            self.assertEqual(row["sample_id"], expected_id)
            self.assertTrue(project_path(row["image_path"]).is_file())

    def test_summary_hash_matches_subset(self):
        digest = hashlib.sha256(self.subset_path.read_bytes()).hexdigest()
        self.assertEqual(self.summary["subset_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
