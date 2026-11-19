"""Regression tests for local fixes applied to the official data script."""

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "StrucTab" / "code" / "training_data" / "get_sequencial_reasoning.py"

SPEC = importlib.util.spec_from_file_location("get_sequencial_reasoning", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParseHtmlTableTests(unittest.TestCase):
    def test_regular_table_keeps_existing_behavior(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"

        rows, cols, parsed = MODULE.parse_html_table(html)

        self.assertEqual((rows, cols), (2, 2))
        self.assertEqual(parsed, [["A", "B"], ["C", "D"]])

    def test_colspan_contributes_to_logical_column_count(self):
        html = '<table><tr><td colspan="3">A</td><td>B</td></tr></table>'

        rows, cols, parsed = MODULE.parse_html_table(html)

        self.assertEqual((rows, cols), (1, 4))
        self.assertEqual(parsed, [["A", "B"]])

    def test_rowspan_occupancy_and_colspan_are_combined(self):
        html = (
            '<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
            "<tr><td>C</td><td>D</td></tr></table>"
        )

        rows, cols, parsed = MODULE.parse_html_table(html)

        self.assertEqual((rows, cols), (2, 3))
        self.assertEqual(parsed, [["A", "B"], ["C", "D"]])

    def test_non_positive_span_is_rejected(self):
        html = '<table><tr><td colspan="0">A</td></tr></table>'

        with self.assertRaisesRegex(ValueError, "必须为正整数"):
            MODULE.parse_html_table(html)


if __name__ == "__main__":
    unittest.main()
