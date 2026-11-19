import unittest

from reproduction.common.protocol import NO_MERGE_TOKEN
from reproduction.structure.parser import parse_html_structure


class ValidStructureTests(unittest.TestCase):
    def test_regular_table_uses_one_grid_for_dimensions_and_coordinates(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        result = parse_html_structure(html)
        self.assertTrue(result["is_valid"], result["errors"])
        self.assertEqual((result["row_count"], result["col_count"]), (2, 2))
        self.assertEqual([(cell["row"], cell["col"]) for cell in result["cells"]], [(1, 1), (1, 2), (2, 1), (2, 2)])
        self.assertEqual(result["merge_info"], NO_MERGE_TOKEN)

    def test_rowspan_and_colspan_have_official_one_based_coordinates(self):
        html = (
            '<table><tr><th rowspan="2">A</th><td colspan="2">B</td></tr>'
            "<tr><td>C</td><td>D</td></tr></table>"
        )
        result = parse_html_structure(html)
        self.assertTrue(result["is_valid"], result["errors"])
        self.assertEqual((result["row_count"], result["col_count"]), (2, 3))
        self.assertEqual(
            result["merge_info"],
            "<merge>(1,1),(2,1)</merge><merge>(1,2),(1,3)</merge>",
        )
        self.assertEqual((result["cells"][2]["row"], result["cells"][2]["col"]), (2, 2))

    def test_two_dimensional_merge_and_empty_second_row(self):
        html = '<table><tr><td rowspan="2" colspan="2"></td></tr><tr></tr></table>'
        result = parse_html_structure(html)
        self.assertTrue(result["is_valid"], result["errors"])
        self.assertTrue(result["cells"][0]["is_empty"])
        self.assertEqual(result["anchor_grid"], [[(0, 0), (0, 0)], [(0, 0), (0, 0)]])
        self.assertEqual(result["merge_info"], "<merge>(1,1),(2,2)</merge>")

    def test_interleaved_merges_skip_rowspan_occupancy(self):
        html = (
            '<table><tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
            '<tr><td>D</td><td>E</td></tr><tr><td colspan="3">F</td></tr></table>'
        )
        result = parse_html_structure(html)
        self.assertTrue(result["is_valid"], result["errors"])
        self.assertEqual((result["row_count"], result["col_count"]), (3, 3))
        self.assertEqual((result["cells"][3]["row"], result["cells"][3]["col"]), (2, 1))
        self.assertEqual((result["cells"][4]["row"], result["cells"][4]["col"]), (2, 3))

    def test_td_th_empty_content_and_inline_formula_are_parsed_without_rewriting_input(self):
        html = "<table><tr><th></th><td>  x <math><mi>y</mi></math>  </td></tr></table>"
        result = parse_html_structure(html)
        self.assertTrue(result["is_valid"], result["errors"])
        self.assertEqual([cell["tag"] for cell in result["cells"]], ["th", "td"])
        self.assertEqual(result["cells"][0]["text"], "")
        self.assertEqual(result["cells"][1]["text"].strip(), "x y")
        self.assertIn("<math><mi>y</mi></math>", html)


class InvalidStructureTests(unittest.TestCase):
    def assert_has_error(self, html: str, error_code: str):
        result = parse_html_structure(html)
        self.assertFalse(result["is_valid"])
        self.assertTrue(any(error.startswith(error_code + ":") for error in result["errors"]), result["errors"])
        return result

    def test_non_integer_and_non_positive_spans_are_rejected(self):
        self.assert_has_error('<table><tr><td colspan="x">A</td></tr></table>', "invalid_colspan")
        self.assert_has_error('<table><tr><td rowspan="0">A</td></tr></table>', "invalid_rowspan")

    def test_span_overlap_is_rejected(self):
        html = (
            '<table><tr><td>A</td><td rowspan="2">B</td></tr>'
            '<tr><td colspan="2">C</td></tr></table>'
        )
        self.assert_has_error(html, "span_overlap")

    def test_irregular_row_width_is_rejected(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        result = self.assert_has_error(html, "irregular_row_width")
        self.assertEqual((result["row_count"], result["col_count"]), (2, 2))

    def test_rowspan_cannot_extend_beyond_declared_rows(self):
        self.assert_has_error(
            '<table><tr><td rowspan="2">A</td></tr></table>', "rowspan_out_of_bounds"
        )

    def test_missing_cell_closing_tag_is_not_silently_repaired(self):
        html = "<table><tr><td>A</tr></table>"
        self.assert_has_error(html, "missing_closing_tag")

    def test_missing_table_closing_tag_is_rejected(self):
        html = "<table><tr><td>A</td></tr>"
        self.assert_has_error(html, "missing_closing_tag")

    def test_nested_or_multiple_tables_are_rejected(self):
        self.assert_has_error("<table><tr><td><table></table></td></tr></table>", "nested_table")
        self.assert_has_error("<table></table><table></table>", "table_count")

    def test_cell_outside_row_is_rejected(self):
        self.assert_has_error("<table><td>A</td></table>", "invalid_hierarchy")


if __name__ == "__main__":
    unittest.main()
