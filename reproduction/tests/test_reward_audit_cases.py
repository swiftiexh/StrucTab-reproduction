import unittest

from reproduction.common.protocol import NO_MERGE_TOKEN, parse_sequential_output
from reproduction.reward.audit_cases import (
    build_audit_cases,
    remove_last_row,
    replace_first_nonempty_cell,
)
from reproduction.structure.parser import parse_html_structure


MERGED_HTML = (
    '<table><tr><td colspan="2">Header</td></tr>'
    '<tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>'
)


class RewardAuditCaseTests(unittest.TestCase):
    def test_content_mutation_preserves_structure(self):
        mutated = replace_first_nonempty_cell(MERGED_HTML)
        reference = parse_html_structure(MERGED_HTML)
        prediction = parse_html_structure(mutated)
        self.assertNotEqual(mutated, MERGED_HTML)
        self.assertEqual(prediction["anchor_grid"], reference["anchor_grid"])
        self.assertEqual(prediction["merge_info"], reference["merge_info"])

    def test_row_removal_is_valid_and_shorter(self):
        mutated = remove_last_row(MERGED_HTML)
        parsed = parse_html_structure(mutated)
        self.assertTrue(parsed["is_valid"])
        self.assertEqual(parsed["row_count"], 2)
        self.assertEqual(parsed["col_count"], 2)

    def test_four_cases_are_strictly_parseable_and_interpretable(self):
        cases = {row["case"]: row for row in build_audit_cases(MERGED_HTML)}
        self.assertEqual(
            set(cases),
            {"exact", "content_error", "structure_error", "merge_declaration_error"},
        )
        for case in cases.values():
            self.assertTrue(parse_sequential_output(case["response"])["is_valid"])
        merge_error = parse_sequential_output(cases["merge_declaration_error"]["response"])
        self.assertEqual(merge_error["merge"], NO_MERGE_TOKEN)
        self.assertEqual(merge_error["html"], MERGED_HTML)


if __name__ == "__main__":
    unittest.main()
