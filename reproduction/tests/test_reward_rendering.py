import unittest

from reproduction.reward.rendering import build_document, fill_empty_cells, stable_render_name


class RenderHelperTests(unittest.TestCase):
    def test_empty_cells_match_official_placeholder_behavior(self):
        self.assertEqual(
            fill_empty_cells("<table><tr><td> </td><th></th></tr></table>"),
            "<table><tr><td>&nbsp;</td><th>&nbsp;</th></tr></table>",
        )

    def test_render_name_is_stable_and_content_sensitive(self):
        first = stable_render_name("<table><tr><td>A</td></tr></table>")
        self.assertEqual(first, stable_render_name("<table><tr><td>A</td></tr></table>"))
        self.assertNotEqual(first, stable_render_name("<table><tr><td>B</td></tr></table>"))

    def test_offline_document_uses_official_table_layout_without_remote_script(self):
        document = build_document("<table><tr><td>A</td></tr></table>")
        self.assertIn("border-collapse: collapse", document)
        self.assertIn("display: inline-block", document)
        self.assertNotIn("MathJax-script", document)


if __name__ == "__main__":
    unittest.main()
