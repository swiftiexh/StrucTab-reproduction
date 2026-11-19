import json
import tempfile
import unittest
from pathlib import Path

from reproduction.data.conversion import ConversionSession
from reproduction.data.html_reconstruction import (
    reconstruct_html,
    validate_pubtables_alignment,
)
from reproduction.data.schema import NormalizedTableSample, validate_single_table_html


def sample(**changes):
    values = {
        "sample_id": "synthtabnet-12345678",
        "source": "synthtabnet",
        "split": "train",
        "image_path": "assets/images/a.png",
        "html": "<table><tr><td>A</td></tr></table>",
        "metadata": {"source_key": "train:1"},
    }
    values.update(changes)
    return NormalizedTableSample(**values)


class SchemaTests(unittest.TestCase):
    def test_round_trip_has_exact_top_level_keys_and_schema_version(self):
        record = sample()
        restored = NormalizedTableSample.from_dict(json.loads(json.dumps(record.to_dict())))
        self.assertEqual(restored, record)
        self.assertEqual(restored.metadata["schema_version"], 1)

    def test_rejects_absolute_or_parent_image_paths(self):
        for image_path in ("/tmp/a.png", "../a.png", "assets/../a.png"):
            with self.subTest(image_path=image_path), self.assertRaises(ValueError):
                sample(image_path=image_path)

    def test_requires_exactly_one_complete_outer_table(self):
        invalid = (
            "<tr><td>A</td></tr>",
            "<table></table><table></table>",
            "prefix<table></table>",
            "<table><table></table></table>",
        )
        for html in invalid:
            with self.subTest(html=html), self.assertRaises(ValueError):
                validate_single_table_html(html)

    def test_rejects_unknown_schema_version(self):
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            sample(metadata={"source_key": "train:1", "schema_version": 2})


class ReconstructionTests(unittest.TestCase):
    def test_injects_text_after_split_colspan_opening_tag(self):
        tokens = ["<tr>", "<td", ' colspan="2"', ">", "</td>", "</tr>"]
        html = reconstruct_html(tokens, [{"tokens": ["A", "&amp;", "B"]}])
        self.assertEqual(html, '<table><tr><td colspan="2">A&amp;B</td></tr></table>')

    def test_escapes_raw_comparison_text_without_double_escaping_entities(self):
        tokens = ["<tr>", "<td>", "</td>", "</tr>"]
        html = reconstruct_html(tokens, [{"tokens": ["p<0.001 &amp; x>2"]}])
        self.assertEqual(
            html,
            "<table><tr><td>p&lt;0.001 &amp; x&gt;2</td></tr></table>",
        )

    def test_thead_is_not_mistaken_for_a_th_cell(self):
        tokens = ["<thead>", "<tr>", "<td>", "</td>", "</tr>", "</thead>"]
        html = reconstruct_html(tokens, [{"tokens": ["Header"]}])
        self.assertEqual(html, "<table><thead><tr><td>Header</td></tr></thead></table>")

    def test_rejects_cell_alignment_mismatch(self):
        with self.assertRaisesRegex(ValueError, "more cell tags"):
            reconstruct_html(["<tr>", "<td>", "</td>", "</tr>"], [])

    def test_validates_rowspan_against_otsl(self):
        html = (
            '<table><tr><td rowspan="2">A</td><td>B</td></tr>'
            "<tr><td>C</td></tr></table>"
        )
        result = validate_pubtables_alignment(
            ["fcel", "fcel", "nl", "ucel", "fcel", "nl"], 2, 2, html, 3
        )
        self.assertEqual(result["merged_slots"], 1)

    def test_validates_two_dimensional_merge(self):
        html = '<table><tr><td rowspan="2" colspan="2">A</td></tr><tr></tr></table>'
        result = validate_pubtables_alignment(
            ["fcel", "lcel", "nl", "ucel", "xcel", "nl"], 2, 2, html, 1
        )
        self.assertEqual(result["merged_slots"], 3)

    def test_rejects_html_otsl_merge_disagreement(self):
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"
        with self.assertRaisesRegex(ValueError, "geometry disagrees"):
            validate_pubtables_alignment(
                ["fcel", "lcel", "nl", "fcel", "fcel", "nl"], 2, 2, html, 3
            )


class ResumeTests(unittest.TestCase):
    def test_session_recovers_source_keys_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.jsonl"
            first = ConversionSession(output, "synthtabnet")
            first.append(sample())
            resumed = ConversionSession(output, "synthtabnet")
            self.assertTrue(resumed.has_source_key("train:1"))
            self.assertEqual(resumed.existing, 1)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resumed.append(sample())


if __name__ == "__main__":
    unittest.main()
