import unittest

from reproduction.common.protocol import format_sequential_target
from reproduction.reward.pipeline import RewardPipeline
from reproduction.structure.parser import parse_html_structure


REF_HTML = "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>"


def target(html=REF_HTML, rows=2, cols=2, merge="未发现合并单元格。"):
    return format_sequential_target(rows, cols, merge, html)


def exact_or_simple_teds(prediction, reference):
    if prediction == reference:
        return {"teds": 1.0, "teds_s": 1.0, "error": None}
    pred = parse_html_structure(prediction)
    ref = parse_html_structure(reference)
    structure_equal = (
        pred["is_valid"] and ref["is_valid"]
        and pred["anchor_grid"] == ref["anchor_grid"]
    )
    return {"teds": 0.5, "teds_s": 1.0 if structure_equal else 0.25, "error": None}


class RewardPipelineTests(unittest.TestCase):
    def test_exact_response_gets_one_for_every_component(self):
        result = RewardPipeline(teds_scorer=exact_or_simple_teds).compute(target(), target())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["validity_reward"], 1.0)
        self.assertEqual(result["structure_reward"], 1.0)
        self.assertEqual(result["content_reward"], 1.0)

    def test_content_error_keeps_validity_and_structure_separate(self):
        pred_html = REF_HTML.replace(">D<", ">WRONG<")
        result = RewardPipeline(teds_scorer=exact_or_simple_teds).compute(
            target(pred_html), target()
        )
        self.assertEqual(result["validity_reward"], 1.0)
        self.assertEqual(result["structure_reward"], 1.0)
        self.assertEqual(result["content_reward"], 0.5)
        self.assertAlmostEqual(result["reward"], 0.85)

    def test_merge_error_zeroes_validity_and_changes_probe(self):
        pred_html = '<table><tr><td colspan="2">A</td></tr><tr><td>C</td><td>D</td></tr></table>'
        pred = target(pred_html, merge="<merge>(1,1),(1,2)</merge>")
        result = RewardPipeline(teds_scorer=exact_or_simple_teds).compute(pred, target())
        self.assertEqual(result["validity_reward"], 0.0)
        self.assertLess(result["structure_reward"], 1.0)

    def test_missing_section_is_official_zero_gate_not_exception(self):
        result = RewardPipeline(teds_scorer=exact_or_simple_teds).compute("bad", target())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["reason"], "response_missing_section_markers")

    def test_dependency_failure_is_isolated(self):
        pipeline = RewardPipeline(
            teds_scorer=lambda pred, ref: {"teds": 0.0, "teds_s": 0.0, "error": "offline"}
        )
        result = pipeline.compute(target(), target())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("teds_dependency_failed", result["reason"])

    def test_anchor_cache_is_used_or_explicitly_falls_back(self):
        cache = {"should_de_stylize": True, "de_stylize_ref": REF_HTML}
        used = RewardPipeline(
            teds_scorer=exact_or_simple_teds,
            anchor_scorer=lambda pred, ref: {"teds": 0.75, "teds_s": 1.0},
        ).compute(target(), target(), ref_destylized=cache)
        self.assertEqual(used["content_reward"], 0.75)
        self.assertTrue(used["content_detail"])

        fallback = RewardPipeline(teds_scorer=exact_or_simple_teds).compute(
            target(), target(), ref_destylized=cache
        )
        self.assertEqual(fallback["content_reward"], 1.0)
        self.assertEqual(fallback["fallbacks"], ["anchor_scorer_unavailable_use_base_teds"])

    def test_repeated_computation_is_deterministic(self):
        pipeline = RewardPipeline(teds_scorer=exact_or_simple_teds)
        first = pipeline.compute(target(), target())
        second = pipeline.compute(target(), target())
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
