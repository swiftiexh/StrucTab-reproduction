import unittest
from pathlib import Path

from reproduction.inference.hunyuan_baseline import (
    build_messages,
    diagnose_generation,
    extract_table,
)
from reproduction.inference.run_baseline_subset import (
    is_cuda_oom,
    resume_config_mismatches,
    summarize_predictions,
)


class HunyuanBaselineHelperTests(unittest.TestCase):
    def test_extract_table_preserves_complete_table(self):
        text = "prefix <table><tr><td>A</td></tr></table> suffix"
        self.assertEqual(extract_table(text), "<table><tr><td>A</td></tr></table>")

    def test_extract_table_returns_empty_for_unclosed_table(self):
        self.assertEqual(extract_table("<table><tr><td>A</td>"), "")

    def test_messages_follow_multimodal_chat_shape(self):
        messages = build_messages(Path("images/table.jpg"), "把图中的表格解析为HTML。")
        self.assertEqual(messages[0], {"role": "system", "content": ""})
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertEqual(messages[1]["content"][1]["type"], "text")

    def test_generation_diagnostics_detect_unclosed_table(self):
        diagnostics = diagnose_generation("<table><tr><td>A</td>", 20, 100)
        self.assertTrue(diagnostics["suspected_truncation"])
        self.assertFalse(diagnostics["suspected_repetition"])

    def test_generation_diagnostics_detect_repeated_table(self):
        text = "<table></table><table></table>"
        diagnostics = diagnose_generation(text, 20, 100)
        self.assertFalse(diagnostics["suspected_truncation"])
        self.assertTrue(diagnostics["suspected_repetition"])

    def test_batch_summary_counts_completed_and_failed(self):
        subset = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "c"}]
        indexed = {
            "a": {
                "sample_id": "a",
                "status": "ok",
                "generation_seconds": 2.0,
                "peak_generation_memory_gib": 2.5,
                "has_complete_table": True,
                "exact_match": True,
                "diagnostics": {
                    "suspected_truncation": False,
                    "suspected_repetition": False,
                },
            },
            "b": {"sample_id": "b", "status": "error"},
        }
        summary = summarize_predictions(subset, indexed)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["pending"], 2)
        self.assertEqual(summary["exact_match_rate"], 1.0)

    def test_cuda_oom_detection(self):
        self.assertTrue(is_cuda_oom(RuntimeError("CUDA error: out of memory")))
        self.assertFalse(is_cuda_oom(RuntimeError("ordinary inference error")))

    def test_resume_config_detects_attention_backend_change(self):
        previous = {"attention_implementation": "eager"}
        current = {"attention_implementation": "sdpa"}
        self.assertIn("attention_implementation", resume_config_mismatches(previous, current))


if __name__ == "__main__":
    unittest.main()
