import unittest
from unittest.mock import Mock

from reproduction.eval.score_baseline import (
    build_tracked_failure_cases,
    summarize_scores,
    validate_completed_predictions,
)
from reproduction.eval.teds_client import TedsClient, normalize_table_response


class TedsClientTests(unittest.TestCase):
    def test_normalization_matches_official_benchmark_behavior(self):
        response = "prefix\n```html\n<table><thead><tr><th>A</th></tr></thead></table>\n```"
        self.assertEqual(
            normalize_table_response(response),
            "<table><tr><td>A</td></tr></table>",
        )

    def test_empty_prediction_is_stable_zero(self):
        result = TedsClient(session=Mock()).score("", "<table></table>")
        self.assertEqual(result["teds"], 0.0)
        self.assertEqual(result["teds_s"], 0.0)
        self.assertEqual(result["diagnostic"], "empty_prediction")

    def test_valid_service_response_is_checked(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"teds": 0.75, "teds_s": 1.0}
        session = Mock()
        session.post.return_value = response
        result = TedsClient(session=session).score("<table></table>", "<table></table>")
        self.assertEqual(result["teds"], 0.75)
        self.assertEqual(result["teds_s"], 1.0)


class TedsAggregationTests(unittest.TestCase):
    def test_incomplete_predictions_are_rejected(self):
        subset = [
            {"sample_id": "a", "primary_stratum": "regular_no_merge"},
            {"sample_id": "b", "primary_stratum": "regular_no_merge"},
        ]
        with self.assertRaises(RuntimeError):
            validate_completed_predictions(subset, [{"sample_id": "a", "status": "ok"}])

    def test_mixed_attention_backends_are_rejected(self):
        subset = [{"sample_id": "a"}, {"sample_id": "b"}]
        predictions = [
            {"sample_id": "a", "status": "ok", "attention_implementation": "sdpa"},
            {"sample_id": "b", "status": "ok", "attention_implementation": "eager"},
        ]
        with self.assertRaises(ValueError):
            validate_completed_predictions(subset, predictions)

    def test_summary_uses_fixed_subset_denominator(self):
        subset = [
            {"sample_id": "a", "primary_stratum": "regular_no_merge"},
            {"sample_id": "b", "primary_stratum": "regular_no_merge"},
        ]
        scores = {
            "a": {"sample_id": "a", "teds": 1.0, "teds_s": 1.0, "error": None},
            "b": {"sample_id": "b", "teds": 0.0, "teds_s": 0.5, "error": None},
        }
        summary = summarize_scores(subset, scores)
        self.assertEqual(summary["teds"], 0.5)
        self.assertEqual(summary["teds_s"], 0.75)
        self.assertEqual(summary["pending_count"], 0)
        self.assertEqual(summary["by_primary_stratum"]["regular_no_merge"]["teds"], 0.5)

    def test_repetition_cases_are_kept_for_cross_stage_tracking(self):
        subset = [
            {
                "sample_id": "a",
                "source_index": 1,
                "image_path": "images/a.jpg",
                "primary_stratum": "large_dense",
            }
        ]
        predictions = {
            "a": {
                "output_tokens": 20,
                "diagnostics": {
                    "suspected_repetition": True,
                    "table_open_count": 2,
                    "table_close_count": 2,
                },
            }
        }
        scores = {"a": {"teds": 0.4, "teds_s": 0.5}}
        tracked = build_tracked_failure_cases(subset, predictions, scores)
        self.assertEqual(tracked[0]["sample_id"], "a")
        self.assertEqual(tracked[0]["table_open_count"], 2)


if __name__ == "__main__":
    unittest.main()
