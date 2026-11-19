import asyncio
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image
from fastapi import HTTPException

from reproduction.demo.app import create_app
from reproduction.demo.catalog import DemoCatalog, sanitize_table_html
from reproduction.demo.scoring import LocalTedsScorer, baseline_reward, one_d_probe_explanation
from reproduction.demo.services import DemoInferenceService, persist_upload, resolve_upload
from reproduction.common.paths import PROJECT_ROOT


TABLE = "<table><tr><td>A</td><td>B</td></tr></table>"


class DemoCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = DemoCatalog()

    def test_fixed_catalog_covers_all_phase1_examples(self):
        rows = self.catalog.list_examples()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["sample_id"] for row in rows}), 100)
        self.assertTrue(all("is_repetition_case" not in row for row in rows))
        self.assertTrue(all("repetition_case" not in row["tags"] for row in rows))

    def test_catalog_exposes_overlapping_filter_tags_without_cached_results(self):
        rows = self.catalog.list_examples()
        all_tags = {tag for row in rows for tag in row["tags"]}
        self.assertTrue(
            {"regular_no_merge", "rowspan", "colspan", "combined_merge", "multiple_merges"}
            <= all_tags
        )
        self.assertTrue(all("reward" not in row and "teds" not in row for row in rows))

    def test_detail_uses_allowlisted_image_and_read_only_sample_id(self):
        sample_id = self.catalog.sample_ids()[0]
        detail = self.catalog.detail(sample_id)
        self.assertEqual(detail["sample_id"], sample_id)
        self.assertEqual(detail["result_origin"], "awaiting_live_inference")
        self.assertIsNone(detail["prediction"])
        self.assertIsNone(detail["reward"])
        self.assertTrue(self.catalog.image_path(sample_id).is_file())


class DemoSafetyTests(unittest.TestCase):
    def test_html_sanitizer_removes_scripts_events_and_unsafe_attributes(self):
        unsafe = (
            '<table onclick="bad()"><tr><td rowspan="2" style="color:red">A'
            '<script>alert(1)</script></td><td><img src=x onerror=bad()>B</td></tr></table>'
        )
        safe = sanitize_table_html(unsafe)
        self.assertNotIn("script", safe)
        self.assertNotIn("onclick", safe)
        self.assertNotIn("style", safe)
        self.assertNotIn("img", safe)
        self.assertIn('rowspan="2"', safe)

    def test_upload_validation_and_allowlisted_resolution(self):
        data = io.BytesIO()
        Image.new("RGB", (8, 6), "white").save(data, format="PNG")
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            upload_dir = Path(directory)
            with patch("reproduction.demo.services.UPLOAD_DIR", upload_dir):
                path, token = persist_upload(data.getvalue())
                self.assertEqual(resolve_upload(token), path)
                with self.assertRaises(FileNotFoundError):
                    resolve_upload("not-a-token")

    def test_reward_components_are_derived_from_html(self):
        result = baseline_reward(TABLE, TABLE, {"teds": 1.0, "teds_s": 1.0, "error": None})
        self.assertTrue(result["available"])
        self.assertTrue(result["derived_from_html"])
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["validity_detail"]["row_col_match"])
        self.assertTrue(result["validity_detail"]["merge_match"])
        self.assertEqual(result["one_d_probe_explanation"]["calculation"], "2 / 2 = 1.0000")
        self.assertAlmostEqual(
            sum(row["value"] for row in result["aggregation"]["contributions"]), 1.0
        )

    def test_anchor_score_replaces_raw_teds_in_proxy_reward(self):
        result = baseline_reward(
            TABLE,
            TABLE,
            {"teds": 0.9, "teds_s": 1.0, "error": None},
            anchor_content_score=0.5,
        )
        self.assertEqual(result["content_variant"], "anchor_destylization")
        self.assertEqual(result["content"], 0.5)
        self.assertAlmostEqual(result["reward"], 0.85)
        self.assertEqual(result["teds"], 0.9)

    def test_live_reward_uses_local_official_teds_scorer(self):
        metrics = LocalTedsScorer()(TABLE, TABLE)
        self.assertIsNone(metrics["error"])
        self.assertEqual(metrics["teds"], 1.0)
        self.assertEqual(metrics["teds_s"], 1.0)

    def test_probe_explanation_marks_first_error_and_unscored_suffix(self):
        prediction = "<table><tr><td></td><th></th><td></td></tr></table>"
        reference = "<table><tr><td></td><td></td><td></td></tr></table>"
        explanation = one_d_probe_explanation(
            prediction,
            reference,
            {"matched_cells": 1, "total_cells": 3, "score": 1 / 3},
        )
        self.assertEqual(explanation["first_mismatch_index"], 2)
        self.assertEqual(
            [item["status"] for item in explanation["prediction"]],
            ["matched", "first_mismatch", "not_scored"],
        )

    def test_probe_explanation_discloses_unpenalized_prediction_suffix(self):
        prediction = "<table><tr><td></td><td></td><td></td></tr></table>"
        reference = "<table><tr><td></td><td></td></tr></table>"
        explanation = one_d_probe_explanation(
            prediction,
            reference,
            {"matched_cells": 2, "total_cells": 2, "score": 1.0},
        )
        self.assertEqual(explanation["extra_prediction_cells"], 1)
        self.assertIsNotNone(explanation["boundary_note"])

    def test_anchor_demo_exposes_input_images_and_destylized_html(self):
        service = DemoInferenceService()

        class FakeRenderer:
            def __init__(self, _output_dir):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def render(self, html):
                side = "prediction" if "A" in html else "reference"
                return {"image_path": f"/tmp/table_render_{side}.png"}

        class FakeEngine:
            def infer(self, path, _prompt, _max_new_tokens):
                value = "A" if "prediction" in path.name else "R"
                return {"pred_html": TABLE.replace("A", value), "generation_seconds": 0.1}

        service._teds = lambda _prediction, _reference: {
            "teds": 0.75,
            "teds_s": 1.0,
            "error": None,
        }
        with (
            patch("reproduction.demo.services.LocalHtmlRenderer", FakeRenderer),
            patch.object(service, "_get_engine", return_value=FakeEngine()),
        ):
            result = service._anchor_destylization_sync(
                TABLE, TABLE.replace("A", "R"), max_new_tokens=128
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content_reward"], 0.75)
        self.assertIn("destylized_html", result["prediction"])
        self.assertEqual(result["reference"]["input_render"], "table_render_reference.png")

    def test_tableverse_inference_runs_anchor_as_content_component(self):
        service = DemoInferenceService()
        inference = {
            "raw_output": TABLE,
            "pred_html": TABLE,
            "has_complete_table": True,
            "diagnostics": {"suspected_truncation": False},
            "input_tokens": 10,
            "output_tokens": 20,
            "generation_seconds": 1.0,
            "model_memory_gib": 1.0,
            "peak_generation_memory_gib": 2.0,
        }
        anchor = {
            "status": "ok",
            "content_reward": 0.75,
            "prediction": {},
            "reference": {},
        }
        service.infer = AsyncMock(return_value=inference)
        service.anchor_destylization = AsyncMock(return_value=anchor)
        service._teds = lambda _prediction, _reference: {
            "teds": 0.9,
            "teds_s": 1.0,
            "error": None,
        }
        async def immediate_to_thread(function, *args):
            return function(*args)

        with patch("reproduction.demo.services.asyncio.to_thread", immediate_to_thread):
            result = asyncio.run(service.infer_tableverse(Path("unused.png"), TABLE, 128))
        self.assertEqual(result["reward"]["content"], 0.75)
        self.assertEqual(result["reward"]["content_variant"], "anchor_destylization")
        self.assertEqual(result["evaluation_metrics"]["teds"], 0.9)


class _FakeCatalog:
    def __len__(self):
        return 1

    def list_examples(self):
        return [{"sample_id": "demo-one", "tags": [], "image_url": "/api/examples/demo-one/image"}]

    def has(self, sample_id):
        return sample_id == "demo-one"

    def detail(self, sample_id):
        return {"sample_id": sample_id, "reference": {"html": TABLE}}

    def image_path(self, sample_id):
        raise FileNotFoundError(sample_id)


class _FakeInference:
    def __init__(self):
        self.model_loaded = False
        self._engine_lock = asyncio.Lock()

    async def infer_upload(self, image_path, max_new_tokens):
        return {
            "raw_output": TABLE,
            "html": TABLE,
            "safe_html": TABLE,
            "structure": {"is_valid": True},
            "has_complete_table": True,
            "diagnostics": {},
            "runtime": {"max_new_tokens": max_new_tokens},
        }


class DemoApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(_FakeCatalog(), _FakeInference())
        self.routes = {
            route.path: route
            for route in self.app.routes
            if getattr(route, "path", "").startswith("/api/")
        }

    def test_config_and_examples_keep_only_baseline_available(self):
        config = self.routes["/api/config"].endpoint()
        self.assertEqual([row["id"] for row in config["stages"] if row["available"]], ["baseline"])
        self.assertEqual(self.routes["/api/examples"].endpoint()["count"], 1)

    def test_unknown_example_is_404(self):
        with self.assertRaises(HTTPException) as raised:
            self.routes["/api/examples/{sample_id}"].endpoint("missing")
        self.assertEqual(raised.exception.status_code, 404)

    def test_raw_image_upload_route_does_not_declare_multipart_form(self):
        route = self.routes["/api/infer-upload"]
        self.assertIn("POST", route.methods)
        self.assertEqual([field.name for field in route.dependant.body_params], [])

if __name__ == "__main__":
    unittest.main()
