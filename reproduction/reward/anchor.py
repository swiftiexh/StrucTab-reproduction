"""Local sequential Anchor OCR adapter for Uni-TabRL content reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reproduction.eval.teds_client import TedsClient
from reproduction.inference.hunyuan_baseline import DEFAULT_MODEL_PATH, HunyuanBaselineEngine
from reproduction.reward.rendering import LocalHtmlRenderer


class LocalAnchorScorer:
    """Render, reparse with frozen HunyuanOCR, then score against cached reference."""

    def __init__(
        self,
        renderer: LocalHtmlRenderer,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        teds_client: TedsClient | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        seed: int = 42,
        max_new_tokens: int = 4096,
    ):
        self.renderer = renderer
        self.model_path = model_path.resolve()
        self.teds_client = teds_client or TedsClient()
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self._engine: HunyuanBaselineEngine | None = None

    def _get_engine(self) -> HunyuanBaselineEngine:
        if self._engine is None:
            self._engine = HunyuanBaselineEngine(
                self.model_path, self.device, self.dtype, self.seed, "sdpa"
            )
        return self._engine

    def parse_reference(self, reference_html: str) -> dict[str, Any]:
        render = self.renderer.render(reference_html)
        inference = self._get_engine().infer(
            Path(render["image_path"]), "把图中的表格解析为HTML。", self.max_new_tokens
        )
        return {"render": render, "inference": inference}

    def __call__(self, pred_table: str, ref_table_destylized: str) -> dict[str, Any] | None:
        parsed = self.parse_reference(pred_table)
        pred_destylized = parsed["inference"]["pred_html"]
        if not pred_destylized:
            return None
        score = self.teds_client.score(pred_destylized, ref_table_destylized)
        if score["error"] is not None:
            return None
        return {
            "parsing_result": pred_destylized,
            "teds": score["teds"],
            "teds_s": score["teds_s"],
            "render_image": parsed["render"]["image_path"],
            "generation_seconds": parsed["inference"]["generation_seconds"],
        }
