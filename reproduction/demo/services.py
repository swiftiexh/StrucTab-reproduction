"""Lazy single-GPU inference service for the demo."""

from __future__ import annotations

import asyncio
import gc
import io
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from reproduction.common.paths import PROJECT_ROOT, project_path
from reproduction.demo.catalog import sanitize_table_html
from reproduction.demo.scoring import LocalTedsScorer, baseline_reward, structure_summary
from reproduction.inference.hunyuan_baseline import (
    DEFAULT_ATTENTION_IMPLEMENTATION,
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT,
    HunyuanBaselineEngine,
)
from reproduction.reward.rendering import LocalHtmlRenderer


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
UPLOAD_DIR = project_path("reproduction/results/demo/uploads")
ANCHOR_RENDER_DIR = project_path("reproduction/results/demo/anchor_rendered")
SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


class DemoInferenceService:
    def __init__(self) -> None:
        self._engine: HunyuanBaselineEngine | None = None
        self._engine_lock = asyncio.Lock()
        self._teds = LocalTedsScorer()

    @property
    def model_loaded(self) -> bool:
        return self._engine is not None

    def _get_engine(self) -> HunyuanBaselineEngine:
        if self._engine is None:
            self._engine = HunyuanBaselineEngine(
                DEFAULT_MODEL_PATH,
                "cuda:0",
                "bfloat16",
                42,
                DEFAULT_ATTENTION_IMPLEMENTATION,
            )
        return self._engine

    def _infer_sync(self, image_path: Path, max_new_tokens: int) -> dict[str, Any]:
        return self._get_engine().infer(image_path, DEFAULT_PROMPT, max_new_tokens)

    async def infer(self, image_path: Path, max_new_tokens: int = 8192) -> dict[str, Any]:
        async with self._engine_lock:
            try:
                return await asyncio.to_thread(self._infer_sync, image_path, max_new_tokens)
            except Exception:
                if self._engine is not None:
                    self._engine = None
                    gc.collect()
                raise

    async def infer_tableverse(
        self, image_path: Path, reference_html: str, max_new_tokens: int = 8192
    ) -> dict[str, Any]:
        inference = await self.infer(image_path, max_new_tokens)
        metrics = await asyncio.to_thread(self._teds, inference["pred_html"], reference_html)
        anchor = (
            await self.anchor_destylization(
                inference["pred_html"], reference_html, min(max_new_tokens, 4096)
            )
            if inference["pred_html"]
            else {"status": "failed", "reason": "Baseline 未生成完整 HTML，无法运行 Anchor。"}
        )
        anchor_score = anchor.get("content_reward") if anchor.get("status") == "ok" else None
        reward = baseline_reward(
            inference["pred_html"],
            reference_html,
            metrics,
            anchor_content_score=anchor_score,
        )
        if anchor_score is None:
            reward.update(
                {
                    "available": False,
                    "reward": None,
                    "content": None,
                    "content_variant": "anchor_destylization_failed",
                    "reason": anchor.get("reason") or "anchor_destylization_failed",
                }
            )
        result = self._format_result(inference, reward=reward, max_new_tokens=max_new_tokens)
        result["anchor_destylization"] = anchor
        result["evaluation_metrics"] = metrics
        return result

    async def infer_upload(self, image_path: Path, max_new_tokens: int = 8192) -> dict[str, Any]:
        inference = await self.infer(image_path, max_new_tokens)
        return self._format_result(inference, reward=None, max_new_tokens=max_new_tokens)

    def _anchor_destylization_sync(
        self, prediction_html: str, reference_html: str, max_new_tokens: int
    ) -> dict[str, Any]:
        engine = self._get_engine()
        with LocalHtmlRenderer(ANCHOR_RENDER_DIR) as renderer:
            prediction_render = renderer.render(prediction_html)
            reference_render = renderer.render(reference_html)
            prediction_ocr = engine.infer(
                Path(prediction_render["image_path"]), DEFAULT_PROMPT, max_new_tokens
            )
            reference_ocr = engine.infer(
                Path(reference_render["image_path"]), DEFAULT_PROMPT, max_new_tokens
            )

        prediction_destylized = prediction_ocr["pred_html"]
        reference_destylized = reference_ocr["pred_html"]
        if not prediction_destylized or not reference_destylized:
            return {
                "status": "failed",
                "reason": "Anchor OCR 未能为预测与参考同时生成完整 HTML。",
            }
        metrics = self._teds(prediction_destylized, reference_destylized)
        if metrics["error"] is not None:
            return {"status": "failed", "reason": f"Anchor TEDS 失败：{metrics['error']}"}
        return {
            "status": "ok",
            "note": "Anchor TEDS 作为方法级代理 Reward 的 Content 分量。",
            "prediction": {
                "input_render": Path(prediction_render["image_path"]).name,
                "destylized_html": prediction_destylized,
                "safe_destylized_html": sanitize_table_html(prediction_destylized),
                "generation_seconds": prediction_ocr["generation_seconds"],
            },
            "reference": {
                "input_render": Path(reference_render["image_path"]).name,
                "destylized_html": reference_destylized,
                "safe_destylized_html": sanitize_table_html(reference_destylized),
                "generation_seconds": reference_ocr["generation_seconds"],
            },
            "content_reward": metrics["teds"],
            "teds": metrics["teds"],
            "teds_s": metrics["teds_s"],
        }

    async def anchor_destylization(
        self, prediction_html: str, reference_html: str, max_new_tokens: int = 4096
    ) -> dict[str, Any]:
        async with self._engine_lock:
            try:
                return await asyncio.to_thread(
                    self._anchor_destylization_sync,
                    prediction_html,
                    reference_html,
                    max_new_tokens,
                )
            except Exception:
                if self._engine is not None:
                    self._engine = None
                    gc.collect()
                raise

    @staticmethod
    def _format_result(
        inference: dict[str, Any],
        reward: dict[str, Any] | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        return {
            "raw_output": inference["raw_output"],
            "html": inference["pred_html"],
            "safe_html": sanitize_table_html(inference["pred_html"]),
            "structure": structure_summary(inference["pred_html"]),
            "has_complete_table": inference["has_complete_table"],
            "diagnostics": inference["diagnostics"],
            "runtime": {
                "input_tokens": inference["input_tokens"],
                "output_tokens": inference["output_tokens"],
                "max_new_tokens": max_new_tokens,
                "generation_seconds": inference["generation_seconds"],
                "model_memory_gib": inference["model_memory_gib"],
                "peak_generation_memory_gib": inference["peak_generation_memory_gib"],
            },
            "reward": reward,
        }


def persist_upload(data: bytes) -> tuple[Path, str]:
    if not data:
        raise ValueError("上传文件为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("图片超过 15 MiB 上限")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("文件不是可识别的 JPEG、PNG 或 WebP 图片") from exc
    if image_format not in SUPPORTED_FORMATS:
        raise ValueError("仅支持 JPEG、PNG 和 WebP 图片")
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("图片尺寸无效或像素数超过 5,000 万")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    path = (UPLOAD_DIR / f"{token}{SUPPORTED_FORMATS[image_format]}").resolve()
    if PROJECT_ROOT not in path.parents:
        raise ValueError("上传路径越界")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path, token


def resolve_upload(token: str) -> Path:
    try:
        normalized = uuid.UUID(hex=token).hex
    except ValueError as exc:
        raise FileNotFoundError("invalid upload token") from exc
    matches = list(UPLOAD_DIR.glob(f"{normalized}.*")) if UPLOAD_DIR.is_dir() else []
    if len(matches) != 1 or matches[0].suffix.lower() not in {".jpg", ".png", ".webp"}:
        raise FileNotFoundError("upload image not found")
    return matches[0].resolve()


def resolve_anchor_render(name: str) -> Path:
    if Path(name).name != name or not name.startswith("table_render_") or not name.endswith(".png"):
        raise FileNotFoundError("invalid anchor render name")
    path = (ANCHOR_RENDER_DIR / name).resolve()
    if ANCHOR_RENDER_DIR.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError("anchor render image not found")
    return path
