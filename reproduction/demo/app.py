"""Single-page local StructTab table-recognition demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reproduction.common.paths import project_path
from reproduction.demo.catalog import DemoCatalog, STAGE_STATUS
from reproduction.demo.services import (
    DemoInferenceService,
    persist_upload,
    resolve_anchor_render,
    resolve_upload,
)


STATIC_DIR = project_path("reproduction/demo/static")


def create_app(
    catalog: DemoCatalog | None = None,
    inference: DemoInferenceService | None = None,
) -> FastAPI:
    catalog = catalog or DemoCatalog()
    inference = inference or DemoInferenceService()
    app = FastAPI(
        title="StructTab Table Recognition Demo",
        description="Local method-level demo; only HunyuanOCR Baseline has a trained checkpoint.",
    )
    app.state.catalog = catalog
    app.state.inference = inference
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "examples": len(catalog),
            "model_loaded": inference.model_loaded,
            "gpu_busy": inference._engine_lock.locked(),
        }

    @app.get("/api/config")
    def config() -> dict:
        return {
            "stages": STAGE_STATUS,
            "default_stage": "baseline",
            "reward_note": (
                "方法级代理 Reward：Baseline 只生成 HTML；Validity 输入由 HTML 派生，"
                "Structure 真实执行 1D Probe，Content 强制 should_de_stylize=True 并真实执行 "
                "Anchor Destylization。该分数展示 Uni-TabRL 设计，不代表已完成 RL 训练。"
            ),
            "upload_reward_note": "自定义图片没有参考标注，无法计算 Reward。",
        }

    @app.get("/api/examples")
    def examples() -> dict:
        return {"count": len(catalog), "examples": catalog.list_examples()}

    @app.get("/api/examples/{sample_id}")
    def example_detail(sample_id: str) -> dict:
        if not catalog.has(sample_id):
            raise HTTPException(status_code=404, detail="TableVerse 示例不存在")
        return catalog.detail(sample_id)

    @app.get("/api/examples/{sample_id}/image")
    def example_image(sample_id: str) -> FileResponse:
        if not catalog.has(sample_id):
            raise HTTPException(status_code=404, detail="TableVerse 示例不存在")
        return FileResponse(catalog.image_path(sample_id))

    @app.post("/api/examples/{sample_id}/infer")
    async def rerun_example(
        sample_id: str,
        max_new_tokens: int = Query(default=8192, ge=64, le=8192),
    ) -> dict:
        if not catalog.has(sample_id):
            raise HTTPException(status_code=404, detail="TableVerse 示例不存在")
        detail = catalog.detail(sample_id)
        try:
            result = await inference.infer_tableverse(
                catalog.image_path(sample_id), detail["reference"]["html"], max_new_tokens
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Baseline 实时推理失败：{type(exc).__name__}: {exc}",
            ) from exc
        detail["prediction"] = {key: value for key, value in result.items() if key != "reward"}
        detail["reward"] = result["reward"]
        detail["result_origin"] = "live_baseline_rerun"
        anchor = detail["prediction"].get("anchor_destylization")
        if anchor and anchor.get("status") == "ok":
            for side in ("prediction", "reference"):
                name = anchor[side]["input_render"]
                anchor[side]["input_render_url"] = f"/api/anchor-renders/{name}"
                del anchor[side]["input_render"]
        return detail

    @app.get("/api/anchor-renders/{name}")
    def anchor_render_image(name: str) -> FileResponse:
        try:
            return FileResponse(resolve_anchor_render(name))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Anchor 渲染图片不存在") from exc

    @app.post("/api/infer-upload")
    async def infer_upload(
        request: Request,
        max_new_tokens: int = Query(default=8192, ge=64, le=8192),
    ) -> dict:
        try:
            path, token = persist_upload(await request.body())
            result = await inference.infer_upload(path, max_new_tokens)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Baseline 实时推理失败：{type(exc).__name__}: {exc}",
            ) from exc
        return {
            "source_type": "upload",
            "sample_id": None,
            "primary_stratum": None,
            "tags": [],
            "image_url": f"/api/uploads/{token}/image",
            "prediction": result,
            "reference": None,
            "reward": None,
            "result_origin": "live_upload",
        }

    @app.get("/api/uploads/{token}/image")
    def upload_image(token: str) -> FileResponse:
        try:
            return FileResponse(resolve_upload(token))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="上传图片不存在") from exc

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
