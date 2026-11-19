"""Read-only TableVerse catalog used by the local demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from reproduction.common.paths import PROJECT_ROOT, project_path
from reproduction.demo.scoring import structure_summary


DEFAULT_SUBSET = project_path("reproduction/data/splits/baseline_test.jsonl")
ALLOWED_TABLE_TAGS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "td", "th", "br", "sup", "sub"})
ALLOWED_CELL_ATTRIBUTES = frozenset({"rowspan", "colspan"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def sanitize_table_html(value: str) -> str:
    """Return a safe table-only fragment suitable for insertion in the demo DOM."""
    soup = BeautifulSoup(value or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return ""
    for unsafe in table.find_all(["script", "style", "iframe", "object", "embed", "svg", "math"]):
        unsafe.decompose()
    for tag in list(table.find_all(True)):
        if tag.name.lower() not in ALLOWED_TABLE_TAGS:
            tag.unwrap()
            continue
        attributes = {}
        if tag.name.lower() in {"td", "th"}:
            for name in ALLOWED_CELL_ATTRIBUTES:
                raw = tag.get(name)
                if raw is not None:
                    try:
                        parsed = int(raw)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= parsed <= 1000:
                        attributes[name] = str(parsed)
        tag.attrs = attributes
    table.attrs = {}
    return str(table)


class DemoCatalog:
    def __init__(
        self,
        subset_path: Path = DEFAULT_SUBSET,
    ) -> None:
        subset = _read_jsonl(subset_path)
        subset_ids = {row["sample_id"] for row in subset}
        if len(subset_ids) != len(subset):
            raise ValueError("baseline subset contains duplicate sample IDs")

        self._records: dict[str, dict[str, Any]] = {}
        for base in subset:
            sample_id = base["sample_id"]
            ref_structure = structure_summary(base["reference_html"])
            tags = set(base["table_stats"]["strata"])
            if ref_structure["merge_count"] > 1:
                tags.add("multiple_merges")
            if "rowspan" in tags and "colspan" in tags:
                tags.add("combined_merge")
            self._records[sample_id] = {
                "base": base,
                "reference_structure": ref_structure,
                "tags": sorted(tags),
            }

    def __len__(self) -> int:
        return len(self._records)

    def sample_ids(self) -> list[str]:
        return list(self._records)

    def has(self, sample_id: str) -> bool:
        return sample_id in self._records

    def image_path(self, sample_id: str) -> Path:
        record = self._records[sample_id]
        path = project_path(record["base"]["image_path"])
        if PROJECT_ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError(f"allowlisted sample image is missing: {sample_id}")
        return path

    def list_examples(self) -> list[dict[str, Any]]:
        rows = []
        for sample_id, record in self._records.items():
            base = record["base"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "primary_stratum": base["primary_stratum"],
                    "tags": record["tags"],
                    "image_url": f"/api/examples/{sample_id}/image",
                    "image_size": base["image_size"],
                    "reference_rows": record["reference_structure"]["row_count"],
                    "reference_cols": record["reference_structure"]["col_count"],
                    "reference_merges": record["reference_structure"]["merge_count"],
                }
            )
        return rows

    def detail(self, sample_id: str) -> dict[str, Any]:
        record = self._records[sample_id]
        base = record["base"]
        return {
            "source_type": "tableverse",
            "sample_id": sample_id,
            "primary_stratum": base["primary_stratum"],
            "tags": record["tags"],
            "image_url": f"/api/examples/{sample_id}/image",
            "image_size": base["image_size"],
            "prompt": base["prompt"],
            "prediction": None,
            "reference": {
                "html": base["reference_html"],
                "safe_html": sanitize_table_html(base["reference_html"]),
                "structure": record["reference_structure"],
            },
            "reward": None,
            "result_origin": "awaiting_live_inference",
        }


STAGE_STATUS = [
    {
        "id": "baseline",
        "label": "HunyuanOCR Baseline",
        "available": True,
        "badge": "可用",
        "description": "原始 HunyuanOCR v1.0；对示例与上传图片执行实时推理。",
    },
    {
        "id": "stage1",
        "label": "Stage 1",
        "available": False,
        "badge": "未训练 · 硬件阻塞",
        "description": "设计目标：分别学习行列计数、合并单元格分析和 HTML 解析。",
    },
    {
        "id": "stage2",
        "label": "Stage 2",
        "available": False,
        "badge": "未训练 · 硬件阻塞",
        "description": "设计目标：按固定顺序生成三段 Sequential Structural Reasoning。",
    },
    {
        "id": "sft",
        "label": "SFT",
        "available": False,
        "badge": "仅完成数据构造",
        "description": "已构造 3,200 条高质量 SFT 数据，但没有正式 checkpoint。",
    },
    {
        "id": "rl",
        "label": "RL",
        "available": False,
        "badge": "Reward 已完成 · RL 未训练",
        "description": "Uni-TabRL Reward Pipeline 已验证；RL optimization 判定为 No-Go。",
    },
]
