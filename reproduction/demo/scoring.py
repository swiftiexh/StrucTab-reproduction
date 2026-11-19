"""Reward helpers for Baseline HTML in the Phase 13 demo."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from reproduction.common.protocol import format_sequential_target
from reproduction.common.paths import project_path
from reproduction.eval.teds_client import normalize_table_response
from reproduction.reward.pipeline import RewardPipeline
from reproduction.reward.official_primitives import (
    clean_table_content,
    official_merge_correct,
    official_row_col_correct,
)
from reproduction.structure.parser import parse_html_structure


OFFICIAL_TEDS_DIR = project_path("StrucTab/code/Uni_TabRL/server/TEDS_judger")


class LocalTedsScorer:
    """Lazy in-process adapter around the exact official simple TEDS function."""

    def __init__(self) -> None:
        self._score = None

    def _get_score(self):
        if self._score is None:
            if str(OFFICIAL_TEDS_DIR) not in sys.path:
                sys.path.insert(0, str(OFFICIAL_TEDS_DIR))
            from TEDS_judger_utils import process_simple_task  # noqa: PLC0415

            self._score = process_simple_task
        return self._score

    def __call__(self, prediction: str, reference: str) -> dict[str, Any]:
        prediction = normalize_table_response(prediction)
        reference = normalize_table_response(reference)
        if not prediction:
            return {"teds": 0.0, "teds_s": 0.0, "error": None}
        if not reference:
            return {"teds": 0.0, "teds_s": 0.0, "error": "empty reference"}
        try:
            result = self._get_score()(prediction, reference)
            teds = float(result["teds"])
            teds_s = float(result["teds_s"])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (teds, teds_s)):
                raise ValueError("TEDS scores must be finite values in [0, 1]")
            return {"teds": teds, "teds_s": teds_s, "error": None}
        except Exception as exc:
            return {"teds": 0.0, "teds_s": 0.0, "error": f"{type(exc).__name__}: {exc}"}


def structure_summary(html: str) -> dict[str, Any]:
    parsed = parse_html_structure(html)
    return {
        "is_valid": parsed["is_valid"],
        "row_count": parsed["row_count"],
        "col_count": parsed["col_count"],
        "physical_cell_count": len(parsed["cells"]),
        "merge_count": len(parsed["merged_cells"]),
        "merge_info": parsed["merge_info"],
        "merged_cells": [
            {
                "start_row": cell["start_row"],
                "start_col": cell["start_col"],
                "end_row": cell["end_row"],
                "end_col": cell["end_col"],
                "protocol": cell["protocol"],
            }
            for cell in parsed["merged_cells"]
        ],
        "errors": parsed["errors"],
    }


def _cell_sequence(html: str) -> list[dict[str, Any]]:
    """Expose the exact row/tag/attribute sequence consumed by official 1D Probe."""
    soup = BeautifulSoup(clean_table_content(html), "html.parser")
    sequence: list[dict[str, Any]] = []
    for row_index, row in enumerate(soup.find_all("tr"), 1):
        for cell in row.find_all(["td", "th"]):
            attributes = {name: str(value) for name, value in sorted(cell.attrs.items())}
            attribute_text = "".join(f' {name}="{value}"' for name, value in attributes.items())
            sequence.append(
                {
                    "index": len(sequence) + 1,
                    "row": row_index,
                    "tag": cell.name,
                    "attributes": attributes,
                    "signature": f"R{row_index} · <{cell.name}{attribute_text}>",
                }
            )
    return sequence


def one_d_probe_explanation(
    prediction_html: str, reference_html: str, probe: dict[str, Any]
) -> dict[str, Any]:
    """Build a visualizable trace without changing the official probe score."""
    prediction = _cell_sequence(prediction_html)
    reference = _cell_sequence(reference_html)
    matched = int(probe["matched_cells"])
    total = int(probe["total_cells"])
    first_mismatch = matched + 1 if matched < total else None

    for item in reference:
        item["status"] = (
            "matched"
            if item["index"] <= matched
            else "first_mismatch"
            if item["index"] == first_mismatch
            else "not_scored"
        )
    for item in prediction:
        if item["index"] <= matched:
            item["status"] = "matched"
        elif item["index"] == first_mismatch:
            item["status"] = "first_mismatch"
        elif item["index"] > total:
            item["status"] = "extra_prediction"
        else:
            item["status"] = "not_scored"

    return {
        "prediction": prediction,
        "reference": reference,
        "matched_cells": matched,
        "total_cells": total,
        "prediction_cells": len(prediction),
        "first_mismatch_index": first_mismatch,
        "score": float(probe["score"]),
        "calculation": f"{matched} / {total} = {float(probe['score']):.4f}",
        "extra_prediction_cells": max(0, len(prediction) - total),
        "boundary_note": (
            "官方 1D Probe 当前不额外惩罚参考序列结束后的预测 cell。"
            if len(prediction) > total and matched == total
            else None
        ),
    }


def baseline_reward(
    prediction_html: str,
    reference_html: str,
    metrics: dict[str, Any],
    *,
    anchor_content_score: float | None = None,
) -> dict[str, Any]:
    """Build a method-level proxy reward from HTML-derived prerequisites."""
    pred = parse_html_structure(prediction_html)
    ref = parse_html_structure(reference_html)
    if not reference_html or not ref["is_valid"]:
        return {"available": False, "reason": "reference_structure_invalid"}
    if not prediction_html or pred["row_count"] < 1 or pred["col_count"] < 1:
        return {"available": False, "reason": "prediction_has_no_derivable_table_structure"}
    if metrics.get("error") is not None:
        return {"available": False, "reason": f"teds_unavailable: {metrics['error']}"}

    response = format_sequential_target(
        pred["row_count"], pred["col_count"], pred["merge_info"], prediction_html
    )
    reference = format_sequential_target(
        ref["row_count"], ref["col_count"], ref["merge_info"], reference_html
    )

    def cached_teds(_prediction: str, _reference: str) -> dict[str, Any]:
        return {
            "teds": float(metrics["teds"]),
            "teds_s": float(metrics["teds_s"]),
            "error": None,
        }

    result = RewardPipeline(
        teds_scorer=cached_teds,
        structure_variant="1d_probe",
        content_variant="teds",
    ).compute(response, reference, split="evaluation")
    weights = [float(value) for value in result["weights"]]
    content_score = (
        float(anchor_content_score)
        if anchor_content_score is not None
        else float(result["content_reward"])
    )
    component_values = [
        float(result["validity_reward"]),
        float(result["structure_reward"]),
        content_score,
    ]
    proxy_reward = sum(weight * score for weight, score in zip(weights, component_values))
    return {
        "available": result["status"] == "ok",
        "derived_from_html": True,
        "method_level_proxy": True,
        "reward": proxy_reward,
        "validity": result["validity_reward"],
        "structure": result["structure_reward"],
        "content": content_score,
        "content_variant": "anchor_destylization" if anchor_content_score is not None else "teds",
        "teds": result["base_metrics"]["teds"],
        "teds_s": result["base_metrics"]["teds_s"],
        "one_d_probe": result["one_d_probe"],
        "one_d_probe_explanation": one_d_probe_explanation(
            prediction_html, reference_html, result["one_d_probe"]
        ),
        "validity_detail": {
            "prediction": {
                "rows": pred["row_count"],
                "cols": pred["col_count"],
                "merge": pred["merge_info"],
            },
            "reference": {
                "rows": ref["row_count"],
                "cols": ref["col_count"],
                "merge": ref["merge_info"],
            },
            "row_col_match": official_row_col_correct(
                f"这是一个{pred['row_count']}行{pred['col_count']}列的表格。",
                f"这是一个{ref['row_count']}行{ref['col_count']}列的表格。",
            ),
            "merge_match": official_merge_correct(pred["merge_info"], ref["merge_info"]),
        },
        "aggregation": {
            "weights": weights,
            "contributions": [
                {"component": name, "weight": weight, "score": score, "value": weight * score}
                for name, weight, score in zip(
                    ("Validity", "Structure", "Content"), weights, component_values
                )
            ],
            "formula": " + ".join(
                f"{weight:.1f} × {score:.4f}" for weight, score in zip(weights, component_values)
            ),
        },
        "reason": result["reason"],
        "fallbacks": result["fallbacks"],
    }
