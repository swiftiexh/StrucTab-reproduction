"""Canonical sequential-reasoning output protocol used by StrucTab."""

from __future__ import annotations

import re
from typing import Any


ROW_COL_TAG = "【行列数分析】:"
MERGE_TAG = "【合并单元格分析】:"
FINAL_HTML_TAG = "【最终表格解析结果】:"
NO_MERGE_TOKEN = "未发现合并单元格。"
SECTION_TAGS = (ROW_COL_TAG, MERGE_TAG, FINAL_HTML_TAG)
ROW_COL_PATTERN = re.compile(r"^这是一个(\d+)行(\d+)列的表格。$")
MERGE_PATTERN = re.compile(r"<merge>\((\d+),(\d+)\),\((\d+),(\d+)\)</merge>")


def format_sequential_target(row_count: int, col_count: int, merge_info: str, html: str) -> str:
    """Build a target accepted by the official Uni-TabRL section parser."""
    if row_count < 1 or col_count < 1:
        raise ValueError("row_count and col_count must be positive")
    if not merge_info or not merge_info.strip():
        raise ValueError("merge_info must not be empty")
    if not html or not html.strip():
        raise ValueError("html must not be empty")

    return (
        f"{ROW_COL_TAG}\n这是一个{row_count}行{col_count}列的表格。\n\n"
        f"{MERGE_TAG}\n{merge_info.strip()}\n\n"
        f"{FINAL_HTML_TAG}\n{html.strip()}"
    )


def parse_sequential_output(text: str) -> dict[str, Any]:
    """Strictly parse the shared three-section protocol without changing HTML."""
    result: dict[str, Any] = {
        "row_col": None,
        "merge": None,
        "html": None,
        "row_count": None,
        "col_count": None,
        "merged_cells": [],
        "is_valid": False,
        "errors": [],
    }
    if not isinstance(text, str) or not text.strip():
        result["errors"].append("empty_output: expected a non-empty string")
        return result
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for tag in SECTION_TAGS:
        count = normalized.count(tag)
        if count != 1:
            result["errors"].append(f"section_count: {tag} expected once, found {count}")
    if result["errors"]:
        return result

    positions = [normalized.index(tag) for tag in SECTION_TAGS]
    if positions != sorted(positions):
        result["errors"].append("section_order: section markers are out of order")
        return result
    if normalized[: positions[0]].strip():
        result["errors"].append("unexpected_prefix: content appears before the first section")

    row_start = positions[0] + len(ROW_COL_TAG)
    merge_start = positions[1] + len(MERGE_TAG)
    html_start = positions[2] + len(FINAL_HTML_TAG)
    row_col = normalized[row_start : positions[1]].strip()
    merge = normalized[merge_start : positions[2]].strip()
    html = normalized[html_start:].strip()
    result.update({"row_col": row_col, "merge": merge, "html": html})

    row_match = ROW_COL_PATTERN.fullmatch(row_col)
    if row_match is None:
        result["errors"].append("row_col_format: expected '这是一个x行y列的表格。'")
    else:
        result["row_count"], result["col_count"] = map(int, row_match.groups())
        if result["row_count"] < 1 or result["col_count"] < 1:
            result["errors"].append("row_col_value: row and column counts must be positive")

    if merge == NO_MERGE_TOKEN:
        merge_matches = []
    else:
        merge_matches = list(MERGE_PATTERN.finditer(merge))
        if not merge_matches or "".join(match.group(0) for match in merge_matches) != merge:
            result["errors"].append("merge_format: invalid merge section")
    for match in merge_matches:
        start_row, start_col, end_row, end_col = map(int, match.groups())
        if min(start_row, start_col) < 1 or end_row < start_row or end_col < start_col:
            result["errors"].append(f"merge_coordinates: invalid region {match.group(0)}")
        elif start_row == end_row and start_col == end_col:
            result["errors"].append(f"merge_coordinates: 1x1 region is not a merge {match.group(0)}")
        result["merged_cells"].append(
            {
                "start_row": start_row,
                "start_col": start_col,
                "end_row": end_row,
                "end_col": end_col,
                "protocol": match.group(0),
            }
        )

    try:
        from reproduction.data.schema import validate_single_table_html

        validate_single_table_html(html)
    except ValueError as exc:
        result["errors"].append(f"html_format: {exc}")
    result["is_valid"] = not result["errors"]
    return result
