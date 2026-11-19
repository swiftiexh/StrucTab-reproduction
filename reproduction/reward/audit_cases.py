"""Deterministic response mutations for the Phase 10 reward audit."""

from __future__ import annotations

import re

from reproduction.common.protocol import NO_MERGE_TOKEN, format_sequential_target
from reproduction.structure.parser import parse_html_structure


CELL_RE = re.compile(r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.IGNORECASE | re.DOTALL)
ROW_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)


def target_from_html(html: str, *, merge_info: str | None = None) -> str:
    parsed = parse_html_structure(html)
    if not parsed["is_valid"]:
        raise ValueError("audit HTML is invalid: " + "; ".join(parsed["errors"]))
    return format_sequential_target(
        parsed["row_count"],
        parsed["col_count"],
        parsed["merge_info"] if merge_info is None else merge_info,
        html,
    )


def replace_first_nonempty_cell(html: str, replacement: str = "CONTENT_ERROR") -> str:
    if not replacement or "<" in replacement or ">" in replacement:
        raise ValueError("replacement must be non-empty plain text")
    replaced = False

    def replace(match: re.Match[str]) -> str:
        nonlocal replaced
        if replaced or not re.sub(r"<[^>]+>", "", match.group(2)).strip():
            return match.group(0)
        replaced = True
        return match.group(1) + replacement + match.group(3)

    result = CELL_RE.sub(replace, html)
    if not replaced:
        raise ValueError("HTML contains no non-empty cell")
    return result


def remove_last_row(html: str) -> str:
    matches = list(ROW_RE.finditer(html))
    if len(matches) < 2:
        raise ValueError("at least two rows are required for the structure mutation")
    last = matches[-1]
    result = html[: last.start()] + html[last.end() :]
    parsed = parse_html_structure(result)
    if not parsed["is_valid"]:
        raise ValueError("removing the last row produced invalid HTML: " + "; ".join(parsed["errors"]))
    return result


def build_audit_cases(reference_html: str) -> list[dict[str, str]]:
    parsed = parse_html_structure(reference_html)
    if not parsed["is_valid"]:
        raise ValueError("reference HTML is invalid: " + "; ".join(parsed["errors"]))
    if not parsed["merged_cells"]:
        raise ValueError("reward audit reference must contain at least one merge")

    reference = target_from_html(reference_html)
    content_html = replace_first_nonempty_cell(reference_html)
    shorter_html = remove_last_row(reference_html)
    return [
        {
            "case": "exact",
            "description": "所有三段输出及 HTML 均与 reference 一致",
            "response": reference,
            "reference": reference,
        },
        {
            "case": "content_error",
            "description": "仅替换首个非空单元格文本，结构与前置分析保持正确",
            "response": target_from_html(content_html),
            "reference": reference,
        },
        {
            "case": "structure_error",
            "description": "删除最后一行并同步预测行数，制造真实的 HTML 结构差异",
            "response": target_from_html(shorter_html),
            "reference": reference,
        },
        {
            "case": "merge_declaration_error",
            "description": "HTML 保持正确，但合并单元格分析错误地声明无 merge",
            "response": target_from_html(reference_html, merge_info=NO_MERGE_TOKEN),
            "reference": reference,
        },
    ]

