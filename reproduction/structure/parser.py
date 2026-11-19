"""Strict HTML table parser backed by one shared occupancy grid.

The parser never rewrites the supplied HTML.  Its one grid is the source of
truth for dimensions, physical-cell coordinates, and merge coordinates.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

from bs4 import BeautifulSoup

from reproduction.common.protocol import NO_MERGE_TOKEN


_STRUCTURAL_TAGS = frozenset({"table", "thead", "tbody", "tfoot", "tr", "td", "th"})
_SECTIONS = frozenset({"thead", "tbody", "tfoot"})


class _StrictTableMarkupValidator(HTMLParser):
    """Check explicit closure and hierarchy for table-structural tags only."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.table_starts = 0
        self.table_ends = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in _STRUCTURAL_TAGS:
            return
        parent = self.stack[-1] if self.stack else None
        if tag == "table":
            self.table_starts += 1
            if "table" in self.stack:
                self.errors.append("nested_table: nested <table> is not supported")
        elif tag in _SECTIONS and parent != "table":
            self.errors.append(f"invalid_hierarchy: <{tag}> must be directly inside <table>")
        elif tag == "tr" and parent not in ({"table"} | _SECTIONS):
            self.errors.append("invalid_hierarchy: <tr> must be inside <table> or a table section")
        elif tag in {"td", "th"} and parent != "tr":
            self.errors.append(f"invalid_hierarchy: <{tag}> must be directly inside <tr>")
        self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in _STRUCTURAL_TAGS:
            self.errors.append(f"self_closing_structural_tag: <{tag}/> is not supported")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag not in _STRUCTURAL_TAGS:
            return
        if tag == "table":
            self.table_ends += 1
        if tag not in self.stack:
            self.errors.append(f"unexpected_closing_tag: </{tag}>")
            return
        while self.stack and self.stack[-1] != tag:
            missing = self.stack.pop()
            self.errors.append(f"missing_closing_tag: </{missing}> before </{tag}>")
        if self.stack:
            self.stack.pop()

    def close(self) -> None:
        super().close()
        while self.stack:
            self.errors.append(f"missing_closing_tag: </{self.stack.pop()}> at end of input")
        if self.table_starts != 1 or self.table_ends != 1:
            self.errors.append(
                "table_count: expected one explicit <table>...</table> pair, "
                f"found starts={self.table_starts}, ends={self.table_ends}"
            )


def _positive_span(cell: Any, name: str, cell_index: int, errors: list[str]) -> int:
    raw = cell.get(name, 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"invalid_{name}: cell={cell_index}, value={raw!r}")
        return 1
    if value < 1:
        errors.append(f"invalid_{name}: cell={cell_index}, value={value}; expected positive integer")
        return 1
    return value


def format_merge(start_row: int, start_col: int, end_row: int, end_col: int) -> str:
    """Format one 1-based merge rectangle using the official protocol."""
    return f"<merge>({start_row},{start_col}),({end_row},{end_col})</merge>"


def format_merge_info(merged_cells: list[dict[str, Any]]) -> str:
    """Format all merge rectangles, or the official no-merge sentence."""
    if not merged_cells:
        return NO_MERGE_TOKEN
    return "".join(cell["protocol"] for cell in merged_cells)


def _empty_result(errors: list[str]) -> dict[str, Any]:
    return {
        "row_count": 0,
        "col_count": 0,
        "cells": [],
        "merged_cells": [],
        "merge_info": NO_MERGE_TOKEN,
        "anchor_grid": [],
        "is_valid": False,
        "errors": errors,
    }


def parse_html_structure(html: str) -> dict[str, Any]:
    """Parse one complete table without canonicalizing or returning rewritten HTML.

    Coordinates are 1-based.  Invalid input returns the best available partial
    structure with ``is_valid=False`` and stable, human-readable error strings.
    """
    if not isinstance(html, str) or not html.strip():
        return _empty_result(["empty_html: expected a non-empty HTML string"])

    validator = _StrictTableMarkupValidator()
    try:
        validator.feed(html)
        validator.close()
    except Exception as exc:
        validator.errors.append(f"html_tokenization_error: {exc}")
    errors = list(validator.errors)

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) != 1:
        errors.append(f"table_count_dom: expected one table, found {len(tables)}")
        return _empty_result(list(dict.fromkeys(errors)))
    table = tables[0]
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    row_count = len(rows)
    if row_count == 0:
        errors.append("empty_table: table contains no rows")

    grid: list[list[tuple[int, int] | None]] = [[] for _ in rows]
    cells: list[dict[str, Any]] = []
    merged_cells: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        col_index = 0
        physical_cells = row.find_all(["td", "th"], recursive=False)
        for cell in physical_cells:
            cell_index = len(cells) + 1
            while col_index < len(grid[row_index]) and grid[row_index][col_index] is not None:
                col_index += 1
            rowspan = _positive_span(cell, "rowspan", cell_index, errors)
            colspan = _positive_span(cell, "colspan", cell_index, errors)
            start_row = row_index + 1
            start_col = col_index + 1
            end_row = start_row + rowspan - 1
            end_col = start_col + colspan - 1
            anchor = (row_index, col_index)

            if end_row > row_count:
                errors.append(
                    f"rowspan_out_of_bounds: cell={cell_index}, end_row={end_row}, "
                    f"row_count={row_count}"
                )

            overlap_positions: list[tuple[int, int]] = []
            for target_row in range(row_index, min(row_index + rowspan, row_count)):
                while len(grid[target_row]) < col_index + colspan:
                    grid[target_row].append(None)
                for target_col in range(col_index, col_index + colspan):
                    if grid[target_row][target_col] is not None:
                        overlap_positions.append((target_row + 1, target_col + 1))
                    else:
                        grid[target_row][target_col] = anchor
            for overlap_row, overlap_col in overlap_positions:
                errors.append(
                    f"span_overlap: cell={cell_index}, row={overlap_row}, col={overlap_col}"
                )

            text = cell.get_text(separator="", strip=False)
            cell_record = {
                "cell_index": cell_index,
                "tag": cell.name.lower(),
                "row": start_row,
                "col": start_col,
                "end_row": end_row,
                "end_col": end_col,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": text,
                "is_empty": not bool(text.strip()),
            }
            cells.append(cell_record)
            if rowspan > 1 or colspan > 1:
                merge_record = {
                    "cell_index": cell_index,
                    "start_row": start_row,
                    "start_col": start_col,
                    "end_row": end_row,
                    "end_col": end_col,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "protocol": format_merge(start_row, start_col, end_row, end_col),
                }
                merged_cells.append(merge_record)
            col_index += colspan

    col_count = max((len(row) for row in grid), default=0)
    if row_count > 0 and col_count == 0:
        errors.append("empty_table: table contains no cells")
    for row_index, row in enumerate(grid):
        occupied = sum(slot is not None for slot in row)
        if len(row) != col_count or occupied != col_count:
            errors.append(
                f"irregular_row_width: row={row_index + 1}, occupied={occupied}, expected={col_count}"
            )

    errors = list(dict.fromkeys(errors))
    anchor_grid = [
        [slot if slot is not None else (-1, -1) for slot in row]
        + [(-1, -1)] * (col_count - len(row))
        for row in grid
    ]
    return {
        "row_count": row_count,
        "col_count": col_count,
        "cells": cells,
        "merged_cells": merged_cells,
        "merge_info": format_merge_info(merged_cells),
        "anchor_grid": anchor_grid,
        "is_valid": not errors,
        "errors": errors,
    }
