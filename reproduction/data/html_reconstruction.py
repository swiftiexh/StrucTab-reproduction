"""Recover full table HTML from PubTabNet-style structure and cell tokens."""

from __future__ import annotations

import re

from reproduction.structure.parser import parse_html_structure


CELL_OPEN_RE = re.compile(r"^<t[dh](?:\s|>|$)", flags=re.IGNORECASE)
ENTITY_RE = re.compile(r"&(?!(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);)", flags=re.IGNORECASE)


def _cell_text(cell: dict) -> str:
    tokens = cell.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("each cell must contain a string-token list")
    # Dataset cell tokens are text, not table markup. Preserve entities already
    # present in the annotation while escaping raw comparison signs such as
    # ``<0.001`` or ``</=4.5`` that HTMLParser would otherwise treat as tags.
    text = "".join(tokens)
    text = ENTITY_RE.sub("&amp;", text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def reconstruct_html(structure_tokens: list[str], cells: list[dict]) -> str:
    """Insert cells in physical opening-tag order and return one full table."""
    if not isinstance(structure_tokens, list) or not all(
        isinstance(token, str) for token in structure_tokens
    ):
        raise ValueError("structure_tokens must be a list of strings")

    output: list[str] = []
    cell_index = 0
    pending_cell_tag = False
    for token in structure_tokens:
        stripped = token.lstrip().lower()
        if not pending_cell_tag and CELL_OPEN_RE.match(stripped):
            pending_cell_tag = True

        output.append(token)
        if pending_cell_tag and ">" in token:
            if cell_index >= len(cells):
                raise ValueError("structure contains more cell tags than cell annotations")
            output.append(_cell_text(cells[cell_index]))
            cell_index += 1
            pending_cell_tag = False

    if pending_cell_tag:
        raise ValueError("unterminated cell opening tag in structure tokens")
    if cell_index != len(cells):
        raise ValueError(
            f"cell alignment mismatch: structure={cell_index}, annotations={len(cells)}"
        )

    html = "".join(output).strip()
    if not html.lower().startswith("<table"):
        html = f"<table>{html}</table>"
    return html


def otsl_anchor_grid(tokens: list[str], rows: int, cols: int) -> list[list[tuple[int, int]]]:
    """Validate OTSL shape/merge directions and return each slot's anchor."""
    if rows < 1 or cols < 1:
        raise ValueError("OTSL rows and cols must be positive")
    expected = rows * (cols + 1)
    if len(tokens) != expected:
        raise ValueError(f"OTSL length mismatch: expected {expected}, found {len(tokens)}")

    grid: list[list[tuple[int, int]]] = []
    for row_index in range(rows):
        row_tokens = tokens[row_index * (cols + 1) : (row_index + 1) * (cols + 1)]
        if row_tokens[-1] != "nl":
            raise ValueError(f"OTSL row {row_index} does not end with nl")
        anchors: list[tuple[int, int]] = []
        for col_index, token in enumerate(row_tokens[:-1]):
            if token in {"fcel", "ecel"}:
                anchor = (row_index, col_index)
            elif token == "lcel":
                if col_index == 0:
                    raise ValueError("lcel cannot occur in the first column")
                anchor = anchors[col_index - 1]
            elif token == "ucel":
                if row_index == 0:
                    raise ValueError("ucel cannot occur in the first row")
                anchor = grid[row_index - 1][col_index]
            elif token == "xcel":
                if row_index == 0 or col_index == 0:
                    raise ValueError("xcel requires both an upper and a left slot")
                left = anchors[col_index - 1]
                upper = grid[row_index - 1][col_index]
                if left != upper:
                    raise ValueError("xcel upper and left anchors disagree")
                anchor = left
            else:
                raise ValueError(f"unknown OTSL token: {token!r}")
            anchors.append(anchor)
        grid.append(anchors)
    return grid


def html_anchor_grid(html: str) -> list[list[tuple[int, int]]]:
    """Return the unified parser's grid, rejecting every structural error."""
    parsed = parse_html_structure(html)
    if not parsed["is_valid"]:
        raise ValueError("invalid HTML structure: " + "; ".join(parsed["errors"]))
    return parsed["anchor_grid"]


def validate_pubtables_alignment(
    otsl: list[str], rows: int, cols: int, html: str, cell_count: int
) -> dict[str, int]:
    """Cross-check OTSL, merged HTML geometry, and physical cell alignment."""
    otsl_grid = otsl_anchor_grid(otsl, rows, cols)
    html_grid = html_anchor_grid(html)
    if len(html_grid) != rows or any(len(row) != cols for row in html_grid):
        raise ValueError("HTML row/column geometry disagrees with PubTables metadata")

    # Anchor coordinates are canonical in both grids, so equality validates the
    # complete rowspan/colspan partition rather than only comparing dimensions.
    if html_grid != otsl_grid:
        raise ValueError("HTML merged-cell geometry disagrees with OTSL")
    physical = sum(token in {"fcel", "ecel"} for token in otsl)
    if physical != cell_count:
        raise ValueError(
            f"OTSL/cell alignment mismatch: OTSL={physical}, annotations={cell_count}"
        )
    return {
        "rows": rows,
        "cols": cols,
        "physical_cells": physical,
        "merged_slots": sum(token in {"lcel", "ucel", "xcel"} for token in otsl),
        "empty_cells": otsl.count("ecel"),
    }
