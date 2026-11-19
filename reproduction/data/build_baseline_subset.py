"""Build the deterministic, stratified TableVerse baseline subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.ids import make_sample_id
from reproduction.common.paths import PROJECT_ROOT, project_path
from reproduction.common.reproducibility import DEFAULT_SEED


DEFAULT_SOURCE = project_path("assets/datasets/TableVerse-5K/TableVerse_5K.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/data/splits/baseline_test.jsonl")
DEFAULT_SUMMARY = project_path("reproduction/data/splits/baseline_test_summary.json")
BASELINE_PROMPT = "把图中的表格解析为HTML。"

# These source rows were visually reviewed from contact sheets. The annotation
# means a captured/scanned real-world document, not an inferred camera label.
REVIEWED_CAPTURE_INDICES = (1501, 1504, 1508, 1511, 1522, 1524, 1529, 1539, 1550, 1557)

PRIMARY_QUOTAS = {
    "reviewed_capture": 10,
    "rowspan": 15,
    "colspan_only": 15,
    "large_dense": 15,
    "formula": 10,
    "chinese": 10,
    "english": 10,
    "mixed_zh_en": 5,
    "regular_no_merge": 10,
}

# Fixed near the empirical 80th percentiles of TableVerse-5K. Keeping explicit
# values makes future rebuilds stable even if exploratory code changes.
LARGE_ROW_THRESHOLD = 21
LARGE_COLUMN_THRESHOLD = 9
DENSE_CELL_THRESHOLD = 130

FORMULA_RE = re.compile(
    r"(?:<math\b|\$[^$\n]+\$|\\\(|\\\[|\\(?:frac|sum|int|sqrt|begin|alpha|beta|gamma)\b|"
    r"[∑∫√±×÷≈≠≤≥∞])",
    flags=re.IGNORECASE,
)
CJK_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _positive_span(cell, name: str) -> int:
    raw = cell.get(name, 1)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive: {value}")
    return value


def analyze_html(html: str) -> dict:
    """Extract structural and content strata from one reference table."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise ValueError("reference does not contain a table")

    rows = table.find_all("tr")
    grid: list[list[object | None]] = []
    physical_cells = 0
    rowspan_cells = 0
    colspan_cells = 0

    for row_index, row in enumerate(rows):
        while len(grid) <= row_index:
            grid.append([])
        column_index = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            physical_cells += 1
            while column_index < len(grid[row_index]) and grid[row_index][column_index] is not None:
                column_index += 1
            rowspan = _positive_span(cell, "rowspan")
            colspan = _positive_span(cell, "colspan")
            rowspan_cells += rowspan > 1
            colspan_cells += colspan > 1
            for target_row in range(row_index, row_index + rowspan):
                while len(grid) <= target_row:
                    grid.append([])
                while len(grid[target_row]) < column_index + colspan:
                    grid[target_row].append(None)
                for target_column in range(column_index, column_index + colspan):
                    if grid[target_row][target_column] is not None:
                        raise ValueError(
                            f"overlapping span at row={target_row + 1}, col={target_column + 1}"
                        )
                    grid[target_row][target_column] = (row_index, column_index)
            column_index += colspan

    text = table.get_text(" ", strip=True)
    has_cjk = bool(CJK_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    row_count = len(rows)
    logical_columns = max((len(row) for row in grid), default=0)
    large_dense = (
        row_count >= LARGE_ROW_THRESHOLD
        or logical_columns >= LARGE_COLUMN_THRESHOLD
        or physical_cells >= DENSE_CELL_THRESHOLD
    )
    regular = not rowspan_cells and not colspan_cells and not large_dense

    strata = []
    if regular:
        strata.append("regular_no_merge")
    if rowspan_cells:
        strata.append("rowspan")
    if colspan_cells:
        strata.append("colspan")
    if large_dense:
        strata.append("large_dense")
    if FORMULA_RE.search(html):
        strata.append("formula")
    if has_cjk and has_latin:
        strata.append("mixed_zh_en")
    elif has_cjk:
        strata.append("chinese")
    elif has_latin:
        strata.append("english")
    else:
        strata.append("other_language_or_numeric")

    return {
        "row_count": row_count,
        "logical_column_count": logical_columns,
        "physical_cell_count": physical_cells,
        "text_char_count": len(text),
        "rowspan_cell_count": rowspan_cells,
        "colspan_cell_count": colspan_cells,
        "strata": strata,
    }


def _matches_primary(record: dict, primary: str) -> bool:
    strata = set(record["table_stats"]["strata"])
    if primary == "colspan_only":
        return "colspan" in strata and "rowspan" not in strata
    return primary in strata


def _choose_records(records: list[dict], seed: int) -> list[dict]:
    by_index = {record["source_index"]: record for record in records}
    chosen: list[dict] = []
    used_images: set[str] = set()

    def add(record: dict, primary: str) -> bool:
        if record["source_image_path"] in used_images:
            return False
        selected = dict(record)
        selected["primary_stratum"] = primary
        selected["table_stats"] = dict(record["table_stats"])
        selected["table_stats"]["strata"] = list(record["table_stats"]["strata"])
        if primary == "reviewed_capture":
            selected["table_stats"]["strata"].append("reviewed_capture")
        chosen.append(selected)
        used_images.add(record["source_image_path"])
        return True

    for index in REVIEWED_CAPTURE_INDICES:
        if index not in by_index:
            raise ValueError(f"reviewed capture source index is missing: {index}")
        if not add(by_index[index], "reviewed_capture"):
            raise ValueError(f"duplicate image in reviewed capture rows: {index}")

    rng = random.Random(seed)
    for primary, quota in PRIMARY_QUOTAS.items():
        if primary == "reviewed_capture":
            continue
        candidates = [record for record in records if _matches_primary(record, primary)]
        rng.shuffle(candidates)
        before = len(chosen)
        for record in candidates:
            if add(record, primary) and len(chosen) - before == quota:
                break
        actual = len(chosen) - before
        if actual != quota:
            raise ValueError(f"could only select {actual}/{quota} records for {primary}")

    return chosen


def build_subset(source_path: Path, seed: int = DEFAULT_SEED) -> list[dict]:
    assert_data_usage("tableverse", "evaluation")
    dataset_root = source_path.parent
    records = []
    with source_path.open(encoding="utf-8") as source_file:
        for source_index, line in enumerate(source_file):
            source = json.loads(line)
            source_image_path = source["image_path"]
            reference_html = source["ref_answer"]
            image_path = dataset_root / source_image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"missing TableVerse image: {image_path}")
            records.append(
                {
                    "sample_id": make_sample_id(
                        "tableverse", source_index, source_image_path, reference_html
                    ),
                    "source": "tableverse",
                    "source_index": source_index,
                    "source_image_path": source_image_path,
                    "image_path": image_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
                    "prompt": BASELINE_PROMPT,
                    "reference_html": reference_html,
                    "table_stats": analyze_html(reference_html),
                }
            )

    chosen = _choose_records(records, seed)
    for record in chosen:
        with Image.open(project_path(record["image_path"])) as image:
            record["image_size"] = [image.width, image.height]
    return chosen


def write_subset(records: list[dict], output_path: Path, summary_path: Path, seed: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    subset_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    primary_counts = Counter(record["primary_stratum"] for record in records)
    label_counts = Counter(
        label for record in records for label in record["table_stats"]["strata"]
    )
    summary = {
        "source": "TableVerse-5K",
        "usage": "evaluation_only",
        "seed": seed,
        "sample_count": len(records),
        "unique_sample_ids": len({record["sample_id"] for record in records}),
        "unique_image_paths": len({record["source_image_path"] for record in records}),
        "primary_quotas": PRIMARY_QUOTAS,
        "primary_counts": dict(sorted(primary_counts.items())),
        "overlapping_label_counts": dict(sorted(label_counts.items())),
        "reviewed_capture_source_indices": list(REVIEWED_CAPTURE_INDICES),
        "subset_sha256": subset_sha256,
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_subset(args.source.resolve(), args.seed)
    summary = write_subset(records, args.output.resolve(), args.summary.resolve(), args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
