"""Audit the unified parser on deterministic complex normalized samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.structure.parser import format_merge, parse_html_structure


DEFAULT_INPUTS = [
    project_path("reproduction/data/normalized/synthtabnet_smoke.jsonl"),
    project_path("reproduction/data/normalized/fintabnet_smoke.jsonl"),
    project_path("reproduction/data/normalized/pubtables_smoke.jsonl"),
]
DEFAULT_OUTPUT = project_path("reproduction/data/normalized/structure_parser_audit.json")


def verify_internal_consistency(parsed: dict) -> None:
    if not parsed["is_valid"]:
        raise ValueError("parser rejected audit sample: " + "; ".join(parsed["errors"]))
    grid = parsed["anchor_grid"]
    cells_by_anchor = {(cell["row"] - 1, cell["col"] - 1): cell for cell in parsed["cells"]}
    observed_anchors = {anchor for row in grid for anchor in row}
    if observed_anchors != set(cells_by_anchor):
        raise ValueError("physical cells and grid anchors disagree")

    expected_merges = []
    for anchor, cell in cells_by_anchor.items():
        for row in range(cell["row"] - 1, cell["end_row"]):
            for col in range(cell["col"] - 1, cell["end_col"]):
                if grid[row][col] != anchor:
                    raise ValueError(f"cell rectangle disagrees with grid: cell={cell['cell_index']}")
        if cell["rowspan"] > 1 or cell["colspan"] > 1:
            expected_merges.append(
                format_merge(cell["row"], cell["col"], cell["end_row"], cell["end_col"])
            )
    if "".join(expected_merges) != parsed["merge_info"]:
        raise ValueError("cell coordinates and merge_info disagree")


def run(inputs: list[Path], output: Path, per_source: int = 10) -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    parsed_by_id: dict[str, dict] = {}
    total_checked = valid = 0
    for path in inputs:
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                parsed = parse_html_structure(record["html"])
                total_checked += 1
                valid += parsed["is_valid"]
                if parsed["is_valid"] and parsed["merged_cells"]:
                    candidates[record["source"]].append(record)
                    parsed_by_id[record["sample_id"]] = parsed

    audit_records = []
    for source in ("synthtabnet", "fintabnet", "pubtables"):
        selected = candidates[source][:per_source]
        if len(selected) != per_source:
            raise ValueError(f"only {len(selected)}/{per_source} merged samples available for {source}")
        for record in selected:
            parsed = parsed_by_id[record["sample_id"]]
            verify_internal_consistency(parsed)
            audit_records.append(
                {
                    "sample_id": record["sample_id"],
                    "source": source,
                    "row_count": parsed["row_count"],
                    "col_count": parsed["col_count"],
                    "physical_cell_count": len(parsed["cells"]),
                    "merge_count": len(parsed["merged_cells"]),
                    "merge_info": parsed["merge_info"],
                    "status": "consistent",
                }
            )

    summary = {
        "normalized_records_checked": total_checked,
        "normalized_records_valid": valid,
        "audit_records": len(audit_records),
        "per_source": per_source,
        "all_coordinates_consistent": True,
        "samples": audit_records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-source", type=int, default=10)
    args = parser.parse_args()
    result = run([path.resolve() for path in args.inputs], args.output.resolve(), args.per_source)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
