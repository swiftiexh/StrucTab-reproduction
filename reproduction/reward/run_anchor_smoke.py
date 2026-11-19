"""Render one fixed table and reparse it with the local HunyuanOCR Anchor."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.reward.anchor import LocalAnchorScorer
from reproduction.reward.rendering import LocalHtmlRenderer
from reproduction.structure.parser import parse_html_structure


DEFAULT_OUTPUT = project_path("reproduction/results/reward/anchor_smoke.json")
DEFAULT_RENDER_DIR = project_path("reproduction/results/reward/render_smoke")
REFERENCE_HTML = (
    "<table><tr><th>项目</th><th>数值</th></tr>"
    "<tr><td>A</td><td>1</td></tr>"
    "<tr><td>B</td><td></td></tr></table>"
)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.perf_counter()
    try:
        with LocalHtmlRenderer(args.render_dir.resolve()) as renderer:
            anchor = LocalAnchorScorer(renderer, max_new_tokens=args.max_new_tokens)
            parsed = anchor.parse_reference(REFERENCE_HTML)
        prediction = parsed["inference"]["pred_html"]
        pred_structure = parse_html_structure(prediction)
        ref_structure = parse_html_structure(REFERENCE_HTML)
        result = {
            "status": "ok" if prediction else "failed",
            "reference_html": REFERENCE_HTML,
            "render": parsed["render"],
            "inference": parsed["inference"],
            "checks": {
                "complete_table": bool(prediction),
                "exact_html": prediction == REFERENCE_HTML,
                "valid_structure": pred_structure["is_valid"],
                "same_dimensions": (
                    pred_structure["row_count"], pred_structure["col_count"]
                ) == (ref_structure["row_count"], ref_structure["col_count"]),
                "same_anchor_grid": pred_structure["anchor_grid"] == ref_structure["anchor_grid"],
            },
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    _write_atomic(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
