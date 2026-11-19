"""Materialize selected Phase 5 source rows into canonical normalized JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reproduction.common.data_policy import assert_data_usage
from reproduction.common.paths import project_path
from reproduction.data.conversion import ConversionSession, project_relative, validate_image
from reproduction.data.convert_fintabnet import run as run_fintabnet
from reproduction.data.convert_synthtabnet import convert_record as convert_synth_record


DEFAULT_SELECTION = project_path("reproduction/data/pools/pretraining_source_selection.json")
DEFAULT_SYNTH_SOURCE = project_path("assets/datasets/pretrain/SynthTabNet/sparse/synthetic_data.jsonl")
DEFAULT_FIN_DIR = project_path("assets/datasets/pretrain/FinTabNet-HTML/en")
DEFAULT_SYNTH_OUTPUT = project_path("reproduction/data/normalized/synthtabnet_pretrain_candidates.jsonl")
DEFAULT_FIN_OUTPUT = project_path("reproduction/data/normalized/fintabnet_pretrain_candidates.jsonl")
DEFAULT_FIN_IMAGES = project_path("reproduction/data/normalized/images/fintabnet")
DEFAULT_SUMMARY = project_path("reproduction/data/pools/pretraining_materialization_summary.json")


def materialize_synth(
    selection: dict, source: Path, output: Path, restart: bool = False
) -> dict:
    assert_data_usage("synthtabnet", "pretrain")
    selected = {int(item["source_row_index"]): item for item in selection["synth_candidates"]}
    session = ConversionSession(output, "synthtabnet", restart=restart)
    scanned = skipped = matched = 0
    max_row = max(selected)
    with source.open(encoding="utf-8") as input_file:
        for row_index, line in enumerate(input_file):
            scanned += 1
            if row_index not in selected:
                if row_index >= max_row:
                    break
                continue
            matched += 1
            expected = selected[row_index]
            source_key = expected["source_key"]
            try:
                if session.has_source_key(source_key):
                    skipped += 1
                else:
                    record = json.loads(line)
                    actual_key = f"{str(record['split']).lower()}:{record['imgid']}"
                    if actual_key != source_key:
                        raise ValueError(
                            f"selection/source mismatch: expected {source_key}, found {actual_key}"
                        )
                    sample = convert_synth_record(record, source, row_index)
                    validate_image(project_path(sample.image_path))
                    session.append(sample)
            except Exception as exc:
                session.record_failure(source_key, exc)
            if matched == len(selected):
                break
    summary = session.write_summary(scanned=scanned, skipped=skipped)
    summary["selected_source_rows"] = len(selected)
    summary["selected_source_rows_matched"] = matched
    return summary


def run(
    selection_path: Path,
    synth_source: Path,
    fin_dir: Path,
    synth_output: Path,
    fin_output: Path,
    fin_images: Path,
    summary_path: Path,
    restart: bool = False,
) -> dict:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    synth_summary = materialize_synth(selection, synth_source, synth_output, restart)
    fin_summary = run_fintabnet(fin_dir, fin_output, fin_images, limit=0, restart=restart)
    expected_synth = len(selection["synth_candidates"])
    expected_fin = len(selection["fintabnet_candidates"])
    result = {
        "selection": project_relative(selection_path),
        "synthtabnet": synth_summary,
        "fintabnet": fin_summary,
        "expected": {"synthtabnet": expected_synth, "fintabnet": expected_fin},
        "complete": (
            synth_summary["records"] == expected_synth
            and fin_summary["records"] == expected_fin
            and synth_summary["failures_this_run"] == 0
            and fin_summary["failures_this_run"] == 0
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    if not result["complete"]:
        raise RuntimeError(f"pretraining materialization incomplete; see {summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--synth-source", type=Path, default=DEFAULT_SYNTH_SOURCE)
    parser.add_argument("--fin-dir", type=Path, default=DEFAULT_FIN_DIR)
    parser.add_argument("--synth-output", type=Path, default=DEFAULT_SYNTH_OUTPUT)
    parser.add_argument("--fin-output", type=Path, default=DEFAULT_FIN_OUTPUT)
    parser.add_argument("--fin-images", type=Path, default=DEFAULT_FIN_IMAGES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    result = run(
        args.selection.resolve(), args.synth_source.resolve(), args.fin_dir.resolve(),
        args.synth_output.resolve(), args.fin_output.resolve(), args.fin_images.resolve(),
        args.summary.resolve(), args.restart,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
