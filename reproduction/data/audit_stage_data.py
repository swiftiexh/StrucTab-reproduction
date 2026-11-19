"""Deterministically audit paired Stage 1/Stage 2 examples for 30 base tables."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.data.training_examples import STAGE1_TASKS, TrainingExample, validate_stage2_semantics
from reproduction.structure.parser import parse_html_structure


DEFAULT_STAGE1 = project_path("reproduction/data/training/stage1_smoke.jsonl")
DEFAULT_STAGE2 = project_path("reproduction/data/training/stage2_smoke.jsonl")
DEFAULT_OUTPUT = project_path("reproduction/data/training/stage_data_audit.json")
OFFICIAL_PARSER = project_path("StrucTab/code/Uni_TabRL/reward/utils/parsing.py")


def _load_examples(path: Path) -> list[TrainingExample]:
    with path.open(encoding="utf-8") as input_file:
        return [TrainingExample.from_dict(json.loads(line)) for line in input_file if line.strip()]


def _load_official_parser():
    spec = importlib.util.spec_from_file_location("structab_official_output_parser", OFFICIAL_PARSER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official reward parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(stage1_path: Path, stage2_path: Path, output: Path, count: int = 30, seed: int = 42) -> dict:
    stage1 = _load_examples(stage1_path)
    stage2 = _load_examples(stage2_path)
    official = _load_official_parser()
    by_base: dict[str, dict[str, TrainingExample]] = defaultdict(dict)
    for example in [*stage1, *stage2]:
        if example.task_type in by_base[example.base_sample_id]:
            raise ValueError(f"duplicate task for {example.base_sample_id}: {example.task_type}")
        by_base[example.base_sample_id][example.task_type] = example

    eligible = sorted(
        base_id
        for base_id, tasks in by_base.items()
        if set(tasks) == {*STAGE1_TASKS, "sequential"}
    )
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)}/{count} complete base groups available")
    validated: dict[str, dict] = {}
    for base_id in eligible:
        tasks = by_base[base_id]
        html = tasks["html"].target
        structure = parse_html_structure(html)
        if not structure["is_valid"]:
            raise ValueError(f"invalid HTML task for {base_id}: {structure['errors']}")
        expected_row_col = f"这是一个{structure['row_count']}行{structure['col_count']}列的表格。"
        if tasks["row_col"].target != expected_row_col:
            raise ValueError(f"wrong row_col target for {base_id}")
        if tasks["merge"].target != structure["merge_info"]:
            raise ValueError(f"wrong merge target for {base_id}")
        semantics = validate_stage2_semantics(tasks["sequential"].target)
        if not semantics["is_valid"]:
            raise ValueError(f"invalid Stage 2 target for {base_id}: {semantics['errors']}")
        official_sections = official.extract_sections(tasks["sequential"].target)
        if not official.has_all_sections(tasks["sequential"].target) or any(
            value is None for value in official_sections.values()
        ):
            raise ValueError(f"official parser rejected {base_id}")
        if official_sections["origin_tab"] != html:
            raise ValueError(f"official parser changed HTML for {base_id}")
        validated[base_id] = structure

    selected = random.Random(seed).sample(eligible, count)
    audit = []
    source_counts: Counter[str] = Counter()
    for base_id in selected:
        tasks = by_base[base_id]
        structure = validated[base_id]
        source_counts[tasks["html"].source] += 1
        audit.append(
            {
                "base_sample_id": base_id,
                "source": tasks["html"].source,
                "row_count": structure["row_count"],
                "col_count": structure["col_count"],
                "merge_count": len(structure["merged_cells"]),
                "stage1_tasks": list(STAGE1_TASKS),
                "stage2_official_parser": "accepted",
                "status": "consistent",
            }
        )

    summary = {
        "seed": seed,
        "available_complete_base_groups": len(eligible),
        "complete_base_groups_validated": len(validated),
        "audited_base_groups": len(audit),
        "audited_stage1_examples": len(audit) * 3,
        "audited_stage2_examples": len(audit),
        "source_counts": dict(sorted(source_counts.items())),
        "all_answers_consistent": True,
        "official_parser_acceptance_all": len(validated),
        "samples": audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--stage2", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(
        args.stage1.resolve(), args.stage2.resolve(), args.output.resolve(), args.count, args.seed
    )
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
