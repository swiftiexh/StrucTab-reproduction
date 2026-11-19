"""Run the guarded 24+8 step Stage 1/2 soak and emit a Go/No-Go report."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from reproduction.common.paths import project_path
from reproduction.train.hunyuan_feasibility import atomic_json


DEFAULT_MODEL = project_path("assets/models/HunyuanOCR/v1.0")
DEFAULT_SOAK_DIR = project_path("reproduction/data/training/soak")
DEFAULT_RUN_RESULT_DIR = project_path("reproduction/results/soak/runs")
DEFAULT_REPORT_DIR = project_path("reproduction/results/training_feasibility")
DEFAULT_CHECKPOINT_DIR = project_path("reproduction/checkpoints/soak")


def stop_group(process: subprocess.Popen, grace: int = 10) -> str:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
        return "sigterm"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return "sigkill"


def run_child(arguments: list[str], timeout: int) -> int:
    command = [sys.executable, "-u", "-m", "reproduction.train.stage_trainer", *arguments]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_group(process)
        return 124
    except KeyboardInterrupt:
        stop_group(process)
        raise


def read_result(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "error": f"missing result: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def memory_drift(result: dict) -> float | None:
    train = result.get("train")
    if not train:
        return None
    return round(
        train["cleanup_allocated_last_gib"] - train["cleanup_allocated_first_gib"], 4
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--soak-dir", type=Path, default=DEFAULT_SOAK_DIR)
    parser.add_argument("--timeout-seconds-per-stage", type=int, default=1800)
    parser.add_argument("--max-projected-hours", type=float, default=7.5)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    args = parser.parse_args()
    if args.timeout_seconds_per_stage < 1 or args.max_projected_hours <= 0:
        parser.error("timeout and projected-hour limit must be positive")

    run_result_dir = DEFAULT_RUN_RESULT_DIR.resolve()
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR.resolve()
    report_path = DEFAULT_REPORT_DIR.resolve() / f"{args.run_name}.json"
    stage1_name = f"{args.run_name}_stage1"
    stage2_name = f"{args.run_name}_stage2"
    stage1_result_path = run_result_dir / f"{stage1_name}.json"
    stage2_result_path = run_result_dir / f"{stage2_name}.json"
    if report_path.exists() or stage1_result_path.exists() or stage2_result_path.exists():
        raise FileExistsError(f"soak run-name already exists: {args.run_name}")

    common = [
        "--learning-rate", str(args.learning_rate),
        "--max-sequence-tokens", "1536",
        "--validation-steps", "0",
        "--result-dir", str(run_result_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-every", "1",
    ]
    report = {
        "status": "running",
        "run_name": args.run_name,
        "paper_alignment": {
            "full_parameter": True,
            "stage1_learning_rate": 4e-5,
            "stage1_epochs_formal": 1,
            "stage2_epochs_formal": 1,
            "stage2_learning_rate_assumption": args.learning_rate,
        },
    }
    atomic_json(report_path, report)
    started = time.time()
    try:
        stage1_code = run_child(
            [
                "--stage", "stage1",
                "--run-name", stage1_name,
                "--model", str(args.model.resolve()),
                "--processor", str(args.model.resolve()),
                "--train-data", str((args.soak_dir / "stage1_soak.jsonl").resolve()),
                "--max-train-steps", "24",
                *common,
            ],
            args.timeout_seconds_per_stage,
        )
        stage1 = read_result(stage1_result_path)
        if stage1_code != 0 or stage1.get("status") != "ok":
            raise RuntimeError(f"Stage 1 soak failed: exit={stage1_code}, status={stage1.get('status')}")
        stage1_checkpoint = checkpoint_dir / stage1_name
        stage2_code = run_child(
            [
                "--stage", "stage2",
                "--run-name", stage2_name,
                "--model", str(stage1_checkpoint),
                "--processor", str(stage1_checkpoint),
                "--train-data", str((args.soak_dir / "stage2_soak.jsonl").resolve()),
                "--max-train-steps", "8",
                *common,
            ],
            args.timeout_seconds_per_stage,
        )
        stage2 = read_result(stage2_result_path)
        if stage2_code != 0 or stage2.get("status") != "ok":
            raise RuntimeError(f"Stage 2 soak failed: exit={stage2_code}, status={stage2.get('status')}")

        train_seconds = (
            stage1["train"]["mean_step_seconds"] * 960
            + stage2["train"]["mean_step_seconds"] * 320
        )
        validation_seconds = (
            stage1["train"]["mean_step_seconds"] * 120 * 0.4
            + stage2["train"]["mean_step_seconds"] * 40 * 0.4
        )
        fixed_seconds = sum(
            result.get("model_load_seconds", 0)
            + result.get("checkpoint_save_seconds", 0)
            + result.get("reload", {}).get("seconds", 0)
            for result in (stage1, stage2)
        )
        projected_seconds = (train_seconds + validation_seconds) * 1.2 + fixed_seconds
        drifts = {"stage1": memory_drift(stage1), "stage2": memory_drift(stage2)}
        memory_stable = all(value is not None and value <= 0.25 for value in drifts.values())
        reload_success = stage1["reload"]["success"] and stage2["reload"]["success"]
        projected_hours = projected_seconds / 3600
        decision = (
            "go"
            if memory_stable
            and reload_success
            and stage1["losses_finite"]
            and stage2["losses_finite"]
            and projected_hours <= args.max_projected_hours
            else "no-go"
        )
        report.update(
            {
                "status": "ok",
                "decision": decision,
                "stage1_result": str(stage1_result_path),
                "stage2_result": str(stage2_result_path),
                "stage1_steps": stage1["train"]["steps"],
                "stage2_steps": stage2["train"]["steps"],
                "stage1_mean_step_seconds": stage1["train"]["mean_step_seconds"],
                "stage2_mean_step_seconds": stage2["train"]["mean_step_seconds"],
                "projected_total_hours": round(projected_hours, 4),
                "projected_limit_hours": args.max_projected_hours,
                "projection_includes_validation_and_20_percent_margin": True,
                "memory_cleanup_drift_gib": drifts,
                "memory_stable": memory_stable,
                "checkpoint_reload_success": reload_success,
                "losses_finite": stage1["losses_finite"] and stage2["losses_finite"],
            }
        )
    except KeyboardInterrupt:
        report.update({"status": "interrupted", "decision": "no-go"})
    except Exception as exc:
        report.update(
            {"status": "failed", "decision": "no-go", "error_type": type(exc).__name__, "error": str(exc)}
        )
    report["actual_soak_seconds"] = round(time.time() - started, 3)
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.get("decision") == "go" else 1)


if __name__ == "__main__":
    main()
