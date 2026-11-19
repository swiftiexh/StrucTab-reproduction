"""Run a feasibility probe in a child process with a hard wall-clock timeout."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from reproduction.train.hunyuan_feasibility import DEFAULT_RESULT_DIR, atomic_json


def _option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 == len(arguments):
                raise ValueError(f"{name} requires a value")
            return arguments[index + 1]
        prefix = name + "="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def result_path_for(arguments: list[str]) -> Path:
    run_name = _option_value(arguments, "--run-name")
    if run_name is None:
        run_name = _option_value(arguments, "--strategy")
    if run_name is None:
        raise ValueError("child arguments require --strategy (and preferably --run-name)")
    result_dir = Path(_option_value(arguments, "--result-dir") or DEFAULT_RESULT_DIR)
    return result_dir.resolve() / f"{run_name}.json"


def _stop_process_group(process: subprocess.Popen, grace_seconds: int) -> str:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return "sigterm"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return "sigkill"


def _record_guard_status(
    result_path: Path, status: str, timeout_seconds: int, termination: str
) -> None:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        result = {}
    result.update(
        {
            "status": status,
            "error_type": "GuardTimeout" if status == "timeout" else "GuardInterrupted",
            "error": (
                f"training child exceeded {timeout_seconds}s wall-clock timeout"
                if status == "timeout"
                else "training child was interrupted by the user"
            ),
            "guard": {
                "timeout_seconds": timeout_seconds,
                "termination": termination,
            },
            "completed_at_unix": time.time(),
        }
    )
    atomic_json(result_path, result)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--terminate-grace-seconds", type=int, default=10)
    args, child_arguments = parser.parse_known_args()
    if args.timeout_seconds < 1 or args.terminate_grace_seconds < 1:
        parser.error("timeout values must be positive")
    if child_arguments and child_arguments[0] == "--":
        child_arguments = child_arguments[1:]
    if not child_arguments:
        parser.error("missing hunyuan_feasibility arguments")
    return args, child_arguments


def main() -> None:
    args, child_arguments = parse_args()
    result_path = result_path_for(child_arguments)
    command = [
        sys.executable,
        "-u",
        "-m",
        "reproduction.train.hunyuan_feasibility",
        *child_arguments,
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        termination = _stop_process_group(process, args.terminate_grace_seconds)
        _record_guard_status(result_path, "timeout", args.timeout_seconds, termination)
        print(
            f"feasibility guard: timed out after {args.timeout_seconds}s; "
            f"child stopped with {termination}",
            file=sys.stderr,
        )
        raise SystemExit(124)
    except KeyboardInterrupt:
        termination = _stop_process_group(process, args.terminate_grace_seconds)
        _record_guard_status(result_path, "interrupted", args.timeout_seconds, termination)
        print(
            f"feasibility guard: interrupted; child stopped with {termination}",
            file=sys.stderr,
        )
        raise SystemExit(130)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
