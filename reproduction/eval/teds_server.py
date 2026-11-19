"""Start exactly one local worker for the official StrucTab TEDS service."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from reproduction.common.paths import project_path


OFFICIAL_SERVER_DIR = project_path("StrucTab/code/Uni_TabRL/server/TEDS_judger")


def load_official_app():
    if not OFFICIAL_SERVER_DIR.is_dir():
        raise FileNotFoundError(f"official TEDS server directory missing: {OFFICIAL_SERVER_DIR}")
    sys.path.insert(0, str(OFFICIAL_SERVER_DIR))
    from TEDS_judger import app  # noqa: PLC0415

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18910)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    # Passing the app object without reload/workers launches exactly one process.
    uvicorn.run(load_official_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
