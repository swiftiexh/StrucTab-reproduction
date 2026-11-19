"""Single-endpoint client for the official StrucTab TEDS judging service."""

from __future__ import annotations

import argparse
import math
import re
import time

import requests


DEFAULT_ENDPOINT = "http://127.0.0.1:18910"
SIMPLE_ROUTE = "/judge/simple"


def normalize_table_response(response: str) -> str:
    """Apply the same lightweight normalization as official benchmark/judge.py."""
    if not response:
        return ""
    response = response.strip()
    fenced = re.search(r"```(?:html)?\s*\n(.*?)\n```", response, re.DOTALL | re.IGNORECASE)
    if fenced:
        response = fenced.group(1).strip()
    response = (
        response.replace("<th>", "<td>")
        .replace("</th>", "</td>")
        .replace("<thead>", "")
        .replace("</thead>", "")
        .replace("<tbody>", "")
        .replace("</tbody>", "")
    )
    table = re.search(r"<table[\s\S]*?</table>", response, re.IGNORECASE)
    return table.group(0).strip() if table else response


class TedsClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 0.5,
        session: requests.Session | None = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.url = endpoint.rstrip("/") + SIMPLE_ROUTE
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.session = session or requests.Session()
        self.session.trust_env = False

    def score(self, prediction: str, reference: str) -> dict:
        prediction = normalize_table_response(prediction)
        reference = normalize_table_response(reference)
        if not prediction:
            return {
                "teds": 0.0,
                "teds_s": 0.0,
                "error": None,
                "diagnostic": "empty_prediction",
            }
        if not reference:
            return {
                "teds": 0.0,
                "teds_s": 0.0,
                "error": "empty reference",
                "diagnostic": None,
            }

        last_error = "TEDS service unavailable"
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.url,
                    json={"response": prediction, "ref_answer": reference},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                teds = float(payload["teds"])
                teds_s = float(payload["teds_s"])
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (teds, teds_s)):
                    raise ValueError(f"scores outside [0, 1]: teds={teds}, teds_s={teds_s}")
                return {"teds": teds, "teds_s": teds_s, "error": None, "diagnostic": None}
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
        return {"teds": 0.0, "teds_s": 0.0, "error": last_error, "diagnostic": None}

    def health_check(self) -> dict:
        probe = "<table><tr><td>health</td></tr></table>"
        result = self.score(probe, probe)
        healthy = (
            result["error"] is None
            and result["teds"] == 1.0
            and result["teds_s"] == 1.0
        )
        return {"healthy": healthy, "endpoint": self.url, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the single TEDS endpoint")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    result = TedsClient(args.endpoint, timeout_seconds=args.timeout).health_check()
    print(result)
    if not result["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
