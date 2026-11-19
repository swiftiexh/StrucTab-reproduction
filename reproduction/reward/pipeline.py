"""Local, failure-isolated integration of the official Uni-TabRL reward."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from reproduction.common.protocol import parse_sequential_output
from reproduction.eval.teds_client import TedsClient
from reproduction.reward.official_primitives import (
    clean_table_content,
    official_extract_sections,
    official_has_all_sections,
    official_validity_reward,
    one_d_probe_reward,
)


VALIDITY_WEIGHT = 0.4
STRUCTURE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.3
DEFAULT_WEIGHTS = (VALIDITY_WEIGHT, STRUCTURE_WEIGHT, CONTENT_WEIGHT)
STRUCTURE_VARIANTS = frozenset({"1d_probe", "teds_s"})
CONTENT_VARIANTS = frozenset({"anchor", "teds"})

TedsScorer = Callable[[str, str], dict[str, Any]]
AnchorScorer = Callable[[str, str], dict[str, Any] | None]


def _default_teds_scorer(prediction: str, reference: str) -> dict[str, Any]:
    return TedsClient().score(prediction, reference)


def _zero_result(reason: str, *, computation_ok: bool, parsing: dict | None = None) -> dict:
    return {
        "status": "ok" if computation_ok else "failed",
        "reward": 0.0,
        "validity_reward": 0.0,
        "structure_reward": 0.0,
        "content_reward": 0.0,
        "weights": list(DEFAULT_WEIGHTS),
        "variants": {"structure": "1d_probe", "content": "anchor"},
        "base_metrics": {"teds": None, "teds_s": None},
        "one_d_probe": None,
        "content_detail": None,
        "fallbacks": [],
        "parsing": parsing,
        "reason": reason,
    }


@dataclass
class RewardPipeline:
    """Compute official components while injecting local service dependencies."""

    teds_scorer: TedsScorer = _default_teds_scorer
    anchor_scorer: AnchorScorer | None = None
    structure_variant: str = "1d_probe"
    content_variant: str = "anchor"
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS

    def __post_init__(self) -> None:
        self.structure_variant = self.structure_variant.lower()
        self.content_variant = self.content_variant.lower()
        if self.structure_variant not in STRUCTURE_VARIANTS:
            raise ValueError(f"invalid structure variant: {self.structure_variant}")
        if self.content_variant not in CONTENT_VARIANTS:
            raise ValueError(f"invalid content variant: {self.content_variant}")
        if len(self.weights) != 3 or not all(
            math.isfinite(value) and value >= 0 for value in self.weights
        ):
            raise ValueError("weights must be three finite non-negative values")

    def _zero(self, reason: str, *, computation_ok: bool, parsing: dict | None = None) -> dict:
        result = _zero_result(reason, computation_ok=computation_ok, parsing=parsing)
        result["weights"] = list(self.weights)
        result["variants"] = {
            "structure": self.structure_variant,
            "content": self.content_variant,
        }
        return result

    def compute(
        self,
        response: str,
        reference: str,
        *,
        split: str = "train",
        ref_destylized: dict[str, Any] | None = None,
    ) -> dict:
        """Return component scores and diagnostics; dependency failures never raise."""
        strict_parsing = parse_sequential_output(response)
        parsing_diagnostic = {
            "strict_valid": strict_parsing["is_valid"],
            "strict_errors": strict_parsing["errors"],
        }
        try:
            # Preserve the official hard gate rather than replacing it with the
            # stricter project parser. The strict result is retained as diagnosis.
            if not official_has_all_sections(response):
                return self._zero(
                    "response_missing_section_markers",
                    computation_ok=True,
                    parsing=parsing_diagnostic,
                )
            parsed_response = official_extract_sections(response)
            pred_table = parsed_response["origin_tab"]
            if not pred_table or not pred_table.endswith("</table>"):
                return self._zero(
                    "response_final_table_missing_ending",
                    computation_ok=True,
                    parsing=parsing_diagnostic,
                )

            if not official_has_all_sections(reference):
                return self._zero(
                    "reference_missing_section_markers",
                    computation_ok=False,
                    parsing=parsing_diagnostic,
                )
            parsed_reference = official_extract_sections(reference)
            ref_table = parsed_reference["origin_tab"]
            if not ref_table or not ref_table.endswith("</table>"):
                return self._zero(
                    "reference_final_table_missing_ending",
                    computation_ok=False,
                    parsing=parsing_diagnostic,
                )

            base = self.teds_scorer(pred_table, ref_table)
            if base.get("error") is not None:
                return self._zero(
                    f"teds_dependency_failed: {base['error']}",
                    computation_ok=False,
                    parsing=parsing_diagnostic,
                )
            teds = float(base["teds"])
            teds_s = float(base["teds_s"])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (teds, teds_s)):
                raise ValueError("TEDS scores must be finite values in [0, 1]")

            validity = official_validity_reward(
                parsed_response["row_col"],
                parsed_reference["row_col"],
                parsed_response["merge"],
                parsed_reference["merge"],
            )
            probe = None
            if self.structure_variant == "1d_probe":
                probe = one_d_probe_reward(
                    clean_table_content(pred_table), clean_table_content(ref_table)
                )
                structure = float(probe["score"])
            else:
                structure = teds_s

            content = teds
            content_detail = None
            fallbacks: list[str] = []
            if split == "train" and self.content_variant == "anchor":
                if ref_destylized and ref_destylized.get("should_de_stylize"):
                    reference_anchor = ref_destylized.get("de_stylize_ref")
                    if self.anchor_scorer is None:
                        fallbacks.append("anchor_scorer_unavailable_use_base_teds")
                    elif not isinstance(reference_anchor, str) or not reference_anchor.strip():
                        fallbacks.append("anchor_reference_missing_use_base_teds")
                    else:
                        anchor = self.anchor_scorer(pred_table, reference_anchor)
                        if anchor and anchor.get("teds") is not None:
                            content = float(anchor["teds"])
                            content_detail = {"anchor": anchor}
                        else:
                            fallbacks.append("anchor_failed_use_base_teds")

            total = (
                self.weights[0] * validity
                + self.weights[1] * structure
                + self.weights[2] * content
            )
            return {
                "status": "ok",
                "reward": total,
                "validity_reward": float(validity),
                "structure_reward": structure,
                "content_reward": content,
                "weights": list(self.weights),
                "variants": {
                    "structure": self.structure_variant,
                    "content": self.content_variant,
                },
                "base_metrics": {"teds": teds, "teds_s": teds_s},
                "one_d_probe": probe,
                "content_detail": content_detail,
                "fallbacks": fallbacks,
                "parsing": parsing_diagnostic,
                "reason": None,
            }
        except Exception as exc:
            return self._zero(
                f"reward_exception: {type(exc).__name__}: {exc}",
                computation_ok=False,
                parsing=parsing_diagnostic,
            )
