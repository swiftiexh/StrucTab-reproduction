"""Load the official Uni-TabRL primitives without importing its service stack."""

from __future__ import annotations

import sys

from reproduction.common.paths import project_path


OFFICIAL_CODE_ROOT = project_path("StrucTab/code")
if str(OFFICIAL_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_CODE_ROOT))

# These packages have no network or environment-variable side effects. Importing
# reward_func/content is intentionally avoided because the official content utils
# instantiate placeholder service clients and require OUTPUT_PATH at import time.
from Uni_TabRL.reward.structure import clean_table_content, one_d_probe_reward  # noqa: E402
from Uni_TabRL.reward.utils.parsing import (  # noqa: E402
    extract_sections as official_extract_sections,
    has_all_sections as official_has_all_sections,
)
from Uni_TabRL.reward.validity import validity_reward as official_validity_reward  # noqa: E402
from Uni_TabRL.reward.validity.utils import (  # noqa: E402
    merge_correct as official_merge_correct,
    row_col_correct as official_row_col_correct,
)


__all__ = [
    "clean_table_content",
    "one_d_probe_reward",
    "official_extract_sections",
    "official_has_all_sections",
    "official_validity_reward",
    "official_merge_correct",
    "official_row_col_correct",
]
