"""Defensive PDP should be positive on average (higher = better defense)."""

from __future__ import annotations

import pandas as pd
import pytest

from drive_strength.constants import DEFAULT_FORMULA_PATH
from drive_strength.formula import load_formula, score_defense_pdp


pytestmark = pytest.mark.skipif(
    not DEFAULT_FORMULA_PATH.exists(),
    reason="formula not trained yet",
)


def test_def_pdp_positive_on_strong_stop_drive():
    formula = load_formula()
    features = formula["defense"]["features"]
    row = {f: 0.0 for f in features}
    row.update(
        {
            "progress_prevented": 12.0,
            "progress_pct_prevented": 0.25,
            "first_downs_allowed_total": 1.0,
            "turnover_forced_interception": 1.0,
            "sack_forced_count": 1.0,
            "negative_play_forced_rate": 0.25,
        }
    )
    score = float(score_defense_pdp(pd.DataFrame([row]), formula)[0])
    assert score > 0.0, f"expected positive def_pdp, got {score}"
