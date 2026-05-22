"""Synthetic Team A vs Team B and turnover narrative checks."""

from __future__ import annotations

import pandas as pd

from ..constants import OFFENSE_FORMULA_FEATURES, OFFENSE_FORMULA_FEATURES_V2
from ..formula import load_formula, score_offense_pdp


def _feature_list() -> list[str]:
    f = load_formula()
    return f.get("offense", {}).get("features", OFFENSE_FORMULA_FEATURES_V2)


def _synthetic_row(**kwargs: float) -> pd.DataFrame:
    base = {k: 0.0 for k in _feature_list()}
    base.update(kwargs)
    return pd.DataFrame([base])


def check_team_a_vs_team_b() -> tuple[bool, str]:
    """
    Team A: long march + fumble. Team B: 1-play explosive TD profile.
    """
    team_a = _synthetic_row(
        progress_yards=75.0,
        progress_pct=0.75,
        plays=10.0,
        yards_total=75.0,
        yards_per_play=7.5,
        first_downs_total=5.0,
        first_downs_on_1st=2.0,
        first_downs_on_2nd=2.0,
        turnover_fumble_lost=1.0,
        turnover_after_progress=75.0,
        turnover_penalty_interaction=75.0,
        unsustained_explosion=0.05,
        explosive_yards_share=0.2,
        max_play_yard_share=0.25,
        start_yardline_100=75.0,
    )
    team_b = _synthetic_row(
        progress_yards=80.0,
        progress_pct=1.0,
        plays=1.0,
        yards_total=80.0,
        yards_per_play=80.0,
        first_downs_total=0.0,
        explosive_play_count=1.0,
        explosive_yards_share=1.0,
        max_play_yard_share=1.0,
        unsustained_explosion=1.0,
        start_yardline_100=80.0,
    )
    formula = load_formula()
    pdp_a = float(score_offense_pdp(team_a, formula)[0])
    pdp_b = float(score_offense_pdp(team_b, formula)[0])
    ok = pdp_a > pdp_b
    msg = f"Team A PDP={pdp_a:.3f} vs Team B PDP={pdp_b:.3f} (need A > B)"
    return ok, msg


def check_turnover_narratives() -> tuple[bool, str]:
    """Goal-line march > 3-and-out; early INT hurts."""
    march = _synthetic_row(
        progress_yards=70.0,
        plays=12.0,
        first_downs_total=6.0,
        turnover_on_downs=1.0,
        turnover_after_progress=70.0,
        turnover_penalty_interaction=70.0,
        start_yardline_100=70.0,
    )
    three_out = _synthetic_row(
        progress_yards=2.0,
        plays=3.0,
        first_downs_total=0.0,
        start_yardline_100=80.0,
    )
    early_int = _synthetic_row(
        progress_yards=0.0,
        plays=1.0,
        turnover_interception=1.0,
        turnover_after_progress=0.0,
        start_yardline_100=80.0,
    )
    formula = load_formula()
    p_march = float(score_offense_pdp(march, formula)[0])
    p_three = float(score_offense_pdp(three_out, formula)[0])
    p_int = float(score_offense_pdp(early_int, formula)[0])
    ok = p_march > p_three and p_three > p_int
    msg = (
        f"march={p_march:.3f} three_out={p_three:.3f} early_int={p_int:.3f} "
        f"(need march > three_out > early_int)"
    )
    return ok, msg
