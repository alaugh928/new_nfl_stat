"""Optional benchmarks vs EPA/play (reference only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..data import load_pbp


def team_epa_per_play(seasons: list[int], *, no_cache: bool = False) -> pd.DataFrame:
    """Mean EPA per scrimmage play by posteam-season (for benchmark comparison)."""
    pbp = load_pbp(seasons, no_cache=no_cache)
    if "epa" not in pbp.columns:
        return pd.DataFrame(columns=["posteam", "season", "epa_per_play"])
    scrim = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_not_null()
        & ~pl.col("play_type").is_in(["kickoff", "kick_off", "extra_point", "no_play"])
    )
    return (
        scrim.group_by(["posteam", "season"])
        .agg(epa_per_play=pl.col("epa").mean())
        .to_pandas()
    )


def compare_pdp_vs_epa_season(
    team_season_pdp: pd.DataFrame,
    seasons: list[int],
    *,
    no_cache: bool = False,
) -> tuple[float, str]:
    """Correlation: off_pdp_mean vs epa_per_play (same season)."""
    epa = team_epa_per_play(seasons, no_cache=no_cache)
    merged = team_season_pdp.merge(
        epa,
        left_on=["team", "season"],
        right_on=["posteam", "season"],
        how="inner",
    )
    if len(merged) < 5:
        return float("nan"), "insufficient teams for EPA benchmark"
    r = float(np.corrcoef(merged["off_pdp_mean"], merged["epa_per_play"])[0, 1])
    return r, f"same-season PDP vs EPA/play r={r:.3f}"
