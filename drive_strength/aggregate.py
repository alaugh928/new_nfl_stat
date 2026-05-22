"""Team-week and team-season PDP aggregates."""

from __future__ import annotations

import pandas as pd


def aggregate_team_week(
    drives: pd.DataFrame,
    *,
    exclude_kneels: bool = False,
) -> pd.DataFrame:
    d = drives.copy()
    if exclude_kneels and "is_kneel_only_drive" in d.columns:
        d = d.loc[~d["is_kneel_only_drive"]]

    off = (
        d.groupby(["posteam", "season", "week"], observed=True)
        .agg(
            off_drives=("off_pdp", "count"),
            off_pdp_mean=("off_pdp", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    de = (
        d.groupby(["defteam", "season", "week"], observed=True)
        .agg(
            def_drives_faced=("def_pdp", "count"),
            def_pdp_mean=("def_pdp", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    return off.merge(de, on=["team", "season", "week"], how="outer")


def _shrink_toward_league(
    values: pd.Series,
    counts: pd.Series,
    league_mean: float,
    *,
    prior_weight: float = 10.0,
) -> pd.Series:
    """Empirical-Bayes shrinkage toward league mean (strength ∝ drives)."""
    n = counts.astype(float)
    return (n * values + prior_weight * league_mean) / (n + prior_weight)


def aggregate_team_season(
    drives: pd.DataFrame,
    *,
    exclude_kneels: bool = False,
    shrink_prior: float = 10.0,
) -> pd.DataFrame:
    d = drives.copy()
    if exclude_kneels and "is_kneel_only_drive" in d.columns:
        d = d.loc[~d["is_kneel_only_drive"]]

    off = (
        d.groupby(["posteam", "season"], observed=True)
        .agg(
            off_drives=("off_pdp", "count"),
            off_pdp_mean=("off_pdp", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    de = (
        d.groupby(["defteam", "season"], observed=True)
        .agg(
            def_drives_faced=("def_pdp", "count"),
            def_pdp_mean=("def_pdp", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    out = off.merge(de, on=["team", "season"], how="outer")
    if shrink_prior > 0 and "off_pdp_mean" in out.columns:
        league = float(d["off_pdp"].mean())
        out["off_pdp_mean_shrunk"] = _shrink_toward_league(
            out["off_pdp_mean"], out["off_drives"], league, prior_weight=shrink_prior
        )
        if "def_pdp_mean" in out.columns:
            league_def = float(d["def_pdp"].mean())
            out["def_pdp_mean_shrunk"] = _shrink_toward_league(
                out["def_pdp_mean"],
                out["def_drives_faced"],
                league_def,
                prior_weight=shrink_prior,
            )
    return out
