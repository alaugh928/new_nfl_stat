"""Team-level benchmark metrics: PPD, EPA/drive, EPA/play, success rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..data import load_pbp
from ..labels import team_season_offense_labels
from ..utils import NON_SCRIMMAGE_TYPES


def _scrimmage_filter() -> pl.Expr:
    pt = pl.col("play_type")
    return (
        pl.col("posteam").is_not_null()
        & pt.is_not_null()
        & ~pt.is_in(sorted(NON_SCRIMMAGE_TYPES))
    )


def team_epa_per_play(seasons: list[int], *, no_cache: bool = False) -> pd.DataFrame:
    pbp = load_pbp(seasons, no_cache=no_cache)
    if "epa" not in pbp.columns:
        return pd.DataFrame(columns=["team", "season", "epa_per_play"])
    return (
        pbp.filter(_scrimmage_filter())
        .group_by(["posteam", "season"])
        .agg(epa_per_play=pl.col("epa").mean())
        .rename({"posteam": "team"})
        .to_pandas()
    )


def team_epa_per_drive(drives: pd.DataFrame, pbp_seasons: list[int], *, no_cache: bool = False) -> pd.DataFrame:
    """Mean sum(epa) per offensive drive by team-season."""
    pbp = load_pbp(pbp_seasons, no_cache=no_cache)
    if "epa" not in pbp.columns:
        return pd.DataFrame(columns=["team", "season", "epa_per_drive"])

    drive_epa = (
        pbp.filter(_scrimmage_filter())
        .group_by(["game_id", "fixed_drive", "posteam", "season"])
        .agg(drive_epa=pl.col("epa").sum())
        .rename({"posteam": "team"})
    )
    return (
        drive_epa.group_by(["team", "season"])
        .agg(epa_per_drive=pl.col("drive_epa").mean())
        .to_pandas()
    )


def team_success_rate(seasons: list[int], *, no_cache: bool = False) -> pd.DataFrame:
    pbp = load_pbp(seasons, no_cache=no_cache)
    if "success" not in pbp.columns:
        return pd.DataFrame(columns=["team", "season", "success_rate"])
    succ = (
        (pl.col("success") == True)  # noqa: E712
        | (pl.col("success") == 1)
        | (pl.col("success") == 1.0)
    ).cast(pl.Float64)
    return (
        pbp.filter(_scrimmage_filter())
        .with_columns(_succ=succ)
        .group_by(["posteam", "season"])
        .agg(success_rate=pl.col("_succ").mean())
        .rename({"posteam": "team"})
        .to_pandas()
    )


def team_ppd(seasons: list[int], drives: pd.DataFrame | None = None, *, exclude_kneels: bool = True) -> pd.DataFrame:
    if drives is not None:
        return (
            team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=exclude_kneels)
            .rename({"posteam": "team", "team_season_pts_per_drive": "ppd"})
            .select("team", "season", "ppd")
            .to_pandas()
        )
    # seasons list unused if drives provided
    _ = seasons
    return pd.DataFrame(columns=["team", "season", "ppd"])


def build_team_season_benchmarks(
    drives: pd.DataFrame,
    seasons: list[int],
    team_season_pdp: pd.DataFrame,
    *,
    no_cache: bool = False,
    exclude_kneels: bool = True,
) -> pd.DataFrame:
    """One row per team-season with PDP, PPD, EPA/play, EPA/drive, success."""
    base = team_season_pdp.copy()
    ppd = team_ppd(seasons, drives, exclude_kneels=exclude_kneels)
    epa_play = team_epa_per_play(seasons, no_cache=no_cache)
    epa_drive = team_epa_per_drive(drives, seasons, no_cache=no_cache)
    succ = team_success_rate(seasons, no_cache=no_cache)

    out = base.merge(ppd, on=["team", "season"], how="left")
    out = out.merge(epa_play, on=["team", "season"], how="left")
    out = out.merge(epa_drive, on=["team", "season"], how="left")
    out = out.merge(succ, on=["team", "season"], how="left")
    return out


def week_level_offense(drives: pd.DataFrame, *, exclude_kneels: bool = False) -> pd.DataFrame:
    d = drives.copy()
    if exclude_kneels and "is_kneel_only_drive" in d.columns:
        d = d.loc[~d["is_kneel_only_drive"]]
    return (
        d.groupby(["posteam", "season", "week"], observed=True)
        .agg(
            off_pdp_mean=("off_pdp", "mean"),
            ppd=("drive_points_scored", lambda s: s.sum() / max(len(s), 1)),
            drives=("off_pdp", "count"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


def autocorrelation_series(x: pd.Series) -> float:
    if len(x) < 3:
        return float("nan")
    return float(x.autocorr(lag=1))
