"""Out-of-sample predictive correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

import polars as pl

from ..labels import team_season_offense_labels


def season_over_season_corr(
    drives: pd.DataFrame,
    *,
    split_week: int | None = 8,
) -> tuple[float, str]:
    """Mean off_pdp in season S (optionally weeks 1..split) vs PPD in season S+1."""
    if "drive_points_scored" not in drives.columns:
        drives = drives.copy()

    d = drives
    if split_week is not None and "week" in d.columns:
        d = d.loc[d["week"] <= split_week]
    off = (
        d.groupby(["posteam", "season"], observed=True)["off_pdp"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "off_pdp": "pdp_mean"})
    )
    actual = team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=True).to_pandas()
    actual = actual.rename(columns={"posteam": "team", "team_season_pts_per_drive": "actual_ppd"})

    off["next_season"] = off["season"] + 1
    merged = off.merge(
        actual[["team", "season", "actual_ppd"]],
        left_on=["team", "next_season"],
        right_on=["team", "season"],
        how="inner",
        suffixes=("", "_y"),
    )
    if len(merged) < 10:
        return float("nan"), "insufficient season pairs"
    r = float(np.corrcoef(merged["pdp_mean"], merged["actual_ppd"])[0, 1])
    return r, f"season-over-season r={r:.3f} (n={len(merged)})"


def rest_of_season_corr(
    drives: pd.DataFrame,
    *,
    split_week: int = 8,
) -> tuple[float, str]:
    """Mean PDP weeks 1-k vs actual pts/drive weeks k+1-17."""
    d = drives.copy()
    if "drive_points_scored" not in d.columns:
        from ..labels import add_drive_points_column

        d = add_drive_points_column(pl.from_pandas(d)).to_pandas()

    early = (
        d.loc[d["week"] <= split_week]
        .groupby(["posteam", "season"], observed=True)["off_pdp"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "off_pdp": "pdp_early"})
    )
    late = (
        d.loc[d["week"] > split_week]
        .groupby(["posteam", "season"], observed=True)
        .agg(points=("drive_points_scored", "sum"), drives=("game_id", "count"))
    )
    late = late.reset_index().rename(columns={"posteam": "team"})
    late["actual_ros_ppd"] = late["points"] / late["drives"].replace(0, np.nan)

    merged = early.merge(late[["team", "season", "actual_ros_ppd"]], on=["team", "season"])
    if len(merged) < 10:
        return float("nan"), "insufficient ROS pairs"
    r = float(np.corrcoef(merged["pdp_early"], merged["actual_ros_ppd"])[0, 1])
    return r, f"ROS (split wk {split_week}) r={r:.3f} (n={len(merged)})"
