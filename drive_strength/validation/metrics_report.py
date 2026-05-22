"""Fair predictive/descriptive metric tables for PDP vs benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..labels import add_drive_points_column, team_season_offense_labels
from ..plots.metrics import (
    autocorrelation_series,
    team_epa_per_play,
    week_level_offense,
)
from .predictive import rest_of_season_corr, season_over_season_corr

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _corr(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def team_early_epa_per_play(
    drives: pd.DataFrame,
    pbp_seasons: list[int],
    *,
    split_week: int = 8,
    no_cache: bool = False,
) -> pd.DataFrame:
    """EPA/play from weeks 1..split only."""
    from ..data import load_pbp
    from ..utils import NON_SCRIMMAGE_TYPES

    pbp = load_pbp(pbp_seasons, no_cache=no_cache)
    if "epa" not in pbp.columns:
        return pd.DataFrame(columns=["team", "season", "epa_early"])
    scrim = pbp.filter(
        pl.col("posteam").is_not_null()
        & pl.col("play_type").is_not_null()
        & ~pl.col("play_type").is_in(sorted(NON_SCRIMMAGE_TYPES))
        & (pl.col("week") <= split_week)
    )
    return (
        scrim.group_by(["posteam", "season"])
        .agg(epa_early=pl.col("epa").mean())
        .rename({"posteam": "team"})
        .to_pandas()
    )


def rest_of_season_corr_metric(
    drives: pd.DataFrame,
    x_col: str,
    *,
    split_week: int = 8,
    group_col: str = "posteam",
) -> float:
    """Correlation: early-window X vs late PPD."""
    d = drives.copy()
    if "drive_points_scored" not in d.columns:
        d = add_drive_points_column(pl.from_pandas(d)).to_pandas()

    early = (
        d.loc[d["week"] <= split_week]
        .groupby([group_col, "season"], observed=True)[x_col]
        .mean()
        .reset_index()
        .rename(columns={group_col: "team", x_col: "x_early"})
    )
    late = (
        d.loc[d["week"] > split_week]
        .groupby([group_col, "season"], observed=True)
        .agg(pts=("drive_points_scored", "sum"), n=("game_id", "count"))
        .reset_index()
        .rename(columns={group_col: "team"})
    )
    late["ppd_ros"] = late["pts"] / late["n"].replace(0, np.nan)
    merged = early.merge(late[["team", "season", "ppd_ros"]], on=["team", "season"])
    return _corr(merged["x_early"], merged["ppd_ros"])


def season_over_season_for_column(
    bench: pd.DataFrame,
    x_col: str,
) -> float:
    cur = bench[["team", "season", x_col]].copy()
    cur["next_season"] = cur["season"] + 1
    nxt = bench[["team", "season", "ppd"]].rename(
        columns={"season": "next_season", "ppd": "ppd_next"}
    )
    m = cur.merge(nxt, on=["team", "next_season"], how="inner", suffixes=("", "_dup"))
    x = m[x_col]
    y = m["ppd_next"]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return _corr(x, y)


def build_metrics_table(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    formula_label: str = "PDP",
    pdp_col: str = "off_pdp",
    no_cache: bool = False,
    season_filter: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Full metrics report for one formula version."""
    if "drive_points_scored" not in drives.columns:
        drives = add_drive_points_column(pl.from_pandas(drives)).to_pandas()

    d = drives.copy()
    ts = team_season.copy()
    if season_filter:
        lo, hi = season_filter
        d = d[(d["season"] >= lo) & (d["season"] <= hi)]
        ts = ts[(ts["season"] >= lo) & (ts["season"] <= hi)]
        seasons = [s for s in seasons if lo <= s <= hi]

    pbp_seasons = seasons if seasons else sorted(d["season"].unique().astype(int).tolist())

    ppd = team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=True).to_pandas()
    ppd = ppd.rename(columns={"posteam": "team", "team_season_pts_per_drive": "ppd"})
    bench = ts.merge(ppd[["team", "season", "ppd"]], on=["team", "season"], how="left")
    bench = bench.rename(columns={"off_pdp_mean": "pdp_mean"})
    if pbp_seasons:
        epa_full = team_epa_per_play(pbp_seasons, no_cache=no_cache)
        if not epa_full.empty:
            bench = bench.merge(epa_full, on=["team", "season"], how="left")

    rows: list[dict] = []

    rows.append(
        {
            "formula": formula_label,
            "metric": "same_season_vs_ppd",
            "r": _corr(bench["pdp_mean"], bench["ppd"]),
        }
    )
    if "epa_per_play" in bench.columns:
        rows.append(
            {
                "formula": formula_label,
                "metric": "same_season_vs_epa_play",
                "r": _corr(bench["pdp_mean"], bench["epa_per_play"]),
            }
        )
    rows.append(
        {
            "formula": formula_label,
            "metric": "season_over_season_pdp",
            "r": season_over_season_for_column(bench, "pdp_mean"),
        }
    )
    if "epa_per_play" in bench.columns:
        rows.append(
            {
                "formula": formula_label,
                "metric": "season_over_season_epa_play",
                "r": season_over_season_for_column(bench, "epa_per_play"),
            }
        )

    for split in (4, 8, 12):
        rows.append(
            {
                "formula": formula_label,
                "metric": f"ros_pdp_weeks_1_{split}",
                "r": rest_of_season_corr_metric(d, pdp_col, split_week=split),
            }
        )
        if not pbp_seasons:
            continue
        epa_e = team_early_epa_per_play(drives, pbp_seasons, split_week=split, no_cache=no_cache)
        be = bench.merge(epa_e, on=["team", "season"], how="left")
        cur = be[["team", "season", "epa_early"]].copy()
        late = (
            d.loc[d["week"] > split]
            .groupby(["posteam", "season"])
            .agg(pts=("drive_points_scored", "sum"), n=("game_id", "count"))
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        late["ppd_ros"] = late["pts"] / late["n"].replace(0, np.nan)
        m = cur.merge(late[["team", "season", "ppd_ros"]], on=["team", "season"])
        rows.append(
            {
                "formula": formula_label,
                "metric": f"ros_epa_early_weeks_1_{split}",
                "r": _corr(m["epa_early"], m["ppd_ros"]),
            }
        )

    weekly = week_level_offense(d, exclude_kneels=True)
    pdp_ac = weekly.groupby(["team", "season"])["off_pdp_mean"].apply(autocorrelation_series).mean()
    ppd_ac = weekly.groupby(["team", "season"])["ppd"].apply(autocorrelation_series).mean()
    rows.append({"formula": formula_label, "metric": "stability_pdp_weekly", "r": float(pdp_ac)})
    rows.append({"formula": formula_label, "metric": "stability_ppd_weekly", "r": float(ppd_ac)})

    rows.append(
        {
            "formula": formula_label,
            "metric": "drive_level_mean_pdp",
            "r": float(d[pdp_col].mean()),
        }
    )
    rows.append(
        {
            "formula": formula_label,
            "metric": "drive_level_mean_ppd",
            "r": float(d["drive_points_scored"].mean()),
        }
    )
    rows.append(
        {
            "formula": formula_label,
            "metric": "team_level_mean_pdp",
            "r": float(bench["pdp_mean"].mean()),
        }
    )

    return pd.DataFrame(rows)


def write_metrics_reports(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    pdp_col: str = "off_pdp",
    no_cache: bool = False,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Write full, holdout, and tune metric CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full = build_metrics_table(
        drives, team_season, seasons, pdp_col=pdp_col, no_cache=no_cache
    )
    holdout = build_metrics_table(
        drives,
        team_season,
        seasons,
        pdp_col=pdp_col,
        no_cache=no_cache,
        season_filter=(2020, 2024),
    )
    tune = build_metrics_table(
        drives,
        team_season,
        seasons,
        pdp_col=pdp_col,
        no_cache=no_cache,
        season_filter=(2013, 2017),
    )
    full.to_csv(output_dir / "metrics_report_all.csv", index=False)
    holdout.to_csv(output_dir / "metrics_report_holdout_2020_2024.csv", index=False)
    tune.to_csv(output_dir / "metrics_report_tune_2013_2017.csv", index=False)
    return full
