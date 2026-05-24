"""Head-to-head PDP vs EPA/play benchmarks (honest reporting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..aggregate import aggregate_team_season
from ..constants import OUTPUT_DIR, TRAIN_HOLDOUT_SEASONS
from ..labels import team_season_offense_labels
from ..plots.metrics import team_epa_per_play
from .metrics_report import build_metrics_table, season_over_season_for_column, _corr


def _z(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    sd = s.std()
    if sd < 1e-12:
        return s * 0.0
    return (s - s.mean()) / sd


def build_benchmark_table(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    holdout: tuple[int, int] | None = TRAIN_HOLDOUT_SEASONS,
) -> pd.DataFrame:
    """PDP vs EPA metrics for all seasons and holdout slice."""
    rows: list[dict] = []
    for label, filt in [("all", None), ("holdout_2020_2024", holdout)]:
        m = build_metrics_table(
            drives,
            team_season,
            seasons,
            formula_label="v2",
            season_filter=filt,
        )
        for _, r in m.iterrows():
            rows.append({"window": label, "metric": r["metric"], "r": r["r"]})
    return pd.DataFrame(rows)


def build_disagreement_table(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    holdout: tuple[int, int] = TRAIN_HOLDOUT_SEASONS,
    z_threshold: float = 0.25,
) -> pd.DataFrame:
    """PDP vs EPA rank disagreements and next-season PPD correlation."""
    lo, hi = holdout
    ppd = (
        team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=True)
        .to_pandas()
        .rename(columns={"posteam": "team", "team_season_pts_per_drive": "ppd"})
    )
    epa = team_epa_per_play(seasons)
    bench = team_season.merge(ppd[["team", "season", "ppd"]], on=["team", "season"], how="left")
    if "epa_per_play" not in bench.columns:
        bench = bench.merge(epa, on=["team", "season"], how="left")
    bench = bench[(bench["season"] >= lo) & (bench["season"] <= hi)]

    pdp_col = "off_pdp_mean_shrunk" if "off_pdp_mean_shrunk" in bench.columns else "off_pdp_mean"
    bench["pdp_z"] = _z(bench[pdp_col])
    bench["epa_z"] = _z(bench["epa_per_play"])
    bench["next_season"] = bench["season"] + 1
    nxt = bench[["team", "season", "ppd"]].rename(
        columns={"season": "next_season", "ppd": "ppd_next"}
    )
    m = bench.merge(nxt, on=["team", "next_season"], how="inner")

    def _row(name: str, sub: pd.DataFrame, x_col: str) -> dict:
        if len(sub) < 8:
            return {"segment": name, "n": len(sub), "s1_r": float("nan")}
        return {
            "segment": name,
            "n": len(sub),
            "s1_r": _corr(sub[x_col], sub["ppd_next"]),
        }

    rows = [
        _row("all_teams", m, pdp_col),
        _row("all_teams_epa", m, "epa_per_play"),
        _row(f"pdp_ahead_of_epa (z diff > {z_threshold})", m[m["pdp_z"] > m["epa_z"] + z_threshold], pdp_col),
        _row(f"epa_ahead_of_pdp (z diff > {z_threshold})", m[m["epa_z"] > m["pdp_z"] + z_threshold], "epa_per_play"),
    ]
    return pd.DataFrame(rows)


def print_benchmark_report(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
) -> None:
    """Log human-readable benchmark summary."""
    metrics = build_benchmark_table(drives, team_season, seasons)
    disagree = build_disagreement_table(drives, team_season, seasons)

    hold = metrics[metrics["window"] == "holdout_2020_2024"]
    key = hold[hold["metric"].isin(
        [
            "season_over_season_pdp",
            "season_over_season_epa_play",
            "ros_pdp_weeks_1_8",
            "ros_epa_early_weeks_1_8",
            "same_season_vs_ppd",
            "same_season_vs_epa_play",
        ]
    )]

    print("\n=== PDP vs EPA (holdout 2020–2024) ===")
    for _, r in key.iterrows():
        print(f"  {r['metric']:32s}  r = {r['r']:.3f}")

    print("\n=== When PDP and EPA disagree (S+1 to next-year PPD) ===")
    for _, r in disagree.iterrows():
        print(f"  {r['segment']:36s}  n={int(r['n']):3d}  r={r['s1_r']:.3f}")

    pdp_col = "off_pdp_mean_shrunk" if "off_pdp_mean_shrunk" in team_season.columns else "off_pdp_mean"
    lo, hi = TRAIN_HOLDOUT_SEASONS
    ppd = (
        team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=True)
        .to_pandas()
        .rename(columns={"posteam": "team", "team_season_pts_per_drive": "ppd"})
    )
    epa = team_epa_per_play(seasons)
    bench = team_season.merge(ppd[["team", "season", "ppd"]], on=["team", "season"], how="left")
    if "epa_per_play" not in bench.columns:
        bench = bench.merge(epa, on=["team", "season"], how="left")
    h = bench[(bench["season"] >= lo) & (bench["season"] <= hi)]
    print(f"\n  PDP–EPA same-season correlation: {_corr(h[pdp_col], h['epa_per_play']):.3f}")
    print(
        "\n  Takeaway: EPA/play wins overall on holdout. PDP is most useful when "
        "process rank exceeds EPA (see pdp_ahead_of_epa row).\n"
    )


def write_benchmark_reports(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    build_benchmark_table(drives, team_season, seasons).to_csv(
        output_dir / "benchmark_metrics.csv", index=False
    )
    build_disagreement_table(drives, team_season, seasons).to_csv(
        output_dir / "benchmark_disagreement.csv", index=False
    )
