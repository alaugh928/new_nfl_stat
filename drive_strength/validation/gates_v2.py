"""Product gates for formula_v2 holdout evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..constants import TRAIN_FEATURE_SPLIT_WEEK, TRAIN_HOLDOUT_SEASONS
from ..labels import team_season_offense_labels
from ..plots.metrics import team_epa_per_play
from ..validation.metrics_report import season_over_season_for_column
from ..validation.narrative import check_team_a_vs_team_b, check_turnover_narratives
from .predictive import rest_of_season_corr


def run_gates(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    holdout: tuple[int, int] = TRAIN_HOLDOUT_SEASONS,
) -> tuple[bool, list[tuple[str, bool, str]]]:
    lo, hi = holdout
    ppd = (
        team_season_offense_labels(pl.from_pandas(drives), exclude_kneels=True)
        .rename({"posteam": "team", "team_season_pts_per_drive": "ppd"})
        .to_pandas()
    )
    split = TRAIN_FEATURE_SPLIT_WEEK
    pdp_col = (
        "off_pdp_mean_shrunk"
        if "off_pdp_mean_shrunk" in team_season.columns
        else "off_pdp_mean"
    )
    bench = team_season.rename(columns={pdp_col: "pdp_mean"}).merge(
        ppd, on=["team", "season"], how="left"
    )
    bench = bench.merge(team_epa_per_play(seasons), on=["team", "season"], how="left")
    h = bench[(bench["season"] >= lo) & (bench["season"] <= hi)]

    results: list[tuple[str, bool, str]] = []

    ok_a, msg_a = check_team_a_vs_team_b()
    results.append(("team_a_vs_team_b", ok_a, msg_a))

    ok_t, msg_t = check_turnover_narratives()
    results.append(("turnover_narratives", ok_t, msg_t))

    pdp_s1 = season_over_season_for_column(bench, "pdp_mean")
    epa_s1 = season_over_season_for_column(bench, "epa_per_play")
    ok_s1 = bool(np.isfinite(pdp_s1) and np.isfinite(epa_s1) and pdp_s1 >= epa_s1)
    results.append(
        (
            "s1_vs_epa",
            ok_s1,
            f"PDP S+1 r={pdp_s1:.3f} vs EPA/play {epa_s1:.3f}",
        )
    )

    d_hold = drives[(drives["season"] >= lo) & (drives["season"] <= hi)]
    drive_mean = float(d_hold["off_pdp"].mean())
    ok_scale = 1.8 <= drive_mean <= 2.8
    results.append(
        ("drive_scale", ok_scale, f"holdout mean off_pdp={drive_mean:.3f} (target ~2.2)"),
    )

    def_mean = float(d_hold["def_pdp"].mean())
    def_min = float(d_hold["def_pdp"].min())
    ok_def = 1.0 <= def_mean <= 2.5 and def_min >= -0.05
    results.append(
        (
            "def_drive_scale",
            ok_def,
            f"holdout mean def_pdp={def_mean:.3f} min={def_min:.3f} (target ~1.5, higher=better)",
        )
    )

    ros_r, ros_msg = rest_of_season_corr(drives, split_week=8)
    from ..validation.metrics_report import team_early_epa_per_play, _corr

    try:
        epa_e = team_early_epa_per_play(drives, seasons, split_week=8)
        if "drive_points_scored" not in drives.columns:
            from ..labels import add_drive_points_column

            drives = add_drive_points_column(pl.from_pandas(drives)).to_pandas()
        late = (
            drives.loc[drives["week"] > 8]
            .groupby(["posteam", "season"])
            .agg(pts=("drive_points_scored", "sum"), n=("game_id", "count"))
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        late["ppd_ros"] = late["pts"] / late["n"].replace(0, np.nan)
        m = epa_e.merge(late[["team", "season", "ppd_ros"]], on=["team", "season"])
        epa_ros_fair = _corr(m["epa_early"], m["ppd_ros"])
    except Exception:
        epa_ros_fair = float("nan")

    results.append(
        (
            "ros_pdp",
            True,
            f"{ros_msg}; fair EPA early→ROS r={epa_ros_fair:.3f}",
        )
    )

    critical = (
        "team_a_vs_team_b",
        "turnover_narratives",
        "drive_scale",
        "def_drive_scale",
    )
    all_ok = all(r[1] for r in results if r[0] in critical)
    return all_ok, results
