"""End-to-end PDP computation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .aggregate import aggregate_team_season, aggregate_team_week
from .constants import DEFAULT_FORMULA_PATH, DEFAULT_SEASONS_END, OUTPUT_DIR
from .data import load_pbp
from .drives import attach_scrimmage_plays, build_drive_index
from .features import build_offense_drive_features
from .formula import load_formula
from .labels import add_drive_points_column
from .score import score_drives

logger = logging.getLogger(__name__)


def run_pipeline(
    seasons: list[int] | None = None,
    *,
    no_cache: bool = False,
    output_dir: Path | None = None,
    exclude_kneels: bool = False,
    formula_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load PBP, build features, score PDP, write CSVs.

    Returns (drives_df, team_week_df, team_season_df).
    """
    fpath = formula_path or DEFAULT_FORMULA_PATH
    if not fpath.exists():
        raise FileNotFoundError(
            f"Missing frozen formula at {fpath}. Run: python -m drive_strength train-v2"
        )
    formula = load_formula(fpath)

    if seasons is None:
        seasons = [DEFAULT_SEASONS_END - 2, DEFAULT_SEASONS_END - 1, DEFAULT_SEASONS_END]

    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pbp = load_pbp(seasons, no_cache=no_cache)
    drives = build_drive_index(pbp)
    joined = attach_scrimmage_plays(pbp, drives)
    off_feat = build_offense_drive_features(joined)
    off_feat = add_drive_points_column(off_feat)
    drives_scored = score_drives(off_feat, formula=formula)

    week_df = aggregate_team_week(drives_scored, exclude_kneels=exclude_kneels)
    season_df = aggregate_team_season(drives_scored, exclude_kneels=exclude_kneels)

    from .plots.metrics import team_epa_per_play

    epa = team_epa_per_play(seasons)
    season_df = season_df.merge(epa, on=["team", "season"], how="left")
    pdp_col = (
        "off_pdp_mean_shrunk"
        if "off_pdp_mean_shrunk" in season_df.columns
        else "off_pdp_mean"
    )
    if pdp_col in season_df.columns and "epa_per_play" in season_df.columns:
        pdp_z = (season_df[pdp_col] - season_df[pdp_col].mean()) / season_df[pdp_col].std()
        epa_z = (season_df["epa_per_play"] - season_df["epa_per_play"].mean()) / season_df[
            "epa_per_play"
        ].std()
        season_df["process_vs_epa"] = pdp_z - epa_z

    drives_scored.to_csv(out_dir / "drives_pdp.csv", index=False)
    week_df.to_csv(out_dir / "team_week_pdp.csv", index=False)
    season_df.to_csv(out_dir / "team_season_pdp.csv", index=False)

    logger.info(
        "Wrote %s drives, %s team-weeks, %s team-seasons",
        len(drives_scored),
        len(week_df),
        len(season_df),
    )
    return drives_scored, week_df, season_df
