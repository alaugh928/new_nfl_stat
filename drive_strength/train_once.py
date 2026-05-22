"""One-time ridge fit (1999-2019) and export formula_v1.json."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .constants import (
    DEFENSE_FORMULA_FEATURES,
    FORMULA_PATH,
    OFFENSE_FORMULA_FEATURES,
    TRAIN_SEASON_END,
    TRAIN_SEASON_START,
)
from .data import load_pbp
from .defense import offense_to_defense_features
from .drives import attach_scrimmage_plays, build_drive_index
from .features import build_offense_drive_features
from .formula import save_formula
from .formula_constraints import enforce_formula_constraints
from .labels import add_drive_points_column, team_season_defense_labels, team_season_offense_labels

logger = logging.getLogger(__name__)


def _team_season_feature_matrix(
    drives: pd.DataFrame,
    features: list[str],
    team_col: str,
) -> pd.DataFrame:
    cols = [team_col, "season"] + [c for c in features if c in drives.columns]
    return drives.groupby([team_col, "season"], observed=True)[
        [c for c in features if c in drives.columns]
    ].mean().reset_index()


def _fit_side(
    team_features: pd.DataFrame,
    labels: pd.DataFrame,
    team_col: str,
    label_col: str,
    feature_names: list[str],
    *,
    defense_mode: bool = False,
    alpha: float = 10.0,
) -> tuple[dict[str, float], float, dict[str, float]]:
    lab = labels.copy()
    if team_col not in lab.columns and "team" in lab.columns:
        lab = lab.rename(columns={"team": team_col})
    merged = team_features.merge(lab, on=[team_col, "season"], how="inner")

    avail = [f for f in feature_names if f in merged.columns]
    x = merged[avail].fillna(0.0).to_numpy(dtype=np.float64)
    y = merged[label_col].to_numpy(dtype=np.float64)
    if defense_mode:
        # Higher label = better defense (fewer points allowed)
        y = -y

    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xs, y)

    # Unscale coefficients back to raw feature units
    coefs: dict[str, float] = {}
    for i, name in enumerate(avail):
        coefs[name] = float(model.coef_[i] / scaler.scale_[i])
    intercept = float(
        model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    )

    pred = intercept + merged[avail].fillna(0.0).to_numpy() @ np.array(
        [coefs[n] for n in avail]
    )
    league_mean_y = float(np.mean(y))
    league_mean_pred = float(np.mean(pred))
    scale = 1.0 if league_mean_pred == 0 else league_mean_y / league_mean_pred
    calibration = {"scale": scale, "shift": 0.0}
    return coefs, intercept, calibration


def train_formula(
    *,
    train_start: int = TRAIN_SEASON_START,
    train_end: int = TRAIN_SEASON_END,
    output_path: Path = FORMULA_PATH,
    no_cache: bool = False,
) -> dict:
    seasons = list(range(train_start, train_end + 1))
    logger.info("Training PDP formula on seasons %s-%s", train_start, train_end)
    pbp = load_pbp(seasons, no_cache=no_cache)
    drives = build_drive_index(pbp)
    joined = attach_scrimmage_plays(pbp, drives)
    off_feat = build_offense_drive_features(joined)
    off_pdf = add_drive_points_column(off_feat).to_pandas()

    off_labels = team_season_offense_labels(off_feat, exclude_kneels=True).to_pandas()
    def_labels = team_season_defense_labels(off_feat, exclude_kneels=True).to_pandas()

    off_team = _team_season_feature_matrix(off_pdf, OFFENSE_FORMULA_FEATURES, "posteam")
    off_coefs, off_intercept, off_cal = _fit_side(
        off_team,
        off_labels,
        "posteam",
        "team_season_pts_per_drive",
        OFFENSE_FORMULA_FEATURES,
        defense_mode=False,
    )

    def_feat = offense_to_defense_features(off_feat).to_pandas()
    def_team = _team_season_feature_matrix(def_feat, DEFENSE_FORMULA_FEATURES, "defteam")
    def_coefs, def_intercept, def_cal = _fit_side(
        def_team,
        def_labels,
        "defteam",
        "team_season_opp_pts_per_drive",
        DEFENSE_FORMULA_FEATURES,
        defense_mode=True,
    )

    formula = enforce_formula_constraints(
        {
        "version": "v1",
        "train_seasons": [train_start, train_end],
        "offense": {
            "features": OFFENSE_FORMULA_FEATURES,
            "coefficients": off_coefs,
            "intercept": off_intercept,
            "calibration": off_cal,
        },
        "defense": {
            "features": DEFENSE_FORMULA_FEATURES,
            "coefficients": def_coefs,
            "intercept": def_intercept,
            "calibration": def_cal,
        },
    }
    )
    save_formula(formula, output_path)
    logger.info("Wrote frozen formula to %s", output_path)
    return formula
