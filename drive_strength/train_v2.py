"""Walk-forward training for formula_v2 (S+1 predictive target)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .constants import (
    DEFENSE_FORMULA_FEATURES_V2,
    DRIVE_CALIBRATION_TARGET,
    FORMULA_V2_PATH,
    OFFENSE_FORMULA_FEATURES_V2,
    TRAIN_FIT_SEASONS,
    TRAIN_TUNE_SEASONS,
    TRAIN_VALIDATE_SEASONS,
)
from .data import load_pbp
from .defense import offense_to_defense_features
from .drives import attach_scrimmage_plays, build_drive_index
from .features import build_offense_drive_features
from .formula import save_formula
from .formula_constraints import enforce_narrative_constraints_v2
from .labels import (
    add_drive_points_column,
    build_defensive_predictive_labels,
    build_predictive_labels,
)

logger = logging.getLogger(__name__)

ALPHAS = [0.1, 1.0, 10.0, 100.0, 500.0]


def _team_season_means(
    pdf: pd.DataFrame,
    features: list[str],
    team_col: str,
) -> pd.DataFrame:
    cols = [c for c in features if c in pdf.columns]
    return (
        pdf.groupby([team_col, "season"], observed=True)[cols]
        .mean()
        .reset_index()
    )


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _fit_ridge(
    merged: pd.DataFrame,
    features: list[str],
    label_col: str,
    *,
    alpha: float,
) -> tuple[dict[str, float], float, dict[str, dict[str, float]]]:
    avail = [f for f in features if f in merged.columns]
    sub = merged.dropna(subset=[label_col])
    raw_x = sub[avail].fillna(0.0).to_numpy(dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}
    x = raw_x.copy()
    for i, name in enumerate(avail):
        mu = float(np.mean(raw_x[:, i]))
        sd = float(np.std(raw_x[:, i]))
        if sd < 1e-8:
            sd = 1.0
        stats[name] = {"mean": mu, "std": sd}
        x[:, i] = (raw_x[:, i] - mu) / sd
    y = sub[label_col].to_numpy(dtype=np.float64)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x, y)
    coefs = {f: float(c) for f, c in zip(avail, model.coef_)}
    return coefs, float(model.intercept_), stats


def _predict(
    df: pd.DataFrame,
    features: list[str],
    coefs: dict[str, float],
    intercept: float,
    *,
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    avail = [f for f in features if f in df.columns]
    x = df[avail].fillna(0.0).to_numpy(dtype=np.float64)
    if feature_stats:
        for i, name in enumerate(avail):
            st = feature_stats.get(name)
            if st:
                x[:, i] = (x[:, i] - st["mean"]) / st["std"]
    w = np.array([coefs.get(f, 0.0) for f in avail], dtype=np.float64)
    return intercept + x @ w


def _evaluate_s1(
    team_feats: pd.DataFrame,
    features: list[str],
    coefs: dict[str, float],
    intercept: float,
    season_range: tuple[int, int],
    label_col: str = "ppd_next_season",
    *,
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> float:
    lo, hi = season_range
    m = team_feats[(team_feats["season"] >= lo) & (team_feats["season"] <= hi)].dropna(
        subset=[label_col]
    )
    if len(m) < 10:
        return float("nan")
    pred = _predict(m, features, coefs, intercept, feature_stats=feature_stats)
    return _corr(pred, m[label_col].to_numpy())


def _evaluate_ros(
    drives: pd.DataFrame,
    features: list[str],
    coefs: dict[str, float],
    intercept: float,
    season_range: tuple[int, int],
    *,
    split: int = 8,
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> float:
    lo, hi = season_range
    d = drives[(drives["season"] >= lo) & (drives["season"] <= hi)]
    early = (
        d[d["week"] <= split]
        .groupby(["posteam", "season"], observed=True)[features]
        .mean()
        .reset_index()
    )
    late = (
        d[d["week"] > split]
        .groupby(["posteam", "season"], observed=True)
        .agg(pts=("drive_points_scored", "sum"), n=("game_id", "count"))
        .reset_index()
    )
    late["ppd_ros"] = late["pts"] / late["n"].replace(0, np.nan)
    m = early.merge(late[["posteam", "season", "ppd_ros"]], on=["posteam", "season"])
    if len(m) < 10:
        return float("nan")
    pred = _predict(m, features, coefs, intercept, feature_stats=feature_stats)
    return _corr(pred, m["ppd_ros"].to_numpy())


def _drive_level_calibration(
    drives: pd.DataFrame,
    features: list[str],
    coefs: dict[str, float],
    intercept: float,
    season_range: tuple[int, int],
    *,
    target: float = DRIVE_CALIBRATION_TARGET,
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    lo, hi = season_range
    d = drives[(drives["season"] >= lo) & (drives["season"] <= hi)]
    raw = _predict(d, features, coefs, intercept, feature_stats=feature_stats)
    mean_raw = float(np.nanmean(raw))
    scale = target / mean_raw if mean_raw > 0 else 1.0
    return {"scale": scale, "shift": 0.0}


def train_formula_v2(
    *,
    output_path: Path = FORMULA_V2_PATH,
    no_cache: bool = False,
) -> dict:
    fit_lo, fit_hi = TRAIN_FIT_SEASONS
    tune_lo, tune_hi = TRAIN_TUNE_SEASONS
    val_lo, val_hi = TRAIN_VALIDATE_SEASONS
    train_end = val_hi

    seasons = list(range(fit_lo, train_end + 1))
    logger.info("Training formula_v2 on seasons %s-%s", fit_lo, train_end)

    pbp = load_pbp(seasons, no_cache=no_cache)
    drives = build_drive_index(pbp)
    joined = attach_scrimmage_plays(pbp, drives)
    off_feat = build_offense_drive_features(joined)
    off_pdf = add_drive_points_column(off_feat).to_pandas()

    pred_labels = build_predictive_labels(off_feat)
    off_team_all = _team_season_means(off_pdf, OFFENSE_FORMULA_FEATURES_V2, "posteam")
    off_team_all = off_team_all.merge(
        pred_labels.rename(columns={"team": "posteam"}),
        on=["posteam", "season"],
        how="left",
    )

    best_alpha = 10.0
    best_s1 = -1.0
    best_ros = -1.0
    best_score = -1.0
    for alpha in ALPHAS:
        fit_mask = (off_team_all["season"] >= fit_lo) & (off_team_all["season"] <= fit_hi)
        fit_df = off_team_all.loc[fit_mask].dropna(subset=["ppd_next_season"])
        coefs, intercept, stats = _fit_ridge(
            fit_df, OFFENSE_FORMULA_FEATURES_V2, "ppd_next_season", alpha=alpha
        )
        s1 = _evaluate_s1(
            off_team_all,
            OFFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (tune_lo, tune_hi),
            feature_stats=stats,
        )
        ros = _evaluate_ros(
            off_pdf,
            OFFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (tune_lo, tune_hi),
            feature_stats=stats,
        )
        score = s1 + 0.15 * ros if np.isfinite(s1) and np.isfinite(ros) else s1
        logger.info("alpha=%s tune S+1 r=%.4f ROS r=%.4f score=%.4f", alpha, s1, ros, score)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_s1 = s1
            best_ros = ros
            best_alpha = alpha

    logger.info(
        "Selected alpha=%s (tune S+1 r=%.4f ROS r=%.4f)",
        best_alpha,
        best_s1,
        best_ros,
    )

    final_mask = off_team_all["season"] <= val_hi
    final_df = off_team_all.loc[final_mask].dropna(subset=["ppd_next_season"])
    off_coefs, off_intercept, off_stats = _fit_ridge(
        final_df, OFFENSE_FORMULA_FEATURES_V2, "ppd_next_season", alpha=best_alpha
    )

    val_s1 = _evaluate_s1(
        off_team_all,
        OFFENSE_FORMULA_FEATURES_V2,
        off_coefs,
        off_intercept,
        (val_lo, val_hi),
        feature_stats=off_stats,
    )
    logger.info("Validate 2018-2019 S+1 r=%.4f", val_s1)

    off_cal = _drive_level_calibration(
        off_pdf,
        OFFENSE_FORMULA_FEATURES_V2,
        off_coefs,
        off_intercept,
        (fit_lo, val_hi),
        feature_stats=off_stats,
    )

    # Defense: next-season opp PPD (lower allowed = better → train on -ppd_next)
    def_feat = offense_to_defense_features(off_feat).to_pandas()
    def_labels = build_defensive_predictive_labels(off_feat)
    def_team = _team_season_means(def_feat, DEFENSE_FORMULA_FEATURES_V2, "defteam")
    def_team = def_team.merge(
        def_labels.rename(columns={"team": "defteam"}),
        on=["defteam", "season"],
        how="left",
    )

    best_def_alpha = 10.0
    best_def_s1 = -1.0
    for alpha in ALPHAS:
        fit_mask = (def_team["season"] >= fit_lo) & (def_team["season"] <= fit_hi)
        fit_df = def_team.loc[fit_mask].dropna(subset=["neg_opp_ppd_next"])
        coefs, intercept, def_stats = _fit_ridge(
            fit_df, DEFENSE_FORMULA_FEATURES_V2, "neg_opp_ppd_next", alpha=alpha
        )
        s1 = _evaluate_s1(
            def_team,
            DEFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (tune_lo, tune_hi),
            label_col="neg_opp_ppd_next",
            feature_stats=def_stats,
        )
        if np.isfinite(s1) and s1 > best_def_s1:
            best_def_s1 = s1
            best_def_alpha = alpha

    def_final = def_team.loc[def_team["season"] <= val_hi].dropna(subset=["neg_opp_ppd_next"])
    def_coefs, def_intercept, def_stats = _fit_ridge(
        def_final, DEFENSE_FORMULA_FEATURES_V2, "neg_opp_ppd_next", alpha=best_def_alpha
    )
    def_cal = _drive_level_calibration(
        def_feat,
        DEFENSE_FORMULA_FEATURES_V2,
        def_coefs,
        def_intercept,
        (fit_lo, val_hi),
        target=1.5,
        feature_stats=def_stats,
    )

    formula = enforce_narrative_constraints_v2(
        {
            "version": "v2",
            "train_seasons": [fit_lo, train_end],
            "train_objective": "ppd_next_season",
            "tune_s1_r": best_s1,
            "tune_ros_r": best_ros,
            "validate_s1_r": val_s1,
            "ridge_alpha_offense": best_alpha,
            "ridge_alpha_defense": best_def_alpha,
            "offense": {
                "features": OFFENSE_FORMULA_FEATURES_V2,
                "coefficients": off_coefs,
                "intercept": off_intercept,
                "feature_stats": off_stats,
                "calibration": off_cal,
            },
            "defense": {
                "features": DEFENSE_FORMULA_FEATURES_V2,
                "coefficients": def_coefs,
                "intercept": def_intercept,
                "feature_stats": def_stats,
                "calibration": def_cal,
            },
        }
    )
    save_formula(formula, output_path)
    logger.info("Wrote %s", output_path)
    return formula
