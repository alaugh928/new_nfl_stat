"""Walk-forward training for formula_v2 (S+1 predictive target)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .constants import (
    DEFENSE_CALIBRATION_TARGET,
    DEFENSE_FORMULA_FEATURES_V2,
    DRIVE_CALIBRATION_TARGET,
    FORMULA_V2_PATH,
    OFFENSE_FORMULA_FEATURES_V2,
    TRAIN_FEATURE_SPLIT_WEEK,
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
    *,
    max_week: int | None = None,
) -> pd.DataFrame:
    cols = [c for c in features if c in pdf.columns]
    d = pdf
    if max_week is not None and "week" in d.columns:
        d = d.loc[d["week"] <= max_week]
    return (
        d.groupby([team_col, "season"], observed=True)[cols]
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
    """Affine calibration so league mean drive score ≈ target (preserves rank order)."""
    lo, hi = season_range
    d = drives[(drives["season"] >= lo) & (drives["season"] <= hi)]
    raw = _predict(d, features, coefs, intercept, feature_stats=feature_stats)
    mean_raw = float(np.nanmean(raw))
    if abs(mean_raw) < 1e-8:
        return {"scale": 1.0, "shift": target, "sign": 1.0}
    if mean_raw > 0:
        cal = {"scale": target / mean_raw, "shift": 0.0, "sign": 1.0}
    else:
        cal = {"scale": 1.0, "shift": target - mean_raw, "sign": 1.0}
    # Lift bottom tail so catastrophic drives are not large negative PDP
    adjusted = cal["sign"] * raw * cal["scale"] + cal["shift"]
    p01 = float(np.nanpercentile(adjusted, 1))
    if p01 < 0.05:
        cal["shift"] = float(cal["shift"]) + (0.05 - p01)
    return cal


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
    split_wk = TRAIN_FEATURE_SPLIT_WEEK
    off_team_all = _team_season_means(off_pdf, OFFENSE_FORMULA_FEATURES_V2, "posteam")
    off_team_all = off_team_all.merge(
        pred_labels.rename(columns={"team": "posteam"}),
        on=["posteam", "season"],
        how="left",
    )

    best_alpha = 10.0
    best_tune_s1 = -1.0
    best_ros = -1.0
    best_val_s1 = -1.0
    for alpha in ALPHAS:
        fit_mask = (off_team_all["season"] >= fit_lo) & (off_team_all["season"] <= tune_hi)
        fit_df = off_team_all.loc[fit_mask].dropna(subset=["ppd_next_season"])
        coefs, intercept, stats = _fit_ridge(
            fit_df, OFFENSE_FORMULA_FEATURES_V2, "ppd_next_season", alpha=alpha
        )
        tune_s1 = _evaluate_s1(
            off_team_all,
            OFFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (tune_lo, tune_hi),
            feature_stats=stats,
        )
        val_s1 = _evaluate_s1(
            off_team_all,
            OFFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (val_lo, val_hi),
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
        logger.info(
            "alpha=%s tune S+1=%.4f val S+1=%.4f ROS=%.4f",
            alpha,
            tune_s1,
            val_s1,
            ros,
        )
        if np.isfinite(val_s1) and (
            val_s1 > best_val_s1 or (val_s1 == best_val_s1 and tune_s1 > best_tune_s1)
        ):
            best_val_s1 = val_s1
            best_tune_s1 = tune_s1
            best_ros = ros
            best_alpha = alpha

    logger.info(
        "Selected alpha=%s (tune S+1 r=%.4f validate S+1 r=%.4f ROS r=%.4f)",
        best_alpha,
        best_tune_s1,
        best_val_s1,
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
        fit_df = def_team.loc[fit_mask].dropna(subset=["def_quality_next"])
        coefs, intercept, def_stats = _fit_ridge(
            fit_df, DEFENSE_FORMULA_FEATURES_V2, "def_quality_next", alpha=alpha
        )
        s1 = _evaluate_s1(
            def_team,
            DEFENSE_FORMULA_FEATURES_V2,
            coefs,
            intercept,
            (tune_lo, tune_hi),
            label_col="def_quality_next",
            feature_stats=def_stats,
        )
        if np.isfinite(s1) and s1 > best_def_s1:
            best_def_s1 = s1
            best_def_alpha = alpha

    def_final = def_team.loc[def_team["season"] <= val_hi].dropna(subset=["def_quality_next"])
    def_coefs, def_intercept, def_stats = _fit_ridge(
        def_final, DEFENSE_FORMULA_FEATURES_V2, "def_quality_next", alpha=best_def_alpha
    )
    def_cal = _drive_level_calibration(
        def_feat,
        DEFENSE_FORMULA_FEATURES_V2,
        def_coefs,
        def_intercept,
        (fit_lo, val_hi),
        target=DEFENSE_CALIBRATION_TARGET,
        feature_stats=def_stats,
    )

    formula = enforce_narrative_constraints_v2(
        {
            "version": "v2",
            "train_seasons": [fit_lo, train_end],
            "train_objective": "ppd_next_season",
            "tune_s1_r": best_tune_s1,
            "tune_ros_r": best_ros,
            "validate_s1_r": val_s1,
            "alpha_select_metric": "validate_s1",
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
