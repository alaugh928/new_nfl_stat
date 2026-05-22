"""Load and apply frozen PDP formula (formula_v1.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_FORMULA_PATH,
    DEFENSE_FORMULA_FEATURES,
    DEFENSE_FORMULA_FEATURES_V2,
    FORMULA_PATH,
    FORMULA_V2_PATH,
    OFFENSE_FORMULA_FEATURES,
    OFFENSE_FORMULA_FEATURES_V2,
)


def load_formula(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_FORMULA_PATH
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_formula(formula: dict[str, Any], path: Path | None = None) -> None:
    path = path or DEFAULT_FORMULA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(formula, f, indent=2)


def _apply_linear(
    df: pd.DataFrame,
    features: list[str],
    coefs: dict[str, float],
    intercept: float,
    calibration: dict[str, float],
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    raw = np.full(len(df), intercept, dtype=np.float64)
    for name in features:
        if name not in coefs:
            continue
        if name not in df.columns:
            raw += coefs[name] * 0.0
            continue
        col = df[name].fillna(0.0).to_numpy(dtype=np.float64)
        if feature_stats and name in feature_stats:
            st = feature_stats[name]
            sd = float(st.get("std", 1.0)) or 1.0
            col = (col - float(st.get("mean", 0.0))) / sd
        raw += coefs[name] * col
    scale = float(calibration.get("scale", 1.0))
    shift = float(calibration.get("shift", 0.0))
    return (raw * scale + shift).astype(np.float32)


def score_offense_pdp(df: pd.DataFrame, formula: dict[str, Any] | None = None) -> np.ndarray:
    f = formula or load_formula()
    off = f["offense"]
    return _apply_linear(
        df,
        off["features"],
        off["coefficients"],
        float(off["intercept"]),
        off.get("calibration", {"scale": 1.0, "shift": 0.0}),
        off.get("feature_stats"),
    )


def score_defense_pdp(df: pd.DataFrame, formula: dict[str, Any] | None = None) -> np.ndarray:
    f = formula or load_formula()
    de = f["defense"]
    return _apply_linear(
        df,
        de["features"],
        de["coefficients"],
        float(de["intercept"]),
        de.get("calibration", {"scale": 1.0, "shift": 0.0}),
        de.get("feature_stats"),
    )


def offense_feature_names() -> list[str]:
    return list(OFFENSE_FORMULA_FEATURES)


def defense_feature_names() -> list[str]:
    return list(DEFENSE_FORMULA_FEATURES)
