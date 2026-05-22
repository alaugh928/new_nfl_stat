"""Attach offensive and defensive PDP columns to drive tables."""

from __future__ import annotations

import pandas as pd
import polars as pl

from .defense import offense_to_defense_features
from .formula import load_formula, score_defense_pdp, score_offense_pdp


def score_drives(
    off_features: pl.DataFrame,
    formula: dict | None = None,
) -> pd.DataFrame:
    """Return pandas drive table with ``off_pdp`` and ``def_pdp``."""
    off_pdf = off_features.to_pandas()
    def_pl = offense_to_defense_features(off_features)
    def_pdf = def_pl.to_pandas()
    formula = formula or load_formula()
    off_pdf["off_pdp"] = score_offense_pdp(off_pdf, formula)
    def_pdf["def_pdp"] = score_defense_pdp(def_pdf, formula)
    out = off_pdf.merge(
        def_pdf[["game_id", "fixed_drive", "def_pdp"]],
        on=["game_id", "fixed_drive"],
        how="left",
    )
    return out
