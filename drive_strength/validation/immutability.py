"""Verify drive PDP unchanged across recomputation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..score import score_drives
from ..features import build_offense_drive_features
from ..drives import attach_scrimmage_plays, build_drive_index
from ..data import load_pbp


def check_immutability(seasons: list[int], *, no_cache: bool = False) -> tuple[bool, str]:
    """Score twice; max absolute diff should be 0."""
    pbp = load_pbp(seasons, no_cache=no_cache)
    drives = build_drive_index(pbp)
    joined = attach_scrimmage_plays(pbp, drives)
    feat = build_offense_drive_features(joined)
    a = score_drives(feat)[["game_id", "fixed_drive", "off_pdp", "def_pdp"]]
    b = score_drives(feat)[["game_id", "fixed_drive", "off_pdp", "def_pdp"]]
    diff = np.abs(a["off_pdp"].values - b["off_pdp"].values).max()
    ok = diff == 0.0
    return ok, f"max off_pdp diff={diff}"
