"""Defense-lens mirrored features and defensive PDP scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from .constants import DEFENSE_FORMULA_FEATURES


def offense_to_defense_features(off: pl.DataFrame) -> pl.DataFrame:
    """
    Map offense-lens drive table to defense-lens columns.

    Higher defensive PDP = better defense. Progress allowed hurts defense scores
    via ``progress_prevented`` (= offense progress_yards; ridge learns negative weight
    or we negate — training uses -opp_pts label).
    """
    pdf = off.to_pandas()
    d = pl.DataFrame(
        {
            "game_id": pdf["game_id"],
            "fixed_drive": pdf["fixed_drive"],
            "defteam": pdf["defteam"],
            "posteam": pdf["posteam"],
            "season": pdf["season"],
            "week": pdf["week"],
            "is_kneel_only_drive": pdf["is_kneel_only_drive"],
            "progress_prevented": pdf["progress_yards"],
            "progress_pct_prevented": pdf["progress_pct"],
            "yards_allowed": pdf["yards_total"],
            "yards_allowed_early_down": pdf["yards_early_down"],
            "yards_allowed_late_down": pdf["yards_late_down"],
            "plays_faced": pdf["plays"],
            "yards_per_play_allowed": pdf["yards_per_play"],
            "first_downs_allowed_total": pdf["first_downs_total"],
            "first_downs_allowed_on_1st": pdf["first_downs_on_1st"],
            "first_downs_allowed_on_2nd": pdf["first_downs_on_2nd"],
            "first_downs_allowed_on_3rd": pdf["first_downs_on_3rd"],
            "first_downs_allowed_on_4th": pdf["first_downs_on_4th"],
            "negative_play_forced_rate": pdf["negative_play_rate"],
            "sack_forced_count": pdf["sack_count"],
            "sack_forced_rate": pdf["sack_rate"],
            "tfl_forced_count": pdf["tfl_count"],
            "tfl_forced_rate": pdf["tfl_rate"],
            "def_penalty_yards_committed": pdf["def_penalty_yards"],
            "def_penalty_count_committed": pdf.get(
                "def_penalty_auto_first_count", pd.Series(0, index=pdf.index)
            ),
            "off_penalty_yards_drawn": pdf["off_penalty_yards"],
            "off_penalty_auto_first_drawn": pdf["def_penalty_auto_first_count"].astype(np.float32),
            "off_penalty_yards_0_5_drawn": pdf["off_penalty_yards"].clip(upper=5),
            "off_penalty_yards_6_15_drawn": 0.0,
            "off_penalty_yards_16_plus_drawn": 0.0,
            "penalty_net_yards_def": -pdf["penalty_net_yards"],
            "explosive_play_count_allowed": pdf["explosive_play_count"],
            "explosive_yards_share_allowed": pdf["explosive_yards_share"],
            "max_play_yard_share_allowed": pdf["max_play_yard_share"],
            "unsustained_explosion_allowed": pdf["unsustained_explosion"],
            "turnover_forced_interception": pdf["turnover_interception"],
            "turnover_forced_fumble": pdf["turnover_fumble_lost"],
            "turnover_forced_on_downs": pdf["turnover_on_downs"],
            "turnover_forced_other": pdf["turnover_other"],
            "turnover_after_progress_allowed": pdf["turnover_after_progress"],
            "turnover_bonus_interaction": (
                pdf["turnover_interception"]
                + pdf["turnover_fumble_lost"]
                + pdf["turnover_on_downs"]
            )
            * pdf["turnover_after_progress"],
            "start_yardline_100": pdf["start_yardline_100"],
            "score_differential": -pdf["score_differential"].fillna(0),
            "qtr": pdf["qtr"],
            "game_seconds_remaining": pdf["game_seconds_remaining"],
        }
    )
    if "off_penalty_count" in pdf.columns:
        d = d.with_columns(
            off_penalty_auto_first_drawn=pl.lit(0.0),
            def_penalty_count_committed=pl.Series(pdf["def_penalty_auto_first_count"].astype(np.float32)),
        )
    return d
