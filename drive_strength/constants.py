"""Frozen training window, cache keys, and PBP column lists."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
FORMULA_PATH = PACKAGE_DIR / "formula_v1.json"
FORMULA_V2_PATH = PACKAGE_DIR / "formula_v2.json"
DEFAULT_FORMULA_PATH = FORMULA_V2_PATH

# Training splits for formula_v2 (walk-forward).
TRAIN_FIT_SEASONS = (1999, 2012)
TRAIN_TUNE_SEASONS = (2013, 2017)
TRAIN_VALIDATE_SEASONS = (2018, 2019)
TRAIN_HOLDOUT_SEASONS = (2020, 2024)
DRIVE_CALIBRATION_TARGET = 2.2
DEFENSE_CALIBRATION_TARGET = 1.5
TRAIN_FEATURE_SPLIT_WEEK = 8
CACHE_DIR = PACKAGE_DIR / "cache"
OUTPUT_DIR = PACKAGE_DIR / "output"

TRAIN_SEASON_START = 1999
TRAIN_SEASON_END = 2019
DEFAULT_SEASONS_END = 2024

RESULTS_CACHE_VERSION = "pdp_v1"
EXPLOSIVE_YARD_THRESHOLD = 15

# Columns requested from nflreadpy (subset of nflverse PBP).
PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "fixed_drive",
    "fixed_drive_result",
    "posteam",
    "defteam",
    "play_type",
    "down",
    "ydstogo",
    "yardline_100",
    "yards_gained",
    "epa",
    "success",
    "first_down",
    "penalty",
    "penalty_team",
    "penalty_type",
    "penalty_yards",
    "interception",
    "fumble_lost",
    "sack",
    "tackled_for_loss",
    "qb_kneel",
    "touchdown",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
    "drive_play_count",
]

# Features entering frozen offensive PDP (diagnostics excluded).
OFFENSE_FORMULA_FEATURES = [
    "progress_yards",
    "progress_pct",
    "yards_total",
    "yards_early_down",
    "yards_late_down",
    "plays",
    "yards_per_play",
    "first_downs_total",
    "first_downs_on_1st",
    "first_downs_on_2nd",
    "first_downs_on_3rd",
    "first_downs_on_4th",
    "negative_play_rate",
    "sack_count",
    "sack_rate",
    "tfl_count",
    "tfl_rate",
    "off_penalty_yards",
    "off_penalty_count",
    "def_penalty_yards",
    "def_penalty_auto_first_count",
    "def_penalty_yards_0_5",
    "def_penalty_yards_6_15",
    "def_penalty_yards_16_plus",
    "penalty_net_yards",
    "explosive_play_count",
    "explosive_yards_share",
    "max_play_yard_share",
    "unsustained_explosion",
    "turnover_interception",
    "turnover_fumble_lost",
    "turnover_on_downs",
    "turnover_other",
    "turnover_after_progress",
    "turnover_penalty_interaction",
    "start_yardline_100",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
]

# Defense: opponent progress hurts; use negated progress + mirrored counts.
DEFENSE_FORMULA_FEATURES = [
    "progress_prevented",
    "progress_pct_prevented",
    "yards_allowed",
    "yards_allowed_early_down",
    "yards_allowed_late_down",
    "plays_faced",
    "yards_per_play_allowed",
    "first_downs_allowed_total",
    "first_downs_allowed_on_1st",
    "first_downs_allowed_on_2nd",
    "first_downs_allowed_on_3rd",
    "first_downs_allowed_on_4th",
    "negative_play_forced_rate",
    "sack_forced_count",
    "sack_forced_rate",
    "tfl_forced_count",
    "tfl_forced_rate",
    "def_penalty_yards_committed",
    "def_penalty_count_committed",
    "off_penalty_yards_drawn",
    "off_penalty_auto_first_drawn",
    "off_penalty_yards_0_5_drawn",
    "off_penalty_yards_6_15_drawn",
    "off_penalty_yards_16_plus_drawn",
    "penalty_net_yards_def",
    "explosive_play_count_allowed",
    "explosive_yards_share_allowed",
    "max_play_yard_share_allowed",
    "unsustained_explosion_allowed",
    "turnover_forced_interception",
    "turnover_forced_fumble",
    "turnover_forced_on_downs",
    "turnover_forced_other",
    "turnover_after_progress_allowed",
    "turnover_bonus_interaction",
    "start_yardline_100",
    "score_differential",
    "qtr",
    "game_seconds_remaining",
]

# Tier A: formula_v2 weighted features (lean process; no yards_total/plays/ypp).
OFFENSE_FORMULA_FEATURES_V2 = [
    "progress_yards",
    "progress_pct",
    "first_downs_total",
    "first_downs_on_1st",
    "first_downs_on_2nd",
    "first_downs_on_3rd",
    "first_downs_on_4th",
    "unsustained_explosion",
    "turnover_interception",
    "turnover_fumble_lost",
    "turnover_on_downs",
    "turnover_after_progress",
    "turnover_penalty_interaction",
    "off_penalty_yards",
    "def_penalty_yards",
    "def_penalty_auto_first_count",
    "def_penalty_yards_16_plus",
    "negative_play_rate",
    "sack_count",
]

OFFENSE_DIAGNOSTIC_FEATURES = [
    "start_yardline_100",
    "best_yardline_100",
    "plays_in_opp40",
    "plays_in_opp20",
    "yards_total",
    "max_play_yard_share",
    "plays",
    "yards_per_play",
]

DEFENSE_FORMULA_FEATURES_V2 = [
    "progress_prevented",
    "progress_pct_prevented",
    "first_downs_allowed_total",
    "first_downs_allowed_on_1st",
    "first_downs_allowed_on_2nd",
    "first_downs_allowed_on_3rd",
    "first_downs_allowed_on_4th",
    "unsustained_explosion_allowed",
    "turnover_forced_interception",
    "turnover_forced_fumble",
    "turnover_forced_on_downs",
    "turnover_after_progress_allowed",
    "turnover_bonus_interaction",
    "def_penalty_yards_committed",
    "off_penalty_yards_drawn",
    "off_penalty_auto_first_drawn",
    "off_penalty_yards_16_plus_drawn",
    "negative_play_forced_rate",
    "sack_forced_count",
]

DEF_AUTO_FIRST_PENALTY_TYPES = frozenset(
    {
        "defensive pass interference",
        "defensive holding",
        "illegal contact",
        "defensive pass interference",
        "pass interference",
        "dpi",
        "holding",
        "illegal use of hands",
        "roughing the passer",
    }
)
