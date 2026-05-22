"""Drive-level signatures from ordered scrimmage plays (offense lens)."""

from __future__ import annotations

import numpy as np
import polars as pl

from .constants import DEF_AUTO_FIRST_PENALTY_TYPES, EXPLOSIVE_YARD_THRESHOLD
from .drives import DRIVE_COL
from .utils import KNEEL_TYPES

# Scrimmage yards_gained: exclude plays where penalty yardage is the primary gain
# (penalty buckets tracked separately).


def _bool_col(name: str) -> pl.Expr:
    c = pl.col(name)
    return (
        (c == True)  # noqa: E712
        | (c == 1)
        | (c == 1.0)
        | (c.cast(pl.Utf8, strict=False) == "1")
    ).cast(pl.Boolean)


def _penalty_type_norm() -> pl.Expr:
    return (
        pl.col("penalty_type")
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .str.strip_chars()
        .fill_null("")
    )


def _is_def_auto_first() -> pl.Expr:
    pt = _penalty_type_norm()
    return pt.str.contains("pass interference") | pt.str.contains("defensive holding") | pt.is_in(
        [x.lower() for x in DEF_AUTO_FIRST_PENALTY_TYPES]
    )


def _drive_play_features(plays: pl.DataFrame) -> pl.DataFrame:
    """Per-play columns used in drive aggregation."""
    yds = pl.col("yards_gained").cast(pl.Float64, strict=False).fill_null(0.0)
    down = pl.col("down").cast(pl.Float64, strict=False)
    is_pen = _bool_col("penalty") if "penalty" in plays.columns else pl.lit(False)
    pen_team = pl.col("penalty_team").cast(pl.Utf8, strict=False)
    pen_yds = pl.col("penalty_yards").cast(pl.Float64, strict=False).fill_null(0.0).abs()
    post = pl.col("posteam").cast(pl.Utf8, strict=False)

    off_pen = is_pen & (pen_team == post)
    def_pen = is_pen & (pen_team != post) & pen_team.is_not_null()

    return plays.with_columns(
        _yds_scrim=pl.when(is_pen).then(0.0).otherwise(yds),
        _early=down.is_in([1.0, 2.0]),
        _late=down.is_in([3.0, 4.0]),
        _explosive=yds >= EXPLOSIVE_YARD_THRESHOLD,
        _neg=(
            (_bool_col("sack") if "sack" in plays.columns else pl.lit(False).cast(pl.Boolean))
            | (yds < 0)
            | (
                _bool_col("tackled_for_loss")
                if "tackled_for_loss" in plays.columns
                else pl.lit(False).cast(pl.Boolean)
            )
        ).cast(pl.Boolean),
        _sack=_bool_col("sack") if "sack" in plays.columns else pl.lit(False),
        _tfl=_bool_col("tackled_for_loss") if "tackled_for_loss" in plays.columns else pl.lit(False),
        _off_pen=off_pen,
        _def_pen=def_pen,
        _off_pen_yds=pl.when(off_pen).then(pen_yds).otherwise(0.0),
        _def_pen_yds=pl.when(def_pen).then(pen_yds).otherwise(0.0),
        _def_auto=def_pen & _is_def_auto_first(),
        _def_pen_0_5=def_pen & (pen_yds <= 5),
        _def_pen_6_15=def_pen & (pen_yds > 5) & (pen_yds <= 15),
        _def_pen_16p=def_pen & (pen_yds > 15),
        _fd=(
            (_bool_col("first_down") if "first_down" in plays.columns else pl.lit(False).cast(pl.Boolean))
            | (
                _bool_col("success")
                if "success" in plays.columns
                else (yds >= pl.col("ydstogo").cast(pl.Float64, strict=False).fill_null(0) * 0.4).cast(
                    pl.Boolean
                )
            )
        ).cast(pl.Boolean),
        _int=_bool_col("interception") if "interception" in plays.columns else pl.lit(False),
        _fum=_bool_col("fumble_lost") if "fumble_lost" in plays.columns else pl.lit(False),
        _yl=pl.col("yardline_100").cast(pl.Float64, strict=False),
    )


def _infer_turnover_on_downs(drive_result: str | None) -> bool:
    if not drive_result:
        return False
    rl = drive_result.lower()
    return "downs" in rl and "turnover" in rl


def build_offense_drive_features(plays_with_drives: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate offense-lens drive signatures.

    ``yards_total`` uses scrimmage yards excluding penalty-flagged plays to avoid
    double-counting with penalty bucket features.
    """
    p = _drive_play_features(plays_with_drives)
    gkeys = ["game_id", DRIVE_COL, "posteam", "defteam", "season", "week", "is_kneel_only_drive", "drive_result_raw"]

    agg = p.group_by(gkeys).agg(
        plays=pl.len(),
        yards_total=pl.col("_yds_scrim").sum(),
        yards_early_down=pl.col("_yds_scrim").filter(pl.col("_early")).sum(),
        yards_late_down=pl.col("_yds_scrim").filter(pl.col("_late")).sum(),
        start_yardline_100=pl.col("_yl").filter(pl.col("_yl").is_finite()).first(),
        best_yardline_100=pl.col("_yl").filter(pl.col("_yl").is_finite()).min(),
        end_yardline_100=pl.col("_yl").filter(pl.col("_yl").is_finite()).last(),
        first_downs_total=pl.col("_fd").sum(),
        first_downs_on_1st=pl.col("_fd").filter(pl.col("down") == 1).sum(),
        first_downs_on_2nd=pl.col("_fd").filter(pl.col("down") == 2).sum(),
        first_downs_on_3rd=pl.col("_fd").filter(pl.col("down") == 3).sum(),
        first_downs_on_4th=pl.col("_fd").filter(pl.col("down") == 4).sum(),
        negative_plays=pl.col("_neg").sum(),
        sack_count=pl.col("_sack").sum(),
        tfl_count=pl.col("_tfl").sum(),
        off_penalty_yards=pl.col("_off_pen_yds").sum(),
        off_penalty_count=pl.col("_off_pen").sum(),
        def_penalty_yards=pl.col("_def_pen_yds").sum(),
        def_penalty_auto_first_count=pl.col("_def_auto").sum(),
        def_penalty_yards_0_5=pl.col("_def_pen_yds").filter(pl.col("_def_pen_0_5")).sum(),
        def_penalty_yards_6_15=pl.col("_def_pen_yds").filter(pl.col("_def_pen_6_15")).sum(),
        def_penalty_yards_16_plus=pl.col("_def_pen_yds").filter(pl.col("_def_pen_16p")).sum(),
        explosive_play_count=pl.col("_explosive").sum(),
        explosive_yards=pl.col("_yds_scrim").filter(pl.col("_explosive")).sum(),
        max_play_yards=pl.col("_yds_scrim").max(),
        turnover_interception=pl.col("_int").max().cast(pl.Int8),
        turnover_fumble_lost=pl.col("_fum").max().cast(pl.Int8),
        turnover_down=pl.when(pl.col("_int") | pl.col("_fum"))
        .then(pl.col("down").cast(pl.Float64, strict=False))
        .otherwise(None)
        .drop_nulls()
        .first(),
        turnover_yardline_100=pl.when(pl.col("_int") | pl.col("_fum"))
        .then(pl.col("_yl"))
        .otherwise(None)
        .drop_nulls()
        .first(),
        score_differential=pl.col("score_differential").cast(pl.Float64, strict=False).first()
        if "score_differential" in p.columns
        else pl.lit(None),
        qtr=pl.col("qtr").cast(pl.Float64, strict=False).first() if "qtr" in p.columns else pl.lit(None),
        game_seconds_remaining=pl.col("game_seconds_remaining").cast(pl.Float64, strict=False).first()
        if "game_seconds_remaining" in p.columns
        else pl.lit(None),
        plays_in_opp40=pl.col("_yl").filter(pl.col("_yl") <= 40).count(),
        plays_in_opp20=pl.col("_yl").filter(pl.col("_yl") <= 20).count(),
    )

    pdf = agg.to_pandas()
    start = pdf["start_yardline_100"].fillna(50.0).clip(1, 99)
    best = pdf["best_yardline_100"].fillna(start)
    pdf["progress_yards"] = (start - best).clip(lower=0)
    denom = np.where(start > 10, start, 10.0)
    pdf["progress_pct"] = (pdf["progress_yards"] / denom).clip(0, 1.5)

    pdf["yards_per_play"] = pdf["yards_total"] / pdf["plays"].replace(0, np.nan)
    pdf["negative_play_rate"] = pdf["negative_plays"] / pdf["plays"].replace(0, np.nan)
    pdf["sack_rate"] = pdf["sack_count"] / pdf["plays"].replace(0, np.nan)
    pdf["tfl_rate"] = pdf["tfl_count"] / pdf["plays"].replace(0, np.nan)
    pdf["penalty_net_yards"] = pdf["def_penalty_yards"] - pdf["off_penalty_yards"]

    ytot = pdf["yards_total"].replace(0, np.nan)
    pdf["explosive_yards_share"] = (pdf["explosive_yards"] / ytot).fillna(0.0)
    pdf["max_play_yard_share"] = (pdf["max_play_yards"] / ytot).fillna(0.0)
    pdf["unsustained_explosion"] = (
        pdf["explosive_yards_share"]
        * pdf["max_play_yard_share"]
        / np.maximum(pdf["plays"].astype(float), 1.0)
    )

    # Turnover on downs from result if not on play row
    tod = pdf["drive_result_raw"].map(_infer_turnover_on_downs).astype(int)
    pdf["turnover_on_downs"] = tod
    pdf["turnover_other"] = 0
    pdf.loc[
        (pdf["turnover_interception"] == 0)
        & (pdf["turnover_fumble_lost"] == 0)
        & (tod == 0),
        "turnover_other",
    ] = 0

    pdf["turnover_after_progress"] = pdf["progress_yards"]
    pdf["turnover_penalty_interaction"] = (
        pdf["turnover_interception"] + pdf["turnover_fumble_lost"] + pdf["turnover_on_downs"]
    ) * pdf["turnover_after_progress"]

    for c in pdf.select_dtypes(include=[np.float64]).columns:
        pdf[c] = pdf[c].astype(np.float32)
    return pl.from_pandas(pdf)
