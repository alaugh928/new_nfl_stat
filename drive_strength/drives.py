"""Assemble offensive drives from fixed_drive; phantom filter and kneel tags."""

from __future__ import annotations

import polars as pl

DRIVE_COL = "fixed_drive"
RESULT_COL = "fixed_drive_result"


def _is_scrimmage_expr() -> pl.Expr:
    pt = pl.col("play_type")
    return (
        pl.col("posteam").is_not_null()
        & pt.is_not_null()
        & ~pt.is_in(list({"kickoff", "kick_off", "kickoff_clock", "extra_point", "no_play", "qb_spike"}))
    )


def build_drive_index(pbp: pl.DataFrame) -> pl.DataFrame:
    """
    One row per (game_id, fixed_drive) with posteam/defteam and kneel tag.

    Drops phantom drives (zero possession plays). Does not exclude kneels.
    """
    if DRIVE_COL not in pbp.columns:
        raise KeyError(f"Missing {DRIVE_COL} in PBP")

    plays = pbp.sort(["game_id", "play_id"])
    possession = (
        pl.col("posteam").is_not_null() & pl.col("play_type").is_not_null()
    ).alias("_poss")

    drives = (
        plays.with_columns(possession)
        .group_by(["game_id", DRIVE_COL])
        .agg(
            min_play_id=pl.col("play_id").min(),
            posteam=pl.col("posteam").drop_nulls().first(),
            defteam=pl.col("defteam").drop_nulls().first(),
            season=pl.col("season").first(),
            week=pl.col("week").first(),
            drive_result_raw=pl.col(RESULT_COL).last() if RESULT_COL in plays.columns else pl.lit(None),
            n_possession_plays=pl.col("_poss").sum(),
            scrimmage_play_types=pl.col("play_type").filter(_is_scrimmage_expr()),
        )
        .filter(pl.col("n_possession_plays") > 0)
        .drop("n_possession_plays")
    )

    drives = drives.with_columns(
        is_kneel_only_drive=pl.when(pl.col("scrimmage_play_types").list.len() == 0)
        .then(pl.lit(False))
        .otherwise(
            pl.col("scrimmage_play_types").list.eval(
                pl.element().is_in(["qb_kneel", "qb_kneel_clock"])
            ).list.all()
        ),
    ).drop("scrimmage_play_types")

    return drives.sort(["game_id", "min_play_id"])


def attach_scrimmage_plays(pbp: pl.DataFrame, drives: pl.DataFrame) -> pl.DataFrame:
    """Join scrimmage play rows to drive keys for feature engineering."""
    scrim = pbp.filter(_is_scrimmage_expr()).sort(["game_id", "play_id"])
    return scrim.join(
        drives.select(
            "game_id",
            DRIVE_COL,
            "posteam",
            "defteam",
            "season",
            "week",
            "drive_result_raw",
            "is_kneel_only_drive",
            "min_play_id",
        ),
        on=["game_id", DRIVE_COL],
        how="inner",
    )
