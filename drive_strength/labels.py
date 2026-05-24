"""Team-season points per drive labels for training (not per-drive outcomes)."""

from __future__ import annotations

import polars as pl


def _drive_points_from_result(result: str | None) -> float:
    if not result:
        return 0.0
    rl = result.lower().strip()
    if "touchdown" in rl and "defensive" not in rl:
        return 7.0
    if "field goal" in rl and "miss" not in rl:
        return 3.0
    return 0.0


def add_drive_points_column(drives: pl.DataFrame) -> pl.DataFrame:
    """Diagnostic only: actual points on drive from result (not used in PDP formula)."""
    return drives.with_columns(
        drive_points_scored=pl.col("drive_result_raw")
        .map_elements(_drive_points_from_result, return_dtype=pl.Float64)
        .alias("drive_points_scored")
    )


def team_season_offense_labels(
    drives: pl.DataFrame,
    *,
    exclude_kneels: bool = True,
) -> pl.DataFrame:
    """
    Offensive team-season pts/drive for ridge training labels.

    Uses sum(drive_points) / n(drives); optional kneel exclusion in denominator.
    """
    d = drives
    if exclude_kneels and "is_kneel_only_drive" in d.columns:
        d = d.filter(~pl.col("is_kneel_only_drive"))

    if "drive_points_scored" not in d.columns:
        d = add_drive_points_column(d)

    return (
        d.group_by(["posteam", "season"])
        .agg(
            drives=pl.len(),
            points=pl.col("drive_points_scored").sum(),
        )
        .with_columns(team_season_pts_per_drive=pl.col("points") / pl.col("drives"))
        .select("posteam", "season", "team_season_pts_per_drive", "drives", "points")
    )


def team_season_defense_labels(
    drives: pl.DataFrame,
    *,
    exclude_kneels: bool = True,
) -> pl.DataFrame:
    """Opponent points per drive allowed (defteam lens) for defensive training."""
    d = drives
    if exclude_kneels and "is_kneel_only_drive" in d.columns:
        d = d.filter(~pl.col("is_kneel_only_drive"))

    if "drive_points_scored" not in d.columns:
        d = add_drive_points_column(d)

    return (
        d.group_by(["defteam", "season"])
        .agg(
            drives_faced=pl.len(),
            opponent_points=pl.col("drive_points_scored").sum(),
        )
        .with_columns(
            team_season_opp_pts_per_drive=pl.col("opponent_points") / pl.col("drives_faced")
        )
        .select(
            "defteam",
            "season",
            "team_season_opp_pts_per_drive",
            "drives_faced",
            "opponent_points",
        )
    )


def build_predictive_labels(drives_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Per team-season: ppd, ppd_next_season, ppd_ros (weeks 9-17), ppd_early (weeks 1-8).

    Requires week column; excludes kneels from PPD aggregates.
    """
    d = drives_pl.filter(~pl.col("is_kneel_only_drive")) if "is_kneel_only_drive" in drives_pl.columns else drives_pl
    if "drive_points_scored" not in d.columns:
        d = add_drive_points_column(d)

    season_ppd = (
        d.group_by(["posteam", "season"])
        .agg(
            ppd=(pl.col("drive_points_scored").sum() / pl.col("drive_points_scored").count()),
        )
        .rename({"posteam": "team"})
    )

    early = (
        d.filter(pl.col("week") <= 8)
        .group_by(["posteam", "season"])
        .agg(
            pts_early=pl.col("drive_points_scored").sum(),
            n_early=pl.len(),
        )
        .with_columns(ppd_early=pl.col("pts_early") / pl.col("n_early"))
        .rename({"posteam": "team"})
        .select("team", "season", "ppd_early")
    )

    late = (
        d.filter(pl.col("week") > 8)
        .group_by(["posteam", "season"])
        .agg(
            pts_late=pl.col("drive_points_scored").sum(),
            n_late=pl.len(),
        )
        .with_columns(ppd_ros=pl.col("pts_late") / pl.col("n_late"))
        .rename({"posteam": "team"})
        .select("team", "season", "ppd_ros")
    )

    out = season_ppd.join(early, on=["team", "season"], how="left").join(
        late, on=["team", "season"], how="left"
    )
    pdf = out.sort(["team", "season"]).to_pandas()
    pdf["ppd_next_season"] = pdf.groupby("team")["ppd"].shift(-1)
    return pdf


def build_defensive_predictive_labels(drives_pl: pl.DataFrame) -> pl.DataFrame:
    """Per defteam-season: opp_ppd allowed and next-season value."""
    d = drives_pl.filter(~pl.col("is_kneel_only_drive")) if "is_kneel_only_drive" in drives_pl.columns else drives_pl
    if "drive_points_scored" not in d.columns:
        d = add_drive_points_column(d)

    opp = (
        d.group_by(["defteam", "season"])
        .agg(opp_ppd=(pl.col("drive_points_scored").sum() / pl.col("drive_points_scored").count()))
        .rename({"defteam": "team"})
    )
    pdf = opp.sort(["team", "season"]).to_pandas()
    pdf["opp_ppd_next"] = pdf.groupby("team")["opp_ppd"].shift(-1)
    league = float(pdf["opp_ppd"].mean())
    pdf["neg_opp_ppd_next"] = -pdf["opp_ppd_next"]
    # Higher = better defense (fewer points allowed next season)
    pdf["def_quality_next"] = league - pdf["opp_ppd_next"]
    return pdf
