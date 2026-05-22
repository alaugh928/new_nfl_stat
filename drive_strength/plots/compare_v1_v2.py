"""Side-by-side v1 vs v2 metric comparison charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..constants import FORMULA_PATH, FORMULA_V2_PATH, TRAIN_HOLDOUT_SEASONS
from ..formula import load_formula
from ..pipeline import run_pipeline
from ..score import score_drives
from ..validation.metrics_report import build_metrics_table

FIGURES_V2 = Path(__file__).resolve().parent / "figures" / "v2"


def compare_metrics_bar(
    drives_v1: pd.DataFrame,
    ts_v1: pd.DataFrame,
    drives_v2: pd.DataFrame,
    ts_v2: pd.DataFrame,
    seasons: list[int],
) -> Path:
    m1 = build_metrics_table(drives_v1, ts_v1, seasons, formula_label="v1", pdp_col="off_pdp")
    m2 = build_metrics_table(drives_v2, ts_v2, seasons, formula_label="v2", pdp_col="off_pdp")
    hold_lo, hold_hi = TRAIN_HOLDOUT_SEASONS
    m1h = build_metrics_table(
        drives_v1,
        ts_v1,
        seasons,
        formula_label="v1",
        pdp_col="off_pdp",
        season_filter=(hold_lo, hold_hi),
    )
    m2h = build_metrics_table(
        drives_v2,
        ts_v2,
        seasons,
        formula_label="v2",
        pdp_col="off_pdp",
        season_filter=(hold_lo, hold_hi),
    )

    key_metrics = [
        "same_season_vs_ppd",
        "season_over_season_pdp",
        "season_over_season_epa_play",
        "ros_pdp_weeks_1_8",
        "ros_epa_early_weeks_1_8",
        "stability_pdp_weekly",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, m1x, m2x, title in [
        (axes[0], m1, m2, "All seasons in data"),
        (axes[1], m1h, m2h, f"Holdout {hold_lo}-{hold_hi}"),
    ]:
        x = range(len(key_metrics))
        w = 0.35
        v1 = [m1x.loc[m1x["metric"] == k, "r"].iloc[0] if k in m1x["metric"].values else 0 for k in key_metrics]
        v2 = [m2x.loc[m2x["metric"] == k, "r"].iloc[0] if k in m2x["metric"].values else 0 for k in key_metrics]
        ax.bar([i - w / 2 for i in x], v1, width=w, label="v1", color="#888888")
        ax.bar([i + w / 2 for i in x], v2, width=w, label="v2", color="#4c72b0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(key_metrics, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Pearson r")
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("PDP v1 vs v2 correlation comparison", fontsize=13)
    fig.tight_layout()
    out = FIGURES_V2 / "v1_vs_v2_correlation_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_comparison(
    seasons: list[int],
    *,
    no_cache: bool = False,
) -> Path:
    """Score same drives with v1 and v2 formulas; plot comparison."""
    from ..data import load_pbp
    from ..drives import attach_scrimmage_plays, build_drive_index
    from ..features import build_offense_drive_features
    from ..aggregate import aggregate_team_season

    pbp = load_pbp(seasons, no_cache=no_cache)
    drives = build_drive_index(pbp)
    joined = attach_scrimmage_plays(pbp, drives)
    off_feat = build_offense_drive_features(joined)

    d1 = score_drives(off_feat, formula=load_formula(FORMULA_PATH))
    d2 = score_drives(off_feat, formula=load_formula(FORMULA_V2_PATH))
    ts1 = aggregate_team_season(d1)
    ts2 = aggregate_team_season(d2)
    return compare_metrics_bar(d1, ts1, d2, ts2, seasons)
