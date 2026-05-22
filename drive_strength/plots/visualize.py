"""PDP correlation and Drive-Quality-style diagnostic figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from ..labels import add_drive_points_column
from ..validation.predictive import rest_of_season_corr, season_over_season_corr
from .metrics import (
    autocorrelation_series,
    build_team_season_benchmarks,
    team_epa_per_drive,
    team_epa_per_play,
    team_success_rate,
    week_level_offense,
)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def _corr(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.sum() < 5:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])


def _scatter_with_fit(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    *,
    highlight_season: int | None = None,
    highlight_df: pd.DataFrame | None = None,
) -> float:
    ax.scatter(x, y, alpha=0.55, s=36, edgecolors="white", linewidths=0.4, c="#2c5f8a")
    if highlight_df is not None and highlight_season is not None:
        h = highlight_df[highlight_df["season"] == highlight_season]
        if not h.empty:
            ax.scatter(
                h[xlabel.split()[0] if False else h.columns[0]],
            )
    r = _corr(pd.Series(x), pd.Series(y))
    if len(x) >= 3 and np.std(x) > 0:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, coef[0] * xs + coef[1], color="#c44e52", lw=2, label=f"r = {r:.3f}")
        ax.legend(loc="lower right", frameon=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return r


def plot_pdp_vs_ppd(
    bench: pd.DataFrame,
    *,
    season: int | None = None,
    output_path: Path | None = None,
) -> Path:
    d = bench.dropna(subset=["off_pdp_mean", "ppd"])
    if season is not None:
        d = d[d["season"] == season]
        title = f"Offensive PDP vs points per drive ({season})"
        fname = f"pdp_vs_ppd_{season}.png"
    else:
        title = "Offensive PDP vs points per drive (all seasons)"
        fname = "pdp_vs_ppd_all_seasons.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_with_fit(
        ax,
        d["off_pdp_mean"].to_numpy(),
        d["ppd"].to_numpy(),
        "Mean offensive PDP per drive",
        "Actual points per drive",
        title,
    )
    if season is not None:
        for _, row in d.iterrows():
            ax.annotate(
                row["team"],
                (row["off_pdp_mean"], row["ppd"]),
                fontsize=7,
                alpha=0.85,
            )
    fig.tight_layout()
    out = output_path or FIGURES_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pdp_vs_epa_drive(
    bench: pd.DataFrame,
    *,
    season: int | None = None,
    output_path: Path | None = None,
) -> Path:
    d = bench.dropna(subset=["off_pdp_mean", "epa_per_drive"])
    if season is not None:
        d = d[d["season"] == season]
        title = f"Offensive PDP vs EPA per drive ({season})"
        fname = f"pdp_vs_epa_per_drive_{season}.png"
    else:
        title = "Offensive PDP vs EPA per drive (all seasons)"
        fname = "pdp_vs_epa_per_drive_all_seasons.png"

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_with_fit(
        ax,
        d["off_pdp_mean"].to_numpy(),
        d["epa_per_drive"].to_numpy(),
        "Mean offensive PDP per drive",
        "EPA per drive (sum EPA on drive, team mean)",
        title,
    )
    if season is not None:
        for _, row in d.iterrows():
            ax.annotate(row["team"], (row["off_pdp_mean"], row["epa_per_drive"]), fontsize=7, alpha=0.85)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_correlation_summary(
    drives: pd.DataFrame,
    bench: pd.DataFrame,
    seasons: list[int],
    *,
    output_path: Path | None = None,
) -> Path:
    """Bar chart: PDP vs PPD/EPA metrics — DQ-style predictive & stability checks."""
    rows: list[dict] = []

    # Same-season descriptive
    rows.append(
        {
            "metric": "PDP vs PPD\n(same season)",
            "r": _corr(bench["off_pdp_mean"], bench["ppd"]),
            "group": "Same season",
        }
    )
    rows.append(
        {
            "metric": "PDP vs EPA/drive\n(same season)",
            "r": _corr(bench["off_pdp_mean"], bench["epa_per_drive"]),
            "group": "Same season",
        }
    )
    rows.append(
        {
            "metric": "PDP vs EPA/play\n(same season)",
            "r": _corr(bench["off_pdp_mean"], bench["epa_per_play"]),
            "group": "Same season",
        }
    )
    if "success_rate" in bench.columns:
        rows.append(
            {
                "metric": "PDP vs success rate\n(same season)",
                "r": _corr(bench["off_pdp_mean"], bench["success_rate"]),
                "group": "Same season",
            }
        )

    # Predictive (PDP)
    r_sos, _ = season_over_season_corr(drives)
    rows.append({"metric": "PDP → next season PPD", "r": r_sos, "group": "Predictive"})
    for split in (4, 8, 12):
        r_ros, _ = rest_of_season_corr(drives, split_week=split)
        rows.append({"metric": f"PDP wk1-{split} → ROS PPD", "r": r_ros, "group": "Predictive"})

    # Benchmarks: EPA/play and PPD → next season PPD
    cur = bench[["team", "season", "epa_per_play", "ppd"]].copy()
    cur["next_season"] = cur["season"] + 1
    nxt = bench[["team", "season", "ppd"]].rename(
        columns={"season": "next_season", "ppd": "ppd_next"}
    )
    m_nxt = cur.merge(nxt, on=["team", "next_season"], how="inner")
    if len(m_nxt) >= 10:
        rows.append(
            {
                "metric": "EPA/play → next season PPD",
                "r": _corr(m_nxt["epa_per_play"], m_nxt["ppd_next"]),
                "group": "Benchmark",
            }
        )
        rows.append(
            {
                "metric": "PPD → next season PPD",
                "r": _corr(m_nxt["ppd"], m_nxt["ppd_next"]),
                "group": "Benchmark",
            }
        )

    # Week-to-week stability
    weekly = week_level_offense(drives, exclude_kneels=True)
    pdp_ac = (
        weekly.groupby(["team", "season"])["off_pdp_mean"]
        .apply(autocorrelation_series)
        .mean()
    )
    ppd_w = (
        weekly.groupby(["team", "season"])
        .apply(lambda g: autocorrelation_series(g["ppd"]))
        .mean()
    )
    rows.append({"metric": "PDP week-to-week\nstability", "r": pdp_ac, "group": "Stability"})
    rows.append({"metric": "PPD week-to-week\nstability", "r": ppd_w, "group": "Stability"})

    if "epa_per_play" in bench.columns:
        # EPA/play weekly from drives only approximate — use team-season rank as proxy N/A
        pass

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"Same season": "#4c72b0", "Predictive": "#55a868", "Stability": "#c44e52", "Benchmark": "#8172b3"}
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["r"],
        color=[colors.get(g, "#888") for g in df["group"]],
        edgecolor="white",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["metric"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r")
    ax.set_title("PDP correlations (Drive Quality–style summary)")
    ax.set_ylim(-0.1, max(0.65, df["r"].max() + 0.05))
    for bar, val in zip(bars, df["r"]):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=c, label=g) for g, c in colors.items()], loc="upper right")
    fig.tight_layout()
    out = output_path or FIGURES_DIR / "correlation_summary_dq_style.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_stability_comparison(drives: pd.DataFrame, *, output_path: Path | None = None) -> Path:
    """Week-to-week autocorrelation: PDP vs PPD vs EPA/play (team-season averages)."""
    weekly = week_level_offense(drives, exclude_kneels=True)
    seasons = sorted(drives["season"].unique())
    epa_play = team_epa_per_play([int(s) for s in seasons])

    records = []
    for (team, season), g in weekly.groupby(["team", "season"]):
        records.append(
            {
                "team": team,
                "season": season,
                "pdp_ac": autocorrelation_series(g["off_pdp_mean"].sort_index()),
                "ppd_ac": autocorrelation_series(g["ppd"].sort_index()),
            }
        )
    stab = pd.DataFrame(records)
    stab = stab.merge(epa_play, on=["team", "season"], how="left")

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        stab["pdp_ac"].dropna(),
        stab["ppd_ac"].dropna(),
    ]
    labels = ["PDP", "PPD (weekly)"]
    if "epa_per_play" in stab.columns:
        # EPA/play has no weekly series in this build — show season metric dispersion as note
        labels.append("EPA/play\n(season mean, no weekly)")
        data.append(stab["epa_per_play"].dropna() * 0 + np.nan)  # skip invalid box

    bp = ax.boxplot(
        [stab["pdp_ac"].dropna(), stab["ppd_ac"].dropna()],
        tick_labels=["PDP", "PPD (weekly)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#4c72b0", "#c44e52"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    medians = [stab["pdp_ac"].median(), stab["ppd_ac"].median()]
    ax.set_ylabel("Week-to-week autocorrelation (lag-1)")
    ax.set_title("Metric stability by team-season (higher = stickier week-to-week)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, m in enumerate(medians):
        ax.text(i + 1, m + 0.02, f"med={m:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / "stability_week_to_week.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_season_predictive_grid(
    drives: pd.DataFrame,
    bench: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> Path:
    """2x2: PDP/EPA/play vs next-season PPD; early PDP vs ROS PPD."""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)

    off = bench[["team", "season", "off_pdp_mean", "epa_per_play", "ppd"]].copy()
    off["next_season"] = off["season"] + 1
    nxt = bench[["team", "season", "ppd"]].rename(
        columns={"season": "next_season", "ppd": "ppd_next"}
    )
    m = off.merge(nxt, on=["team", "next_season"], how="inner")

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_with_fit(
        ax1,
        m["off_pdp_mean"].to_numpy(),
        m["ppd_next"].to_numpy(),
        "PDP (season S)",
        "PPD (season S+1)",
        "PDP → next season scoring",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_with_fit(
        ax2,
        m["epa_per_play"].to_numpy(),
        m["ppd_next"].to_numpy(),
        "EPA/play (season S)",
        "PPD (season S+1)",
        "EPA/play → next season scoring",
    )

    split = 8
    early = (
        drives.loc[drives["week"] <= split]
        .groupby(["posteam", "season"])["off_pdp"]
        .mean()
        .reset_index()
        .rename(columns={"posteam": "team", "off_pdp": "pdp_early"})
    )
    late = (
        drives.loc[drives["week"] > split]
        .groupby(["posteam", "season"])
        .agg(pts=("drive_points_scored", "sum"), n=("game_id", "count"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    late["ppd_ros"] = late["pts"] / late["n"].replace(0, np.nan)
    ros = early.merge(late[["team", "season", "ppd_ros"]], on=["team", "season"])

    ax3 = fig.add_subplot(gs[1, 0])
    _scatter_with_fit(
        ax3,
        ros["pdp_early"].to_numpy(),
        ros["ppd_ros"].to_numpy(),
        f"PDP weeks 1–{split}",
        f"PPD weeks {split + 1}–17",
        "Rest-of-season predictiveness (PDP)",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    if not ros.empty:
        from ..validation.metrics_report import team_early_epa_per_play

        seasons_list = sorted(drives["season"].unique().astype(int).tolist())
        epa_early = team_early_epa_per_play(
            drives, seasons_list, split_week=split, no_cache=False
        )
        epa_ros = epa_early.merge(ros[["team", "season", "ppd_ros"]], on=["team", "season"])
        _scatter_with_fit(
            ax4,
            epa_ros["epa_early"].to_numpy(),
            epa_ros["ppd_ros"].to_numpy(),
            f"EPA/play weeks 1–{split}",
            f"PPD weeks {split + 1}–17",
            "ROS predictiveness (EPA/play, fair)",
        )

    fig.suptitle("Predictive comparisons (Drive Quality–style)", fontsize=13, y=1.02)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / "predictive_grid_dq_style.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_team_drive_distributions(
    drives: pd.DataFrame,
    season: int,
    *,
    output_path: Path | None = None,
) -> Path:
    """DQ-style: distribution of drive-level PDP by team for one season."""
    d = drives[(drives["season"] == season) & (~drives["is_kneel_only_drive"])].copy()
    teams = (
        d.groupby("posteam")["off_pdp"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    data = [d.loc[d["posteam"] == t, "off_pdp"].values for t in teams]

    fig, ax = plt.subplots(figsize=(14, 6))
    parts = ax.violinplot(data, positions=range(len(teams)), showmeans=True, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c72b0")
        pc.set_alpha(0.55)
    ax.set_xticks(range(len(teams)))
    ax.set_xticklabels(teams, rotation=45, ha="right")
    ax.set_ylabel("Drive-level offensive PDP")
    ax.set_title(f"Drive-level PDP distributions by team ({season})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / f"team_pdp_distributions_{season}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_team_rankings_bar(
    bench: pd.DataFrame,
    season: int,
    *,
    output_path: Path | None = None,
) -> Path:
    """Horizontal bar: team offensive PDP mean vs PPD for a season."""
    d = bench[bench["season"] == season].sort_values("off_pdp_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
    axes[0].barh(d["team"], d["off_pdp_mean"], color="#4c72b0", alpha=0.85)
    axes[0].set_xlabel("Mean offensive PDP")
    axes[0].set_title(f"PDP ranking ({season})")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(d["team"], d["ppd"], color="#c44e52", alpha=0.85)
    axes[1].set_xlabel("Points per drive")
    axes[1].set_title(f"Actual PPD ({season})")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / f"team_rankings_pdp_ppd_{season}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ros_by_split_week(drives: pd.DataFrame, *, output_path: Path | None = None) -> Path:
    """PDP early-window vs ROS PPD correlation across split weeks (DQ short-term)."""
    splits = list(range(3, 14))
    pdp_rs = []
    for k in splits:
        r, _ = rest_of_season_corr(drives, split_week=k)
        pdp_rs.append(r)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(splits, pdp_rs, marker="o", color="#4c72b0", lw=2, label="PDP")
    ax.set_xlabel("Split week k (PDP from weeks 1–k)")
    ax.set_ylabel("Correlation with PPD weeks k+1–17")
    ax.set_title("Rest-of-season predictive correlation by split week")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = output_path or FIGURES_DIR / "ros_correlation_by_split.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_2025_comparison_panel(
    bench: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> Path:
    """Side-by-side 2025: PDP vs PPD and PDP vs EPA/drive with team labels."""
    d = bench[bench["season"] == 2025].dropna(subset=["off_pdp_mean", "ppd", "epa_per_drive"])
    if d.empty:
        raise ValueError("No 2025 benchmark rows — run pipeline for season 2025 first")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_with_fit(
        axes[0],
        d["off_pdp_mean"].to_numpy(),
        d["ppd"].to_numpy(),
        "Mean offensive PDP",
        "Points per drive",
        "2025: PDP vs PPD",
    )
    _scatter_with_fit(
        axes[1],
        d["off_pdp_mean"].to_numpy(),
        d["epa_per_drive"].to_numpy(),
        "Mean offensive PDP",
        "EPA per drive",
        "2025: PDP vs EPA/drive",
    )
    for _, row in d.iterrows():
        axes[0].annotate(row["team"], (row["off_pdp_mean"], row["ppd"]), fontsize=8, alpha=0.85)
        axes[1].annotate(row["team"], (row["off_pdp_mean"], row["epa_per_drive"]), fontsize=8, alpha=0.85)
    fig.tight_layout()
    out = output_path or FIGURES_DIR / "pdp_2025_comparison_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all_figures(
    drives: pd.DataFrame,
    team_season: pd.DataFrame,
    seasons: list[int],
    *,
    focus_season: int = 2025,
    no_cache: bool = False,
) -> list[Path]:
    """Generate full figure set; returns paths written."""
    import polars as pl

    if "drive_points_scored" not in drives.columns:
        drives = add_drive_points_column(pl.from_pandas(drives)).to_pandas()

    bench = build_team_season_benchmarks(
        drives, seasons, team_season, no_cache=no_cache, exclude_kneels=True
    )
    paths: list[Path] = []

    paths.append(plot_pdp_vs_ppd(bench))
    paths.append(plot_pdp_vs_epa_drive(bench))
    paths.append(plot_correlation_summary(drives, bench, seasons))
    paths.append(plot_stability_comparison(drives))
    paths.append(plot_season_predictive_grid(drives, bench))
    paths.append(plot_ros_by_split_week(drives))

    if focus_season in bench["season"].values:
        paths.append(plot_pdp_vs_ppd(bench, season=focus_season))
        paths.append(plot_pdp_vs_epa_drive(bench, season=focus_season))
        paths.append(plot_team_drive_distributions(drives, focus_season))
        paths.append(plot_team_rankings_bar(bench, focus_season))
        paths.append(plot_2025_comparison_panel(bench))

    # EPA/play scatter all seasons
    fig, ax = plt.subplots(figsize=(7, 6))
    d = bench.dropna(subset=["off_pdp_mean", "epa_per_play"])
    _scatter_with_fit(
        ax,
        d["off_pdp_mean"].to_numpy(),
        d["epa_per_play"].to_numpy(),
        "Mean offensive PDP",
        "EPA per play",
        "PDP vs EPA/play (all seasons)",
    )
    p = FIGURES_DIR / "pdp_vs_epa_per_play_all_seasons.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Defensive: higher def_pdp = better; vs opponent points allowed
    if "def_pdp_mean" in bench.columns:
        d25 = bench[bench["season"] == focus_season].dropna(
            subset=["def_pdp_mean", "ppd"]
        )
        if not d25.empty:
            opp_ppd = (
                drives[drives["season"] == focus_season]
                .groupby("defteam")
                .agg(opp_ppd=("drive_points_scored", lambda s: s.sum() / max(len(s), 1)))
                .reset_index()
                .rename(columns={"defteam": "team"})
            )
            dm = d25.merge(opp_ppd, on="team")
            fig, ax = plt.subplots(figsize=(7, 6))
            _scatter_with_fit(
                ax,
                dm["def_pdp_mean"].to_numpy(),
                dm["opp_ppd"].to_numpy(),
                "Mean defensive PDP (higher = better)",
                "Opponent points per drive allowed",
                f"{focus_season}: Defensive PDP vs opponent PPD",
            )
            for _, row in dm.iterrows():
                ax.annotate(row["team"], (row["def_pdp_mean"], row["opp_ppd"]), fontsize=8, alpha=0.85)
            pdef = FIGURES_DIR / f"def_pdp_vs_opp_ppd_{focus_season}.png"
            fig.tight_layout()
            fig.savefig(pdef, dpi=150)
            plt.close(fig)
            paths.append(pdef)

    # Offense vs defense PDP same season (team identity)
    d_id = bench[bench["season"] == focus_season].dropna(
        subset=["off_pdp_mean", "def_pdp_mean"]
    )
    if not d_id.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_with_fit(
            ax,
            d_id["off_pdp_mean"].to_numpy(),
            d_id["def_pdp_mean"].to_numpy(),
            "Mean offensive PDP",
            "Mean defensive PDP",
            f"{focus_season}: Offense vs defense PDP by team",
        )
        for _, row in d_id.iterrows():
            ax.annotate(row["team"], (row["off_pdp_mean"], row["def_pdp_mean"]), fontsize=8, alpha=0.85)
        p2 = FIGURES_DIR / f"off_vs_def_pdp_{focus_season}.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        paths.append(p2)

    # Export numeric summary
    summary_rows = []
    for name, r in [
        ("pdp_vs_ppd_same_season", _corr(bench["off_pdp_mean"], bench["ppd"])),
        ("pdp_vs_epa_drive_same_season", _corr(bench["off_pdp_mean"], bench["epa_per_drive"])),
        ("pdp_vs_epa_play_same_season", _corr(bench["off_pdp_mean"], bench["epa_per_play"])),
        ("pdp_season_over_season", season_over_season_corr(drives)[0]),
        ("pdp_ros_week8", rest_of_season_corr(drives, split_week=8)[0]),
    ]:
        summary_rows.append({"metric": name, "r": r})
    b25 = bench[bench["season"] == focus_season]
    if not b25.empty:
        summary_rows.append(
            {
                "metric": f"pdp_vs_ppd_{focus_season}",
                "r": _corr(b25["off_pdp_mean"], b25["ppd"]),
            }
        )
        summary_rows.append(
            {
                "metric": f"pdp_vs_epa_drive_{focus_season}",
                "r": _corr(b25["off_pdp_mean"], b25["epa_per_drive"]),
            }
        )
    summary_path = FIGURES_DIR / "correlation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    paths.append(summary_path)

    return paths
