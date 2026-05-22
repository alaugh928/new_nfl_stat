"""Generate PDP visualization suite.

Run: ``python -m drive_strength.plots --seasons 2018 2019 2020 2021 2022 2023 2024 2025``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..aggregate import aggregate_team_season
from ..pipeline import run_pipeline
from .visualize import FIGURES_DIR, generate_all_figures

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDP figures")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--focus-season", type=int, default=2025)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--use-existing-output",
        action="store_true",
        help="Skip pipeline; read drives_pdp.csv from output dir",
    )
    args = parser.parse_args()

    if args.use_existing_output:
        from ..constants import OUTPUT_DIR

        drives = pd.read_csv(OUTPUT_DIR / "drives_pdp.csv")
        seasons = sorted(drives["season"].unique().astype(int).tolist())
        season = aggregate_team_season(drives)
    else:
        seasons = args.seasons
        drives, _, season = run_pipeline(seasons, no_cache=args.no_cache)

    from ..validation.metrics_report import write_metrics_reports

    paths = generate_all_figures(
        drives,
        season,
        seasons,
        focus_season=args.focus_season,
        no_cache=args.no_cache,
    )
    write_metrics_reports(drives, season, seasons, no_cache=args.no_cache)

    v2_dir = args.figures_dir / "v2"
    import drive_strength.plots.visualize as viz

    old_dir = viz.FIGURES_DIR
    viz.FIGURES_DIR = v2_dir
    try:
        v2_paths = generate_all_figures(
            drives,
            season,
            seasons,
            focus_season=args.focus_season,
            no_cache=args.no_cache,
        )
        paths.extend(v2_paths)
    finally:
        viz.FIGURES_DIR = old_dir

    logger.info("Wrote %d figures to %s (+ v2/)", len(paths), args.figures_dir)
    for p in paths:
        logger.info("  %s", p)


if __name__ == "__main__":
    main()
