"""CLI: compute PDP, train formula, or run validation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .constants import DEFAULT_SEASONS_END, TRAIN_SEASON_END, TRAIN_SEASON_START
from .pipeline import run_pipeline
from .train_once import train_formula
from .train_v2 import train_formula_v2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Path Drive Points (PDP)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Score drives and write output CSVs")
    run_p.add_argument("--seasons", nargs="+", type=int, default=None)
    run_p.add_argument("--no-cache", action="store_true")
    run_p.add_argument("--output-dir", type=Path, default=None)
    run_p.add_argument(
        "--exclude-kneels",
        action="store_true",
        help="Exclude kneel-only drives from team aggregates only",
    )

    train_p = sub.add_parser("train", help="One-time fit and write formula_v1.json")
    train_p.add_argument("--train-start", type=int, default=TRAIN_SEASON_START)
    train_p.add_argument("--train-end", type=int, default=TRAIN_SEASON_END)
    train_p.add_argument("--no-cache", action="store_true")

    train2_p = sub.add_parser("train-v2", help="Walk-forward fit and write formula_v2.json")
    train2_p.add_argument("--no-cache", action="store_true")

    val_p = sub.add_parser("validate", help="Run validation harness")
    val_p.add_argument("--seasons", nargs="+", type=int, default=None)
    val_p.add_argument("--no-cache", action="store_true")

    gates_p = sub.add_parser("gates-v2", help="Run v2 product gates on holdout")
    gates_p.add_argument("--seasons", nargs="+", type=int, default=None)
    gates_p.add_argument("--no-cache", action="store_true")

    cmp_p = sub.add_parser("compare-v1-v2", help="v1 vs v2 correlation charts")
    cmp_p.add_argument("--seasons", nargs="+", type=int, default=None)
    cmp_p.add_argument("--no-cache", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        seasons = args.seasons
        if seasons is None:
            seasons = [
                DEFAULT_SEASONS_END - 2,
                DEFAULT_SEASONS_END - 1,
                DEFAULT_SEASONS_END,
            ]
        run_pipeline(
            seasons,
            no_cache=args.no_cache,
            output_dir=args.output_dir,
            exclude_kneels=args.exclude_kneels,
        )
    elif args.command == "train":
        train_formula(
            train_start=args.train_start,
            train_end=args.train_end,
            no_cache=args.no_cache,
        )
    elif args.command == "train-v2":
        train_formula_v2(no_cache=args.no_cache)
    elif args.command == "validate":
        from .validation import run_all_validations

        seasons = args.seasons
        if seasons is None:
            seasons = list(range(2020, DEFAULT_SEASONS_END + 1))
        ok = run_all_validations(seasons, no_cache=args.no_cache)
        sys.exit(0 if ok else 1)
    elif args.command == "gates-v2":
        from .validation.gates_v2 import run_gates

        seasons = args.seasons or list(range(2020, DEFAULT_SEASONS_END + 1))
        drives, _, ts = run_pipeline(seasons, no_cache=args.no_cache)
        ok, results = run_gates(drives, ts, seasons)
        for name, passed, msg in results:
            logging.info("[%s] %s: %s", "PASS" if passed else "FAIL", name, msg)
        sys.exit(0 if ok else 1)
    elif args.command == "compare-v1-v2":
        from .plots.compare_v1_v2 import run_comparison

        seasons = args.seasons or list(range(2018, DEFAULT_SEASONS_END + 1))
        path = run_comparison(seasons, no_cache=args.no_cache)
        logging.info("Wrote %s", path)


if __name__ == "__main__":
    main()
