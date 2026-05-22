"""Run all validation checks."""

from __future__ import annotations

import logging

from ..constants import DEFAULT_FORMULA_PATH
from ..pipeline import run_pipeline
from .immutability import check_immutability
from .narrative import check_team_a_vs_team_b, check_turnover_narratives
from .predictive import rest_of_season_corr, season_over_season_corr

logger = logging.getLogger(__name__)


def run_all_validations(seasons: list[int], *, no_cache: bool = False) -> bool:
    if not DEFAULT_FORMULA_PATH.exists():
        logger.error("Missing %s — run: python -m drive_strength train-v2", DEFAULT_FORMULA_PATH)
        return False

    results: list[tuple[str, bool, str]] = []

    ok_a, msg_a = check_team_a_vs_team_b()
    results.append(("team_a_vs_b", ok_a, msg_a))

    ok_t, msg_t = check_turnover_narratives()
    results.append(("turnover_narratives", ok_t, msg_t))

    ok_i, msg_i = check_immutability(seasons[:1], no_cache=no_cache)
    results.append(("immutability", ok_i, msg_i))

    drives, _, _ = run_pipeline(seasons, no_cache=no_cache)
    r_sos, msg_sos = season_over_season_corr(drives)
    results.append(("season_over_season", not (r_sos != r_sos) and r_sos > 0, msg_sos))

    r_ros, msg_ros = rest_of_season_corr(drives)
    results.append(("rest_of_season", not (r_ros != r_ros) and r_ros > 0, msg_ros))

    all_ok = True
    for name, ok, msg in results:
        status = "PASS" if ok else "FAIL"
        logger.info("[%s] %s: %s", status, name, msg)
        all_ok = all_ok and ok

    return all_ok
