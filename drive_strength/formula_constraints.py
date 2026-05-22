"""Sign constraints on frozen coefficients (v1 full, v2 soft narrative-only)."""

from __future__ import annotations

from typing import Any

# v1: legacy broad constraints (avoid for v2 training).
OFFENSE_SIGN_RULES: dict[str, str] = {
    "progress_yards": "positive",
    "progress_pct": "positive",
    "plays": "positive",
    "yards_per_play": "positive",
    "first_downs_total": "positive",
    "first_downs_on_1st": "positive",
    "first_downs_on_2nd": "positive",
    "explosive_play_count": "non_negative",
    "explosive_yards_share": "positive",
    "penalty_net_yards": "non_negative",
    "def_penalty_yards": "non_negative",
    "def_penalty_auto_first_count": "non_negative",
    "turnover_penalty_interaction": "positive",
    "unsustained_explosion": "negative",
    "turnover_interception": "negative",
    "turnover_fumble_lost": "negative",
    "turnover_on_downs": "non_positive",
    "negative_play_rate": "negative",
    "sack_count": "negative",
    "off_penalty_yards": "negative",
    "off_penalty_count": "negative",
}

# v2: narrative-critical only (Team A vs B, turnover attenuation).
OFFENSE_NARRATIVE_RULES_V2: dict[str, str] = {
    "unsustained_explosion": "negative",
    "turnover_interception": "negative",
    "turnover_fumble_lost": "negative",
    "turnover_on_downs": "non_positive",
    "turnover_penalty_interaction": "positive",
}

DEFENSE_NARRATIVE_RULES_V2: dict[str, str] = {
    "unsustained_explosion_allowed": "negative",
    "turnover_forced_interception": "positive",
    "turnover_forced_fumble": "positive",
    "turnover_bonus_interaction": "positive",
}


def _apply_rule(value: float, rule: str) -> float:
    if rule == "positive" and value <= 0:
        return abs(value) if value != 0 else 0.01
    if rule == "negative" and value >= 0:
        return -abs(value) if value != 0 else -0.01
    if rule == "non_negative" and value < 0:
        return 0.0
    if rule == "non_positive" and value > 0:
        return 0.0
    return value


def enforce_formula_constraints(formula: dict[str, Any]) -> dict[str, Any]:
    """v1 full sign clamp."""
    for name, rule in OFFENSE_SIGN_RULES.items():
        coefs = formula["offense"]["coefficients"]
        if name in coefs:
            coefs[name] = _apply_rule(float(coefs[name]), rule)
    return formula


def enforce_narrative_constraints_v2(formula: dict[str, Any]) -> dict[str, Any]:
    """v2 soft narrative-only sign clamp."""
    for name, rule in OFFENSE_NARRATIVE_RULES_V2.items():
        coefs = formula["offense"]["coefficients"]
        if name in coefs:
            coefs[name] = _apply_rule(float(coefs[name]), rule)
    for name, rule in DEFENSE_NARRATIVE_RULES_V2.items():
        coefs = formula["defense"]["coefficients"]
        if name in coefs:
            coefs[name] = _apply_rule(float(coefs[name]), rule)
    return formula
