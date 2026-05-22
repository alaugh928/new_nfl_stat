"""Unit tests for narrative fixtures (requires formula_v1.json)."""

from __future__ import annotations

from pathlib import Path

import pytest

from drive_strength.constants import DEFAULT_FORMULA_PATH
from drive_strength.validation.narrative import check_team_a_vs_team_b, check_turnover_narratives

pytestmark = pytest.mark.skipif(
    not DEFAULT_FORMULA_PATH.exists(),
    reason="formula not trained yet",
)


def test_team_a_beats_team_b():
    ok, msg = check_team_a_vs_team_b()
    assert ok, msg


def test_turnover_narratives():
    ok, msg = check_turnover_narratives()
    assert ok, msg
