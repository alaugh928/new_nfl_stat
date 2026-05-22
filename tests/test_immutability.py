"""Immutability smoke test."""

from __future__ import annotations

import pytest

from drive_strength.constants import FORMULA_PATH
from drive_strength.validation.immutability import check_immutability

pytestmark = pytest.mark.skipif(
    not FORMULA_PATH.exists(),
    reason="formula_v1.json missing",
)


def test_immutability_2023():
    ok, msg = check_immutability([2023], no_cache=False)
    assert ok, msg
