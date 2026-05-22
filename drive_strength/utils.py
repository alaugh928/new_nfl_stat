"""Play-type helpers."""

from __future__ import annotations

KICKOFF_TYPES = frozenset({"kickoff", "kick_off", "kickoff_clock"})

NON_SCRIMMAGE_TYPES = KICKOFF_TYPES | frozenset(
    {
        "extra_point",
        "no_play",
        "qb_spike",
        "timeout",
        "comment",
    }
)

KNEEL_TYPES = frozenset({"qb_kneel", "qb_kneel_clock"})
