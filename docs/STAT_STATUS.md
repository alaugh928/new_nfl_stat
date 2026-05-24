# PDP vs EPA: honest status (2020–2024 holdout)

This document reflects **fair** benchmarks from `python -m drive_strength benchmark`.

## Bottom line

**PDP is not a better general predictor than EPA/play** on recent seasons. Offensive PDP and EPA/play are **highly correlated (~0.88)** because both mostly capture “how well did this offense move the ball.”

PDP is still useful as:

1. **Drive-level process score** with intentional narratives (march > one-play TD, turnover ordering).
2. **A second opinion** when process rank runs ahead of EPA — that subset predicts next-year scoring modestly better than EPA overall.
3. **Research / tooling** with frozen, interpretable coefficients and no team priors.

Do **not** market PDP as “beats EPA everywhere.” Use EPA for default team strength; use PDP where process matters or where PDP and EPA disagree.

## Holdout 2020–2024 (fair comparisons)

| Metric | PDP v2 | EPA/play | Winner |
|--------|--------|----------|--------|
| Same-season vs PPD | ~0.85 | ~0.88 | EPA |
| Season-over-season (S+1) | ~0.36 | ~0.38 | EPA (close) |
| ROS: early metric → late PPD (wk 1–8) | ~0.48 | ~0.55 | **EPA** |
| Week-to-week stability | ~−0.08 | — | Weak for both |

Earlier claims that PDP beat EPA on ROS used mismatched windows in some charts. The **fair** early-EPA → late-PPD benchmark favors EPA on 2020–24.

## Where PDP adds value

**Disagreement segments** (team-season z-scores, S+1 to next-year PPD):

| Segment | Approx. S+1 r |
|---------|----------------|
| All teams, PDP | ~0.36 |
| All teams, EPA | ~0.38 |
| PDP clearly ahead of EPA (process > results proxy) | **~0.42** |
| EPA clearly ahead of PDP | EPA path ~0.26 on that slice |

When a team’s **drive process** ranks well above what EPA/play implies, PDP is the more informative forward signal. When EPA ranks higher, trust EPA.

**Combined ridge (PDP + EPA)** → S+1 r ~0.37: small lift over EPA alone, not a full replacement.

## v1 vs v2

| | v1 | v2 |
|--|----|----|
| Training target | Same-season PPD | Next-season PPD |
| Same-season fit | Higher (~0.79) | Lower (~0.73–0.85) |
| Holdout S+1 | **~0.39** | ~0.36 |
| Drive scale | Misleading (~6–8) | ~2.2 (fixed) |
| Defensive PDP | Often negative | Non-negative (fixed) |

v2 improved **semantics and evaluation**; it did **not** clearly beat v1 or EPA on year-ahead prediction.

## Should we keep iterating?

**Low ROI** for another full formula search unless you change the goal:

- **Stop** if the goal is “one number that beats EPA on all horizons.”
- **Continue lightly** if the goal is “drive-quality storytelling + disagreement flag for analysts.”

## Recommended next steps

1. Run `python -m drive_strength benchmark` after each scoring run.
2. Export `process_vs_epa` on `team_season_pdp.csv` (PDP z − EPA z) and filter to **pdp_ahead_of_epa** teams for forward-looking views.
3. For prediction products, default to **EPA/play**; show PDP as process context, not primary rank.
4. Optional v2.1: train only on **non-yard** features (turnovers, penalties, `unsustained_explosion`) to reduce EPA redundancy — experimental, may hurt overall r.
