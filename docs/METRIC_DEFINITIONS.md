# Metric definitions

## Points per drive (PPD)

Team-season offensive scoring rate: total points scored on offensive drives divided by offensive drive count. Kneel-only drives can be excluded from aggregates via `--exclude-kneels`. Used as the **benchmark** for predictive evaluation, not as a PDP input on the same drive.

## Path Drive Points (PDP)

A deterministic linear score from the full play sequence of a drive (progress, first downs by down, turnovers, penalties, negative plays, etc.). Same frozen coefficients for every team; no matchup or opponent-strength priors.

- **`off_pdp`**: higher = better offense on that drive.
- **`def_pdp`**: higher = better defense on that drive (trained so better defense aligns with lower opponent scoring next season).

**Scale (v2):** Drive-level calibration targets league mean `off_pdp` ≈ 2.2 on training years (PPD-like units). Team-season means can differ slightly from drive means because of aggregation.

## EPA per play / EPA per drive

From nflverse play-by-play `epa`, aggregated to team-season means. Scrims only (excludes kickoffs, punts, etc.). Not used as PDP inputs.

## Same-season descriptive correlation

Pearson *r* between mean team-season PDP and same-season PPD. High values are expected when the model fits current results; v2 intentionally trades some of this for forward fit.

## Season-over-season (S+1)

Mean PDP in season *S* vs actual PPD in season *S+1*. Primary training label for `formula_v2.json`.

## Rest-of-season (ROS)

Mean PDP in weeks 1–*k* vs PPD in weeks *k*+1–17 within the same season. Reported for *k* ∈ {4, 8, 12}. Fair EPA benchmark: **EPA/play weeks 1–*k*** vs **PPD weeks *k*+1–17** (not full-season EPA vs partial-season PPD).

## Week-to-week stability

Mean lag-1 autocorrelation of weekly team PDP (or PPD) within each team-season.

## Invalid benchmark (v1 charts only)

Plotting **full-season EPA/play** against **weeks 9–17 PPD** inflates correlation (~0.80) because full-season EPA already embeds late-season performance. Do not use for model comparison; v2 predictive grid uses early EPA vs ROS PPD instead.

## Training splits (formula_v2)

| Window | Seasons | Role |
|--------|---------|------|
| Fit | 1999–2012 | Ridge coefficients |
| Tune | 2013–2017 | Pick ridge α by S+1 *r* (tie-break: ROS) |
| Validate | 2018–2019 | Pre-freeze report |
| Holdout | 2020–2024 | Product gates only |
| Scoring | 2025+ | User-facing outputs |
