# Path Drive Points (PDP)

Deterministic, drive-level NFL offensive/defensive strength from nflverse play-by-play. Each drive is scored from its full play sequence via a **frozen** linear formula, not from same-drive points scored and not from EPA.

## Principles

- **Process over results:** Long productive drives score high even without points; one-play TDs score lower than sustained marches.
- **No matchup / team priors:** Same formula for every team; scores never revise when opponents play worse later.
- **Higher = better** for both `off_pdp` and `def_pdp`.

## formula_v1 vs formula_v2

| | v1 (`formula_v1.json`) | v2 (`formula_v2.json`, default) |
|--|------------------------|----------------------------------|
| Training label | Same-season PPD | Next-season PPD (S+1) |
| Features | 38 overlapping yard proxies | Lean process set (~20) |
| Constraints | Broad sign clamps (distorts ridge) | Narrative-critical signs only |
| Calibration | Team-season mean ≈ PPD | **Drive-level** mean ≈ 2.2 |
| Typical use | Descriptive ranking | Forward scoring prediction |

**v1 issues:** Strong same-season fit (r≈0.79 vs PPD) but weaker forward tests (S+1 r≈0.37 vs EPA/play ~0.40). Team means were ~6–8 while PPD ~2 because calibration was team-season only. One predictive chart compared full-season EPA to late-season PPD (misleading ~0.80 *r*).

**v2 goal:** Optimize walk-forward S+1 and ROS; accept lower same-season *r* if forward *r* improves. See [docs/METRIC_DEFINITIONS.md](docs/METRIC_DEFINITIONS.md) for benchmark definitions.

## Setup

Requires **Python 3.9+**. On 3.10+, `nflreadpy` is used; on 3.9, `nfl_data_py` is used automatically.

```bash
cd "new_NFL_stat"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Train frozen formula

**v2 (recommended):**

```bash
python -m drive_strength train-v2
```

Writes `drive_strength/formula_v2.json` (tune 2013–2017, validate 2018–2019).

**v1 (legacy comparison):**

```bash
python -m drive_strength train
```

## Score drives

```bash
python -m drive_strength run --seasons 2022 2023 2024 2025
python -m drive_strength run --seasons 2024 --exclude-kneels
```

Outputs under `drive_strength/output/`:

- `drives_pdp.csv` — per drive features + `off_pdp` + `def_pdp`
- `team_week_pdp.csv` — team-week aggregates (includes shrunk season means when aggregated)
- `team_season_pdp.csv` — team-season aggregates

## Validate

```bash
python -m drive_strength validate --seasons 2020 2021 2022 2023 2024
python -m drive_strength gates-v2 --seasons 2020 2021 2022 2023 2024
python -m drive_strength compare-v1-v2 --seasons 2018 2019 2020 2021 2022 2023 2024
```

`gates-v2` checks narratives, offensive/defensive drive scale, and reports holdout S+1 / ROS vs EPA/play.

**Defensive PDP:** trained on `def_quality_next` (league mean − next-season opp PPD). Drive scores are affine-calibrated then floored at 0 so `def_pdp` is always non-negative (higher = better defense).

**Holdout 2020–2024 (typical):** ROS PDP r≈0.47 (beats EPA early→ROS benchmark); S+1 r≈0.36 vs EPA/play ≈0.38 (close; full-season team aggregates). Validate-window S+1 on 2018–2019 is higher (~0.49) before freeze.

## Kneels

Kneel-only drives are scored and tagged `is_kneel_only_drive`. Use `--exclude-kneels` to omit them from team aggregates only.

## Visualizations

```bash
python -m drive_strength.plots --seasons 2018 2019 2020 2021 2022 2023 2024 2025
python -m drive_strength.plots --use-existing-output
```

Charts: `drive_strength/plots/figures/` (main) and `drive_strength/plots/figures/v2/` (duplicate bundle for v2 scoring). Metrics tables: `drive_strength/output/metrics_report_*.csv`. v1 vs v2 bars: `plots/figures/v2/v1_vs_v2_correlation_bars.png`.

The predictive grid bottom-right panel uses **EPA/play weeks 1–8 → PPD weeks 9–17** (fair ROS). Do not compare to old v1 charts that used full-season EPA vs late PPD.

## Tests

```bash
pytest tests/
```
