# Path Drive Points (PDP)

Deterministic, drive-level NFL offensive/defensive strength from nflverse play-by-play. Each drive is scored from its full play sequence via a **frozen** linear formula, not from same-drive points scored and not from EPA.

## What PDP is (and is not)

**PDP is not a drop-in replacement for EPA/play.** On 2020–2024 holdout, EPA/play is slightly better at predicting next-year scoring and rest-of-season scoring. PDP and EPA are ~0.88 correlated — they mostly measure the same underlying ball movement.

**PDP is useful when:**

- You want **drive-level process** scores with explicit narratives (long march > one-play TD; turnover ordering).
- You want a **second opinion** when process rank runs **ahead** of EPA (`process_vs_epa` on `team_season_pdp.csv`). That segment predicts next-year PPD modestly better (~0.42 r vs ~0.36 overall).
- You need a **frozen, interpretable** formula with no team priors.

See **[docs/STAT_STATUS.md](docs/STAT_STATUS.md)** for the full honest benchmark write-up.

## Principles

- **Process over results:** Long productive drives score high even without points; one-play TDs score lower than sustained marches.
- **No matchup / team priors:** Same formula for every team.
- **Higher = better** for both `off_pdp` and `def_pdp`.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

python -m drive_strength train-v2
python -m drive_strength run --seasons 2020 2021 2022 2023 2024 2025
python -m drive_strength benchmark --seasons 2018 2019 2020 2021 2022 2023 2024
```

## Commands

| Command | Purpose |
|---------|---------|
| `train-v2` | Fit `formula_v2.json` (S+1 label, walk-forward) |
| `run` | Score drives → `output/drives_pdp.csv`, team aggregates |
| `benchmark` | PDP vs EPA tables → `output/benchmark_*.csv` + console summary |
| `gates-v2` | Narratives, drive scale, S+1/ROS reports |
| `compare-v1-v2` | Side-by-side correlation charts |
| `validate` | Narrative + immutability + predictive harness |

## Outputs

- `drives_pdp.csv` — per-drive `off_pdp`, `def_pdp`, features
- `team_season_pdp.csv` — aggregates + `epa_per_play` + **`process_vs_epa`** (PDP z − EPA z)
- `benchmark_metrics.csv`, `benchmark_disagreement.csv` — from `benchmark`

## formula v1 vs v2

| | v1 | v2 (default) |
|--|----|--------------|
| Training label | Same-season PPD | Next-season PPD |
| Holdout S+1 vs EPA | ~0.39 vs ~0.38 | ~0.36 vs ~0.38 |
| Drive scale | Broken (~6–8) | ~2.2 |
| Defensive PDP | Often negative | Non-negative |

For **year-ahead team ranks**, EPA/play or v1 may edge v2. For **process storytelling** and **disagreement flags**, use v2.

## Docs

- [docs/STAT_STATUS.md](docs/STAT_STATUS.md) — performance reality check
- [docs/METRIC_DEFINITIONS.md](docs/METRIC_DEFINITIONS.md) — PPD, S+1, ROS definitions

## Tests

```bash
pytest tests/
```
