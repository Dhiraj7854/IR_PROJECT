# Forecast-Aware Evolutionary Game Theory

This repository evaluates a daily prosumer sharing game driven by rolling, 96-step solar forecasts. Its reproducible pipeline trains CNN ensembles, calibrates forecast intervals using validation-only split conformal, settles daily markets against actual generation and load, and simulates strategy adoption with a memory-based replicator.

## Research scope

- **Forecasts:** a plain 1D CNN baseline and a persistence-residual CNN with known time-of-day and pvlib clear-sky covariates. The model family is selected on validation MAE only. Five configured seeds are evaluated against same-time-yesterday persistence on the held-out test split.
- **Intervals:** per-lead 90% split-conformal prediction intervals. Test reporting includes PICP, mean prediction interval width (MPIW), interval pinball loss, and Gaussian CRPS. No MC-dropout scale multiplier is used.
- **Day-ahead archive:** each validation and test day is forecast at midnight for the full next 96 intervals and saved with forecast mean, validation-calibrated residual scale, and actual generation.
- **Game:** Oracle, persistence, CNN-naive, and CNN-aware providers share a single forecast-market/actual-settlement payoff implementation. A decision epoch is one day; fitness is relative to the hoard bill, and the replicator is bounded to a 0.1 maximum share change per day.
- **Uncertainty analyses:** the pipeline reports phase lines, interior and boundary equilibrium stability, initial-condition trajectories, block-bootstrap scenarios, confidence intervals, parameter sensitivity, and unforecast versus forecast-visible weather shocks.

This is a stylized simulation, not a field deployment or a causal estimate of household behavior. Synthetic household demand, a single plant, a short chronological holdout, assumed tariffs/costs, and a normal-error approximation limit external validity. The configured location used for clear-sky features is in [config.yaml](./config.yaml).

## Quickstart

Tested with Python 3.13. From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies and run the tests and full pipeline:

```bash
python -m pip install -r requirements.txt
python -m pytest tests
python run_pipeline.py
```

`run_pipeline.py` resolves data and output paths from the repository root, so it can also be launched from another working directory. It reads the raw Kaggle CSVs from `data/raw/`, writes processed data and the day-ahead archive to `data/processed/`, and regenerates all figures and tables under `results/`. Model weights are generated locally and ignored by Git.

## Configuration

All model, tariff, population, uncertainty, simulation, seed, data-path, and output-path settings are in [config.yaml](./config.yaml). The pipeline uses explicit PyTorch seeds and local `numpy.random.default_rng` generators; importing or constructing a community does not reseed NumPy's process-global RNG.

## Current measured findings

Numbers below are from the committed CSV outputs and use the configured data split and parameters. Dollar amounts are **per household** unless noted.

### Forecast skill

The selected persistence-residual CNN was chosen by validation MAE. On the test set its nMAE was **6.49%**, versus **6.39%** for same-time-yesterday persistence; its overall MAE was 1,891 kW versus 1,863 kW. CNN skill is negative at all five reported leads. Mean skill ± standard deviation across five training seeds:

| Lead | Skill vs persistence |
|---|---:|
| 15 minutes | -1.95% ± 1.86% |
| 1 hour | -1.46% ± 1.17% |
| 6 hours | -2.97% ± 2.33% |
| 12 hours | -1.63% ± 2.35% |
| 24 hours | -1.36% ± 1.40% |

The 90% split-conformal intervals are calibrated by lead on validation residuals. Test coverage is close to, but not exactly, nominal; see the complete per-lead PICP, MPIW, pinball, and CRPS values in [table_2_uncertainty_calibration.csv](./results/tables/table_2_uncertainty_calibration.csv). Do not interpret a validation conformal guarantee as a guarantee under distribution shift.

### Forecast providers and daily game

All four providers use the same payoff code and actual settlement. The rolling evaluation contains 10 complete day-ahead issuance days (5 validation and 5 test).

| Provider | Final share | Matched trade (kWh) | Savings vs hoard bill | Savings (% of hoard bill) |
|---|---:|---:|---:|---:|
| Oracle | 0.514 | 168.11 | $0.39 | 0.92% |
| Persistence | 0.232 | 63.55 | -$2.94 | -6.94% |
| CNN-naive | 0.131 | 54.05 | -$2.92 | -6.88% |
| CNN-aware | 0.067 | 24.93 | -$3.59 | -8.46% |

The observed fitness curve has an unstable interior equilibrium at share 0.155 for uncertainty multiplier κ=0. The lower boundary is stable there, while the upper boundary is also stable. At κ values 0.5–2.0, the lower boundary is stable and the upper boundary is unstable; no interior equilibrium is found on that grid. The configured grid therefore shows a stability change between κ=0 and 0.5, but does **not** establish a precise critical threshold or a stable interior tipping branch. Starting-share trajectories and all tested roots/classifications are in [table_5_equilibria.csv](./results/tables/table_5_equilibria.csv) and [table_6_hysteresis.csv](./results/tables/table_6_hysteresis.csv).

The 20-seed, 200-day block-bootstrap scenario mode reports mean and 95% confidence intervals. At κ=1, the value of probabilistic information is **$0.494 per household per day** (95% CI: **$0.4938–$0.4944**), or **11.75% of the mean daily absolute hoard bill** (95% CI: **11.71–11.79%**). This narrow interval reflects the limited set of 10 observed days and should not be read as broad population uncertainty. Full κ-specific costs and CIs are in [table_10_value_of_information_ci.csv](./results/tables/table_10_value_of_information_ci.csv); seed-level provider outcomes are in [table_8_scenario_seed_results.csv](./results/tables/table_8_scenario_seed_results.csv).

An actual-only 75% generation shock over six hours caused **no share response** (shock-day change 0), **$1.11 per-household** additional sharing bill (2.62% of the hoard bill), and no lost trade in this run; recovery is not applicable because adoption did not deviate. With the same shock visible in the forecast, the shock-day share change was **-0.000049**, trade loss was **0.0023 kWh**, and the bill increase was **$1.15 per household** (2.72% of hoard bill); the share did not recover to 95% of its no-shock reference during the four remaining observed days. These small observed responses are not described as a strategy “collapse.” Full trajectories and metrics are in [table_12_weather_shock.csv](./results/tables/table_12_weather_shock.csv) and [table_13_weather_shock_trajectories.csv](./results/tables/table_13_weather_shock_trajectories.csv).

The Plant 1-trained ensemble's zero-shot Plant 2 test nMAE was **6.36%**; see [table_plant2_generalization.csv](./results/tables/table_plant2_generalization.csv). This is a single short transfer test, not broad external validation.

## Repository layout

```text
config.yaml
run_pipeline.py
requirements.txt
src/
  preprocess.py
  train_cnn.py
  uncertainty.py
  payoffs.py                 # legacy hourly model
  payoffs_v2.py              # daily forecast-aware payoff
  replicator.py              # legacy and daily replicators
  experiments.py             # current provider/scenario analyses
data/
  raw/                       # four source CSVs
  processed/                 # generated clean data and day_ahead_forecasts.npz
results/
  figures/                   # regenerated pipeline plots
  tables/                    # CSV metrics, simulations, and analyses
tests/
```

## Generated result index

- Forecast error and lead skill: [table_1_forecast_metrics.csv](./results/tables/table_1_forecast_metrics.csv), [table_1b_lead_skill.csv](./results/tables/table_1b_lead_skill.csv).
- Per-lead interval calibration and scores: [table_2_uncertainty_calibration.csv](./results/tables/table_2_uncertainty_calibration.csv).
- Provider comparison, fitness curves, equilibria, trajectories, VOI, bootstrap CIs, sensitivity, shocks, and Plant 2 transfer: tables `table_3_provider_comparison.csv` through `table_13_weather_shock_trajectories.csv` plus `table_plant2_generalization.csv` in `results/tables/`.
- Figures: forecast and interval diagnostics (`figure_1`–`figure_3`), provider trajectories (`figure_4_provider_trajectories.png`), phase line and hysteresis (`figure_5_phase_line_bifurcation.png`, `figure_5b_hysteresis.png`), weather shocks (`figure_6_weather_shock.png`), scenario CIs (`figure_7_scenario_intervals.png`), and sensitivity (`figure_8_sensitivity.png`).

## Legacy material

`payoffs.py` and the original replicator class remain for comparison only; the active pipeline uses `payoffs_v2.py` and `DailyReplicator`. The checked-in notebook contains historical outputs and is not the source of current numerical findings. Use `python run_pipeline.py` and the CSVs under `results/tables/` for current results.
