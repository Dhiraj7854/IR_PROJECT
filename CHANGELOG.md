# Changelog

## Research pipeline redesign

| Area | Before | After |
|---|---|---|
| Forecast checkpointing | `state_dict().copy()` kept shallow tensor references, so early stopping could restore final-epoch weights | Deep-copy best weights and verify that the restored validation loss equals the recorded best loss |
| Forecast baseline | A single CNN was reported against same-time-yesterday persistence | Plain CNN and persistence-residual/known-covariate CNN are compared on validation; test skill is reported by lead and averaged over five seeds |
| Uncertainty | MC-dropout uncertainty was scaled by a validation-fitted multiplier | Per-lead split-conformal intervals are calibrated on validation residuals and reported with test PICP, MPIW, pinball loss, and CRPS |
| Forecast issue time | Lead-1 forecasts from overlapping windows were used in game runs | Each midnight validation/test issuance stores a complete 96-step mean, residual scale, and actual series |
| Payoffs | Hourly, x-independent net bills allowed no population-dependent evolutionary equilibrium | Daily x-dependent utility is relative to the hoard bill; forecast market decisions settle against actual generation/load; common solar and diversifying load errors are separated |
| Uncertainty-aware decisions | A dimensionless sharer-only penalty could only reduce sharing | Mean commitments and newsvendor commitments use explicit dollar imbalance costs; value of information is reported with bootstrap intervals |
| Evolutionary dynamics | A per-15-minute memoryless update and M0–M3 comparison | Daily memory-based replicator with normalized fitness, a 0.1 maximum share step, four common-code forecast providers, and interior/boundary stability analysis |
| Scenario reporting | Deterministic sweeps and an overclaimed tipping/collapse interpretation | 200-day block-bootstrap scenarios over 20 seeds, 95% confidence intervals, sensitivity/LHS sweeps, and separate actual-only/forecast-visible shocks |
| Repository reproducibility | Raw CSVs at repository root, empty ignore file, tracked bytecode and generated weights, and Windows-only quickstart | Raw CSVs under `data/raw`, config-driven paths/parameters, pinned direct dependencies, generated artifacts ignored where appropriate, and cross-platform commands |
| Reported conclusions | CNN, tipping-point, collapse, and cross-plant claims were not aligned with reproduced tables | README findings quote regenerated tables; negative CNN skill and unobserved stable interior tipping branches are stated explicitly |

The implementation does not tune parameters to produce a desired adoption or equilibrium result. If a parameter grid does not resolve a threshold, the report describes the tested range rather than claiming a precise critical value.
