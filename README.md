# Forecast-Aware Evolutionary Game Theory for Renewable Energy Communities

This repository implements the complete end-to-end research framework coupling **1D-CNN Multi-Step Solar Generation Forecasting**, **Monte Carlo Dropout Uncertainty Quantification**, and **Dynamic Evolutionary Game Theory (Replicator Dynamics)** for Peer-to-Peer (P2P) renewable energy trading.

---

## 1. Central Research Story & Novelty
Rather than using static payoffs or decoupled machine learning predictions, this framework investigates the causal interaction:

$$\text{Renewable Forecast Uncertainty } (\sigma_t) \longrightarrow \text{Prosumer Risk Perception } [R(\sigma_t)] \longrightarrow \text{Dynamic Payoffs } (U_S, U_H) \longrightarrow \text{Replicator Dynamics } \left(\frac{dx}{dt}\right) \longrightarrow \text{Strategy Evolution}$$

- **Novelty A**: Renewable-generation multi-step forecasting (1D-CNN) directly parameterizing community P2P game mechanics.
- **Novelty B**: Predictive uncertainty $\sigma_t$ directly modifies the prosumer fitness through risk-adjusted payoffs ($U_i = \pi_i - \lambda_i R(\sigma_t)$).
- **Novelty C**: Time-varying, non-autonomous payoff matrices reflecting real diurnal solar irradiance, residential load demand, and dynamic mid-market P2P pricing (Supply-Demand Ratio).
- **Novelty D**: Discovery of the critical uncertainty tipping point $\sigma^*$ beyond which community cooperation collapses into conservative hoarding/grid backup.

---

## 2. Repository Structure
```
Research/
│
├── Forecast_Aware_Evolutionary_Game_Theory.ipynb   # Complete interactive Jupyter Notebook (executed with all inline outputs & plots)
├── run_pipeline.py                                 # Master end-to-end Python script
├── requirements.txt                                # Python library dependencies
├── README.md                                       # Comprehensive project documentation
│
├── src/                                            # Modular Python implementation package
│   ├── __init__.py                                 # Package initialization
│   ├── preprocess.py                               # Timestamp parsing, inverter aggregation, leak-free windowing
│   ├── train_cnn.py                                # 1D-CNN PyTorch model, Huber loss, training loop & metrics
│   ├── uncertainty.py                              # Monte Carlo Dropout, prediction intervals, PICP/MPIW calibration
│   ├── payoffs.py                                  # Prosumer community, diurnal load, P2P SDR pricing, dynamic payoffs
│   ├── replicator.py                               # Replicator dynamics Euler solver, community energy balance
│   └── experiments.py                              # M0-M3 benchmarks, tipping point sweeps, shock simulations, plotting
│
├── data/
│   ├── Plant_1_Generation_Data.csv                 # Raw Plant 1 inverter generation data
│   ├── Plant_1_Weather_Sensor_Data.csv             # Raw Plant 1 weather sensor data
│   ├── Plant_2_Generation_Data.csv                 # Raw Plant 2 inverter generation data
│   ├── Plant_2_Weather_Sensor_Data.csv             # Raw Plant 2 weather sensor data
│   └── processed/
│       ├── plant_1_cleaned.csv                     # Processed 15-minute aligned continuous Plant 1 time series
│       ├── plant_2_cleaned.csv                     # Processed 15-minute aligned continuous Plant 2 time series
│       └── cnn_model_weights.pt                    # Trained PyTorch 1D-CNN model checkpoint
│
└── results/
    ├── figures/                                    # Publication-ready 300 DPI figures
    │   ├── figure_1_actual_vs_predicted.png        # Actual vs 1D-CNN Forecast with 90% MC Dropout uncertainty band
    │   ├── figure_2_horizon_error.png              # MAE error across lead times (15m, 1h, 6h, 12h, 24h)
    │   ├── figure_3_uncertainty_calibration.png    # Empirical PICP vs Nominal coverage curve
    │   ├── figure_4_comparative_trajectories.png   # Strategy evolution x(t) for M0, M1, M2, and M3
    │   ├── figure_5_tipping_point.png              # Bifurcation curves: Final Share & Savings vs Uncertainty Level
    │   └── figure_6_weather_shock.png              # Strategy collapse & recovery dynamics during weather shock
    │
    └── tables/                                     # Exported CSV metric tables
        ├── table_1_forecast_metrics.csv            # MAE, RMSE, nMAE, nRMSE, Daylight errors (Persistence vs 1D-CNN)
        ├── table_2_uncertainty_calibration.csv     # PICP, MPIW, ACE, validation calibration factor
        ├── table_3_model_comparison.csv            # Comparative summary across M0, M1, M2, and M3
        ├── table_4_tipping_point_sweep.csv         # Uncertainty scaling sweep and equilibrium adoption rates
        ├── table_5_weather_shock_metrics.csv       # Pre/min/post shock share, lost trades, cost impact
        ├── table_6_plant2_generalization.csv       # Zero-shot transferability evaluation on Plant 2
        ├── simulation_M0.csv                       # Interval-by-interval simulation trace for M0
        ├── simulation_M1.csv                       # Interval-by-interval simulation trace for M1
        ├── simulation_M2.csv                       # Interval-by-interval simulation trace for M2
        └── simulation_M3.csv                       # Interval-by-interval simulation trace for M3
```

---

## 3. Quickstart & Execution

### Option A: Open the Interactive Jupyter Notebook
The notebook is already fully executed and self-contained with all cells, tables, and inline plots:
```powershell
.\venv\Scripts\jupyter.exe notebook Forecast_Aware_Evolutionary_Game_Theory.ipynb
```

### Option B: Run via Python Master Script
To execute all stages, retrain models, and update output figures/tables:
```powershell
.\venv\Scripts\python.exe run_pipeline.py
```

---

## 4. Summary of Empirical Findings

| Metric / Dimension | M0 (Persistence) | M1 (CNN Static Game) | M2 (CNN Dynamic, $\lambda=0$) | M3 (Proposed Forecast-Aware, $\lambda=2.5$) |
|:---|:---:|:---:|:---:|:---:|
| **Forecasting Method** | Lag-96 Persistence | Deterministic 1D-CNN | Deterministic 1D-CNN | 1D-CNN + MC Dropout |
| **Payoff Formulation** | Static / Persistence | Static Mean Solar | Dynamic SDR (Zero Risk) | Dynamic SDR + Uncertainty Risk |
| **Final Share Adoption $x(T)$** | 0.505 | 0.513 | 0.507 | **0.465** |
| **Mean Share Adoption $\bar{x}$** | 0.502 | 0.506 | 0.504 | **0.484** |
| **Total P2P Traded (kWh)** | 64.65 | 169.04 | 94.51 | **91.01** |
| **Community Savings ($)** | $9.70 | $25.36 | $14.18 | **$13.65** |

### Key Experimental Discoveries:
1. **Uncertainty Drives Conservative Defection**: When prosumers account for forecast uncertainty ($\sigma_t$), their risk-adjusted payoffs penalize P2P commitment risks, shifting equilibrium adoption from 0.507 to 0.465.
2. **Critical Uncertainty Tipping Point**: Sweeping uncertainty scaling factor $\kappa \in [0.0, 4.0]$ reveals a clear tipping point: beyond $\approx 10\%$ normalized uncertainty, community sharing crashes precipitously towards zero ($x(T) = 0.066$).
3. **Severe Meteorological Shock Response**: A 6-hour severe cloudburst shock (75% solar drop + 3x uncertainty spike) induces immediate behavioral withdrawal from P2P contracts with measurable lost community trading volume before recovering.
4. **Generalization Across Inverters & Plants**: The 1D-CNN trained on Plant 1 achieves a strong 9.14% nMAE when evaluated zero-shot on Plant 2 without fine-tuning.
