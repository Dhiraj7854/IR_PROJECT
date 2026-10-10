"""
experiments.py
Comprehensive experimental suite for Forecast-Aware Evolutionary Game Theory:
Runs controlled model comparisons (M0, M1, M2, M3), sensitivity sweeps,
bifurcation/tipping point analysis, weather shock experiments, cross-plant testing,
and exports publication-quality figures and summary tables.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    """Sets clean academic plotting styles with seaborn."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8


def run_comparative_experiment(community, test_dates, y_true, y_cnn, sigma_cnn,
                               y_persist, x0=0.5, lambda_s=2.5):
    """
    Runs controlled comparison between models M0, M1, M2, and M3:
    - M0: Persistence Baseline Forecast + Baseline Game
    - M1: Deterministic 1D-CNN + Baseline Game (static mean solar)
    - M2: 1D-CNN + Dynamic Payoff without Uncertainty Penalty (lambda=0)
    - M3: Proposed 1D-CNN + Uncertainty-Aware Dynamic Payoffs (lambda=lambda_s)
    """
    from .replicator import ReplicatorEngine

    engine = ReplicatorEngine(step_size=0.1)
    T = len(test_dates)
    loads = community.generate_diurnal_load(test_dates)

    # Flatten sequential 24-hour test forecasts into single time series
    # Using 1-step ahead / lead-in from successive windows
    gen_actual = y_true[:, 0]
    gen_cnn = y_cnn[:, 0]
    sigma = sigma_cnn[:, 0]
    gen_persist = y_persist[:, 0]
    zero_sigma = np.zeros_like(sigma)
    static_mean_gen = np.full_like(gen_actual, np.mean(gen_cnn))

    models = {}

    # M0: Persistence baseline
    df_m0, sum_m0 = engine.simulate(
        community, gen_series=gen_persist, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0
    )
    models['M0 (Persistence Baseline)'] = (df_m0, sum_m0)

    # M1: Deterministic 1D-CNN + Static Game
    df_m1, sum_m1 = engine.simulate(
        community, gen_series=static_mean_gen, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0
    )
    models['M1 (1D-CNN Static Game)'] = (df_m1, sum_m1)

    # M2: 1D-CNN + Dynamic Payoff (No Uncertainty Penalty)
    df_m2, sum_m2 = engine.simulate(
        community, gen_series=gen_cnn, sigma_series=zero_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=0.0
    )
    models['M2 (1D-CNN Dynamic, No Uncertainty)'] = (df_m2, sum_m2)

    # M3: Proposed 1D-CNN + Uncertainty-Aware Dynamic Payoff
    df_m3, sum_m3 = engine.simulate(
        community, gen_series=gen_cnn, sigma_series=sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
    )
    models['M3 (Proposed Forecast-Aware Game)'] = (df_m3, sum_m3)

    return models


def run_uncertainty_sweep(community, test_dates, gen_series, sigma_series,
                          loads, x0=0.5, lambda_s=2.5,
                          multipliers=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
    """
    Sweeps uncertainty scaling factor to identify bifurcation and critical tipping points.
    """
    from .replicator import ReplicatorEngine
    engine = ReplicatorEngine(step_size=0.1)

    records = []
    trajectories = {}

    for mult in multipliers:
        scaled_sigma = sigma_series * mult
        df_sim, summary = engine.simulate(
            community, gen_series=gen_series, sigma_series=scaled_sigma,
            load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
        )
        norm_unc_pct = (np.mean(scaled_sigma) / community.plant_capacity) * 100.0
        records.append({
            'Uncertainty_Multiplier': mult,
            'Uncertainty_Scale_Pct': norm_unc_pct,
            'Final_Share': summary['Final_Share'],
            'Mean_Share': summary['Mean_Share'],
            'Time_Above_Threshold': summary['Time_Above_Threshold'],
            'P2P_Traded_kWh': summary['Total_P2P_Traded_kWh'],
            'Community_Savings ($)': summary['Total_Savings ($)'],
            'Savings_Pct (%)': summary['Savings_Pct (%)']
        })
        trajectories[mult] = df_sim['x_share'].values

    return pd.DataFrame(records), trajectories


def run_weather_shock_experiment(community, test_dates, gen_series, sigma_series,
                                 loads, x0=0.85, lambda_s=2.5,
                                 shock_start_idx=96, shock_duration=24, shock_drop=0.75):
    """
    Simulates a sudden severe weather shock (e.g. cloudburst / sudden storm):
    Generation drops by 75% and forecast uncertainty surges by 3x for 24 steps (6 hours).
    Tracks behavioral strategy collapse and recovery time.
    """
    from .replicator import ReplicatorEngine
    engine = ReplicatorEngine(step_size=0.1)

    shock_gen = gen_series.copy()
    shock_sigma = sigma_series.copy()

    end_idx = min(len(gen_series), shock_start_idx + shock_duration)
    # Apply shock
    shock_gen[shock_start_idx:end_idx] *= (1.0 - shock_drop)
    shock_sigma[shock_start_idx:end_idx] *= 3.0

    # Normal simulation
    df_normal, sum_normal = engine.simulate(
        community, gen_series=gen_series, sigma_series=sigma_series,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
    )

    # Shock simulation
    df_shock, sum_shock = engine.simulate(
        community, gen_series=shock_gen, sigma_series=shock_sigma,
        load_matrix=loads, timestamps=test_dates, x0=x0, lambda_s=lambda_s
    )

    trade_lost = sum_normal['Total_P2P_Traded_kWh'] - sum_shock['Total_P2P_Traded_kWh']
    cost_increase = sum_shock['Total_Cost ($)'] - sum_normal['Total_Cost ($)']

    metrics = {
        'Shock_Start_Interval': shock_start_idx,
        'Shock_Duration_Hours': shock_duration * 0.25,
        'Pre_Shock_Share': df_shock['x_share'].iloc[shock_start_idx - 1],
        'Min_Shock_Share': df_shock['x_share'].iloc[shock_start_idx:end_idx].min(),
        'Post_Shock_Final_Share': df_shock['x_share'].iloc[-1],
        'Trade_Lost_kWh': trade_lost,
        'Cost_Increase ($)': cost_increase
    }

    return df_normal, df_shock, metrics, (shock_start_idx, end_idx)


# --- Plotting Functions for the 6 Core Figures ---

def plot_forecast_actual_vs_pred(timestamps, y_true_day, y_pred_day, lower_day, upper_day,
                                save_path=None):
    """Figure 1: Actual vs Predicted Plant AC Power with 90% Prediction Intervals."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    x_axis = np.arange(len(y_true_day))

    ax.plot(x_axis, y_true_day, label='Actual AC Power', color='#1f77b4', linewidth=2.0)
    ax.plot(x_axis, y_pred_day, label='1D-CNN Forecast (Point)', color='#d62728', linestyle='--', linewidth=2.0)
    ax.fill_between(x_axis, lower_day, upper_day, color='#ff7f0e', alpha=0.25, label='90% Prediction Interval (MC Dropout)')

    ax.set_title('Figure 1: 1D-CNN 24-Hour Solar AC Power Forecast with Calibrated Uncertainty Band', fontweight='bold', fontsize=13)
    ax.set_xlabel('15-Minute Forecast Lead Steps (0 to 96 = 24 Hours)')
    ax.set_ylabel('Plant AC Power (kW)')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_error_by_horizon(horizon_dict, save_path=None):
    """Figure 2: Forecast Error by Lead Time (15m, 1h, 6h, 12h, 24h)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = list(horizon_dict.keys())
    values = list(horizon_dict.values())

    bars = ax.bar(names, values, color='#2ca02c', alpha=0.85, edgecolor='black', width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f} kW',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Figure 2: 1D-CNN Forecast Error (MAE) Across Forecasting Horizons', fontweight='bold', fontsize=13)
    ax.set_ylabel('Mean Absolute Error (kW)')
    ax.set_ylim(0, max(values) * 1.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_uncertainty_calibration(calib_results, save_path=None):
    """Figure 3: Prediction Interval Calibration (Empirical PICP vs Nominal Coverage)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    nominal_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    picp_levels = [calib_results.get(lvl, lvl * 100) for lvl in nominal_levels]

    ax.plot([50, 95], [50, 95], 'k--', label='Ideal Calibration Line', linewidth=1.8)
    ax.plot([lvl * 100 for lvl in nominal_levels], picp_levels, 'o-', color='#e377c2',
            linewidth=2.5, markersize=8, label='Empirical Coverage (MC Dropout)')

    ax.set_title('Figure 3: Prediction Interval Calibration Curve (PICP vs Nominal)', fontweight='bold', fontsize=13)
    ax.set_xlabel('Nominal Confidence Level (%)')
    ax.set_ylabel('Empirical Coverage PICP (%)')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_comparative_trajectories(models_dict, save_path=None):
    """Figure 4: Comparative Evolutionary Trajectories x(t) for M0, M1, M2, and M3."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = {
        'M0 (Persistence Baseline)': '#7f7f7f',
        'M1 (1D-CNN Static Game)': '#1f77b4',
        'M2 (1D-CNN Dynamic, No Uncertainty)': '#2ca02c',
        'M3 (Proposed Forecast-Aware Game)': '#d62728'
    }

    for name, (df_sim, _) in models_dict.items():
        time_hours = np.arange(len(df_sim)) * 0.25
        ax.plot(time_hours, df_sim['x_share'], label=name, color=colors.get(name, 'black'),
                linewidth=2.2, alpha=0.9)

    ax.axhline(0.80, color='gray', linestyle=':', label='High-Cooperation Threshold (80%)')
    ax.set_title('Figure 4: Prosumer Strategy Evolution x(t) Under Different Forecast-Game Frameworks',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Simulation Time (Hours)')
    ax.set_ylabel('Population Share Choosing "Share" Strategy x(t)')
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_tipping_point(sweep_df, save_path=None):
    """Figure 5: Final Share Adoption vs Uncertainty Level (Bifurcation / Tipping Point)."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sweep_df['Uncertainty_Scale_Pct'], sweep_df['Final_Share'], 'o-',
             color='#9467bd', linewidth=2.5, markersize=7)
    ax1.axhline(0.5, color='gray', linestyle='--')
    ax1.set_title('(A) Evolutionary Equilibrium x(T) vs Uncertainty', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Normalized Forecast Uncertainty (%)')
    ax1.set_ylabel('Final Share Adoption x(T)')
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(sweep_df['Uncertainty_Scale_Pct'], sweep_df['Community_Savings ($)'], 's-',
             color='#8c564b', linewidth=2.5, markersize=7)
    ax2.set_title('(B) Community Financial Savings vs Uncertainty', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Normalized Forecast Uncertainty (%)')
    ax2.set_ylabel('Cumulative P2P Savings ($)')

    plt.suptitle('Figure 5: Sensitivity Analysis & Critical Uncertainty Tipping Point',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_weather_shock_response(df_normal, df_shock, shock_range, save_path=None):
    """Figure 6: Weather Shock Response and Behavioral Recovery."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    time_hours = np.arange(len(df_shock)) * 0.25
    start_h = shock_range[0] * 0.25
    end_h = shock_range[1] * 0.25

    # Share trajectory
    ax1.plot(time_hours, df_normal['x_share'], label='Baseline Condition (No Shock)',
             color='#2ca02c', linestyle='--', linewidth=2.0)
    ax1.plot(time_hours, df_shock['x_share'], label='Severe Weather Shock Response',
             color='#d62728', linewidth=2.5)
    ax1.axvspan(start_h, end_h, color='red', alpha=0.15, label='Shock Duration (Sudden Storm & High Uncertainty)')
    ax1.set_ylabel('Share Adoption x(t)')
    ax1.set_title('Figure 6: Community Behavioral Response and Strategy Collapse During Weather Shock',
                  fontweight='bold', fontsize=13)
    ax1.legend(loc='lower left', frameon=True)
    ax1.set_ylim(-0.02, 1.05)

    # P2P traded energy
    ax2.plot(time_hours, df_normal['Q_P2P_kWh'], label='Normal P2P Traded Energy',
             color='#1f77b4', linestyle='--', linewidth=1.8)
    ax2.plot(time_hours, df_shock['Q_P2P_kWh'], label='P2P Traded Under Shock',
             color='#ff7f0e', linewidth=2.0)
    ax2.axvspan(start_h, end_h, color='red', alpha=0.15)
    ax2.set_ylabel('P2P Traded Energy (kWh)')
    ax2.set_xlabel('Simulation Time (Hours)')
    ax2.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
