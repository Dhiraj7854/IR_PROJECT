"""
run_pipeline.py
Master execution script running the full end-to-end pipeline:
1. Data Ingestion & Preprocessing (Plant 1 + Plant 2)
2. Chronological Splitting & Windowing (192-step history -> 96-step forecast)
3. 1D-CNN Model Training with Huber Loss & Early Stopping
4. Persistence Baseline & Forecast Accuracy Evaluation (MAE, RMSE, nMAE, Daylight metrics)
5. Monte Carlo Dropout Uncertainty Quantification (PICP, MPIW, Prediction Intervals)
6. Dynamic Payoffs and Replicator Dynamics Simulation (M0, M1, M2, M3)
7. Uncertainty Sensitivity & Tipping Point Sweep
8. Weather Shock Response & Behavioral Resilience
9. Cross-Plant Generalization Test (Plant 2)
10. Publication-Quality Figures & CSV Summary Tables Export
"""

import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from src.preprocess import load_and_clean_plant_data, prepare_chronological_splits
from src.train_cnn import (Conv1DForecaster, train_model, compute_forecast_metrics,
                           persistence_baseline)
from src.uncertainty import (predict_mc_dropout, compute_prediction_intervals,
                             evaluate_uncertainty_calibration, calibrate_intervals_on_validation)
from src.payoffs import ProsumerCommunity
from src.replicator import ReplicatorEngine
from src.experiments import (run_comparative_experiment, run_uncertainty_sweep,
                             run_weather_shock_experiment, plot_forecast_actual_vs_pred,
                             plot_error_by_horizon, plot_uncertainty_calibration,
                             plot_comparative_trajectories, plot_tipping_point,
                             plot_weather_shock_response)


def main():
    print("=" * 80)
    print("FORECAST-AWARE EVOLUTIONARY GAME THEORY PIPELINE")
    print("1D-CNN Solar Forecasting + MC Dropout + Dynamic Replicator Dynamics")
    print("=" * 80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device}")

    # Paths
    p1_gen_path = 'Plant_1_Generation_Data.csv'
    p1_weather_path = 'Plant_1_Weather_Sensor_Data.csv'
    p2_gen_path = 'Plant_2_Generation_Data.csv'
    p2_weather_path = 'Plant_2_Weather_Sensor_Data.csv'

    # Step 1: Preprocessing Plant 1
    print("\n--- STEP 1: Data Ingestion & Preprocessing (Plant 1) ---")
    df_p1 = load_and_clean_plant_data(p1_gen_path, p1_weather_path, is_plant1=True)
    print(f"Plant 1 processed: {len(df_p1)} 15-minute intervals from {df_p1['DATE_TIME'].min()} to {df_p1['DATE_TIME'].max()}")
    df_p1.to_csv('data/processed/plant_1_cleaned.csv', index=False)

    feature_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION',
                    'sin_tod', 'cos_tod', 'sin_dow', 'cos_dow']
    
    print("\n--- STEP 2: Chronological Splitting & Windowing (192 in -> 96 out) ---")
    data_dict = prepare_chronological_splits(df_p1, feature_cols, history_len=192, horizon_len=96)
    
    X_train, y_train, dates_train = data_dict['train']
    X_val, y_val, dates_val = data_dict['val']
    X_test, y_test, dates_test = data_dict['test']
    scalers = data_dict['scalers']
    rated_capacity = scalers['rated_capacity']

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    print(f"Plant Rated AC Capacity: {rated_capacity:.2f} kW")

    # DataLoader
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_tensor_X = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Step 3: Train 1D-CNN
    print("\n--- STEP 3: Training 1D-CNN with Huber Loss & Early Stopping ---")
    torch.manual_seed(42)
    model = Conv1DForecaster(in_channels=len(feature_cols), seq_len=192, horizon=96, dropout_rate=0.2)
    model, history = train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device=device, patience=5)
    torch.save(model.state_dict(), 'data/processed/cnn_model_weights.pt')
    print("Model trained and saved to data/processed/cnn_model_weights.pt")

    # Step 4: Forecasting & Baselines
    print("\n--- STEP 4: Forecasting & Baseline Evaluation ---")
    # Persistence baseline
    y_test_unscaled = y_test * scalers['target_std'] + scalers['target_mean']
    y_persist = persistence_baseline(X_test, scalers['target_scaler'], feature_cols)
    metrics_persist = compute_forecast_metrics(y_test_unscaled, y_persist, rated_capacity, name="M0: Persistence Baseline")

    # Deterministic CNN prediction
    model.eval()
    with torch.no_grad():
        y_cnn_scaled = model(test_tensor_X.to(device)).cpu().numpy()
    y_cnn = np.clip(y_cnn_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None)
    metrics_cnn = compute_forecast_metrics(y_test_unscaled, y_cnn, rated_capacity, name="1D-CNN Deterministic")

    # Metrics table
    metrics_df = pd.DataFrame([metrics_persist, metrics_cnn])
    display_cols = ['Model', 'MAE (kW)', 'RMSE (kW)', 'nMAE (%)', 'nRMSE (%)', 'Daylight MAE (kW)', 'Daylight RMSE (kW)']
    print("\nForecast Performance Comparison:")
    print(metrics_df[display_cols].to_string(index=False))
    metrics_df[display_cols].to_csv('results/tables/table_1_forecast_metrics.csv', index=False)

    # Horizon breakdown plot
    plot_error_by_horizon(metrics_cnn['Horizon_MAE'], save_path='results/figures/figure_2_horizon_error.png')
    print("Saved Figure 2: Horizon error breakdown")

    # Step 5: Uncertainty Quantification with Monte Carlo Dropout
    print("\n--- STEP 5: Monte Carlo Dropout Uncertainty Quantification ---")
    mean_pred, std_pred, all_runs = predict_mc_dropout(model, test_tensor_X, scalers['target_scaler'], K=50, device=device)
    
    # Validation calibration
    val_tensor_X = torch.tensor(X_val, dtype=torch.float32)
    val_y_unscaled = y_val * scalers['target_std'] + scalers['target_mean']
    val_mean, val_std, _ = predict_mc_dropout(model, val_tensor_X, scalers['target_scaler'], K=30, device=device)
    calib_factor = calibrate_intervals_on_validation(val_y_unscaled, val_mean, val_std, nominal_coverage=0.90)
    print(f"Validation-calibrated scale factor for 90% interval: {calib_factor:.3f}")

    # Prediction intervals on test set
    lower_90, upper_90 = compute_prediction_intervals(mean_pred, std_pred, confidence_level=0.90, calibration_factor=calib_factor)
    eval_unc = evaluate_uncertainty_calibration(y_test_unscaled, lower_90, upper_90, nominal_coverage=0.90)
    print(f"Test PICP: {eval_unc['PICP (%)']:.2f}% (Nominal: 90%) | MPIW: {eval_unc['MPIW (kW)']:.2f} kW | ACE: {eval_unc['ACE (%)']:.2f}%")

    unc_df = pd.DataFrame([{
        'Nominal Coverage (%)': 90.0,
        'Empirical PICP (%)': eval_unc['PICP (%)'],
        'MPIW (kW)': eval_unc['MPIW (kW)'],
        'ACE (%)': eval_unc['ACE (%)'],
        'Calibration Factor': calib_factor
    }])
    unc_df.to_csv('results/tables/table_2_uncertainty_calibration.csv', index=False)

    # Calibration across multiple nominal levels for Figure 3
    calib_curve = {}
    for lvl in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        low, high = compute_prediction_intervals(mean_pred, std_pred, confidence_level=lvl, calibration_factor=calib_factor)
        calib_curve[lvl] = float(np.mean((y_test_unscaled >= low) & (y_test_unscaled <= high)) * 100.0)
    plot_uncertainty_calibration(calib_curve, save_path='results/figures/figure_3_uncertainty_calibration.png')
    print("Saved Figure 3: Uncertainty calibration curve")

    # Figure 1: Representative Day Actual vs Predicted + Band
    sample_day_idx = min(96, len(y_test_unscaled) - 1)
    plot_forecast_actual_vs_pred(
        dates_test[sample_day_idx],
        y_test_unscaled[sample_day_idx],
        mean_pred[sample_day_idx],
        lower_90[sample_day_idx],
        upper_90[sample_day_idx],
        save_path='results/figures/figure_1_actual_vs_predicted.png'
    )
    print("Saved Figure 1: Actual vs Predicted with Uncertainty Band")

    # Step 6: Evolutionary Game & Replicator Dynamics (M0, M1, M2, M3)
    print("\n--- STEP 6: Dynamic Evolutionary Game & Replicator Dynamics ---")
    community = ProsumerCommunity(num_prosumers=100, plant_capacity_kw=rated_capacity, p_grid=0.22, p_fit=0.06, c_tx=0.01)
    
    # Flatten test timestamps (using 1st step of each consecutive window)
    test_dates_flat = pd.to_datetime(dates_test[:, 0])
    
    models_dict = run_comparative_experiment(
        community,
        test_dates=test_dates_flat,
        y_true=y_test_unscaled,
        y_cnn=mean_pred,
        sigma_cnn=std_pred,
        y_persist=y_persist,
        x0=0.50,
        lambda_s=2.5
    )

    # Comparison Table
    summary_rows = []
    for name, (df_sim, summary) in models_dict.items():
        summary['Model'] = name
        summary_rows.append(summary)
        # Save individual simulation trace
        clean_name = name.split(' ')[0].replace('(', '').replace(')', '')
        df_sim.to_csv(f'results/tables/simulation_{clean_name}.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    cols_order = ['Model', 'Final_Share', 'Mean_Share', 'Time_Above_Threshold',
                  'Total_P2P_Traded_kWh', 'Total_Cost ($)', 'Total_Savings ($)', 'Savings_Pct (%)']
    print("\nEvolutionary Game Model Comparison:")
    print(summary_df[cols_order].to_string(index=False))
    summary_df[cols_order].to_csv('results/tables/table_3_model_comparison.csv', index=False)

    # Figure 4: Comparative Trajectories
    plot_comparative_trajectories(models_dict, save_path='results/figures/figure_4_comparative_trajectories.png')
    print("Saved Figure 4: Comparative evolutionary trajectories")

    # Step 7: Uncertainty Sensitivity & Tipping Point Analysis
    print("\n--- STEP 7: Sensitivity & Critical Tipping Point Sweep ---")
    loads = community.generate_diurnal_load(test_dates_flat)
    gen_cnn = mean_pred[:, 0]
    sigma_cnn = std_pred[:, 0]

    sweep_df, _ = run_uncertainty_sweep(
        community, test_dates=test_dates_flat,
        gen_series=gen_cnn, sigma_series=sigma_cnn,
        loads=loads, x0=0.50, lambda_s=2.5,
        multipliers=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    )
    print("\nUncertainty Sweep Results:")
    print(sweep_df[['Uncertainty_Multiplier', 'Uncertainty_Scale_Pct', 'Final_Share', 'Community_Savings ($)']].to_string(index=False))
    sweep_df.to_csv('results/tables/table_4_tipping_point_sweep.csv', index=False)

    plot_tipping_point(sweep_df, save_path='results/figures/figure_5_tipping_point.png')
    print("Saved Figure 5: Bifurcation and tipping point curves")

    # Step 8: Weather Shock Response
    print("\n--- STEP 8: Weather Shock & Behavioral Resilience Experiment ---")
    df_normal, df_shock, shock_metrics, shock_range = run_weather_shock_experiment(
        community, test_dates=test_dates_flat,
        gen_series=gen_cnn, sigma_series=sigma_cnn,
        loads=loads, x0=0.85, lambda_s=2.5,
        shock_start_idx=96, shock_duration=24, shock_drop=0.75
    )
    print("\nWeather Shock Response Metrics:")
    for k, v in shock_metrics.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    pd.DataFrame([shock_metrics]).to_csv('results/tables/table_5_weather_shock_metrics.csv', index=False)
    plot_weather_shock_response(df_normal, df_shock, shock_range, save_path='results/figures/figure_6_weather_shock.png')
    print("Saved Figure 6: Weather shock response and recovery dynamics")

    # Step 9: Cross-Plant Generalization Test (Plant 2)
    print("\n--- STEP 9: Cross-Plant Generalization Test (Plant 2) ---")
    df_p2 = load_and_clean_plant_data(p2_gen_path, p2_weather_path, is_plant1=False)
    print(f"Loaded Plant 2: {len(df_p2)} intervals")
    df_p2.to_csv('data/processed/plant_2_cleaned.csv', index=False)

    # Use Plant 1 scalers to evaluate zero-shot transfer
    data_dict_p2 = prepare_chronological_splits(df_p2, feature_cols, history_len=192, horizon_len=96)
    X_test_p2, y_test_p2, dates_test_p2 = data_dict_p2['test']
    y_test_p2_unscaled = y_test_p2 * data_dict_p2['scalers']['target_std'] + data_dict_p2['scalers']['target_mean']

    test_tensor_p2 = torch.tensor(X_test_p2, dtype=torch.float32)
    with torch.no_grad():
        p2_pred_scaled = model(test_tensor_p2.to(device)).cpu().numpy()
    p2_pred_unscaled = np.clip(p2_pred_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None)

    metrics_p2 = compute_forecast_metrics(y_test_p2_unscaled, p2_pred_unscaled, data_dict_p2['scalers']['rated_capacity'], name="Plant 2 Transfer Test")
    print(f"Plant 2 Zero-Shot Transfer: MAE = {metrics_p2['MAE (kW)']:.2f} kW | nMAE = {metrics_p2['nMAE (%)']:.2f}%")
    pd.DataFrame([metrics_p2])[display_cols].to_csv('results/tables/table_6_plant2_generalization.csv', index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Generated Artifacts:")
    print("  Figures: results/figures/ (Figures 1 to 6)")
    print("  Tables:  results/tables/  (Tables 1 to 6)")
    print("=" * 80)


if __name__ == '__main__':
    main()
