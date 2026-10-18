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
import yaml

from src.preprocess import (load_and_clean_plant_data, make_future_covariates,
                            prepare_chronological_splits)
from src.train_cnn import (Conv1DForecaster, train_model, compute_forecast_metrics,
                           persistence_baseline)
from src.uncertainty import (calibrate_conformal_half_width, compute_conformal_intervals,
                             evaluate_probabilistic_by_lead)
from src.payoffs import ProsumerCommunity
from src.replicator import ReplicatorEngine
from src.experiments import (run_comparative_experiment, run_uncertainty_sweep,
                             run_weather_shock_experiment, plot_forecast_actual_vs_pred,
                             plot_error_by_horizon, plot_uncertainty_calibration,
                             plot_comparative_trajectories, plot_tipping_point,
                             plot_weather_shock_response)


def main():
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

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
    latitude = config['forecast']['latitude']
    longitude = config['forecast']['longitude']
    timezone = config['forecast']['timezone']
    future_train = make_future_covariates(dates_train, latitude, longitude, timezone)
    future_val = make_future_covariates(dates_val, latitude, longitude, timezone)
    future_test = make_future_covariates(dates_test, latitude, longitude, timezone)
    target_index = feature_cols.index('AC_POWER')
    persistence_train_scaled = X_train[:, -96:, target_index]
    persistence_val_scaled = X_val[:, -96:, target_index]
    persistence_test_scaled = X_test[:, -96:, target_index]
    train_values = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.float32)
    val_values = torch.tensor(X_val, dtype=torch.float32)
    val_targets = torch.tensor(y_val, dtype=torch.float32)
    test_tensor_X = torch.tensor(X_test, dtype=torch.float32)

    # Train both the plain CNN baseline and a persistence-residual CNN with
    # future-known solar geometry covariates; compare on validation only.
    print("\n--- STEP 3: Five-seed CNN and persistence-residual model comparison ---")
    model_specs = {
        'plain_cnn': {'future_features': 0, 'predict_residual': False},
        'future_covariate_residual_cnn': {'future_features': 3, 'predict_residual': True},
    }
    seeds = config['forecast']['seeds']
    trained = {}
    for model_name, model_options in model_specs.items():
        models, val_predictions, test_predictions, histories = [], [], [], []
        for seed in seeds:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = Conv1DForecaster(
                in_channels=len(feature_cols), seq_len=192, horizon=96, dropout_rate=0.2,
                **model_options,
            )
            if model_options['future_features']:
                train_ds = TensorDataset(
                    train_values, train_targets, torch.tensor(future_train),
                    torch.tensor(persistence_train_scaled, dtype=torch.float32),
                )
                val_ds = TensorDataset(
                    val_values, val_targets, torch.tensor(future_val),
                    torch.tensor(persistence_val_scaled, dtype=torch.float32),
                )
            else:
                train_ds = TensorDataset(train_values, train_targets)
                val_ds = TensorDataset(val_values, val_targets)
            train_loader = DataLoader(
                train_ds, batch_size=config['forecast']['batch_size'], shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            val_loader = DataLoader(val_ds, batch_size=config['forecast']['batch_size'], shuffle=False)
            model, history = train_model(
                model, train_loader, val_loader, epochs=config['forecast']['epochs'],
                lr=config['forecast']['learning_rate'], device=device,
                patience=config['forecast']['early_stopping_patience'],
            )
            models.append(model)
            histories.append(history)
            model.eval()
            with torch.no_grad():
                if model_options['future_features']:
                    val_scaled = model(
                        val_values.to(device), torch.tensor(future_val, dtype=torch.float32, device=device),
                        torch.tensor(persistence_val_scaled, dtype=torch.float32, device=device),
                    ).cpu().numpy()
                    test_scaled = model(
                        test_tensor_X.to(device), torch.tensor(future_test, dtype=torch.float32, device=device),
                        torch.tensor(persistence_test_scaled, dtype=torch.float32, device=device),
                    ).cpu().numpy()
                else:
                    val_scaled = model(val_values.to(device)).cpu().numpy()
                    test_scaled = model(test_tensor_X.to(device)).cpu().numpy()
            val_predictions.append(np.clip(
                val_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None
            ))
            test_predictions.append(np.clip(
                test_scaled * scalers['target_std'] + scalers['target_mean'], 0.0, None
            ))
            print(f"{model_name}, seed={seed}: best validation Huber={history['best_val_loss']:.5f}")
        trained[model_name] = {
            'models': models,
            'val_predictions': np.stack(val_predictions),
            'test_predictions': np.stack(test_predictions),
            'histories': histories,
            'validation_mae': np.mean(np.abs(
                np.mean(val_predictions, axis=0)
                - (y_val * scalers['target_std'] + scalers['target_mean'])
            )),
        }

    selected_name = min(trained, key=lambda name: trained[name]['validation_mae'])
    selected = trained[selected_name]
    print(f"Selected on validation MAE: {selected_name} ({selected['validation_mae']:.3f} kW)")
    models = selected['models']
    val_mean = np.mean(selected['val_predictions'], axis=0)
    mean_pred = np.mean(selected['test_predictions'], axis=0)
    torch.save({
        'architecture': selected_name,
        'seeds': seeds,
        'state_dicts': [
            {name: value.detach().cpu() for name, value in member.state_dict().items()}
            for member in models
        ],
    }, 'data/processed/cnn_model_weights.pt')

    # Step 4: Report lead-specific test skill against same-time-yesterday persistence.
    print("\n--- STEP 4: Five-seed forecast skill against persistence ---")
    y_test_unscaled = y_test * scalers['target_std'] + scalers['target_mean']
    y_persist = persistence_baseline(X_test, scalers['target_scaler'], feature_cols)
    display_cols = ['Model', 'MAE (kW)', 'RMSE (kW)', 'nMAE (%)', 'nRMSE (%)', 'Daylight MAE (kW)', 'Daylight RMSE (kW)']
    metrics_df = pd.DataFrame([
        compute_forecast_metrics(y_test_unscaled, y_persist, rated_capacity, name='Persistence'),
        compute_forecast_metrics(y_test_unscaled, mean_pred, rated_capacity, name=selected_name),
    ])
    print("\nForecast Performance Comparison:")
    print(metrics_df[display_cols].to_string(index=False))
    metrics_df[display_cols].to_csv('results/tables/table_1_forecast_metrics.csv', index=False)

    lead_points = {'15m': 0, '1h': 3, '6h': 23, '12h': 47, '24h': 95}
    skill_rows = []
    for label, lead in lead_points.items():
        persist_mae = np.mean(np.abs(y_test_unscaled[:, lead] - y_persist[:, lead]))
        for seed_index, seed in enumerate(seeds):
            cnn_mae = np.mean(np.abs(
                y_test_unscaled[:, lead] - selected['test_predictions'][seed_index, :, lead]
            ))
            skill_rows.append({
                'Model': selected_name, 'Lead': label, 'Seed': seed,
                'Persistence_MAE_kW': persist_mae, 'CNN_MAE_kW': cnn_mae,
                'Skill_vs_persistence': 1.0 - cnn_mae / max(persist_mae, 1e-12),
            })
    skill_by_seed = pd.DataFrame(skill_rows)
    skill_summary = skill_by_seed.groupby(['Model', 'Lead'], sort=False).agg(
        Skill_mean=('Skill_vs_persistence', 'mean'),
        Skill_std=('Skill_vs_persistence', 'std'),
        Persistence_MAE_kW=('Persistence_MAE_kW', 'first'),
        CNN_MAE_kW=('CNN_MAE_kW', 'mean'),
    ).reset_index()
    skill_summary['Skill_std'] = skill_summary['Skill_std'].fillna(0.0)
    skill_summary.to_csv('results/tables/table_1b_lead_skill.csv', index=False)
    print("\nCNN skill vs persistence by lead (mean +/- std across five seeds):")
    print(skill_summary.to_string(index=False))

    # Horizon breakdown plot
    plot_error_by_horizon(metrics_df.iloc[1]['Horizon_MAE'], save_path='results/figures/figure_2_horizon_error.png')
    print("Saved Figure 2: Horizon error breakdown")

    # Step 5: Per-lead split-conformal intervals calibrated on validation residuals.
    print("\n--- STEP 5: Split-conformal uncertainty calibration ---")
    val_y_unscaled = y_val * scalers['target_std'] + scalers['target_mean']
    coverage = config['forecast']['interval_coverage']
    conformal_width = calibrate_conformal_half_width(val_y_unscaled, val_mean, coverage=coverage)
    sigma_by_lead = np.std(val_y_unscaled - val_mean, axis=0, ddof=1)
    std_pred = np.broadcast_to(sigma_by_lead, mean_pred.shape).copy()
    lower_90, upper_90 = compute_conformal_intervals(mean_pred, conformal_width)
    lead_uncertainty = evaluate_probabilistic_by_lead(
        y_test_unscaled, mean_pred, lower_90, upper_90, sigma_by_lead, nominal_coverage=coverage,
    )
    lead_uncertainty.to_csv('results/tables/table_2_uncertainty_calibration.csv', index=False)
    print(lead_uncertainty.iloc[[0, 3, 23, 47, 95]].to_string(index=False))

    # Calibration across multiple nominal levels for Figure 3
    calib_curve = {}
    for lvl in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        level_width = calibrate_conformal_half_width(val_y_unscaled, val_mean, coverage=lvl)
        low, high = compute_conformal_intervals(mean_pred, level_width)
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

    # Save non-overlapping 00:00 issuance forecasts and outcomes for validation and test days.
    rolling = []
    for split_name, split_dates, split_actual, split_mean in (
        ('validation', dates_val, val_y_unscaled, val_mean),
        ('test', dates_test, y_test_unscaled, mean_pred),
    ):
        target_starts = pd.DatetimeIndex(pd.to_datetime(split_dates[:, 0]))
        day_indices = np.flatnonzero((target_starts.hour == 0) & (target_starts.minute == 0))
        for index in day_indices:
            rolling.append((
                target_starts[index].to_datetime64(), split_name, split_mean[index],
                sigma_by_lead, split_actual[index],
            ))
    rolling.sort(key=lambda item: item[0])
    np.savez_compressed(
        'data/processed/day_ahead_forecasts.npz',
        date=np.array([row[0] for row in rolling]),
        split=np.array([row[1] for row in rolling]),
        mu=np.stack([row[2] for row in rolling]),
        sigma=np.stack([row[3] for row in rolling]),
        actual=np.stack([row[4] for row in rolling]),
    )
    print(f"Saved {len(rolling)} rolling 00:00 day-ahead forecasts.")

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
    p2_raw = data_dict_p2['scalers']['feature_scaler'].inverse_transform(
        X_test_p2.reshape(-1, len(feature_cols))
    ).reshape(X_test_p2.shape)
    X_test_p2_model = scalers['feature_scaler'].transform(
        p2_raw.reshape(-1, len(feature_cols))
    ).reshape(p2_raw.shape).astype(np.float32)
    future_test_p2 = make_future_covariates(dates_test_p2, latitude, longitude, timezone)
    persistence_p2_scaled = X_test_p2_model[:, -96:, target_index]
    test_tensor_p2 = torch.tensor(X_test_p2_model, dtype=torch.float32)
    p2_forecasts = []
    with torch.no_grad():
        for member in models:
            member.eval()
            if selected_name == 'future_covariate_residual_cnn':
                p2_pred_scaled = member(
                    test_tensor_p2.to(device),
                    torch.tensor(future_test_p2, dtype=torch.float32, device=device),
                    torch.tensor(persistence_p2_scaled, dtype=torch.float32, device=device),
                ).cpu().numpy()
            else:
                p2_pred_scaled = member(test_tensor_p2.to(device)).cpu().numpy()
            p2_forecasts.append(p2_pred_scaled)
    p2_pred_unscaled = np.clip(
        np.mean(p2_forecasts, axis=0) * scalers['target_std'] + scalers['target_mean'],
        0.0, None,
    )

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
