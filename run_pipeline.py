"""
run_pipeline.py
Master execution script running the full end-to-end pipeline:
1. Data Ingestion & Preprocessing
2. Chronological Splitting & Windowing (192-step history -> 96-step forecast)
3. Five-seed CNN training and persistence skill evaluation
4. Validation-calibrated split-conformal forecast intervals
5. Rolling day-ahead forecast archive
6. Forecast-provider evolutionary game, equilibrium, sensitivity, and shock analyses
"""

import os
from pathlib import Path

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
from src.experiments import (plot_error_by_horizon, plot_forecast_actual_vs_pred,
                             plot_uncertainty_calibration, run_research_experiments)


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    paths = config['paths']
    forecast_config = config['forecast']
    processed_dir = Path(paths['processed_dir'])
    results_dir = Path(paths['results_dir'])
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FORECAST-AWARE EVOLUTIONARY GAME THEORY PIPELINE")
    print("Five-seed CNN Forecasting + Conformal Intervals + Daily Evolutionary Game")
    print("=" * 80)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device}")

    # Paths
    p1_gen_path = paths['plant1_generation']
    p1_weather_path = paths['plant1_weather']

    # Step 1: Preprocessing Plant 1
    print("\n--- STEP 1: Data Ingestion & Preprocessing (Plant 1) ---")
    df_p1 = load_and_clean_plant_data(p1_gen_path, p1_weather_path, is_plant1=True)
    print(f"Plant 1 processed: {len(df_p1)} 15-minute intervals from {df_p1['DATE_TIME'].min()} to {df_p1['DATE_TIME'].max()}")
    df_p1.to_csv(processed_dir / 'plant_1_cleaned.csv', index=False)

    feature_cols = forecast_config['feature_columns']
    history_steps = forecast_config['history_steps']
    horizon_steps = forecast_config['horizon_steps']
    
    print(f"\n--- STEP 2: Chronological Splitting ({history_steps} in -> {horizon_steps} out) ---")
    data_dict = prepare_chronological_splits(
        df_p1, feature_cols,
        history_len=history_steps,
        horizon_len=horizon_steps,
        train_ratio=forecast_config['train_ratio'],
        val_ratio=forecast_config['validation_ratio'],
    )
    
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
    persistence_train_scaled = X_train[:, -horizon_steps:, target_index]
    persistence_val_scaled = X_val[:, -horizon_steps:, target_index]
    persistence_test_scaled = X_test[:, -horizon_steps:, target_index]
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
                in_channels=len(feature_cols), seq_len=history_steps,
                horizon=horizon_steps, dropout_rate=forecast_config['dropout_rate'],
                conv_channels=tuple(forecast_config['conv_channels']),
                dense_width=forecast_config['dense_width'],
                kernel_size=forecast_config['kernel_size'],
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
                weight_decay=forecast_config['weight_decay'],
                scheduler_factor=forecast_config['scheduler_factor'],
                scheduler_patience=forecast_config['scheduler_patience'],
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
    }, processed_dir / 'cnn_model_weights.pt')

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
    metrics_df[display_cols].to_csv(tables_dir / 'table_1_forecast_metrics.csv', index=False)

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
    skill_summary.to_csv(tables_dir / 'table_1b_lead_skill.csv', index=False)
    print("\nCNN skill vs persistence by lead (mean +/- std across five seeds):")
    print(skill_summary.to_string(index=False))

    # Horizon breakdown plot
    plot_error_by_horizon(metrics_df.iloc[1]['Horizon_MAE'], save_path=figures_dir / 'figure_2_horizon_error.png')
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
    lead_uncertainty.to_csv(tables_dir / 'table_2_uncertainty_calibration.csv', index=False)
    print(lead_uncertainty.iloc[[0, 3, 23, 47, 95]].to_string(index=False))

    # Calibration across multiple nominal levels for Figure 3
    calib_curve = {}
    for lvl in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        level_width = calibrate_conformal_half_width(val_y_unscaled, val_mean, coverage=lvl)
        low, high = compute_conformal_intervals(mean_pred, level_width)
        calib_curve[lvl] = float(np.mean((y_test_unscaled >= low) & (y_test_unscaled <= high)) * 100.0)
    plot_uncertainty_calibration(calib_curve, save_path=figures_dir / 'figure_3_uncertainty_calibration.png')
    print("Saved Figure 3: Uncertainty calibration curve")

    # Figure 1: Representative Day Actual vs Predicted + Band
    sample_day_idx = min(96, len(y_test_unscaled) - 1)
    plot_forecast_actual_vs_pred(
        dates_test[sample_day_idx],
        y_test_unscaled[sample_day_idx],
        mean_pred[sample_day_idx],
        lower_90[sample_day_idx],
        upper_90[sample_day_idx],
        save_path=figures_dir / 'figure_1_actual_vs_predicted.png'
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
        processed_dir / 'day_ahead_forecasts.npz',
        date=np.array([row[0] for row in rolling]),
        split=np.array([row[1] for row in rolling]),
        mu=np.stack([row[2] for row in rolling]),
        sigma=np.stack([row[3] for row in rolling]),
        actual=np.stack([row[4] for row in rolling]),
    )
    print(f"Saved {len(rolling)} rolling 00:00 day-ahead forecasts.")

    print("\n--- STEP 6: Forecast-provider evolutionary game and scenario analyses ---")
    run_research_experiments(
        forecast_path=processed_dir / "day_ahead_forecasts.npz",
        cleaned_path=processed_dir / "plant_1_cleaned.csv",
        config=config,
        output_root=results_dir,
    )

    print("\n--- Plant 2 zero-shot generalization check ---")
    df_p2 = load_and_clean_plant_data(
        paths['plant2_generation'], paths['plant2_weather'], is_plant1=False,
    )
    df_p2.to_csv(processed_dir / 'plant_2_cleaned.csv', index=False)
    data_dict_p2 = prepare_chronological_splits(
        df_p2, feature_cols,
        history_len=history_steps,
        horizon_len=horizon_steps,
        train_ratio=forecast_config['train_ratio'],
        val_ratio=forecast_config['validation_ratio'],
    )
    X_test_p2, y_test_p2, _ = data_dict_p2['test']
    p2_scaled_features = data_dict_p2['scalers']['feature_scaler'].inverse_transform(
        X_test_p2.reshape(-1, len(feature_cols))
    )
    X_test_p2_model = scalers['feature_scaler'].transform(
        p2_scaled_features
    ).reshape(X_test_p2.shape).astype(np.float32)
    future_test_p2 = make_future_covariates(
        data_dict_p2['test'][2], latitude, longitude, timezone,
    )
    persistence_test_p2 = X_test_p2_model[:, -horizon_steps:, target_index]
    y_test_p2_unscaled = (
        y_test_p2 * data_dict_p2['scalers']['target_std']
        + data_dict_p2['scalers']['target_mean']
    )
    p2_predictions = []
    with torch.no_grad():
        for member in models:
            member.eval()
            if selected_name == 'future_covariate_residual_cnn':
                prediction = member(
                    torch.tensor(X_test_p2_model, dtype=torch.float32, device=device),
                    torch.tensor(future_test_p2, dtype=torch.float32, device=device),
                    torch.tensor(persistence_test_p2, dtype=torch.float32, device=device),
                ).cpu().numpy()
            else:
                prediction = member(
                    torch.tensor(X_test_p2_model, dtype=torch.float32, device=device),
                ).cpu().numpy()
            p2_predictions.append(prediction)
    p2_mean = np.clip(
        np.mean(p2_predictions, axis=0) * scalers['target_std'] + scalers['target_mean'],
        0.0, None,
    )
    p2_metrics = compute_forecast_metrics(
        y_test_p2_unscaled,
        p2_mean,
        data_dict_p2['scalers']['rated_capacity'],
        name=f"Plant 2 transfer ({selected_name})",
    )
    pd.DataFrame([{
        'Model': p2_metrics['Model'],
        'MAE (kW)': p2_metrics['MAE (kW)'],
        'RMSE (kW)': p2_metrics['RMSE (kW)'],
        'nMAE (%)': p2_metrics['nMAE (%)'],
        'nRMSE (%)': p2_metrics['nRMSE (%)'],
        'Daylight MAE (kW)': p2_metrics['Daylight MAE (kW)'],
        'Daylight RMSE (kW)': p2_metrics['Daylight RMSE (kW)'],
    }]).to_csv(tables_dir / 'table_plant2_generalization.csv', index=False)
    print("\nAll forecasting and research experiments completed.")


if __name__ == '__main__':
    main()
