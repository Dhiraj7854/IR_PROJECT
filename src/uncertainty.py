"""
uncertainty.py
Forecast uncertainty quantification using Monte Carlo Dropout (Gal & Ghahramani).
Generates predictive distributions, standard deviations (uncertainty proxy),
calibrated prediction intervals, and evaluates PICP and MPIW metrics.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


def enable_mc_dropout(model):
    """
    Sets Dropout layers to training mode while keeping BatchNorm layers in eval mode.
    This enables stochastic forward passes without altering running statistics.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


def predict_mc_dropout(model, X_tensor, target_scaler, K=100, device='cpu', batch_size=64):
    """
    Performs K stochastic forward passes using Monte Carlo Dropout.
    Returns:
    - mean_pred: Point forecast in physical power units (kW) [num_samples, horizon]
    - std_pred: Predictive standard deviation / uncertainty proxy in physical power units (kW) [num_samples, horizon]
    - all_preds: Full ensemble of K predictions [K, num_samples, horizon]
    """
    enable_mc_dropout(model)
    model.to(device)

    all_runs = []
    num_samples = len(X_tensor)

    with torch.no_grad():
        for k in range(K):
            preds_k = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = X_tensor[start:end].to(device)
                out = model(x_batch).cpu().numpy()
                preds_k.append(out)
            preds_k = np.concatenate(preds_k, axis=0)
            all_runs.append(preds_k)

    # Shape: (K, num_samples, horizon)
    all_runs = np.array(all_runs)

    # Unscale predictions back to actual physical units (kW)
    mean = target_scaler.mean_[0]
    std = target_scaler.scale_[0]
    all_runs_unscaled = all_runs * std + mean
    # Clip negative physical power to 0
    all_runs_unscaled = np.clip(all_runs_unscaled, 0.0, None)

    mean_pred = np.mean(all_runs_unscaled, axis=0)
    std_pred = np.std(all_runs_unscaled, axis=0, ddof=1)

    return mean_pred, std_pred, all_runs_unscaled


def compute_prediction_intervals(mean_pred, std_pred, confidence_level=0.90, calibration_factor=1.0):
    """
    Calculates symmetric prediction intervals around point forecast:
    [max(0, mean - z * sigma * calib), mean + z * sigma * calib]
    """
    from scipy.stats import norm
    alpha = 1.0 - confidence_level
    z_score = norm.ppf(1.0 - alpha / 2.0)

    half_width = z_score * std_pred * calibration_factor
    lower_bound = np.clip(mean_pred - half_width, 0.0, None)
    upper_bound = mean_pred + half_width

    return lower_bound, upper_bound


def evaluate_uncertainty_calibration(y_true, lower_bound, upper_bound, nominal_coverage=0.90):
    """
    Computes standard probabilistic forecasting metrics:
    - PICP: Prediction Interval Coverage Probability (fraction of actuals in interval)
    - MPIW: Mean Prediction Interval Width (sharpness)
    - ACE: Average Coverage Error (|PICP - nominal|)
    """
    in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp = np.mean(in_interval)
    mpiw = np.mean(upper_bound - lower_bound)
    ace = picp - nominal_coverage

    # Coverage by horizon (e.g. 15min, 1h, 6h, 12h, 24h)
    horizons = {'15-min': 0, '1-hr': 3, '6-hr': 23, '12-hr': 47, '24-hr': 95}
    coverage_by_lead = {}
    for h_name, h_idx in horizons.items():
        if h_idx < y_true.shape[1]:
            coverage_by_lead[h_name] = float(np.mean(in_interval[:, h_idx]))

    return {
        'Nominal Coverage': nominal_coverage,
        'PICP (%)': picp * 100.0,
        'MPIW (kW)': mpiw,
        'ACE (%)': ace * 100.0,
        'Coverage_by_Lead': coverage_by_lead
    }


def calibrate_intervals_on_validation(val_y_true, val_mean, val_std, nominal_coverage=0.90):
    """
    Calibrates prediction interval scale factor on validation residuals
    so that empirical PICP matches nominal coverage.
    """
    from scipy.optimize import minimize_scalar

    def objective(factor):
        low, high = compute_prediction_intervals(val_mean, val_std, nominal_coverage, factor)
        picp = np.mean((val_y_true >= low) & (val_y_true <= high))
        return (picp - nominal_coverage) ** 2

    res = minimize_scalar(objective, bounds=(0.5, 5.0), method='bounded')
    return float(res.x)


def calibrate_conformal_half_width(val_y_true, val_mean, coverage=0.90):
    """Fit a finite-sample split-conformal absolute-residual radius per lead."""
    if val_y_true.shape != val_mean.shape or val_y_true.ndim != 2:
        raise ValueError("Validation targets and forecasts must be matching 2D arrays")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")

    residuals = np.abs(val_y_true - val_mean)
    n = residuals.shape[0]
    rank = min(int(np.ceil((n + 1) * coverage)), n)
    return np.partition(residuals, rank - 1, axis=0)[rank - 1, :]


def compute_conformal_intervals(mean_pred, half_width):
    """Apply per-lead split-conformal radii to nonnegative power forecasts."""
    mean_pred = np.asarray(mean_pred)
    half_width = np.asarray(half_width)
    if mean_pred.ndim != 2 or half_width.shape != (mean_pred.shape[1],):
        raise ValueError("half_width must have one value for each forecast lead")
    return np.maximum(0.0, mean_pred - half_width), mean_pred + half_width


def evaluate_probabilistic_by_lead(y_true, mean_pred, lower, upper,
                                   predictive_sigma, nominal_coverage=0.90):
    """Report interval calibration, interval pinball loss, and Gaussian CRPS per lead."""
    from scipy.stats import norm

    y_true = np.asarray(y_true)
    mean_pred = np.asarray(mean_pred)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if not (y_true.shape == mean_pred.shape == lower.shape == upper.shape):
        raise ValueError("Targets, means, and interval bounds must have matching shapes")
    if y_true.ndim != 2:
        raise ValueError("Forecast arrays must be two-dimensional")

    sigma = np.broadcast_to(np.asarray(predictive_sigma), mean_pred.shape)
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mean_pred) / sigma
    crps = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )

    alpha = 1.0 - nominal_coverage
    def pinball(actual, quantile, level):
        error = actual - quantile
        return np.maximum(level * error, (level - 1.0) * error)

    rows = []
    for lead in range(y_true.shape[1]):
        covered = (y_true[:, lead] >= lower[:, lead]) & (y_true[:, lead] <= upper[:, lead])
        interval_pinball = (
            pinball(y_true[:, lead], lower[:, lead], alpha / 2.0)
            + pinball(y_true[:, lead], upper[:, lead], 1.0 - alpha / 2.0)
        ) / 2.0
        rows.append({
            'Lead (15-min steps)': lead + 1,
            'PICP (%)': np.mean(covered) * 100.0,
            'MPIW (kW)': np.mean(upper[:, lead] - lower[:, lead]),
            'Pinball (kW)': np.mean(interval_pinball),
            'CRPS (kW)': np.mean(crps[:, lead]),
        })
    return pd.DataFrame(rows)
