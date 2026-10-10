"""
train_cnn.py
1D-Convolutional Neural Network (1D-CNN) implementation for 24-hour ahead
solar power forecasting (96 time steps) using 48-hour history (192 time steps).
Includes Persistence baselines, model training with early stopping,
and comprehensive multi-horizon & daylight-only evaluation metrics.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd


class Conv1DForecaster(nn.Module):
    """
    1D-CNN model architecture for multi-step solar forecasting with Dropout
    compatible with Monte Carlo Dropout uncertainty estimation.
    """
    def __init__(self, in_channels=9, seq_len=192, horizon=96, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_stack = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        # 192 / 4 = 48 feature maps of length 48
        self.flatten_dim = 128 * 48
        self.fc_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, horizon)
        )

    def forward(self, x):
        # Input shape: (Batch, Seq_Len, Channels) -> transpose to (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2)
        feat = self.conv_stack(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_head(feat)
        return out


def train_model(model, train_loader, val_loader, epochs=25, lr=1e-3, device='cpu', patience=5):
    """
    Trains the 1D-CNN model with Huber Loss, Adam optimizer, LR scheduling and Early Stopping.
    """
    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs or patience_counter >= patience:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history


def compute_forecast_metrics(y_true, y_pred, rated_capacity, name="Model"):
    """
    Computes all standard solar forecasting metrics as specified in the Implementation Plan:
    MAE, RMSE, nMAE (%), nRMSE (%), Daylight-only MAE/RMSE.
    """
    # Non-negativity physical constraint
    y_pred_clipped = np.clip(y_pred, 0.0, None)

    mae = np.mean(np.abs(y_true - y_pred_clipped))
    rmse = np.sqrt(np.mean((y_true - y_pred_clipped) ** 2))
    nmae = (mae / rated_capacity) * 100.0
    nrmse = (rmse / rated_capacity) * 100.0

    # Daylight-only mask (solar generation > 1% of capacity)
    daylight_mask = y_true > (0.01 * rated_capacity)
    if np.any(daylight_mask):
        daylight_mae = np.mean(np.abs(y_true[daylight_mask] - y_pred_clipped[daylight_mask]))
        daylight_rmse = np.sqrt(np.mean((y_true[daylight_mask] - y_pred_clipped[daylight_mask]) ** 2))
    else:
        daylight_mae, daylight_rmse = mae, rmse

    # Horizon breakdown (15 min = step 0, 1h = step 3, 6h = step 23, 12h = step 47, 24h = step 95)
    horizons = {'15-min (h=1)': 0, '1-hr (h=4)': 3, '6-hr (h=24)': 23, '12-hr (h=48)': 47, '24-hr (h=96)': 95}
    horizon_mae = {}
    for h_name, h_idx in horizons.items():
        if h_idx < y_true.shape[1]:
            horizon_mae[h_name] = np.mean(np.abs(y_true[:, h_idx] - y_pred_clipped[:, h_idx]))

    return {
        'Model': name,
        'MAE (kW)': mae,
        'RMSE (kW)': rmse,
        'nMAE (%)': nmae,
        'nRMSE (%)': nrmse,
        'Daylight MAE (kW)': daylight_mae,
        'Daylight RMSE (kW)': daylight_rmse,
        'Horizon_MAE': horizon_mae
    }


def persistence_baseline(X_scaled, target_scaler, feature_cols, target_col='AC_POWER', horizon_len=96):
    """
    Same-time-previous-day persistence baseline (Lag 96 steps).
    Uses the previous 24-hour profile from the history window.
    """
    target_idx = feature_cols.index(target_col)
    # The last 96 steps of history corresponds to t-96 to t (previous 24 hours)
    last_day_scaled = X_scaled[:, -horizon_len:, target_idx]
    # Unscale
    mean = target_scaler.mean_[0]
    std = target_scaler.scale_[0]
    last_day_actual = last_day_scaled * std + mean
    return np.clip(last_day_actual, 0.0, None)
