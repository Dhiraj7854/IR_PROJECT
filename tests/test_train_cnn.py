import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train_cnn import Conv1DForecaster, train_model
from src.preprocess import prepare_chronological_splits
from src.uncertainty import (
    calibrate_conformal_half_width,
    compute_conformal_intervals,
    evaluate_probabilistic_by_lead,
)


def test_early_stopping_restores_weights_with_best_validation_loss():
    torch.manual_seed(11)
    features = torch.randn(48, 4)
    targets = features[:, :1] * 2.0 - features[:, 1:2]
    loader = DataLoader(TensorDataset(features, targets), batch_size=8)
    model = nn.Linear(4, 1)

    model, history = train_model(model, loader, loader, epochs=8, lr=0.05, patience=2)
    model.eval()
    with torch.no_grad():
        restored_loss = nn.functional.smooth_l1_loss(model(features), targets).item()

    assert np.isclose(restored_loss, history['best_val_loss'], rtol=1e-6, atol=1e-7)


def test_future_covariate_residual_forecaster_requires_inputs():
    model = Conv1DForecaster(future_features=3, predict_residual=True)
    with torch.no_grad():
        model.fc_head[-1].weight.zero_()
        model.fc_head[-1].bias.zero_()
    history = torch.zeros(2, 192, 9)
    future = torch.zeros(2, 96, 3)
    persistence = torch.ones(2, 96)

    prediction = model(history, future, persistence)

    assert prediction.shape == (2, 96)
    assert torch.equal(prediction, persistence)


def test_split_conformal_intervals_are_per_lead_and_report_metrics():
    actual = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
    mean = actual - np.array([0.25, 0.5])
    width = calibrate_conformal_half_width(actual, mean, coverage=0.75)
    lower, upper = compute_conformal_intervals(mean, width)

    assert width.shape == (2,)
    assert np.all(lower <= actual)
    assert np.all(actual <= upper)
    metrics = evaluate_probabilistic_by_lead(
        actual, mean, lower, upper, predictive_sigma=np.array([0.5, 1.0]),
    )
    assert list(metrics.columns) == [
        'Lead (15-min steps)', 'PICP (%)', 'MPIW (kW)', 'Pinball (kW)', 'CRPS (kW)',
    ]
    assert len(metrics) == 2


def test_chronological_validation_and_test_start_on_complete_days():
    dates = torch.arange(12 * 96).numpy()
    timestamps = np.datetime64('2020-01-01T00:00') + dates * np.timedelta64(15, 'm')
    frame = pd.DataFrame({
        'DATE_TIME': timestamps,
        'AC_POWER': np.sin(dates / 10.0) + 2.0,
    })

    split = prepare_chronological_splits(
        frame, ['AC_POWER'], history_len=4, horizon_len=96,
        train_ratio=0.5, val_ratio=0.25,
    )

    for partition in ('val', 'test'):
        target_starts = pd.to_datetime(split[partition][2][:, 0])
        midnight_starts = target_starts[(target_starts.hour == 0) & (target_starts.minute == 0)]
        assert len(midnight_starts) == 3
