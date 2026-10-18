"""
src package initialization
"""
from .preprocess import load_and_clean_plant_data, make_future_covariates, prepare_chronological_splits
from .train_cnn import Conv1DForecaster, train_model, compute_forecast_metrics, persistence_baseline
from .uncertainty import (predict_mc_dropout, compute_prediction_intervals,
                          evaluate_uncertainty_calibration, calibrate_conformal_half_width,
                          compute_conformal_intervals, evaluate_probabilistic_by_lead)
from .payoffs import ProsumerCommunity
from .payoffs_v2 import ForecastAwarePayoffs
from .replicator import DailyReplicator, ReplicatorEngine
from .experiments import run_research_experiments
