import numpy as np

from src.payoffs_v2 import ForecastAwarePayoffs
from src.payoffs import ProsumerCommunity
from src.replicator import DailyReplicator


def make_market_inputs(days=1, households=20):
    intervals = np.arange(96)
    generation = np.maximum(0.0, 100.0 * np.sin(np.pi * (intervals - 20) / 60.0))
    forecast = np.broadcast_to(generation, (days, 96)).copy()
    actual = np.broadcast_to(generation * 0.92, (days, 96)).copy()
    sigma = np.full((days, 96), 8.0)
    loads = np.full((days, 96, households), 2.0)
    return forecast, actual, sigma, loads


def make_payoff_model(households=20, **overrides):
    arguments = {
        "num_prosumers": households,
        "plant_capacity_kw": 1000.0,
        "p_grid": 0.22,
        "p_fit": 0.06,
        "c_tx": 0.01,
        "c_imb": 0.08,
        "platform_fee": 0.02,
        "n0": 8.0,
        "sigma_load_kwh": 0.02,
        "seed": 4,
    }
    arguments.update(overrides)
    return ForecastAwarePayoffs(**arguments)


def test_daily_sharing_payoff_depends_on_population_share():
    model = make_payoff_model()
    forecast, actual, sigma, loads = make_market_inputs()
    kwargs = dict(
        generation_forecast_kw=forecast[0],
        generation_actual_kw=actual[0],
        sigma_forecast_kw=sigma[0],
        load_mean_kw=loads[0],
        load_actual_kw=loads[0],
    )

    payoff_low, _, _ = model.calculate_day_payoffs(x_share=0.1, **kwargs)
    payoff_high, _, _ = model.calculate_day_payoffs(x_share=0.9, **kwargs)

    assert payoff_low != payoff_high


def test_hoard_payoff_is_zero_in_relative_payoff_definition():
    model = make_payoff_model()
    forecast, actual, sigma, loads = make_market_inputs()

    _, u_h, _ = model.calculate_day_payoffs(
        0.5, forecast[0], actual[0], sigma[0], loads[0], loads[0],
    )

    assert u_h == 0.0


def test_matched_seller_and_buyer_energy_is_equal():
    model = make_payoff_model()
    forecast, actual, sigma, loads = make_market_inputs()
    _, _, metrics = model.calculate_day_payoffs(
        0.5, forecast[0], actual[0], sigma[0], loads[0], loads[0],
    )

    assert metrics["matched_sell_kwh"] == metrics["matched_buy_kwh"]
    assert metrics["matched_trade_kwh"] == metrics["matched_sell_kwh"]


def test_optimal_newsvendor_cost_never_exceeds_mean_commitment_cost():
    for c_u in (0.001, 0.02, 0.08, 0.5):
        for c_o in (0.001, 0.02, 0.08, 0.5):
            z = ForecastAwarePayoffs.optimal_news_vendor_z(c_u, c_o)
            for sigma in (1e-6, 0.01, 0.2, 5.0):
                naive = ForecastAwarePayoffs.expected_newsvendor_cost(
                    sigma, c_u, c_o, 0.0,
                )
                aware = ForecastAwarePayoffs.expected_newsvendor_cost(
                    sigma, c_u, c_o, z,
                )
                assert aware <= naive + 1e-12


def test_zero_solar_uncertainty_leaves_diversifiable_load_error():
    model = make_payoff_model(sigma_load_kwh=0.3)
    for n_s in (1.0, 4.0, 20.0):
        sigma_e = model.forecast_error_std(0.0, n_s, model.sigma_load_kwh)
        assert np.isclose(sigma_e, model.sigma_load_kwh / np.sqrt(n_s))


def test_oracle_forecasts_do_not_charge_uncommitted_surplus_as_imbalance():
    model = make_payoff_model()
    _, actual, _, loads = make_market_inputs()
    rng = np.random.default_rng(22)
    realized_load = np.clip(loads[0] + rng.normal(0.0, 0.08, loads[0].shape), 0.3, None)

    _, _, metrics = model.calculate_day_payoffs(
        0.5, actual[0], actual[0], np.zeros(96), loads[0], realized_load,
    )

    assert metrics["realized_imbalance_cost"] < 0.1


def test_daily_replicator_keeps_share_inside_unit_interval():
    model = make_payoff_model()
    forecast, actual, sigma, loads = make_market_inputs(days=6)
    replicator = DailyReplicator(beta=0.5, eta=100.0, max_share_step=0.1)

    trace, _ = replicator.simulate(
        model,
        generation_forecast_kw=forecast,
        generation_actual_kw=actual,
        sigma_forecast_kw=sigma,
        load_mean_kw=loads,
        load_actual_kw=loads,
        x0=0.5,
    )

    assert np.all((trace["x_share"] >= 1e-3) & (trace["x_share"] <= 1.0 - 1e-3))
    assert np.all(np.abs(trace["dx"]) <= 0.1 + 1e-12)


def test_unforecast_actual_shock_changes_settlement_not_forecast_based_strategy():
    model = make_payoff_model()
    forecast, actual, sigma, loads = make_market_inputs(days=2)
    shock_actual = actual.copy()
    shock_actual[1, 40:64] *= 0.25
    replicator = DailyReplicator()
    bill_scale = np.mean([
        abs(model.daily_hoard_bill(actual[day], loads[day]))
        for day in range(2)
    ])
    kwargs = {
        "payoff_model": model,
        "generation_forecast_kw": forecast,
        "sigma_forecast_kw": sigma,
        "load_mean_kw": loads,
        "load_actual_kw": loads,
        "normalization_bill": bill_scale,
    }

    normal_trace, _ = replicator.simulate(
        generation_actual_kw=actual, **kwargs,
    )
    shock_trace, shock_summary = replicator.simulate(
        generation_actual_kw=shock_actual, **kwargs,
    )

    assert np.allclose(normal_trace["x_share"], shock_trace["x_share"])
    assert not np.isclose(
        normal_trace["bill_savings"].sum(),
        shock_trace["bill_savings"].sum(),
    )


def test_legacy_community_does_not_reseed_global_numpy_rng():
    state = np.random.get_state()
    expected = np.random.random()
    np.random.set_state(state)

    ProsumerCommunity(num_prosumers=5, seed=99)
    actual = np.random.random()

    assert actual == expected
