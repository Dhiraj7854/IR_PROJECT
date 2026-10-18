"""Forecast-provider comparisons and equilibrium, uncertainty, and shock analyses."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import qmc, t as student_t

from .payoffs_v2 import ForecastAwarePayoffs
from .replicator import DailyReplicator


def _payoff_model(game_config, plant_capacity_kw, seed, overrides=None):
    values = {
        "num_prosumers": game_config["num_prosumers"],
        "plant_capacity_kw": plant_capacity_kw,
        "p_grid": game_config["grid_price"],
        "p_fit": game_config["feed_in_price"],
        "c_tx": game_config["transaction_cost"],
        "c_imb": game_config["imbalance_cost"],
        "platform_fee": game_config["daily_platform_fee"],
        "n0": game_config["thin_market_n0"],
        "sigma_load_kwh": game_config["load_sigma_kwh"],
        "dt_hours": game_config["interval_hours"],
        "seed": seed,
    }
    if overrides:
        values.update(overrides)
    return ForecastAwarePayoffs(**values)


def _load_profiles(payoff_model, dates, seed):
    rng = np.random.default_rng(seed)
    means, actuals = [], []
    for day in dates:
        times = pd.date_range(pd.Timestamp(day), periods=96, freq="15min")
        mean_load = payoff_model.expected_diurnal_load(times)
        means.append(mean_load)
        actuals.append(payoff_model.generate_diurnal_load(times, rng=rng))
    return np.stack(means), np.stack(actuals)


def _load_day_ahead_data(forecast_path, cleaned_path):
    archive = np.load(forecast_path, allow_pickle=False)
    dates = pd.to_datetime(archive["date"])
    cnn_mean = np.asarray(archive["mu"], dtype=float)
    cnn_sigma = np.asarray(archive["sigma"], dtype=float)
    actual = np.asarray(archive["actual"], dtype=float)
    splits = np.asarray(archive["split"])
    if cnn_mean.shape != cnn_sigma.shape or cnn_mean.shape != actual.shape:
        raise ValueError("Day-ahead forecast archive arrays must have matching shapes")
    if cnn_mean.ndim != 2 or cnn_mean.shape[1] != 96 or len(dates) != len(cnn_mean):
        raise ValueError("Day-ahead forecasts must have one 96-step row per date")

    cleaned = pd.read_csv(cleaned_path, parse_dates=["DATE_TIME"])
    generation = cleaned.set_index("DATE_TIME")["AC_POWER"].sort_index()
    persistence = []
    for date in dates:
        yesterday = pd.date_range(date - pd.Timedelta(days=1), periods=96, freq="15min")
        values = generation.reindex(yesterday).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing prior-day generation for persistence forecast at {date}")
        persistence.append(np.maximum(values, 0.0))
    persistence = np.stack(persistence)

    validation = splits == "validation"
    if not np.any(validation):
        raise ValueError("Validation day-ahead rows are required to estimate persistence sigma")
    persistence_sigma = np.std(actual[validation] - persistence[validation], axis=0, ddof=1)
    providers = {
        "Oracle": {
            "mean": actual.copy(),
            "sigma": np.zeros_like(actual),
            "commitment": "aware",
        },
        "Persistence": {
            "mean": persistence,
            "sigma": np.broadcast_to(persistence_sigma, actual.shape).copy(),
            "commitment": "naive",
        },
        "CNN-naive": {
            "mean": cnn_mean,
            "sigma": cnn_sigma,
            "commitment": "naive",
        },
        "CNN-aware": {
            "mean": cnn_mean,
            "sigma": cnn_sigma,
            "commitment": "aware",
        },
    }
    return {
        "dates": dates,
        "splits": splits,
        "actual": actual,
        "cnn_mean": cnn_mean,
        "cnn_sigma": cnn_sigma,
        "persistence": persistence,
        "providers": providers,
        "plant_capacity_kw": float(cleaned["AC_POWER"].max()),
    }


def _run_provider(payoff, provider, data, load_mean, load_actual, dates, config, x0):
    game = config["game"]
    engine = DailyReplicator(
        beta=game["replicator_beta"],
        eta=game["replicator_eta"],
        max_share_step=game["max_share_step"],
    )
    return engine.simulate(
        payoff_model=payoff,
        generation_forecast_kw=provider["mean"],
        generation_actual_kw=data["actual"],
        sigma_forecast_kw=provider["sigma"],
        load_mean_kw=load_mean,
        load_actual_kw=load_actual,
        timestamps=dates,
        x0=x0,
        commitment=provider["commitment"],
    )


def _block_bootstrap_indices(day_count, sample_count, block_length, rng):
    if day_count <= 0 or sample_count <= 0 or block_length <= 0:
        raise ValueError("Bootstrap dimensions and block length must be positive")
    sampled = []
    while len(sampled) < sample_count:
        start = int(rng.integers(0, day_count))
        sampled.extend((start + offset) % day_count for offset in range(block_length))
    return np.asarray(sampled[:sample_count], dtype=int)


def _confidence_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(student_t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def _find_equilibria(curves):
    roots = []
    for kappa, group in curves.groupby("kappa", sort=True):
        x = group["x"].to_numpy()
        d_f = group["mean_dF"].to_numpy()
        slope_grid = np.gradient(d_f, x)
        for index in range(len(x) - 1):
            if d_f[index] == 0.0:
                root = x[index]
            elif d_f[index] * d_f[index + 1] < 0.0:
                root = brentq(
                    lambda value: float(np.interp(value, x, d_f)),
                    x[index], x[index + 1],
                )
            else:
                continue
            slope = float(np.interp(root, x, slope_grid))
            roots.append({
                "kappa": float(kappa),
                "x_star": float(root),
                "slope_dF_dx": slope,
                "stability": "stable" if slope < 0.0 else "unstable",
                "equilibrium_type": "interior",
            })
        d_f_zero = float(np.interp(0.0, x, d_f))
        d_f_one = float(np.interp(1.0, x, d_f))
        roots.extend([
            {
                "kappa": float(kappa),
                "x_star": 0.0,
                "slope_dF_dx": np.nan,
                "stability": "stable" if d_f_zero < 0 else "unstable",
                "equilibrium_type": "boundary",
            },
            {
                "kappa": float(kappa),
                "x_star": 1.0,
                "slope_dF_dx": np.nan,
                "stability": "stable" if d_f_one > 0 else "unstable",
                "equilibrium_type": "boundary",
            },
        ])
    return pd.DataFrame(
        roots, columns=[
            "kappa", "x_star", "slope_dF_dx", "stability", "equilibrium_type",
        ]
    )


def _equilibrium_analysis(payoff, data, load_mean, load_actual, config):
    experiment = config["experiment"]
    game = config["game"]
    reference_bills = [
        payoff.daily_hoard_bill(data["actual"][i], load_actual[i])
        for i in range(len(data["dates"]))
    ]
    bill_scale = float(np.mean(np.abs(reference_bills)))
    grid = np.linspace(1e-3, 1.0 - 1e-3, experiment["equilibrium_grid_size"])
    curve_rows = []
    for kappa in experiment["kappa_multipliers"]:
        for x in grid:
            utilities = []
            for day in range(len(data["dates"])):
                u_s, _, _ = payoff.calculate_day_payoffs(
                    x,
                    data["cnn_mean"][day],
                    data["actual"][day],
                    data["cnn_sigma"][day] * kappa,
                    load_mean[day],
                    load_actual[day],
                    commitment="aware",
                )
                utilities.append(u_s)
            curve_rows.append({
                "kappa": kappa,
                "x": x,
                "mean_dF": float(np.mean(utilities) / bill_scale),
            })
    curves = pd.DataFrame(curve_rows)
    equilibria = _find_equilibria(curves)

    trajectory_rows = []
    for kappa in experiment["kappa_multipliers"]:
        curve = curves[curves["kappa"] == kappa]
        x_grid = curve["x"].to_numpy()
        d_f_grid = curve["mean_dF"].to_numpy()
        for initial_share in (0.1, 0.5, 0.9):
            share = initial_share
            memory = 0.0
            for epoch in range(experiment["equilibrium_epochs"]):
                daily_advantage = float(np.interp(share, x_grid, d_f_grid))
                memory = (1.0 - game["replicator_beta"]) * memory + game["replicator_beta"] * daily_advantage
                factor = share * (1.0 - share)
                eta = game["replicator_eta"]
                if factor > 0 and abs(memory) > 0:
                    eta = min(eta, game["max_share_step"] / (factor * abs(memory)))
                share = float(np.clip(
                    share + eta * factor * memory, 1e-3, 1.0 - 1e-3
                ))
                trajectory_rows.append({
                    "kappa": kappa,
                    "x0": initial_share,
                    "epoch": epoch,
                    "x_share": share,
                })
    trajectories = pd.DataFrame(trajectory_rows)
    return curves, equilibria, trajectories


def _value_of_information_by_day(payoff, data, load_mean, load_actual, kappas):
    rows = []
    for kappa in kappas:
        for day in range(len(data["dates"])):
            _, _, metrics = payoff.calculate_day_payoffs(
                0.5,
                data["cnn_mean"][day],
                data["actual"][day],
                data["cnn_sigma"][day] * kappa,
                load_mean[day],
                load_actual[day],
                commitment="aware",
            )
            bill = abs(metrics["hoard_bill"])
            rows.append({
                "kappa": kappa,
                "day_index": day,
                "naive_cost": metrics["newsvendor_naive_cost"],
                "aware_cost": metrics["newsvendor_aware_cost"],
                "value_of_information": metrics["value_of_information"],
                "value_of_information_pct_bill": (
                    metrics["value_of_information"] / max(bill, 1e-12) * 100.0
                ),
            })
    return pd.DataFrame(rows)


def _sensitivity_analysis(data, config, dates):
    game = config["game"]
    sensitivity_config = config["experiment"]["sensitivity_values"]
    seed = config["experiment"]["random_seed"]
    baseline = {
        "c_imb": game["imbalance_cost"],
        "platform_fee": game["daily_platform_fee"],
        "n0": game["thin_market_n0"],
        "c_tx": game["transaction_cost"],
        "num_prosumers": game["num_prosumers"],
        "beta": game["replicator_beta"],
        "eta": game["replicator_eta"],
    }
    values = sensitivity_config
    records = []

    def evaluate(parameter, value, override_lhs=None):
        payoff_overrides = {}
        if parameter == "c_imb":
            payoff_overrides["c_imb"] = value
        elif parameter == "platform_fee":
            payoff_overrides["platform_fee"] = value
        elif parameter == "n0":
            payoff_overrides["n0"] = value
        elif parameter == "c_tx":
            payoff_overrides["c_tx"] = value
        elif parameter == "num_prosumers":
            payoff_overrides["num_prosumers"] = int(value)
        elif parameter in {"beta", "eta"}:
            pass
        if override_lhs:
            payoff_overrides.update({
                "c_imb": override_lhs["c_imb"],
                "platform_fee": override_lhs["platform_fee"],
                "n0": override_lhs["n0"],
                "c_tx": override_lhs["c_tx"],
                "num_prosumers": int(round(override_lhs["num_prosumers"])),
            })
        payoff = _payoff_model(
            game, data["plant_capacity_kw"], seed, payoff_overrides,
        )
        mean_load, actual_load = _load_profiles(payoff, dates, seed)
        beta = override_lhs["beta"] if override_lhs else (
            value if parameter == "beta" else baseline["beta"]
        )
        eta = override_lhs["eta"] if override_lhs else (
            value if parameter == "eta" else baseline["eta"]
        )
        engine = DailyReplicator(
            beta=beta, eta=eta, max_share_step=game["max_share_step"],
        )
        trace, summary = engine.simulate(
            payoff,
            data["cnn_mean"],
            data["actual"],
            data["cnn_sigma"],
            mean_load,
            actual_load,
            timestamps=dates,
            x0=config["experiment"]["x0"],
            commitment="aware",
        )
        records.append({
            "analysis": "latin_hypercube" if override_lhs else "one_at_a_time",
            "parameter": parameter,
            "value": value,
            "relative_to_baseline": (
                value / baseline[parameter] if parameter in baseline else np.nan
            ),
            "Final_Share": summary["Final_Share"],
            "Mean_Share": summary["Mean_Share"],
            "Total_Savings_Per_Household ($)": summary["Total_Savings_Per_Household ($)"],
            "Savings_Pct_of_Hoard_Bill (%)": summary["Savings_Pct_of_Hoard_Bill (%)"],
            "Total_Matched_Trade_kWh": summary["Total_Matched_Trade_kWh"],
        })
        if override_lhs:
            records[-1].update({
                f"lhs_{name}": value for name, value in override_lhs.items()
            })

    for parameter, parameter_values in values.items():
        for value in parameter_values:
            evaluate(parameter, value)

    dimensions = ["c_imb", "platform_fee", "n0", "c_tx", "num_prosumers", "beta", "eta"]
    lower = np.array([
        baseline[name] * config["experiment"]["lhs_lower_factor"]
        for name in dimensions
    ])
    upper = np.array([
        baseline[name] * config["experiment"]["lhs_upper_factor"]
        for name in dimensions
    ])
    upper[dimensions.index("num_prosumers")] = (
        baseline["num_prosumers"]
        * config["experiment"]["lhs_population_upper_factor"]
    )
    sampler = qmc.LatinHypercube(d=len(dimensions), seed=seed)
    lhs = qmc.scale(
        sampler.random(n=config["experiment"]["lhs_samples"]), lower, upper,
    )
    for index, sample in enumerate(lhs):
        sample_values = dict(zip(dimensions, sample))
        payoff_override = {
            key: sample_values[key]
            for key in ("c_imb", "platform_fee", "n0", "c_tx", "num_prosumers")
        }
        evaluate(
            f"lhs-{index + 1}",
            float(sample_values["num_prosumers"]),
            override_lhs={
                **payoff_override,
                "beta": sample_values["beta"],
                "eta": sample_values["eta"],
            },
        )
    return pd.DataFrame(records)


def _weather_shock_analysis(payoff, data, load_mean, load_actual, dates, config):
    game = config["game"]
    experiment = config["experiment"]
    base_provider = data["providers"]["CNN-aware"]
    normal_trace, normal_summary = _run_provider(
        payoff, base_provider, data, load_mean, load_actual, dates,
        config, experiment["x0"],
    )
    normalization_bill = float(np.mean([
        abs(payoff.daily_hoard_bill(data["actual"][day], load_actual[day]))
        for day in range(len(dates))
    ]))
    shock_day = min(experiment["shock_day_index"], len(dates) - 1)
    start = int(experiment["shock_start_hour"] / game["interval_hours"])
    duration = int(experiment["shock_duration_hours"] / game["interval_hours"])
    end = min(start + duration, 96)
    factor = 1.0 - experiment["shock_drop_fraction"]
    rows = []
    traces = [normal_trace.assign(scenario="normal")]
    for visible in (False, True):
        provider = {
            "mean": base_provider["mean"].copy(),
            "sigma": base_provider["sigma"].copy(),
            "commitment": "aware",
        }
        actual = data["actual"].copy()
        actual[shock_day, start:end] *= factor
        if visible:
            provider["mean"][shock_day, start:end] *= factor
        trace, summary = DailyReplicator(
            beta=game["replicator_beta"],
            eta=game["replicator_eta"],
            max_share_step=game["max_share_step"],
        ).simulate(
            payoff,
            provider["mean"],
            actual,
            provider["sigma"],
            load_mean,
            load_actual,
            timestamps=dates,
            x0=experiment["x0"],
            commitment="aware",
            normalization_bill=normalization_bill,
        )
        pre_share = float(normal_trace["x_share"].iloc[max(shock_day - 1, 0)])
        normal_shock_day_share = float(normal_trace["x_share"].iloc[shock_day])
        shock_day_share = float(trace["x_share"].iloc[shock_day])
        has_share_deviation = not np.isclose(
            shock_day_share, normal_shock_day_share, rtol=0.0, atol=1e-12,
        )
        threshold = normal_shock_day_share * experiment["shock_recovery_fraction"]
        recovery = (
            next(
                (day - shock_day for day in range(shock_day + 1, len(trace))
                 if trace["x_share"].iloc[day] >= threshold),
                np.nan,
            )
            if has_share_deviation else 0
        )
        rows.append({
            "shock_visibility": "forecast_visible" if visible else "actual_only_unforecast",
            "shock_day": dates[shock_day],
            "shock_start_hour": experiment["shock_start_hour"],
            "shock_duration_hours": experiment["shock_duration_hours"],
            "generation_drop_fraction": experiment["shock_drop_fraction"],
            "pre_shock_share": pre_share,
            "normal_share_after_shock_day": normal_shock_day_share,
            "share_after_shock_day": shock_day_share,
            "shock_day_share_difference": shock_day_share - normal_shock_day_share,
            "minimum_share_from_shock_day": float(trace["x_share"].iloc[shock_day:].min()),
            "final_share": summary["Final_Share"],
            "lost_trade_kWh": max(
                normal_summary["Total_Matched_Trade_kWh"] - summary["Total_Matched_Trade_kWh"],
                0.0,
            ),
            "additional_trade_kWh": max(
                summary["Total_Matched_Trade_kWh"] - normal_summary["Total_Matched_Trade_kWh"],
                0.0,
            ),
            "sharing_bill_increase_per_household ($)": (
                summary["Total_Sharing_Bill_Per_Household ($)"]
                - normal_summary["Total_Sharing_Bill_Per_Household ($)"]
            ),
            "bill_increase_pct_of_hoard (%)": (
                summary["Total_Sharing_Bill_Per_Household ($)"]
                - normal_summary["Total_Sharing_Bill_Per_Household ($)"]
            ) / max(abs(normal_summary["Total_Hoard_Bill_Per_Household ($)"]), 1e-12) * 100.0,
            "recovery_days": recovery,
            "recovery_status": (
                "no_share_deviation"
                if not has_share_deviation
                else "recovered" if np.isfinite(recovery)
                else "not_recovered_in_observed_period"
            ),
        })
        traces.append(trace.assign(
            scenario="forecast_visible" if visible else "actual_only_unforecast"
        ))
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), normal_trace


def _plot_trajectories(provider_traces, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, trace in provider_traces.items():
        ax.plot(trace["date"], trace["x_share"], marker="o", label=name)
    ax.set(ylabel="Share strategy population x", xlabel="Day",
           title="Daily replicator trajectories by forecast provider", ylim=(0, 1))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dynamics(curves, equilibria, trajectories, output_path):
    fig, (phase_ax, branch_ax) = plt.subplots(1, 2, figsize=(12, 5))
    for kappa, group in curves.groupby("kappa"):
        phase_ax.plot(group["x"], group["mean_dF"], label=f"k={kappa:g}")
    phase_ax.axhline(0.0, color="black", linewidth=1)
    phase_ax.set(xlabel="Share x", ylabel="Mean normalized fitness difference dF(x)",
                 title="Phase line", xlim=(0, 1))
    phase_ax.legend()
    for stability, marker in (("stable", "o"), ("unstable", "x")):
        subset = equilibria[equilibria["stability"] == stability]
        if len(subset):
            branch_ax.scatter(
                subset["kappa"], subset["x_star"], marker=marker,
                label=f"{stability} equilibrium",
            )
    branch_ax.set(xlabel="Uncertainty multiplier kappa", ylabel="Equilibrium share x*",
                  title="Interior equilibrium branches", ylim=(0, 1))
    if len(equilibria):
        branch_ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    final = trajectories[trajectories["epoch"] == trajectories["epoch"].max()]
    for (kappa, x0), group in final.groupby(["kappa", "x0"]):
        ax.plot(kappa, group["x_share"].iloc[0], "o", label=f"x0={x0:g}")
    ax.set(xlabel="Uncertainty multiplier kappa", ylabel="Final share after repeated mean dynamics",
           title="Initial-condition / hysteresis check", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path.with_name("figure_5b_hysteresis.png"), dpi=180)
    plt.close(fig)


def _plot_scenario_summary(scenario_summary, output_path):
    subset = scenario_summary[
        scenario_summary["metric"].isin([
            "Final_Share",
            "Savings_Pct_of_Hoard_Bill (%)",
        ])
    ]
    providers = list(subset["provider"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric in zip(axes, ["Final_Share", "Savings_Pct_of_Hoard_Bill (%)"]):
        rows = subset[subset["metric"] == metric].set_index("provider").loc[providers]
        y = rows["mean"].to_numpy()
        errors = np.vstack((y - rows["ci_low"].to_numpy(), rows["ci_high"].to_numpy() - y))
        axis.errorbar(providers, y, yerr=errors, fmt="o", capsize=4)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Block-bootstrap scenario results (mean and 95% CI)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_sensitivity(sensitivity, output_path):
    subset = sensitivity[sensitivity["analysis"] == "one_at_a_time"]
    metrics = [
        ("Final_Share", "Final share"),
        ("Savings_Pct_of_Hoard_Bill (%)", "Savings (% of hoard bill)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (metric, title) in zip(axes, metrics):
        for parameter, group in subset.groupby("parameter"):
            group = group.sort_values("relative_to_baseline")
            axis.plot(
                group["relative_to_baseline"], group[metric],
                marker="o", label=parameter,
            )
        axis.set(title=title, xlabel="Parameter / baseline value")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_weather_shock(traces, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in traces.groupby("scenario", sort=False):
        ax.plot(group["date"], group["x_share"], marker="o", label=name)
    ax.set(xlabel="Day", ylabel="Share strategy population x",
           title="Unforecast and forecast-visible generation shock", ylim=(0, 1))
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_forecast_actual_vs_pred(timestamps, y_true_day, y_pred_day, lower_day, upper_day,
                                save_path=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(y_true_day))
    ax.plot(x, y_true_day, label="Actual")
    ax.plot(x, y_pred_day, "--", label="CNN ensemble")
    ax.fill_between(x, lower_day, upper_day, alpha=0.25, label="90% split-conformal interval")
    ax.set(xlabel="15-minute lead step", ylabel="Plant AC power (kW)",
           title=f"Day-ahead forecast issued {pd.Timestamp(timestamps[0])}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_error_by_horizon(horizon_errors, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(horizon_errors)
    values = list(horizon_errors.values())
    ax.bar(labels, values)
    ax.set(ylabel="Mean absolute error (kW)", title="CNN ensemble test MAE by lead time")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_uncertainty_calibration(calibration, save_path=None):
    levels = np.asarray(list(calibration), dtype=float)
    observed = np.asarray([calibration[level] for level in levels], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(levels * 100.0, levels * 100.0, "--", color="black", label="Ideal")
    ax.plot(levels * 100.0, observed, "o-", label="Test PICP")
    ax.set(xlabel="Nominal coverage (%)", ylabel="Observed coverage (%)",
           title="Split-conformal interval calibration")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run_research_experiments(forecast_path, cleaned_path, config, output_root="results"):
    """Run the same payoff engine across providers and all requested analyses."""
    output_root = Path(output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    data = _load_day_ahead_data(forecast_path, cleaned_path)
    seed = config["experiment"]["random_seed"]
    payoff = _payoff_model(
        config["game"], data["plant_capacity_kw"], seed,
    )
    load_mean, load_actual = _load_profiles(payoff, data["dates"], seed)
    provider_traces, historical_rows = {}, []
    for name, provider in data["providers"].items():
        trace, summary = _run_provider(
            payoff, provider, data, load_mean, load_actual, data["dates"],
            config, config["experiment"]["x0"],
        )
        provider_traces[name] = trace
        trace.to_csv(tables / f"simulation_{name.lower().replace('-', '_')}.csv", index=False)
        historical_rows.append({"provider": name, **summary})
    historical = pd.DataFrame(historical_rows)
    historical.to_csv(tables / "table_3_provider_comparison.csv", index=False)
    _plot_trajectories(provider_traces, figures / "figure_4_provider_trajectories.png")

    curves, equilibria, equilibrium_trajectories = _equilibrium_analysis(
        payoff, data, load_mean, load_actual, config,
    )
    curves.to_csv(tables / "table_4_dF_curves.csv", index=False)
    equilibria.to_csv(tables / "table_5_equilibria.csv", index=False)
    equilibrium_trajectories.to_csv(tables / "table_6_hysteresis.csv", index=False)
    _plot_dynamics(
        curves, equilibria, equilibrium_trajectories,
        figures / "figure_5_phase_line_bifurcation.png",
    )

    voi_daily = _value_of_information_by_day(
        payoff, data, load_mean, load_actual,
        config["experiment"]["kappa_multipliers"],
    )
    voi_daily.to_csv(tables / "table_7_value_of_information_by_day.csv", index=False)

    scenario_config = config["experiment"]
    scenario_rows, voi_seed_rows = [], []
    scenario_days = scenario_config["scenario_days"]
    block_length = scenario_config["block_length_days"]
    for scenario_seed in scenario_config["scenario_seeds"]:
        rng = np.random.default_rng(scenario_seed)
        indices = _block_bootstrap_indices(
            len(data["dates"]), scenario_days, block_length, rng,
        )
        boot_dates = data["dates"].to_numpy()[indices]
        boot_mean_load = load_mean[indices]
        boot_actual_load = np.stack([
            payoff.generate_diurnal_load(
                pd.date_range(pd.Timestamp(day), periods=96, freq="15min"),
                rng=rng,
            )
            for day in boot_dates
        ])
        boot_data = {
            **data,
            "dates": pd.to_datetime(boot_dates),
            "actual": data["actual"][indices],
        }
        for name, provider in data["providers"].items():
            boot_provider = {
                "mean": provider["mean"][indices],
                "sigma": provider["sigma"][indices],
                "commitment": provider["commitment"],
            }
            _, summary = _run_provider(
                payoff, boot_provider, boot_data, boot_mean_load, boot_actual_load,
                boot_data["dates"], config, scenario_config["x0"],
            )
            scenario_rows.append({"seed": scenario_seed, "provider": name, **summary})
        for kappa, group in voi_daily.groupby("kappa"):
            sample = group.set_index("day_index").loc[indices]
            voi_seed_rows.append({
                "seed": scenario_seed,
                "kappa": kappa,
                "naive_cost_per_day": sample["naive_cost"].mean(),
                "aware_cost_per_day": sample["aware_cost"].mean(),
                "value_of_information_per_day": sample["value_of_information"].mean(),
                "value_of_information_pct_bill": sample["value_of_information_pct_bill"].mean(),
            })

    scenario_runs = pd.DataFrame(scenario_rows)
    scenario_runs.to_csv(tables / "table_8_scenario_seed_results.csv", index=False)
    scenario_metrics = [
        "Final_Share",
        "Mean_Share",
        "Total_Hoard_Bill_Per_Household ($)",
        "Total_Sharing_Bill_Per_Household ($)",
        "Total_Savings_Per_Household ($)",
        "Savings_Pct_of_Hoard_Bill (%)",
        "Total_Matched_Trade_kWh",
        "Total_Value_of_Information ($)",
    ]
    scenario_summary_rows = []
    for provider_name, provider_group in scenario_runs.groupby("provider", sort=False):
        for metric in scenario_metrics:
            mean, low, high = _confidence_interval(provider_group[metric])
            scenario_summary_rows.append({
                "provider": provider_name, "metric": metric,
                "mean": mean, "ci_low": low, "ci_high": high,
                "scenario_seeds": len(provider_group),
            })
    scenario_summary = pd.DataFrame(scenario_summary_rows)
    scenario_summary.to_csv(tables / "table_9_scenario_summary.csv", index=False)
    _plot_scenario_summary(scenario_summary, figures / "figure_7_scenario_intervals.png")

    voi_runs = pd.DataFrame(voi_seed_rows)
    voi_summary_rows = []
    for kappa, group in voi_runs.groupby("kappa", sort=True):
        row = {"kappa": kappa, "scenario_seeds": len(group)}
        for source_column, label in (
            ("naive_cost_per_day", "naive_cost"),
            ("aware_cost_per_day", "aware_cost"),
            ("value_of_information_per_day", "value_of_information"),
            ("value_of_information_pct_bill", "value_of_information_pct_bill"),
        ):
            mean, low, high = _confidence_interval(group[source_column])
            row[f"{label}_mean"] = mean
            row[f"{label}_ci_low"] = low
            row[f"{label}_ci_high"] = high
        voi_summary_rows.append(row)
    voi_summary = pd.DataFrame(voi_summary_rows)
    voi_summary.to_csv(tables / "table_10_value_of_information_ci.csv", index=False)

    sensitivity = _sensitivity_analysis(data, config, data["dates"])
    sensitivity.to_csv(tables / "table_11_sensitivity.csv", index=False)
    _plot_sensitivity(sensitivity, figures / "figure_8_sensitivity.png")

    shock_metrics, shock_traces, normal_shock_trace = _weather_shock_analysis(
        payoff, data, load_mean, load_actual, data["dates"], config,
    )
    shock_metrics.to_csv(tables / "table_12_weather_shock.csv", index=False)
    shock_traces.to_csv(tables / "table_13_weather_shock_trajectories.csv", index=False)
    _plot_weather_shock(shock_traces, figures / "figure_6_weather_shock.png")

    print("\nProvider comparison (rolling validation + test days):")
    print(historical[[
        "provider", "Final_Share", "Total_Savings_Per_Household ($)",
        "Savings_Pct_of_Hoard_Bill (%)", "Total_Matched_Trade_kWh",
    ]].to_string(index=False))
    print("\nInterior equilibria:")
    print(equilibria.to_string(index=False) if len(equilibria) else "No interior dF(x)=0 roots found.")
    print("\nValue of information with 95% scenario confidence intervals:")
    print(voi_summary.to_string(index=False))
    return {
        "historical": historical,
        "scenario_summary": scenario_summary,
        "equilibria": equilibria,
        "value_of_information": voi_summary,
        "sensitivity": sensitivity,
        "weather_shock": shock_metrics,
    }
