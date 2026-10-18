"""Daily forecast-aware P2P payoffs with explicit forecast and settlement quantities."""

import numpy as np
from scipy.stats import norm


class ForecastAwarePayoffs:
    """Evaluate daily sharing payoffs relative to the same-day hoard bill."""

    def __init__(
        self,
        num_prosumers=100,
        plant_capacity_kw=29150.0,
        p_grid=0.22,
        p_fit=0.06,
        c_tx=0.01,
        c_imb=0.08,
        platform_fee=0.02,
        n0=8.0,
        sigma_load_kwh=0.02,
        dt_hours=0.25,
        seed=42,
    ):
        if num_prosumers <= 0 or plant_capacity_kw <= 0 or n0 <= 0:
            raise ValueError("Community size, capacity, and n0 must be positive")
        if not 0 <= p_fit < p_grid or c_tx < 0 or 2 * c_tx > p_grid - p_fit:
            raise ValueError("Tariffs and transaction costs do not define a valid market")
        if c_imb < 0 or platform_fee < 0 or sigma_load_kwh < 0 or dt_hours <= 0:
            raise ValueError("Costs, uncertainty, and interval duration must be nonnegative")

        self.num_prosumers = int(num_prosumers)
        self.plant_capacity_kw = float(plant_capacity_kw)
        self.p_grid = float(p_grid)
        self.p_fit = float(p_fit)
        self.c_tx = float(c_tx)
        self.c_imb = float(c_imb)
        self.platform_fee = float(platform_fee)
        self.n0 = float(n0)
        self.sigma_load_kwh = float(sigma_load_kwh)
        self.dt_hours = float(dt_hours)
        self.community_scale_factor = 500.0 / self.plant_capacity_kw
        self.rng = np.random.default_rng(seed)
        weights = self.rng.uniform(0.6, 1.4, size=self.num_prosumers)
        self.solar_weights = weights / weights.sum()
        self.demand_multipliers = self.rng.uniform(0.6, 1.5, size=self.num_prosumers)

    @staticmethod
    def forecast_error_std(sigma_sol_kwh, n_s, sigma_load_kwh):
        """Combine common solar error with diversifying household load error."""
        if n_s <= 0 or sigma_sol_kwh < 0 or sigma_load_kwh < 0:
            raise ValueError("Pool size and forecast standard deviations are invalid")
        return float(np.sqrt(sigma_sol_kwh**2 + sigma_load_kwh**2 / n_s))

    @staticmethod
    def optimal_news_vendor_z(c_u, c_o):
        """Return the optimal standardized commitment offset."""
        if c_u < 0 or c_o < 0 or c_u + c_o <= 0:
            raise ValueError("Newsvendor underage and overage costs must be nonnegative")
        if c_u == 0:
            return -np.inf
        if c_o == 0:
            return np.inf
        return float(norm.ppf(c_u / (c_u + c_o)))

    @staticmethod
    def expected_newsvendor_cost(sigma, c_u, c_o, z):
        """Expected overage and underage costs for a normal forecast error."""
        sigma = np.asarray(sigma, dtype=float)
        c_u = np.asarray(c_u, dtype=float)
        c_o = np.asarray(c_o, dtype=float)
        z = np.asarray(z, dtype=float)
        if np.any(sigma < 0) or np.any(c_u < 0) or np.any(c_o < 0):
            raise ValueError("Newsvendor inputs must be nonnegative")
        sigma, c_u, c_o, z = np.broadcast_arrays(sigma, c_u, c_o, z)
        finite_z = np.isfinite(z)
        safe_z = np.where(finite_z, z, 0.0)
        phi = norm.pdf(safe_z)
        cdf = norm.cdf(safe_z)
        expected_cost = sigma * (
            c_o * (phi + safe_z * cdf)
            + c_u * (phi - safe_z * (1.0 - cdf))
        )
        expected_cost = np.where(finite_z, np.maximum(expected_cost, 0.0), 0.0)
        return float(expected_cost) if expected_cost.ndim == 0 else expected_cost

    @staticmethod
    def _optimal_z_array(c_u, c_o):
        ratio = np.divide(c_u, c_u + c_o, out=np.zeros_like(c_u), where=(c_u + c_o) > 0)
        z = norm.ppf(ratio)
        return np.where(c_u == 0.0, -np.inf, z)

    @staticmethod
    def _newsvendor_cost_array(sigma, c_u, c_o, z):
        finite_z = np.isfinite(z)
        safe_z = np.where(finite_z, z, 0.0)
        phi = norm.pdf(safe_z)
        cdf = norm.cdf(safe_z)
        cost = sigma * (
            c_o * (phi + safe_z * cdf)
            + c_u * (phi - safe_z * (1.0 - cdf))
        )
        return np.where(finite_z, np.maximum(cost, 0.0), 0.0)

    def expected_diurnal_load(self, timestamps):
        """Return expected household loads without scenario noise."""
        hours = np.asarray([ts.hour + ts.minute / 60.0 for ts in timestamps])
        morning = 1.2 * np.exp(-((hours - 8.0) ** 2) / (2 * 1.5**2))
        evening = 2.2 * np.exp(-((hours - 19.5) ** 2) / (2 * 2.0**2))
        base = 0.8 + 0.3 * np.sin(np.pi * (hours - 6.0) / 12.0)
        return np.outer(base + morning + evening, self.demand_multipliers)

    def generate_diurnal_load(self, timestamps, rng=None):
        """Generate one reproducible noisy 15-minute profile per household."""
        rng = self.rng if rng is None else rng
        expected_kw = self.expected_diurnal_load(timestamps)
        return np.clip(expected_kw + rng.normal(0.0, 0.08, expected_kw.shape), 0.3, None)

    def _load_matrix(self, loads_kw, steps, name):
        loads = np.asarray(loads_kw, dtype=float)
        if loads.ndim == 1:
            if loads.shape[0] != steps:
                raise ValueError(f"{name} must have one load value per interval")
            loads = np.broadcast_to(loads[:, None], (steps, self.num_prosumers))
        if loads.shape != (steps, self.num_prosumers):
            raise ValueError(f"{name} must have shape ({steps}, {self.num_prosumers})")
        if not np.isfinite(loads).all() or np.any(loads < 0):
            raise ValueError(f"{name} must contain finite nonnegative loads")
        return loads

    def _surplus_deficit_kwh(self, generation_kw, loads_kw):
        household_generation_kw = (
            generation_kw * self.community_scale_factor * self.solar_weights
        )
        net_kwh = (household_generation_kw - loads_kw) * self.dt_hours
        return np.maximum(net_kwh, 0.0), np.maximum(-net_kwh, 0.0)

    def daily_hoard_bill(self, generation_actual_kw, load_actual_kw):
        """Return the realized average household bill under the hoard strategy."""
        generation = np.asarray(generation_actual_kw, dtype=float)
        if generation.ndim != 1 or not np.isfinite(generation).all() or np.any(generation < 0):
            raise ValueError("Actual generation must be a finite nonnegative 1D series")
        loads = self._load_matrix(load_actual_kw, len(generation), "load_actual_kw")
        household_generation = (
            generation[:, None] * self.community_scale_factor * self.solar_weights[None, :]
        )
        net = (household_generation - loads) * self.dt_hours
        surplus = np.maximum(net, 0.0)
        deficit = np.maximum(-net, 0.0)
        return float(np.mean(np.sum(self.p_grid * deficit - self.p_fit * surplus, axis=0)))

    def calculate_day_payoffs(
        self,
        x_share,
        generation_forecast_kw,
        generation_actual_kw,
        sigma_forecast_kw,
        load_mean_kw,
        load_actual_kw,
        commitment="aware",
    ):
        """Compute forecast-based fitness and actual settled bills for one day."""
        if not np.isfinite(x_share) or not 0.0 <= x_share <= 1.0:
            raise ValueError("x_share must be in [0, 1]")
        forecast = np.asarray(generation_forecast_kw, dtype=float)
        actual = np.asarray(generation_actual_kw, dtype=float)
        sigma = np.asarray(sigma_forecast_kw, dtype=float)
        if forecast.ndim != 1 or actual.shape != forecast.shape or sigma.shape != forecast.shape:
            raise ValueError("Generation forecasts, actuals, and sigmas must be matching 1D series")
        if not (
            np.isfinite(forecast).all()
            and np.isfinite(actual).all()
            and np.isfinite(sigma).all()
            and np.all(forecast >= 0)
            and np.all(actual >= 0)
            and np.all(sigma >= 0)
        ):
            raise ValueError("Generation and uncertainty series must be finite and nonnegative")
        if commitment not in {"naive", "aware"}:
            raise ValueError("commitment must be 'naive' or 'aware'")

        steps = len(forecast)
        if steps == 0:
            raise ValueError("A day must contain at least one interval")
        mean_loads = self._load_matrix(load_mean_kw, steps, "load_mean_kw")
        actual_loads = self._load_matrix(load_actual_kw, steps, "load_actual_kw")
        n_s = max(x_share * self.num_prosumers, 1.0)
        thin_market_efficiency = float(-np.expm1(-n_s / self.n0))

        household_generation_forecast = (
            forecast[:, None] * self.community_scale_factor * self.solar_weights[None, :]
        )
        forecast_net = (household_generation_forecast - mean_loads) * self.dt_hours
        forecast_surplus = np.mean(np.maximum(forecast_net, 0.0), axis=1)
        forecast_deficit = np.mean(np.maximum(-forecast_net, 0.0), axis=1)
        s_pool = n_s * forecast_surplus
        d_pool = n_s * forecast_deficit
        sdr = s_pool / (d_pool + 1e-12)
        price = np.clip(
            self.p_fit + (self.p_grid - self.p_fit) / (1.0 + sdr),
            self.p_fit + self.c_tx,
            self.p_grid - self.c_tx,
        )
        rho_s = np.divide(
            d_pool, s_pool, out=np.zeros_like(s_pool), where=s_pool > 0
        ).clip(max=1.0)
        rho_d = np.divide(
            s_pool, d_pool, out=np.zeros_like(d_pool), where=d_pool > 0
        ).clip(max=1.0)
        gain = thin_market_efficiency * (
            rho_s * forecast_surplus * (price - self.c_tx - self.p_fit)
            + rho_d * forecast_deficit * (self.p_grid - price - self.c_tx)
        )
        sigma_sol = sigma * self.community_scale_factor / self.num_prosumers * self.dt_hours
        sigma_e = np.sqrt(sigma_sol**2 + self.sigma_load_kwh**2 / n_s)
        c_u_seller = np.maximum(price - self.c_tx - self.p_fit, 0.0)
        c_u_buyer = np.maximum(self.p_grid - price - self.c_tx, 0.0)
        z_seller = self._optimal_z_array(c_u_seller, np.full(steps, self.c_imb))
        z_buyer = self._optimal_z_array(c_u_buyer, np.full(steps, self.c_imb))
        naive_seller_cost = self._newsvendor_cost_array(
            sigma_e, c_u_seller, np.full(steps, self.c_imb), np.zeros(steps)
        )
        naive_buyer_cost = self._newsvendor_cost_array(
            sigma_e, c_u_buyer, np.full(steps, self.c_imb), np.zeros(steps)
        )
        aware_seller_cost = self._newsvendor_cost_array(
            sigma_e, c_u_seller, np.full(steps, self.c_imb), z_seller
        )
        aware_buyer_cost = self._newsvendor_cost_array(
            sigma_e, c_u_buyer, np.full(steps, self.c_imb), z_buyer
        )
        newsvendor_naive_cost = float(np.sum(naive_seller_cost + naive_buyer_cost))
        aware_imbalance_cost = float(np.sum(aware_seller_cost + aware_buyer_cost))
        value_of_information = float(np.sum(np.maximum(
            naive_seller_cost + naive_buyer_cost
            - aware_seller_cost - aware_buyer_cost,
            0.0,
        )))
        naive_imbalance_cost = float(np.sum(
            self.c_imb * sigma_e * np.sqrt(2.0 / np.pi)
        ))
        if commitment == "aware":
            q_seller = np.where(
                sigma_e == 0.0,
                forecast_surplus,
                np.maximum(forecast_surplus + z_seller * sigma_e, 0.0),
            )
            q_buyer = np.where(
                sigma_e == 0.0,
                forecast_deficit,
                np.maximum(forecast_deficit + z_buyer * sigma_e, 0.0),
            )
        else:
            q_seller = forecast_surplus
            q_buyer = forecast_deficit

        household_generation_actual = (
            actual[:, None] * self.community_scale_factor * self.solar_weights[None, :]
        )
        actual_net = (household_generation_actual - actual_loads) * self.dt_hours
        actual_surplus = np.mean(np.maximum(actual_net, 0.0), axis=1)
        actual_deficit = np.mean(np.maximum(-actual_net, 0.0), axis=1)
        available_sell = np.minimum(n_s * q_seller, n_s * actual_surplus)
        available_buy = np.minimum(n_s * q_buyer, n_s * actual_deficit)
        matched = thin_market_efficiency * np.minimum(available_sell, available_buy)
        realized_gain = float(np.sum((matched / n_s) * (
            (price - self.c_tx - self.p_fit)
            + (self.p_grid - price - self.c_tx)
        )))
        realized_imbalance = float(self.c_imb * np.sum(
            np.maximum(q_seller - actual_surplus, 0.0)
            + np.maximum(q_buyer - actual_deficit, 0.0)
        ))
        matched_volume = float(np.sum(matched))
        matched_sell_volume = matched_volume
        matched_buy_volume = matched_volume
        price_sum = float(np.sum(price))
        total_gain = float(np.sum(gain))

        platform_fee_per_sharer = self.platform_fee / n_s
        expected_imbalance = (
            aware_imbalance_cost if commitment == "aware" else naive_imbalance_cost
        )
        u_s = total_gain - expected_imbalance - platform_fee_per_sharer
        u_h = 0.0
        hoard_bill = self.daily_hoard_bill(actual, actual_loads)
        share_bill = (
            hoard_bill - realized_gain + realized_imbalance + platform_fee_per_sharer
        )
        metrics = {
            "n_sharers": n_s,
            "thin_market_efficiency": thin_market_efficiency,
            "expected_gain": total_gain,
            "imbalance_cost": expected_imbalance,
            "naive_absolute_error_cost": naive_imbalance_cost,
            "newsvendor_naive_cost": newsvendor_naive_cost,
            "newsvendor_aware_cost": aware_imbalance_cost,
            "value_of_information": value_of_information,
            "platform_fee_per_sharer": platform_fee_per_sharer,
            "mean_p2p_price": price_sum / steps,
            "matched_sell_kwh": matched_sell_volume,
            "matched_buy_kwh": matched_buy_volume,
            "matched_trade_kwh": matched_volume,
            "hoard_bill": hoard_bill,
            "share_bill": share_bill,
            "bill_savings": hoard_bill - share_bill,
            "bill_savings_pct": (hoard_bill - share_bill) / max(abs(hoard_bill), 1e-12) * 100.0,
            "realized_imbalance_cost": realized_imbalance,
            "u_s": u_s,
            "u_h": u_h,
        }
        return float(u_s), float(u_h), metrics
