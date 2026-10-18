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
        if sigma < 0 or c_u < 0 or c_o < 0:
            raise ValueError("Newsvendor inputs must be nonnegative")
        if sigma == 0 or np.isinf(z):
            return 0.0
        phi = norm.pdf(z)
        cdf = norm.cdf(z)
        expected_cost = sigma * (
            c_o * (phi + z * cdf)
            + c_u * (phi - z * (1.0 - cdf))
        )
        return float(max(expected_cost, 0.0))

    def generate_diurnal_load(self, timestamps, rng=None):
        """Generate one reproducible 15-minute load profile per household."""
        rng = self.rng if rng is None else rng
        hours = np.asarray([ts.hour + ts.minute / 60.0 for ts in timestamps])
        morning = 1.2 * np.exp(-((hours - 8.0) ** 2) / (2 * 1.5**2))
        evening = 2.2 * np.exp(-((hours - 19.5) ** 2) / (2 * 2.0**2))
        base = 0.8 + 0.3 * np.sin(np.pi * (hours - 6.0) / 12.0)
        expected_kw = np.outer(base + morning + evening, self.demand_multipliers)
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
        bills = np.zeros(self.num_prosumers)
        for generation_t, loads_t in zip(generation, loads):
            surplus, deficit = self._surplus_deficit_kwh(generation_t, loads_t)
            bills += self.p_grid * deficit - self.p_fit * surplus
        return float(np.mean(bills))

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

        total_gain = 0.0
        naive_imbalance_cost = 0.0
        aware_imbalance_cost = 0.0
        newsvendor_naive_cost = 0.0
        value_of_information = 0.0
        realized_gain = 0.0
        realized_imbalance = 0.0
        matched_volume = 0.0
        matched_sell_volume = 0.0
        matched_buy_volume = 0.0
        price_sum = 0.0

        for forecast_t, actual_t, sigma_t, mean_load_t, actual_load_t in zip(
            forecast, actual, sigma, mean_loads, actual_loads
        ):
            forecast_surplus, forecast_deficit = self._surplus_deficit_kwh(
                forecast_t, mean_load_t
            )
            actual_surplus, actual_deficit = self._surplus_deficit_kwh(
                actual_t, actual_load_t
            )
            s_hat = float(np.mean(forecast_surplus))
            d_hat = float(np.mean(forecast_deficit))
            s_pool = n_s * s_hat
            d_pool = n_s * d_hat
            sdr = s_pool / (d_pool + 1e-12)
            raw_price = self.p_fit + (self.p_grid - self.p_fit) / (1.0 + sdr)
            price = float(np.clip(
                raw_price, self.p_fit + self.c_tx, self.p_grid - self.c_tx
            ))
            rho_s = min(1.0, d_pool / s_pool) if s_pool > 0 else 0.0
            rho_d = min(1.0, s_pool / d_pool) if d_pool > 0 else 0.0
            gain_t = thin_market_efficiency * (
                rho_s * s_hat * (price - self.c_tx - self.p_fit)
                + rho_d * d_hat * (self.p_grid - price - self.c_tx)
            )
            total_gain += gain_t

            sigma_sol = sigma_t * self.community_scale_factor / self.num_prosumers * self.dt_hours
            sigma_e = self.forecast_error_std(
                sigma_sol, n_s, self.sigma_load_kwh
            )
            c_u_seller = max(price - self.c_tx - self.p_fit, 0.0)
            c_u_buyer = max(self.p_grid - price - self.c_tx, 0.0)
            z_seller = self.optimal_news_vendor_z(c_u_seller, self.c_imb)
            z_buyer = self.optimal_news_vendor_z(c_u_buyer, self.c_imb)

            cost_seller_naive = self.expected_newsvendor_cost(
                sigma_e, c_u_seller, self.c_imb, 0.0
            )
            cost_buyer_naive = self.expected_newsvendor_cost(
                sigma_e, c_u_buyer, self.c_imb, 0.0
            )
            cost_seller_aware = self.expected_newsvendor_cost(
                sigma_e, c_u_seller, self.c_imb, z_seller
            )
            cost_buyer_aware = self.expected_newsvendor_cost(
                sigma_e, c_u_buyer, self.c_imb, z_buyer
            )
            newsvendor_naive_cost += cost_seller_naive + cost_buyer_naive
            aware_imbalance_cost += cost_seller_aware + cost_buyer_aware
            value_of_information += max(
                (cost_seller_naive + cost_buyer_naive)
                - (cost_seller_aware + cost_buyer_aware),
                0.0,
            )
            direct_imbalance = self.c_imb * sigma_e * np.sqrt(2.0 / np.pi)
            naive_imbalance_cost += direct_imbalance

            if commitment == "aware":
                q_seller = (
                    s_hat if sigma_e == 0.0
                    else max(s_hat + z_seller * sigma_e, 0.0)
                )
                q_buyer = (
                    d_hat if sigma_e == 0.0
                    else max(d_hat + z_buyer * sigma_e, 0.0)
                )
                interval_cost = cost_seller_aware + cost_buyer_aware
            else:
                q_seller = s_hat
                q_buyer = d_hat
                interval_cost = direct_imbalance

            actual_s = float(np.mean(actual_surplus))
            actual_d = float(np.mean(actual_deficit))
            available_sell = min(n_s * q_seller, n_s * actual_s)
            available_buy = min(n_s * q_buyer, n_s * actual_d)
            matched = thin_market_efficiency * min(available_sell, available_buy)
            matched_volume += matched
            matched_sell_volume += matched
            matched_buy_volume += matched
            realized_gain += (matched / n_s) * (
                (price - self.c_tx - self.p_fit)
                + (self.p_grid - price - self.c_tx)
            )
            realized_imbalance += self.c_imb * (
                abs(q_seller - actual_s) + abs(q_buyer - actual_d)
            )
            price_sum += price

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
