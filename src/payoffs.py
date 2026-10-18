"""
payoffs.py
Economic model and Dynamic Payoff functions for Peer-to-Peer (P2P) energy sharing
versus Hoarding in a Renewable Energy Community (REC).
Couples solar generation forecasts and predictive uncertainty into risk-adjusted utilities.
"""

import numpy as np
import pandas as pd


class ProsumerCommunity:
    """
    Simulates a heterogeneous prosumer community of N households with
    solar allocations and diurnal residential demand profiles.
    """
    def __init__(self, num_prosumers=100, plant_capacity_kw=29150.0,
                 p_grid=0.22, p_fit=0.06, c_tx=0.01, seed=42):
        self.num_prosumers = num_prosumers
        self.p_grid = p_grid    # Grid import retail tariff ($/kWh)
        self.p_fit = p_fit      # Feed-in tariff ($/kWh)
        self.c_tx = c_tx        # Platform transaction fee ($/kWh)
        self.plant_capacity = plant_capacity_kw

        self.rng = np.random.default_rng(seed)
        # Scaled community capacity: ~500 kW peak community solar (~5 kW per household)
        self.comm_scale_factor = 500.0 / plant_capacity_kw

        # Heterogeneous solar share allocations (Dirichlet distribution)
        alpha = self.rng.uniform(0.6, 1.4, size=num_prosumers)
        self.solar_weights = alpha / np.sum(alpha)

        # Heterogeneous base demand levels (mean 1.8 kW per household, peak ~3.5 kW)
        self.demand_multipliers = self.rng.uniform(0.6, 1.5, size=num_prosumers)

    def generate_diurnal_load(self, timestamps):
        """
        Synthesizes realistic 15-minute residential load profile (kW)
        with morning peak (7:00-9:00), evening peak (18:00-21:00), and base load.
        """
        hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
        # Dual-Gaussian residential curve + baseload
        morning_peak = 1.2 * np.exp(-((hours - 8.0) ** 2) / (2 * (1.5 ** 2)))
        evening_peak = 2.2 * np.exp(-((hours - 19.5) ** 2) / (2 * (2.0 ** 2)))
        base_load = 0.8 + 0.3 * np.sin(np.pi * (hours - 6.0) / 12.0)
        base_curve = base_load + morning_peak + evening_peak  # Shape: (T,)

        # Expand to (T, num_prosumers) with household heterogeneity
        loads = np.outer(base_curve, self.demand_multipliers)
        # Add stochastic household noise
        noise = self.rng.normal(0, 0.08, size=loads.shape)
        loads = np.clip(loads + noise, 0.3, None)
        return loads

    def calculate_step_payoffs(self, gen_kw, sigma_kw, loads_kw, x_share,
                               lambda_s=2.5, lambda_h=0.0, dt_hours=0.25):
        """
        Calculates time-varying payoffs U_S and U_H for a given 15-minute interval:
        - gen_kw: plant solar generation (kW)
        - sigma_kw: forecast uncertainty (predictive std in kW)
        - loads_kw: prosumer load vector (kW) of length N
        - x_share: current population fraction choosing Share in [0, 1]
        - lambda_s: risk penalty sensitivity for Sharers
        - lambda_h: risk penalty sensitivity for Hoarders
        - dt_hours: interval duration (0.25 h for 15-min)
        """
        # Prosumer solar generation
        comm_solar = gen_kw * self.comm_scale_factor
        solar_i = comm_solar * self.solar_weights  # shape (N,)
        load_i = loads_kw                          # shape (N,)

        # Energy quantities (kWh)
        e_solar_i = solar_i * dt_hours
        e_load_i = load_i * dt_hours
        net_i = e_solar_i - e_load_i

        surplus_i = np.maximum(0.0, net_i)
        deficit_i = np.maximum(0.0, -net_i)

        # Aggregate community surplus and deficit
        total_surplus = np.sum(surplus_i)
        total_deficit = np.sum(deficit_i)

        # P2P trading market quantities available from Sharers
        # Sharers participate with probability x_share (or population share x_share)
        p2p_surplus_avail = x_share * total_surplus
        p2p_deficit_avail = x_share * total_deficit

        # Supply-Demand Ratio (SDR)
        eps = 1e-6
        sdr = p2p_surplus_avail / (p2p_deficit_avail + eps)

        # Dynamic P2P clearing price: mid-market SDR formula
        # Price lies between feed-in tariff and grid retail rate
        price_spread = self.p_grid - self.p_fit
        # If surplus is high (SDR >> 1), price approaches p_fit + c_tx
        # If deficit is high (SDR << 1), price approaches p_grid - c_tx
        sdr_factor = 1.0 - (sdr / (1.0 + sdr))
        p_p2p_raw = self.p_fit + price_spread * sdr_factor
        p_p2p = np.clip(p_p2p_raw, self.p_fit + self.c_tx, self.p_grid - self.c_tx)

        # Traded P2P energy volume
        q_p2p = min(p2p_surplus_avail, p2p_deficit_avail)

        # Ratios of peer fulfillment
        rho_s = q_p2p / (p2p_surplus_avail + eps) if p2p_surplus_avail > 0 else 0.0
        rho_d = q_p2p / (p2p_deficit_avail + eps) if p2p_deficit_avail > 0 else 0.0

        # --- Payoff for HOARD Strategy ---
        # Hoarders never participate in P2P:
        # Sell all surplus to grid at p_fit, import all deficit from grid at p_grid
        pi_h_i = surplus_i * self.p_fit - deficit_i * self.p_grid
        pi_h = np.mean(pi_h_i)

        # --- Payoff for SHARE Strategy ---
        # Sharers trade in P2P:
        # P2P surplus sold at (p_p2p - c_tx), unfulfilled surplus sold to grid at p_fit
        # P2P deficit bought at p_p2p, unfulfilled deficit bought from grid at p_grid
        rev_s_i = (rho_s * surplus_i) * (p_p2p - self.c_tx) + ((1.0 - rho_s) * surplus_i) * self.p_fit
        cost_s_i = (rho_d * deficit_i) * p_p2p + ((1.0 - rho_d) * deficit_i) * self.p_grid
        pi_s_i = rev_s_i - cost_s_i
        pi_s = np.mean(pi_s_i)

        # --- Risk Penalty Term ---
        # Normalized uncertainty measure R_t = (sigma_t / Capacity)^2
        r_t = ((sigma_kw * self.comm_scale_factor) / 500.0) ** 2

        u_s = pi_s - lambda_s * r_t
        u_h = pi_h - lambda_h * r_t

        # Community metrics
        grid_import_kwh = (1.0 - x_share) * total_deficit + (p2p_deficit_avail - q_p2p)
        total_demand_kwh = np.sum(e_load_i)
        grid_dep_ratio = grid_import_kwh / (total_demand_kwh + eps)

        metrics = {
            'p_p2p': p_p2p,
            'sdr': sdr,
            'q_p2p_kwh': q_p2p,
            'grid_import_kwh': grid_import_kwh,
            'grid_dep_ratio': grid_dep_ratio,
            'pi_s': pi_s,
            'pi_h': pi_h,
            'r_t': r_t,
            'u_s': u_s,
            'u_h': u_h
        }

        return u_s, u_h, metrics
