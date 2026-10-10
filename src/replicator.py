"""
replicator.py
Numerical Replicator Dynamics engine for Evolutionary Game Theory simulation.
Models the time-varying evolutionary trajectory of community prosumers
adapting between Share and Hoard strategies under forecast uncertainty.
"""

import numpy as np
import pandas as pd


class ReplicatorEngine:
    """
    Simulates Replicator Dynamics:
    dx/dt = x(t) * (1 - x(t)) * (U_S(t) - U_H(t))
    using discrete Euler time integration with numerical clamping to [0, 1].
    """
    def __init__(self, step_size=0.1, cooperation_threshold=0.80):
        self.eta = step_size
        self.coop_threshold = cooperation_threshold

    def simulate(self, community, gen_series, sigma_series, load_matrix,
                 timestamps, x0=0.5, lambda_s=2.5, lambda_h=0.0):
        """
        Runs step-by-step evolutionary simulation across the time series.
        - community: ProsumerCommunity instance
        - gen_series: array of solar generation (kW) [T]
        - sigma_series: array of forecast uncertainty (kW) [T]
        - load_matrix: prosumer load array (kW) [T, N]
        - timestamps: datetime index [T]
        - x0: initial fraction of sharers in [0, 1]
        """
        T = len(gen_series)
        x_history = np.zeros(T)
        us_history = np.zeros(T)
        uh_history = np.zeros(T)
        delta_u_history = np.zeros(T)
        p2p_price_history = np.zeros(T)
        p2p_traded_history = np.zeros(T)
        grid_import_history = np.zeros(T)
        grid_dep_history = np.zeros(T)
        cost_p2p_history = np.zeros(T)
        cost_nop2p_history = np.zeros(T)

        curr_x = float(x0)

        for t in range(T):
            gen_t = float(gen_series[t])
            sigma_t = float(sigma_series[t])
            loads_t = load_matrix[t]

            # Calculate dynamic payoffs at current state
            u_s, u_h, met = community.calculate_step_payoffs(
                gen_kw=gen_t,
                sigma_kw=sigma_t,
                loads_kw=loads_t,
                x_share=curr_x,
                lambda_s=lambda_s,
                lambda_h=lambda_h
            )

            # Record state
            x_history[t] = curr_x
            us_history[t] = u_s
            uh_history[t] = u_h
            delta_u = u_s - u_h
            delta_u_history[t] = delta_u
            p2p_price_history[t] = met['p_p2p']
            p2p_traded_history[t] = met['q_p2p_kwh']
            grid_import_history[t] = met['grid_import_kwh']
            grid_dep_history[t] = met['grid_dep_ratio']

            # Cost with current P2P state
            cost_p2p = -(curr_x * met['pi_s'] + (1.0 - curr_x) * met['pi_h']) * community.num_prosumers
            cost_p2p_history[t] = cost_p2p

            # Baseline cost with zero P2P sharing (x=0)
            _, _, met_zero = community.calculate_step_payoffs(
                gen_kw=gen_t,
                sigma_kw=sigma_t,
                loads_kw=loads_t,
                x_share=0.0,
                lambda_s=lambda_s,
                lambda_h=lambda_h
            )
            cost_nop2p = -met_zero['pi_h'] * community.num_prosumers
            cost_nop2p_history[t] = cost_nop2p

            # Replicator dynamics Euler update
            # dx = x * (1 - x) * delta_u
            dx = curr_x * (1.0 - curr_x) * delta_u
            next_x = curr_x + self.eta * dx
            # Numerical clamping to unit interval
            curr_x = float(np.clip(next_x, 1e-4, 1.0 - 1e-4))

        results_df = pd.DataFrame({
            'DATE_TIME': timestamps,
            'x_share': x_history,
            'U_S': us_history,
            'U_H': uh_history,
            'Delta_U': delta_u_history,
            'P_P2P': p2p_price_history,
            'Q_P2P_kWh': p2p_traded_history,
            'Grid_Import_kWh': grid_import_history,
            'Grid_Dep_Ratio': grid_dep_history,
            'Cost_Community': cost_p2p_history,
            'Cost_NoP2P_Baseline': cost_nop2p_history
        })

        summary = {
            'Initial_Share': x0,
            'Final_Share': x_history[-1],
            'Mean_Share': np.mean(x_history),
            'Time_Above_Threshold': np.mean(x_history >= self.coop_threshold),
            'Total_P2P_Traded_kWh': np.sum(p2p_traded_history),
            'Total_Grid_Import_kWh': np.sum(grid_import_history),
            'Mean_Grid_Dep_Ratio': np.mean(grid_dep_history),
            'Total_Cost ($)': np.sum(cost_p2p_history),
            'Baseline_Cost ($)': np.sum(cost_nop2p_history),
            'Total_Savings ($)': np.sum(cost_nop2p_history) - np.sum(cost_p2p_history),
            'Savings_Pct (%)': (np.sum(cost_nop2p_history) - np.sum(cost_p2p_history)) / (np.sum(cost_nop2p_history) + 1e-6) * 100.0
        }

        return results_df, summary
