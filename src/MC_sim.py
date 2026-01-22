"""
Monte Carlo simulation module for log-health process.

Simulates paths of h(t) = log( b_X * w_X * X(t) / (w_Y * Y(t)) )
using jump-diffusion processes with spectrally negative jumps.

Outputs:
- log-health paths
- optional collateral ratios
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional
from params_MLE import LogHealthParameters


@dataclass
class MonteCarloParams:
    """
    Parameters for Monte Carlo simulation
    """
    n_paths: int = 10000         # Number of simulated paths
    n_steps: int = 252           # Number of time steps (e.g., daily over 1 year)
    dt: float = 1/252            # Time step in years
    seed: Optional[int] = None    # Random seed for reproducibility


class MonteCarloSimulator:
    def __init__(self, lh_params: LogHealthParameters, mc_params: MonteCarloParams):
        self.lh = lh_params
        self.mc = mc_params
        if mc_params.seed is not None:
            np.random.seed(mc_params.seed)

    def simulate(self):
        """
        Simulate log-health paths using Euler-Maruyama + jump approximation.
        Returns:
            h_paths: ndarray of shape (n_paths, n_steps+1)
        """
        n_paths, n_steps, dt = self.mc.n_paths, self.mc.n_steps, self.mc.dt
        h0 = np.log(self.lh.b_X * self.lh.w_X / self.lh.w_Y)  # normalize X0=Y0=1
        h_paths = np.zeros((n_paths, n_steps+1))
        h_paths[:, 0] = h0

        # Precompute diffusion parameters
        mu_diff = self.lh.drift_diffusion
        sigma_diff = np.sqrt(self.lh.variance_diffusion)

        # Jump parameters
        lam_X, delta_X, eta_X = self.lh.pX.lam, self.lh.pX.delta, self.lh.pX.eta
        lam_Y, delta_Y, eta_Y = self.lh.pY.lam, self.lh.pY.delta, self.lh.pY.eta

        for t in range(1, n_steps+1):
            # Diffusion increment
            dW = np.random.normal(0, 1, size=n_paths)
            dh_diff = mu_diff * dt + sigma_diff * np.sqrt(dt) * dW

            # Jump increments
            # X downward jumps
            n_jumps_X = np.random.poisson(lam_X * dt, size=n_paths)
            jump_X = np.sum(-(delta_X + np.random.exponential(1/eta_X, size=(n_paths, n_jumps_X.max())) *
                             (np.arange(n_jumps_X.max()) < n_jumps_X[:, None])), axis=1)
            # Y upward jumps
            n_jumps_Y = np.random.poisson(lam_Y * dt, size=n_paths)
            jump_Y = np.sum((delta_Y + np.random.exponential(1/eta_Y, size=(n_paths, n_jumps_Y.max())) *
                             (np.arange(n_jumps_Y.max()) < n_jumps_Y[:, None])), axis=1)

            # Update h
            h_paths[:, t] = h_paths[:, t-1] + dh_diff + jump_X + jump_Y

        return h_paths

    def collateral_ratios(self, h_paths: np.ndarray):
        """
        Convert log-health to actual collateral ratios:
            CR(t) = exp(h(t))
        """
        return np.exp(h_paths)

    def summarize(self, h_paths: np.ndarray):
        """
        Compute basic statistics over simulated paths
        """
        cr = self.collateral_ratios(h_paths)
        mean_cr = np.mean(cr, axis=0)
        median_cr = np.median(cr, axis=0)
        prob_liquidation = np.mean(cr < 1.0, axis=0)  # fraction below 1

        summary = {
            "mean_collateral_ratio": mean_cr,
            "median_collateral_ratio": median_cr,
            "prob_liquidation": prob_liquidation,
            "final_mean": mean_cr[-1],
            "final_median": median_cr[-1],
            "final_liquidation_prob": prob_liquidation[-1]
        }
        return summary