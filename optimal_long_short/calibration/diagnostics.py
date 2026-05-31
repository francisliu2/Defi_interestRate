"""
Post-calibration diagnostics for the bivariate Kou ECF estimator.

diagnose_calibration(result, r1, r2) returns a dict with:
    "moments_r1"               -- empirical moments of r1
    "moments_r2"               -- empirical moments of r2
    "moments_z"                -- empirical moments of Z = r1 - r2
    "objective_by_group"       -- normalized ECF error per frequency group
    "spread_quantile_comparison" -- empirical vs model-simulated Z quantiles
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from .calibrate import ECFCalibrationResult


def _moments(x: np.ndarray) -> dict[str, float]:
    return {
        "mean":     float(np.mean(x)),
        "std":      float(np.std(x)),
        "skewness": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "q01":      float(np.percentile(x, 1)),
        "q05":      float(np.percentile(x, 5)),
        "q95":      float(np.percentile(x, 95)),
        "q99":      float(np.percentile(x, 99)),
    }


def _simulate_spread_increments(
    result: "ECFCalibrationResult",
    n_sim: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate n_sim one-step spread increments dZ = dX1 - dX2 from the
    fitted parameters over dt = result.dt_years.
    """
    p = result.params
    dt = result.dt_years
    rng = np.random.default_rng(seed)

    sqrt_dt     = np.sqrt(dt)
    sqrt_1_r2   = np.sqrt(max(1.0 - p.rho ** 2, 0.0))
    W1 = rng.standard_normal(n_sim)
    W2 = rng.standard_normal(n_sim)
    dB1 = sqrt_dt * W1
    dB2 = sqrt_dt * (p.rho * W1 + sqrt_1_r2 * W2)

    def _kou_jumps(lam: float, pi: float, ep: float, en: float) -> np.ndarray:
        n_j = rng.poisson(lam * dt, size=n_sim)
        total = np.zeros(n_sim)
        for i in np.nonzero(n_j)[0]:
            k = int(n_j[i])
            sg = rng.choice([1, -1], size=k, p=[pi, 1.0 - pi])
            sz = np.where(sg == 1, rng.exponential(ep, k), -rng.exponential(en, k))
            total[i] = sz.sum()
        return total

    dr1 = (p.muX1 * dt + p.sigma1 * dB1
           + _kou_jumps(p.lam1, p.p1, p.eta1_pos, p.eta1_neg))
    dr2 = (p.muX2 * dt + p.sigma2 * dB2
           + _kou_jumps(p.lam2, p.p2, p.eta2_pos, p.eta2_neg))
    return dr1 - dr2


def diagnose_calibration(
    result: "ECFCalibrationResult",
    r1: np.ndarray,
    r2: np.ndarray,
    n_sim: int = 50_000,
    sim_seed: int = 0,
) -> dict:
    """
    Compute post-calibration diagnostics.

    Parameters
    ----------
    result   : ECFCalibrationResult from calibrate_ecf.
    r1, r2   : (N,) log-return arrays used for calibration.
    n_sim    : Number of one-step increments to simulate for spread comparison.
    sim_seed : RNG seed for the simulation.

    Returns
    -------
    dict with the keys documented at the top of this module.
    """
    z = r1 - r2

    quantiles = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    emp_q = {f"q{int(100*q):02d}": float(np.percentile(z, 100 * q)) for q in quantiles}

    dz_sim = _simulate_spread_increments(result, n_sim, sim_seed)
    sim_q  = {f"q{int(100*q):02d}": float(np.percentile(dz_sim, 100 * q)) for q in quantiles}

    return {
        "moments_r1": _moments(r1),
        "moments_r2": _moments(r2),
        "moments_z":  _moments(z),
        "objective_by_group": result.objective_by_group,
        "spread_quantile_comparison": {
            "empirical": emp_q,
            "model_sim": sim_q,
        },
    }
