"""CIR process helpers: simulation, simple Euler-based parameter fitting, and plotting.

Model: dX_t = kappa*(theta - X_t) dt + sigma * sqrt(X_t) dW_t

This module provides:
- simulate_cir_euler: forward-simulate with Euler-Maruyama (keeps non-negativity by max(0, x)).
- fit_cir_euler: estimate (kappa, theta, sigma) from a sampled series using an Euler discretization
  regression: Δx = a + b * x + noise, where b = -kappa*dt, a = kappa*theta*dt.
- fit_and_plot: convenience wrapper that fits and plots data vs. a simulated path.

Notes: The estimator is a simple, robust approach suitable for exploratory work. For
high-quality inference use exact transition-density MLE (not implemented here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import math
try:
    from scipy.stats import ncx2
    from scipy.optimize import minimize
except Exception:
    ncx2 = None
    minimize = None


def simulate_cir_euler(x0: float, kappa: float, theta: float, sigma: float,
                       dt: float, n_steps: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Simulate CIR via Euler-Maruyama, reflecting negatives to zero.

    Returns array of length n_steps+1 (including x0).
    """
    if rng is None:
        rng = np.random.default_rng()
    xs = np.empty(n_steps + 1, dtype=float)
    xs[0] = float(x0)
    for i in range(n_steps):
        xi = xs[i]
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt))
        dx = kappa * (theta - xi) * dt + sigma * np.sqrt(max(xi, 0.0)) * dW
        xn = xi + dx
        # enforce non-negativity
        xs[i + 1] = max(xn, 0.0)
    return xs


def fit_cir_euler(x: np.ndarray, dt: float) -> Dict[str, float]:
    """Fit CIR parameters from observations x[0], x[1], ... using Euler discretization.

    Model for estimation:
      Δx_i = x_{i+1} - x_i = a + b * x_i + residual_i,
    where b = -kappa * dt, a = kappa * theta * dt.

    sigma is estimated from residuals via Var(residual) ≈ sigma^2 * x_i * dt.

    Returns dict {'kappa':..., 'theta':..., 'sigma':...}.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1-D array with at least 3 observations")
    # lagged design
    x_lag = x[:-1]
    dx = x[1:] - x_lag

    # design matrix [1, x_lag]
    A = np.vstack([np.ones_like(x_lag), x_lag]).T
    # least squares
    coeffs, *_ = np.linalg.lstsq(A, dx, rcond=None)
    a_hat, b_hat = coeffs

    # translate to parameters
    kappa_hat = -b_hat / dt
    if abs(b_hat) < 1e-12:
        raise ValueError("Estimated slope near zero; cannot recover kappa/theta reliably")
    theta_hat = a_hat / (kappa_hat * dt)

    # residuals
    resid = dx - (a_hat + b_hat * x_lag)
    # estimate sigma^2 using E[resid^2] ≈ sigma^2 * x_lag * dt  => sum resid^2 ≈ sigma^2 * dt * sum x_lag
    denom = dt * np.sum(np.where(x_lag > 0, x_lag, 0.0))
    if denom <= 0:
        sigma_hat = 0.0
    else:
        sigma2_hat = np.sum(resid ** 2) / denom
        sigma_hat = float(np.sqrt(max(0.0, sigma2_hat)))

    return {"kappa": float(kappa_hat), "theta": float(theta_hat), "sigma": float(sigma_hat)}


def fit_and_plot(series: pd.Series, dt: float, n_sim: Optional[int] = None,
                 title: Optional[str] = None, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Fit CIR params from a `pandas.Series` and plot data vs. a simulated path.

    series: index can be datetime, values are positive APY numbers.
    dt: sampling interval in time units consistent with process time (e.g., years).
    n_sim: number of simulated steps; if None uses len(series)-1.
    Returns the fitted parameter dict.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    x = series.dropna().values
    if n_sim is None:
        n_sim = max(1, x.size - 1)

    params = fit_cir_euler(x, dt)

    # simulate one path using fitted params
    sim = simulate_cir_euler(x0=x[0], kappa=params["kappa"], theta=params["theta"],
                            sigma=params["sigma"], dt=dt, n_steps=n_sim, rng=rng)

    # plot
    plt.figure(figsize=(10, 3.5))
    t = np.arange(0, sim.size) * dt
    plt.plot(series.index[: sim.size], x[: sim.size], label="data", lw=0.9)
    plt.plot(series.index[: sim.size], sim, label="CIR sim (fitted)", lw=0.9, alpha=0.8)
    plt.title(title or "CIR fit vs data")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return params


def _cir_transition_logpdf(x_next: float, x_curr: float, kappa: float, theta: float, sigma: float, dt: float) -> float:
    """Log-pdf of CIR transition X_{t+dt} | X_t = x_curr using non-central chi-square.

    Uses the parameterization where
      c = sigma^2 * (1 - exp(-kappa dt)) / (4 kappa)
      d = 4 kappa theta / sigma^2
      lambda = 4 kappa x_curr * exp(-kappa dt) / (sigma^2 * (1 - exp(-kappa dt)))
    Then X_{t+dt} ~ c * ncx2(df=d, nc=lambda).
    """
    if ncx2 is None:
        raise RuntimeError("scipy is required for CIR exact likelihood (scipy.stats.ncx2)")
    if x_next < 0:
        return -np.inf
    if kappa <= 0 or sigma <= 0 or theta < 0:
        return -np.inf
    expm = math.exp(-kappa * dt)
    c = (sigma ** 2) * (1 - expm) / (4.0 * kappa)
    if c <= 0:
        return -np.inf
    d = 4.0 * kappa * theta / (sigma ** 2)
    lam = 4.0 * kappa * x_curr * expm / (sigma ** 2 * (1 - expm))
    # argument for ncx2 is x_next / c
    try:
        z = x_next / c
        # use logpdf with df=d and non-centrality lam
        lp = ncx2.logpdf(z, df=d, nc=lam)
        # adjust for scale c
        return float(lp - math.log(c))
    except Exception:
        return -np.inf


def fit_cir_mle(x: np.ndarray, dt: float, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
                initial: Dict[str, float] = None, method: str = 'L-BFGS-B') -> Dict[str, float]:
    """Fit CIR parameters by maximizing exact likelihood using the transition density.

    Parameters:
      x: 1-D array of observations (X_0,...,X_n)
      dt: sampling interval in time units (e.g., years)
      bounds: optional ((kappa_min,kappa_max),(theta_min,theta_max),(sigma_min,sigma_max))
      initial: optional initial guess dict {'kappa':..,'theta':..,'sigma':..}

    Returns dict with fitted parameters.
    """
    if minimize is None or ncx2 is None:
        raise RuntimeError("scipy is required for fit_cir_mle")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1-D array with at least 3 observations")

    # default bounds
    if bounds is None:
        bounds = ((1e-6, 2000.0), (0.0, 2.0), (1e-6, 10.0))
    # initial via Euler estimator if not provided
    if initial is None:
        try:
            e_est = fit_cir_euler(x, dt)
            init_k = max(1e-6, e_est['kappa'])
            init_th = max(0.0, e_est['theta'])
            init_sig = max(1e-6, e_est['sigma'])
        except Exception:
            init_k, init_th, init_sig = 1.0, float(x.mean()), float(np.std(x))
    else:
        init_k = initial.get('kappa', 1.0)
        init_th = initial.get('theta', float(x.mean()))
        init_sig = initial.get('sigma', float(np.std(x)))

    def neg_loglik(pars):
        kappa, theta, sigma = pars
        # enforce positivity
        if kappa <= 0 or sigma <= 0 or theta < 0:
            return 1e20
        ll = 0.0
        # iterate transitions
        for i in range(len(x) - 1):
            lp = _cir_transition_logpdf(x[i + 1], x[i], kappa, theta, sigma, dt)
            if not np.isfinite(lp):
                return 1e20
            ll += lp
        return -ll

    x0 = np.array([init_k, init_th, init_sig], dtype=float)
    bnds = [bounds[0], bounds[1], bounds[2]]
    res = minimize(neg_loglik, x0, bounds=bnds, method=method)
    if not res.success:
        # try different starting point: perturb
        x0b = x0 * np.array([0.5, 1.0, 1.5])
        res2 = minimize(neg_loglik, x0b, bounds=bnds, method=method)
        res = res2 if res2.success else res

    kf, tf, sf = res.x
    return {'kappa': float(kf), 'theta': float(tf), 'sigma': float(sf), 'success': bool(res.success), 'message': res.message}


if __name__ == "__main__":
    # tiny self-check / demonstration
    rng = np.random.default_rng(12345)
    params_true = {"kappa": 1.5, "theta": 0.03, "sigma": 0.08}
    dt = 1.0 / 365.0
    sim = simulate_cir_euler(x0=0.02, kappa=params_true["kappa"], theta=params_true["theta"],
                             sigma=params_true["sigma"], dt=dt, n_steps=1000, rng=rng)
    fitted = fit_cir_euler(sim, dt)
    print("demo fitted:", fitted)
