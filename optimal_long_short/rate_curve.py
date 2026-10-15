"""Aave utilization dynamics and the piecewise-linear borrow-rate curve.

This module replaces the CIR / CIR-with-jumps fits in
``results/cir_usdc_weth_aave_v3.csv`` and ``results/jcir_usdc_weth_aave_v3.csv``,
which are numerically pathological (``kappa`` of 2089 and 1399 per year -- a
three-hour mean-reversion half-life -- ``theta`` pinned at its optimizer bound,
and a jump-diffusion variant that violates the Feller condition, ``2*kappa*theta
= 13.9`` against ``sigma^2 = 72.3``).

The diagnosis is structural rather than numerical.  Aave's variable borrow rate
is not a stochastic process in its own right: it is a *deterministic*,
piecewise-linear, invertible function of the reserve's utilization,

    r_b(U) = base + slope1 * U / U_opt,                          U <= U_opt
    r_b(U) = base + slope1 + slope2 * (U - U_opt)/(1 - U_opt),   U >  U_opt.

Fitting an affine mean-reverting process to a kinked transform of a bounded
state manufactures both the apparent jumps (the kink) and the apparent extreme
mean reversion (the flat segment).  The fix is to model the bounded state ``U``
on ``[0, 1]`` and push it through the known curve.

Two consequences are used throughout:

*Inversion.*  Because ``r_b`` is strictly increasing in ``U``, the observed
borrow rate is a **sufficient statistic** for utilization -- utilization never
needs to be measured directly.  This matters because the naive ratio
``totalBorrowBalanceUSD / totalDepositBalanceUSD`` from the subgraph is biased
upward: for USDC it reports a mean of 0.881 and a maximum above 1, against a
curve-implied mean of 0.677 and maximum of 0.931, and it misclassifies 25.4% of
hours as above ``U_opt`` when the true figure is 10.4%.  In the flat region,
where the two measures should agree exactly, they do (WBTC: correlation 1.0000).

*Convexity.*  The kink makes the supplier's rate convex in ``U``.  A supplier
earns ``U * r_b(U) * (1 - reserve_factor)``, which is the payoff of a long
position in a call on utilization struck at ``U_opt``; the borrower is short it.
This is the leg the barbell buys.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimal_long_short.aave_hourly import RateCurve

__all__ = [
    "RateCurve",
    "implied_utilization",
    "fit_rate_curve",
    "JacobiParams",
    "fit_jacobi",
    "regime_summary",
]


def implied_utilization(curve: RateCurve, borrow_rate) -> np.ndarray:
    """Invert ``r_b(U)`` for utilization.

    Parameters
    ----------
    borrow_rate : array-like
        Annualized variable borrow rate as a **decimal** (0.027 for 2.7%).

    Returns
    -------
    ndarray of utilization in ``[0, 1]``.
    """
    rate = np.asarray(borrow_rate, dtype=float)
    kink_rate = curve.base + curve.slope1
    with np.errstate(divide="ignore", invalid="ignore"):
        below = curve.u_opt * (rate - curve.base) / curve.slope1
        above = curve.u_opt + (1.0 - curve.u_opt) * (rate - kink_rate) / curve.slope2
    return np.clip(np.where(rate <= kink_rate, below, above), 0.0, 1.0)


def fit_rate_curve(
    utilization,
    borrow_rate,
    *,
    u_opt_grid: np.ndarray | None = None,
) -> tuple[RateCurve, float]:
    """Estimate the rate curve from data by profiled least squares.

    The kink location ``u_opt`` is profiled over a grid; for each candidate the
    remaining three parameters enter linearly and are solved by OLS on the
    design ``[1, min(U,u)/u, max(U-u,0)/(1-u)]``.

    Provided for comparison against the protocol's published parameters
    (:func:`optimal_long_short.aave_hourly.published_curve`).  The two agree
    only where the data actually visit both segments -- a reserve that never
    approaches its kink cannot identify ``slope2``, which is itself a finding.

    Returns
    -------
    (curve, rmse)
    """
    u = np.asarray(utilization, dtype=float)
    y = np.asarray(borrow_rate, dtype=float)
    mask = np.isfinite(u) & np.isfinite(y)
    u, y = u[mask], y[mask]
    if u.size < 10:
        raise ValueError("Need at least 10 observations to fit a rate curve.")

    if u_opt_grid is None:
        lo, hi = np.quantile(u, [0.02, 0.98])
        u_opt_grid = np.linspace(max(lo, 0.05), min(hi, 0.99), 95)

    best: tuple[RateCurve, float] | None = None
    for u_opt in u_opt_grid:
        if not 0.0 < u_opt < 1.0:
            continue
        design = np.column_stack(
            [
                np.ones_like(u),
                np.minimum(u, u_opt) / u_opt,
                np.maximum(u - u_opt, 0.0) / (1.0 - u_opt),
            ]
        )
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        rmse = float(np.sqrt(np.mean((design @ coef - y) ** 2)))
        if best is None or rmse < best[1]:
            best = (
                RateCurve(
                    u_opt=float(u_opt),
                    base=float(coef[0]),
                    slope1=float(coef[1]),
                    slope2=float(coef[2]),
                ),
                rmse,
            )
    assert best is not None
    return best


@dataclass(frozen=True)
class JacobiParams:
    """Jacobi (Wright-Fisher) diffusion for utilization on ``[0, 1]``.

        dU_t = kappa (theta - U_t) dt + sigma sqrt(U_t (1 - U_t)) dW_t

    Chosen over CIR because utilization is bounded above as well as below; CIR
    is a half-line process and has no mechanism to respect ``U <= 1``.

    Attributes
    ----------
    kappa, theta, sigma : float
        Mean-reversion speed (per year), long-run level, volatility.
    """

    kappa: float
    theta: float
    sigma: float

    @property
    def boundaries_unattainable(self) -> tuple[bool, bool]:
        """Whether ``0`` and ``1`` are unattainable.

        The Jacobi analogue of the Feller condition requires
        ``2 kappa theta >= sigma^2`` at the lower boundary and
        ``2 kappa (1 - theta) >= sigma^2`` at the upper one.
        """
        s2 = self.sigma**2
        return (
            2.0 * self.kappa * self.theta >= s2,
            2.0 * self.kappa * (1.0 - self.theta) >= s2,
        )

    @property
    def stationary_moments(self) -> tuple[float, float]:
        """Stationary mean and variance.

        The stationary law is Beta(a, b) with ``a = 2 kappa theta / sigma^2``
        and ``b = 2 kappa (1 - theta) / sigma^2``.
        """
        s2 = self.sigma**2
        a = 2.0 * self.kappa * self.theta / s2
        b = 2.0 * self.kappa * (1.0 - self.theta) / s2
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1.0))
        return float(mean), float(var)


def fit_jacobi(utilization, dt_years: float) -> JacobiParams:
    """Euler pseudo-likelihood estimator for the Jacobi diffusion.

    The drift is estimated by OLS of ``dU`` on ``U``, which is exact for the
    Euler scheme; ``sigma`` follows from the residual variance normalized by the
    diffusion factor ``U(1 - U)``.

    Unlike the CIR fits this replaces, the estimator is bound-free: ``kappa``
    and ``theta`` are read off a linear regression rather than pushed against
    optimizer constraints.
    """
    u = np.asarray(utilization, dtype=float)
    u = u[np.isfinite(u)]
    if u.size < 50:
        raise ValueError("Need at least 50 observations to fit a Jacobi diffusion.")

    u_prev, u_next = u[:-1], u[1:]
    du = u_next - u_prev

    # du = kappa*theta*dt - kappa*dt*u + noise  ->  slope = -kappa*dt
    design = np.column_stack([np.ones_like(u_prev), u_prev])
    coef, *_ = np.linalg.lstsq(design, du, rcond=None)
    kappa = -coef[1] / dt_years
    theta = coef[0] / (kappa * dt_years) if kappa != 0.0 else float(np.mean(u))

    resid = du - design @ coef
    denom = u_prev * (1.0 - u_prev)
    ok = denom > 1e-10
    sigma = float(np.sqrt(np.mean(resid[ok] ** 2 / denom[ok]) / dt_years))

    return JacobiParams(
        kappa=float(kappa), theta=float(np.clip(theta, 0.0, 1.0)), sigma=sigma
    )


def regime_summary(
    utilization,
    curve: RateCurve,
    *,
    reserve_factor: float = 0.0,
) -> pd.Series:
    """Summarize time spent either side of the kink and the rates earned there.

    Produces the inputs for the two-state switching model: the occupation
    fraction of the stressed regime, the mean borrow rate in each regime, and
    the supplier's mean rate in each regime -- the last pair quantifying the
    convexity the barbell's supplier leg is long.
    """
    u = np.asarray(utilization, dtype=float)
    u = u[np.isfinite(u)]
    stressed = u > curve.u_opt
    rb = np.array([curve.borrow_rate(x) for x in u])
    rs = rb * u * (1.0 - reserve_factor)

    def _mean(values, mask):
        return float(np.mean(values[mask])) if mask.any() else float("nan")

    return pd.Series(
        {
            "n_obs": int(u.size),
            "u_opt": curve.u_opt,
            "u_mean": float(np.mean(u)),
            "u_p95": float(np.quantile(u, 0.95)),
            "u_max": float(np.max(u)),
            "frac_stressed": float(np.mean(stressed)),
            "r_b_mean_normal": _mean(rb, ~stressed),
            "r_b_mean_stressed": _mean(rb, stressed),
            "r_b_max": float(np.max(rb)),
            "r_s_mean_normal": _mean(rs, ~stressed),
            "r_s_mean_stressed": _mean(rs, stressed),
            "r_s_max": float(np.max(rs)),
            "convexity_ratio": _mean(rs, stressed) / _mean(rs, ~stressed)
            if (~stressed).any() and stressed.any()
            else float("nan"),
        }
    )
