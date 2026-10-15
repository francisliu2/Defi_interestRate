r"""Bayesian regularization of the spread drift and the induced sizing distribution.

Section 8 of the paper reports a negative result: on the calibrated benchmark the
optimal health buffer sits at the unlevered boundary because the spread drift
``delta = mu1 - mu2`` is negative, and its sign --- which is all that decides
whether an interior optimum exists --- is weakly identified.  This module turns
that point estimate into a posterior and propagates it through the evaluation map.

Model
-----
Prior: ``delta ~ Normal(m0, s0^2)``, anchored on the protocol carry (the book earns
the supply APR on collateral and pays the borrow APR on debt), so ``m0`` is the net
carry and ``s0`` is the a priori scale of a directional relative-growth view.

Likelihood: the window of ``n`` observations over ``T_obs`` years estimates ``delta``
with standard error ``sqrt(Var(Z_1) / T_obs)``, where ``Var(Z_1)`` is the total
(diffusion + jump) instantaneous variance of ``Z = X1 - X2`` per unit time.

Posterior: conjugate Gaussian.  Mapping each posterior draw through
:func:`optimal_buffer` --- which returns the unlevered boundary below a
risk-aversion-dependent threshold ``delta*(alpha)`` and an interior buffer above it
--- yields a mixture distribution over ``H0*``: an atom at the unlevered position
with probability ``P(delta < delta*)`` and a continuous interior part above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optimal_long_short.model.drift_service import with_price_drift_view
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.overshoot_recovery import exact_wealth_moments
from optimal_long_short.recovery import wealth_moments

__all__ = [
    "spread_drift_standard_error",
    "gaussian_posterior",
    "optimal_buffer",
    "interior_threshold",
    "sizing_distribution",
]


def spread_drift_standard_error(params: KouParams, T_obs: float) -> dict[str, float]:
    """Standard error of the realized annualized spread drift over ``T_obs`` years.

    Returns the diffusion-only and total (diffusion + jump) standard errors; the
    latter is ``sqrt(Var(Z_1) / T_obs)`` with ``Var(Z_1)`` the total instantaneous
    variance of ``Z`` per unit time.
    """
    p = params
    sig_z2 = p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2

    def jump_var(lam, prob, ep, en):
        # Var contribution of the asset jump per year: lam * E[J^2],
        # E[J^2] = 2 (prob*ep^2 + (1-prob)*en^2) for the exponential mixture.
        return lam * 2.0 * (prob * ep**2 + (1 - prob) * en**2)

    var_jump = (
        jump_var(p.lam1, p.p1, p.eta1_pos, p.eta1_neg)
        + jump_var(p.lam2, p.p2, p.eta2_pos, p.eta2_neg)
    )
    var_total = sig_z2 + var_jump
    return {
        "se_diffusion": math.sqrt(sig_z2 / T_obs),
        "se_total": math.sqrt(var_total / T_obs),
        "var_Z_total": var_total,
        "sigma_Z": math.sqrt(sig_z2),
    }


def gaussian_posterior(
    prior_mean: float, prior_sd: float, data: float, se: float
) -> tuple[float, float]:
    """Conjugate Gaussian posterior ``(mean, sd)`` for a scalar drift."""
    tau0, taud = 1.0 / prior_sd**2, 1.0 / se**2
    mean = (tau0 * prior_mean + taud * data) / (tau0 + taud)
    sd = math.sqrt(1.0 / (tau0 + taud))
    return mean, sd


@dataclass
class _BufferGrid:
    """Cache of the unconditional-wealth mean/variance along a buffer grid.

    ``recovery`` selects the liquidation-recovery convention in ``W_T``:
    ``"exact"`` (default) uses the jump-overshoot recovery of
    :func:`optimal_long_short.overshoot_recovery.exact_wealth_moments`;
    ``"cc"`` uses the continuous-crossing upper bound of
    :func:`optimal_long_short.recovery.wealth_moments`.
    """

    params: KouParams
    b: float
    liq_bonus: float
    T: float
    H0_lo: float = 1.0684931506849316
    H0_hi: float = 2.0
    n_grid: int = 40
    recovery: str = "exact"

    def __post_init__(self) -> None:
        if self.recovery not in ("exact", "cc"):
            raise ValueError(f"recovery must be 'exact' or 'cc', got {self.recovery!r}")
        self.h_grid = np.linspace(math.log(self.H0_lo), math.log(self.H0_hi), self.n_grid)
        self._cache: dict[float, list[tuple[float, float]]] = {}

    def curve(self, delta: float) -> list[tuple[float, float]]:
        key = round(float(delta), 4)
        if key not in self._cache:
            base_delta = self.params.mu1 - self.params.mu2
            pv = with_price_drift_view(self.params, delta_mu1=(key - base_delta))
            moments = exact_wealth_moments if self.recovery == "exact" else wealth_moments
            self._cache[key] = [
                (
                    (wm := moments(pv, h, b=self.b, liq_bonus=self.liq_bonus, T=self.T))["mean_wealth"],
                    wm["var_wealth"],
                )
                for h in self.h_grid
            ]
        return self._cache[key]


def optimal_buffer(
    grid: _BufferGrid, delta: float, alpha: float
) -> tuple[float, bool]:
    """Maximiser of unconditional-wealth mean--variance ``E[W] - (alpha/2)Var(W)``.

    Returns ``(H0_star, is_interior)``.
    """
    rows = grid.curve(delta)
    vals = [m - 0.5 * alpha * v for (m, v) in rows]
    i = int(np.argmax(vals))
    return math.exp(grid.h_grid[i]), 0 < i < len(grid.h_grid) - 1


def interior_threshold(
    grid: _BufferGrid, alpha: float, deltas: np.ndarray | None = None
) -> float | None:
    """Smallest spread drift on ``deltas`` giving an interior optimum, else ``None``."""
    if deltas is None:
        deltas = np.round(np.arange(-0.30, 0.605, 0.02), 4)
    for d in deltas:
        if optimal_buffer(grid, float(d), alpha)[1]:
            return float(d)
    return None


def sizing_distribution(
    params: KouParams,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    post_mean: float,
    post_sd: float,
    alpha: float,
    n_draws: int = 20_000,
    seed: int = 0,
    grid: _BufferGrid | None = None,
) -> dict[str, float]:
    """Posterior distribution of ``H0*`` induced by ``delta ~ N(post_mean, post_sd^2)``."""
    grid = grid or _BufferGrid(params=params, b=b, liq_bonus=liq_bonus, T=T)
    deltas = np.round(np.arange(-0.30, 0.605, 0.02), 4)
    for d in deltas:  # warm the cache on the shared delta grid
        grid.curve(float(d))
    rng = np.random.default_rng(seed)
    draws = rng.normal(post_mean, post_sd, size=n_draws)
    H = np.empty(n_draws)
    interior = np.empty(n_draws, dtype=bool)
    lut: dict[float, tuple[float, bool]] = {}
    for j, d in enumerate(draws):
        key = float(deltas[int(np.argmin(np.abs(deltas - d)))])
        if key not in lut:
            lut[key] = optimal_buffer(grid, key, alpha)
        H[j], interior[j] = lut[key]
    out = {
        "p_interior": float(interior.mean()),
        "H0_median": float(np.median(H)),
    }
    if interior.any():
        out["H0_interior_median"] = float(np.median(H[interior]))
        out["H0_interior_q25"] = float(np.quantile(H[interior], 0.25))
        out["H0_interior_q75"] = float(np.quantile(H[interior], 0.75))
    return out
