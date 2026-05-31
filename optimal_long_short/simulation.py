"""Simulation helpers for Kou jump-diffusion models."""
from __future__ import annotations

import numpy as np

from optimal_long_short.model_params import KouParams


def simulate_kou_returns(
    params: KouParams,
    n: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n i.i.d. bivariate Kou log-return observations."""
    p = params
    sqrt_dt = np.sqrt(dt)
    sqrt_1mr2 = np.sqrt(max(1.0 - p.rho ** 2, 0.0))

    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    db1 = sqrt_dt * w1
    db2 = sqrt_dt * (p.rho * w1 + sqrt_1mr2 * w2)

    def _kou_jumps(lam: float, prob_pos: float, eta_pos: float, eta_neg: float) -> np.ndarray:
        n_jumps = rng.poisson(lam * dt, size=n)
        total = np.zeros(n)
        for i in np.where(n_jumps > 0)[0]:
            nj = n_jumps[i]
            signs = rng.choice([1, -1], size=nj, p=[prob_pos, 1 - prob_pos])
            sizes = np.where(
                signs == 1,
                rng.exponential(eta_pos, size=nj),
                -rng.exponential(eta_neg, size=nj),
            )
            total[i] = sizes.sum()
        return total

    j1 = _kou_jumps(p.lam1, p.p1, p.eta1_pos, p.eta1_neg)
    j2 = _kou_jumps(p.lam2, p.p2, p.eta2_pos, p.eta2_neg)

    r1 = p.muX1 * dt + p.sigma1 * db1 + j1
    r2 = p.muX2 * dt + p.sigma2 * db2 + j2
    return r1, r2

