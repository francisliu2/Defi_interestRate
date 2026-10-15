"""Validation suite for the liquidation-recovery first-passage transform.

Two checks:

A. **Internal consistency at k = 0.**  The transform (FP) of
   ``optimal_long_short.recovery`` returns ``E[exp(k X_2,tau) 1{tau<=T}]``.  At
   ``k = 0`` the integrand is one, so it must reproduce the liquidation
   probability ``1 - p_surv`` computed independently from the survival
   resolvent.  This is exact and costs nothing.  A second free check is that
   ``Psi(0,-i)`` reproduces ``mu_2``, confirming the drift normalization.
B. **Monte Carlo at k = 1.**  Compare against simulated
   ``E[exp(X_2,tau) 1{tau<=T}]``.  As in the regime validation, the residual is
   dominated by discrete barrier monitoring, which biases the simulated
   liquidation count downward.

    python jobs/validate_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from optimal_long_short.inversion import TalbotInverter  # noqa: E402
from optimal_long_short.kou_model import KouZTiltedDynamics  # noqa: E402
from optimal_long_short.laplace_resolvent import (  # noqa: E402
    GeneralSolution,
    HomogeneousSolution,
    ParticularSolution,
)
from optimal_long_short.market_params import MarketParams  # noqa: E402
from optimal_long_short.model_params import KouParams  # noqa: E402
from optimal_long_short.recovery import (  # noqa: E402
    hitting_moment_transform,
    tilt_exponent,
)
from optimal_long_short.strategy import UnitExposureLongShortStrategy  # noqa: E402

BENCH = KouParams(
    mu1=0.05, sigma1=0.20, lam1=1.0, p1=0.5, eta1_pos=0.10, eta1_neg=0.125,
    mu2=0.03, sigma2=0.15, lam2=0.8, p2=0.5, eta2_pos=1 / 12, eta2_neg=1 / 9,
    rho=0.3,
)
B, T = 0.78, 0.25


def check_k0() -> list[dict]:
    """A. k=0 transform must equal 1 - p_surv from the survival resolvent."""
    inverter = TalbotInverter(M=32)
    rows = []
    for h0 in (0.10, 0.25, 0.50):
        strategy = UnitExposureLongShortStrategy(
            h0=h0, market=MarketParams(b=B, S10=1.0, S20=1.0), T=T
        )
        general = GeneralSolution(
            HomogeneousSolution(
                ParticularSolution(KouZTiltedDynamics(params=BENCH, k=0), strategy)
            )
        )
        reference = 1.0 - inverter.invert(general.evaluate_at_origin, T)
        from_fp = inverter.invert(
            lambda q: hitting_moment_transform(BENCH, 0, h0, q), T
        )
        rows.append(
            {
                "h0": h0,
                "p_liq_survival_resolvent": reference,
                "p_liq_first_passage": from_fp,
                "abs_err": abs(from_fp - reference),
            }
        )
    return rows


def simulate_hitting(
    params: KouParams, h0: float, k: int, n_paths: int, n_steps: int, seed: int
) -> float:
    """Simulate ``E[exp(k X_2,tau) 1{tau <= T}]``."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    chol = np.linalg.cholesky(np.array([[1.0, params.rho], [params.rho, 1.0]]))

    x1 = np.zeros(n_paths)
    x2 = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    payoff = np.zeros(n_paths)

    for _ in range(n_steps):
        noise = (chol @ rng.standard_normal((2, n_paths))) * np.sqrt(dt)
        d1 = params.muX1 * dt + params.sigma1 * noise[0]
        d2 = params.muX2 * dt + params.sigma2 * noise[1]
        for lam, p_up, eta_p, eta_n, leg in (
            (params.lam1, params.p1, params.eta1_pos, params.eta1_neg, 1),
            (params.lam2, params.p2, params.eta2_pos, params.eta2_neg, 2),
        ):
            counts = rng.poisson(lam * dt, n_paths)
            if counts.max() == 0:
                continue
            total = np.zeros(n_paths)
            for _ in range(int(counts.max())):
                active = counts > 0
                draw = np.where(
                    rng.random(n_paths) < p_up,
                    rng.exponential(eta_p, n_paths),
                    -rng.exponential(eta_n, n_paths),
                )
                total += np.where(active, draw, 0.0)
                counts = counts - 1
            if leg == 1:
                d1 = d1 + total
            else:
                d2 = d2 + total

        x1, x2 = x1 + d1, x2 + d2
        hit = alive & ((h0 + x1 - x2) < 0.0)
        payoff[hit] = np.exp(k * x2[hit])
        alive &= ~hit

    return float(payoff.mean())


def check_k1(n_paths: int = 200_000, n_steps: int = 2_000) -> list[dict]:
    """B. k=1 transform against Monte Carlo."""
    inverter = TalbotInverter(M=32)
    rows = []
    for h0 in (0.10, 0.25):
        semi_analytic = inverter.invert(
            lambda q: hitting_moment_transform(BENCH, 1, h0, q), T
        )
        mc = simulate_hitting(BENCH, h0, 1, n_paths, n_steps, 11)
        rows.append(
            {
                "h0": h0,
                "semi_analytic": semi_analytic,
                "monte_carlo": mc,
                "diff": semi_analytic - mc,
            }
        )
    return rows


def main() -> None:
    print("A. k=0 consistency: first-passage transform vs survival resolvent")
    for row in check_k0():
        print(
            f"   h0={row['h0']:.2f}  resolvent={row['p_liq_survival_resolvent']:.8f}  "
            f"first-passage={row['p_liq_first_passage']:.8f}  "
            f"err={row['abs_err']:.2e}"
        )
    print(
        f"\n   Psi(0,-i) = {tilt_exponent(BENCH, 1):.6f}  "
        f"(must equal mu2 = {BENCH.mu2})"
    )

    print("\nB. k=1 transform vs Monte Carlo, E[exp(X_2,tau) 1{tau<=T}]")
    for row in check_k1():
        print(
            f"   h0={row['h0']:.2f}  SA={row['semi_analytic']:.6f}  "
            f"MC={row['monte_carlo']:.6f}  diff={row['diff']:+.6f}"
        )


if __name__ == "__main__":
    main()
