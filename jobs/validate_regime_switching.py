"""Validation suite for the 6x6 regime-switching killed resolvent.

Three independent checks, in increasing order of severity:

A. **Constant-carry limit.**  With ``c1 == c2`` the regime is economically
   irrelevant even though the chain is still switching, so both states must
   reproduce the single-regime 3x3 answer.  Run with ``q1 != q2`` so the
   coupling is genuinely active rather than trivially zero.
B. **Decoupled limit.**  With ``q1 == q2 == 0`` and ``c1 != c2`` the two states
   never communicate, so each must reproduce the single-regime answer at its own
   carry.
C. **Monte Carlo.**  With switching active, compare against a simulated
   regime-modulated Kou process.  The residual here is dominated by discrete
   barrier monitoring, so the check is a convergence check: the gap must shrink
   as the time grid is refined.

A and B are exact identities and should hold to machine precision; C is
statistical.

    python jobs/validate_regime_switching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from optimal_long_short.drift import with_price_drift_view  # noqa: E402
from optimal_long_short.inversion import TalbotInverter  # noqa: E402
from optimal_long_short.kou_model import KouZTiltedDynamics  # noqa: E402
from optimal_long_short.laplace_resolvent import (  # noqa: E402
    GeneralSolution,
    HomogeneousSolution,
    ParticularSolution,
)
from optimal_long_short.market_params import MarketParams  # noqa: E402
from optimal_long_short.model_params import KouParams  # noqa: E402
from optimal_long_short.regime_switching import (  # noqa: E402
    RegimeResolvent,
    RegimeSpec,
    RegimeSwitchingDynamics,
)
from optimal_long_short.strategy import UnitExposureLongShortStrategy  # noqa: E402

BENCH = KouParams(
    mu1=0.05, sigma1=0.20, lam1=1.0, p1=0.5, eta1_pos=0.10, eta1_neg=0.125,
    mu2=0.03, sigma2=0.15, lam2=0.8, p2=0.5, eta2_pos=1 / 12, eta2_neg=1 / 9,
    rho=0.3,
)
B, H0, Q = 0.78, 0.30, 0.9


def _single_regime(params: KouParams, k: int, q: complex, h0: float, T: float) -> complex:
    strategy = UnitExposureLongShortStrategy(
        h0=h0, market=MarketParams(b=B, S10=1.0, S20=1.0), T=T
    )
    dynamics = KouZTiltedDynamics(params=params, k=k)
    return GeneralSolution(
        HomogeneousSolution(ParticularSolution(dynamics, strategy))
    ).evaluate_at_origin(q)


def check_constant_carry() -> list[dict]:
    """A. c1 == c2, coupling active -> single-regime answer in both states."""
    rows = []
    for k in (0, 1, 2):
        for c in (0.0, 0.05):
            shifted = with_price_drift_view(BENCH, delta_mu1=c, delta_mu2=0.0)
            reference = _single_regime(shifted, k, Q, H0, 1 / 12)
            dynamics = RegimeSwitchingDynamics(
                base=KouZTiltedDynamics(params=BENCH, k=k),
                spec=RegimeSpec(q1=3.0, q2=7.0, c1=c, c2=c),
            )
            got = RegimeResolvent(dynamics, h0=H0, b=B).transform(Q)
            rows.append(
                {
                    "k": k,
                    "c": c,
                    "reference": reference.real,
                    "rel_err_state1": abs(got[0] - reference) / abs(reference),
                    "rel_err_state2": abs(got[1] - reference) / abs(reference),
                }
            )
    return rows


def check_decoupled(c1: float = 0.02, c2: float = -0.40) -> list[dict]:
    """B. q1 == q2 == 0 -> each state matches single-regime at its own carry."""
    rows = []
    for k in (0, 1, 2):
        ref1 = _single_regime(
            with_price_drift_view(BENCH, delta_mu1=c1, delta_mu2=0.0), k, Q, H0, 1 / 12
        )
        ref2 = _single_regime(
            with_price_drift_view(BENCH, delta_mu1=c2, delta_mu2=0.0), k, Q, H0, 1 / 12
        )
        dynamics = RegimeSwitchingDynamics(
            base=KouZTiltedDynamics(params=BENCH, k=k),
            spec=RegimeSpec(q1=0.0, q2=0.0, c1=c1, c2=c2),
        )
        got = RegimeResolvent(dynamics, h0=H0, b=B).transform(Q)
        rows.append(
            {
                "k": k,
                "rel_err_state1": abs(got[0] - ref1) / abs(ref1),
                "rel_err_state2": abs(got[1] - ref2) / abs(ref2),
            }
        )
    return rows


def check_conditioning(spec: RegimeSpec | None = None) -> list[dict]:
    """Condition number of the 6x6 across the Laplace argument."""
    spec = spec or RegimeSpec(q1=2.0, q2=25.0, c1=0.02, c2=-0.40)
    dynamics = RegimeSwitchingDynamics(
        base=KouZTiltedDynamics(params=BENCH, k=0), spec=spec
    )
    resolvent = RegimeResolvent(dynamics, h0=H0, b=B)
    return [{"q": q, "cond": resolvent.condition_number(q)} for q in (0.1, 1.0, 10.0, 100.0)]


def simulate_survival(
    params: KouParams,
    spec: RegimeSpec,
    h0: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int,
    start_regime: int,
) -> float:
    """Euler-Poisson simulation of the regime-modulated log-ratio process."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sigma_z = np.sqrt(
        params.sigma1**2 + params.sigma2**2
        - 2 * params.rho * params.sigma1 * params.sigma2
    )
    mu_z = params.muX1 - params.muX2

    z = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    regime = np.full(n_paths, start_regime, dtype=np.int8)
    carry = np.array([spec.c1, spec.c2])
    exit_rate = np.array([spec.q1, spec.q2])

    for _ in range(n_steps):
        flip = rng.random(n_paths) < 1.0 - np.exp(-exit_rate[regime - 1] * dt)
        regime[flip] = 3 - regime[flip]

        dz = (mu_z + carry[regime - 1]) * dt + sigma_z * np.sqrt(dt) * rng.standard_normal(
            n_paths
        )
        for lam, p_up, eta_p, eta_n, sign in (
            (params.lam1, params.p1, params.eta1_pos, params.eta1_neg, +1.0),
            (params.lam2, params.p2, params.eta2_pos, params.eta2_neg, -1.0),
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
            dz += sign * total

        z = z + dz
        alive &= z > -h0
    return float(alive.mean())


def check_monte_carlo(n_paths: int = 100_000) -> list[dict]:
    """C. Switching active; the gap must shrink as the grid is refined."""
    spec = RegimeSpec(q1=2.0, q2=25.0, c1=0.02, c2=-0.60)
    T, h0 = 0.25, 0.10
    dynamics = RegimeSwitchingDynamics(
        base=KouZTiltedDynamics(params=BENCH, k=0), spec=spec
    )
    resolvent = RegimeResolvent(dynamics, h0=h0, b=B)
    semi_analytic = TalbotInverter(M=32).invert(
        lambda q: resolvent.transform(q)[0], T
    )
    rows = []
    for n_steps in (1_000, 4_000, 16_000):
        mc = simulate_survival(BENCH, spec, h0, T, n_paths, n_steps, 7, 1)
        rows.append(
            {
                "n_steps": n_steps,
                "semi_analytic": semi_analytic,
                "monte_carlo": mc,
                "bias": mc - semi_analytic,
                "mc_stderr": float(np.sqrt(mc * (1 - mc) / n_paths)),
            }
        )
    return rows


def main() -> None:
    print("A. Constant-carry limit (c1 == c2, q1 != q2): expect ~1e-14")
    for row in check_constant_carry():
        print(
            f"   k={row['k']} c={row['c']:.2f}  ref={row['reference']:.10f}  "
            f"err1={row['rel_err_state1']:.2e} err2={row['rel_err_state2']:.2e}"
        )

    print("\nB. Decoupled limit (q1 = q2 = 0, c1 != c2): expect ~1e-15")
    for row in check_decoupled():
        print(
            f"   k={row['k']}  err1={row['rel_err_state1']:.2e} "
            f"err2={row['rel_err_state2']:.2e}"
        )

    print("\n   Conditioning of the 6x6 barrier matrix")
    for row in check_conditioning():
        print(f"   q={row['q']:7.1f}  cond={row['cond']:.3e}")

    print("\nC. Monte Carlo with switching active (bias must shrink with n_steps)")
    for row in check_monte_carlo():
        print(
            f"   steps={row['n_steps']:6d}  SA={row['semi_analytic']:.6f} "
            f"MC={row['monte_carlo']:.6f}  bias={row['bias']:+.6f} "
            f"se={row['mc_stderr']:.6f}"
        )


if __name__ == "__main__":
    main()
