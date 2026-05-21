"""
Bivariate Kou ECF calibration demo.

Generates N synthetic daily log-return observations from the benchmark
parameter set (Table 1 of the paper) and recovers the parameters via the
Singleton-style empirical characteristic function estimator implemented in
optimal_long_short.calibration.

Usage:  python jobs/calibrate_kou.py
Output: latex/fig_ecf_fit.pdf   (ECF fit along the spread direction)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_long_short.model_params import KouParams
from optimal_long_short.kou_model import BivariateKouModel
from optimal_long_short.calibration import (
    CalibrationGrid,
    ECFCalibrationResult,
    calibrate_ecf,
    empirical_cf,
    params_to_theta,
)

# ---------------------------------------------------------------------------
# True parameter set (crypto-realistic: higher jump intensity than Table 1
# to give sufficient jump observations for identification with daily data)
# ---------------------------------------------------------------------------
TRUE_PARAMS = KouParams(
    mu1=0.10, sigma1=0.30, lam1=10.0, p1=0.45, eta1_pos=0.06, eta1_neg=0.05,
    mu2=0.08, sigma2=0.25, lam2=8.0,  p2=0.45, eta2_pos=0.05, eta2_neg=0.06,
    rho=0.60,
)

N_OBS = 4000   # number of daily observations (~16 years of daily crypto data)
DT = 1.0 / 365
SEED = 42


# ---------------------------------------------------------------------------
# Simulate bivariate Kou log-returns
# ---------------------------------------------------------------------------

def simulate_returns(params: KouParams, n: int, dt: float,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n i.i.d. bivariate Kou log-return observations.

    Each observation is one time step of (X1, X2) with correlated Brownian
    part and independent Kou jump components.
    """
    p = params
    rho = p.rho
    sqrt_dt = np.sqrt(dt)
    sqrt_1mr2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    W1 = rng.standard_normal(n)
    W2 = rng.standard_normal(n)
    dB1 = sqrt_dt * W1
    dB2 = sqrt_dt * (rho * W1 + sqrt_1mr2 * W2)

    def _kou_jumps(lam, pi, ep, en):
        n_jumps = rng.poisson(lam * dt, size=n)
        total = np.zeros(n)
        idx = np.where(n_jumps > 0)[0]
        for i in idx:
            nj = n_jumps[i]
            signs = rng.choice([1, -1], size=nj, p=[pi, 1 - pi])
            sizes = np.where(
                signs == 1,
                rng.exponential(ep, size=nj),
                -rng.exponential(en, size=nj),
            )
            total[i] = sizes.sum()
        return total

    J1 = _kou_jumps(p.lam1, p.p1, p.eta1_pos, p.eta1_neg)
    J2 = _kou_jumps(p.lam2, p.p2, p.eta2_pos, p.eta2_neg)

    r1 = p.effective_mu1 * dt + p.sigma1 * dB1 + J1
    r2 = p.effective_mu2 * dt + p.sigma2 * dB2 + J2
    return r1, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    r1, r2 = simulate_returns(TRUE_PARAMS, N_OBS, DT, rng)

    print(f"Simulated {N_OBS} daily observations (dt={DT:.6f})\n")

    result: ECFCalibrationResult = calibrate_ecf(r1, r2, DT)

    _print_comparison(TRUE_PARAMS, result)
    _plot_ecf_fit(r1, r2, TRUE_PARAMS, result.params, DT)


def _print_comparison(true: KouParams, result: ECFCalibrationResult) -> None:
    est  = result.params
    init = result.theta0
    rows = [
        ("mu1",      true.mu1,      init.mu1,      est.mu1),
        ("sigma1",   true.sigma1,   init.sigma1,   est.sigma1),
        ("lam1",     true.lam1,     init.lam1,     est.lam1),
        ("p1",       true.p1,       init.p1,       est.p1),
        ("eta1_pos", true.eta1_pos, init.eta1_pos, est.eta1_pos),
        ("eta1_neg", true.eta1_neg, init.eta1_neg, est.eta1_neg),
        ("mu2",      true.mu2,      init.mu2,      est.mu2),
        ("sigma2",   true.sigma2,   init.sigma2,   est.sigma2),
        ("lam2",     true.lam2,     init.lam2,     est.lam2),
        ("p2",       true.p2,       init.p2,       est.p2),
        ("eta2_pos", true.eta2_pos, init.eta2_pos, est.eta2_pos),
        ("eta2_neg", true.eta2_neg, init.eta2_neg, est.eta2_neg),
        ("rho",      true.rho,      init.rho,      est.rho),
    ]
    print(f"{'Parameter':<12} {'True':>10} {'Init':>10} {'Estimated':>12} {'Rel. error':>12}")
    print("-" * 60)
    for name, tv, iv, ev in rows:
        rel = abs(ev - tv) / (abs(tv) + 1e-12)
        print(f"{name:<12} {tv:10.4f} {iv:10.4f} {ev:12.4f} {rel:11.2%}")
    print("-" * 60)
    print(f"Objective Q_N = {result.objective:.6e}")
    print(f"Converged: {result.success}  ({result.message})")
    print(f"Iterations: {result.n_iter}  |  best start: {result.best_start_index}")


def _plot_ecf_fit(r1: np.ndarray, r2: np.ndarray,
                  true_params: KouParams, est_params: KouParams,
                  dt: float) -> None:
    """
    Plot |phi_hat(s,-s)| and phase along the spread direction s in [0, 5],
    comparing the empirical CF to the true and estimated model CFs.
    """
    s_grid = np.linspace(0.01, 5.0, 200)
    freqs_spread = np.column_stack([s_grid, -s_grid])   # (200, 2) spread direction

    phi_hat = empirical_cf(r1, r2, freqs_spread)

    true_model = BivariateKouModel(true_params)
    est_model  = BivariateKouModel(est_params)
    phi_true = np.array([np.exp(dt * true_model.levy_khintchine(s, -s)) for s in s_grid])
    phi_est  = np.array([np.exp(dt * est_model.levy_khintchine(s, -s))  for s in s_grid])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, qty, label in [
        (axes[0], np.abs,   "Modulus $|\\phi(s,-s)|$"),
        (axes[1], np.angle, "Phase $\\arg\\phi(s,-s)$"),
    ]:
        ax.plot(s_grid, qty(phi_hat),  "k.",  ms=2,  label="Empirical", alpha=0.6)
        ax.plot(s_grid, qty(phi_true), "b--", lw=1.5, label="True model")
        ax.plot(s_grid, qty(phi_est),  "r-",  lw=1.5, label="ECF estimate")
        ax.set_xlabel("$s$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"ECF fit along the spread direction $(s,-s)$\n"
        f"$N={len(r1)}$ daily obs., $\\Delta t=1/252$",
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(__file__).parent.parent / "latex" / "fig_ecf_fit.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
