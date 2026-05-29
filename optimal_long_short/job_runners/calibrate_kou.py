"""
Bivariate Kou ECF calibration demo.

Generates N synthetic daily log-return observations from the benchmark
parameter set (Table 1 of the paper) and recovers the parameters via the
Singleton-style empirical characteristic function estimator implemented in
optimal_long_short.calibration.

Usage:  python jobs/calibrate_kou.py
Output: latex/fig_ecf_fit.pdf   (ECF fit along the spread direction)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

from optimal_long_short.model_params import KouParams
from optimal_long_short.calibration import (
    ECFCalibrationResult,
    calibrate_ecf,
    empirical_cf,
    params_to_theta,
)
from optimal_long_short.job_runners.common import LATEX_DIR, jump_cumulants, plot_ecf_spread_fit
from optimal_long_short.simulation import simulate_kou_returns

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
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_q(params: KouParams, phi_hat: np.ndarray, dt: float,
                freqs: np.ndarray, weights: np.ndarray) -> float:
    """Plain (un-anchored) normalized ECF objective Q_N at given KouParams."""
    from optimal_long_short.calibration.ecf_objective import objective_unc
    from optimal_long_short.calibration.transforms import nat_to_unc, _DEFAULT_BOUNDS
    tau = nat_to_unc(params_to_theta(params), _DEFAULT_BOUNDS)
    return objective_unc(tau, phi_hat, dt, freqs, weights, _DEFAULT_BOUNDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    r1, r2 = simulate_kou_returns(TRUE_PARAMS, N_OBS, DT, rng)

    print(f"Simulated {N_OBS} daily observations (dt={DT:.6f})\n")

    result: ECFCalibrationResult = calibrate_ecf(
        r1, r2, DT,
        n_starts=30,  # 6 ridge + 1 base + 23 random scenario starts
    )

    _print_comparison(TRUE_PARAMS, result, r1, r2)
    _plot_ecf_fit(r1, r2, TRUE_PARAMS, result.params, DT)


def _print_comparison(true: KouParams, result: ECFCalibrationResult,
                      r1: np.ndarray, r2: np.ndarray) -> None:
    est  = result.params
    init = result.theta0

    # ----- Parameter table -----
    rows = [
        ("mu1",        true.mu1,           init.mu1,           est.mu1),
        ("eff_mu1",    true.effective_mu1,  init.effective_mu1, est.effective_mu1),
        ("sigma1",     true.sigma1,         init.sigma1,        est.sigma1),
        ("lam1",       true.lam1,           init.lam1,          est.lam1),
        ("p1",         true.p1,             init.p1,            est.p1),
        ("eta1_pos",   true.eta1_pos,       init.eta1_pos,      est.eta1_pos),
        ("eta1_neg",   true.eta1_neg,       init.eta1_neg,      est.eta1_neg),
        ("mu2",        true.mu2,            init.mu2,           est.mu2),
        ("eff_mu2",    true.effective_mu2,  init.effective_mu2, est.effective_mu2),
        ("sigma2",     true.sigma2,         init.sigma2,        est.sigma2),
        ("lam2",       true.lam2,           init.lam2,          est.lam2),
        ("p2",         true.p2,             init.p2,            est.p2),
        ("eta2_pos",   true.eta2_pos,       init.eta2_pos,      est.eta2_pos),
        ("eta2_neg",   true.eta2_neg,       init.eta2_neg,      est.eta2_neg),
        ("rho",        true.rho,            init.rho,           est.rho),
    ]

    print(f"{'Parameter':<12} {'True':>10} {'Init':>10} {'Estimated':>12} {'Rel. error':>12}")
    print("-" * 60)
    for name, tv, iv, ev in rows:
        rel = abs(ev - tv) / (abs(tv) + 1e-12)
        print(f"{name:<12} {tv:10.4f} {iv:10.4f} {ev:12.4f} {rel:11.2%}")
    print("-" * 60)

    # ----- Objective values -----
    phi_hat = empirical_cf(r1, r2, result.freqs)
    q_true  = _evaluate_q(true, phi_hat, result.dt_years, result.freqs, result.weights)
    q_init  = _evaluate_q(init, phi_hat, result.dt_years, result.freqs, result.weights)
    q_est   = result.objective

    print(f"\nECF objective Q_N (plain, normalized):")
    print(f"  Q(true params) = {q_true:.6e}")
    print(f"  Q(init params) = {q_init:.6e}")
    print(f"  Q(est  params) = {q_est:.6e}  (optimizer target incl. anchor penalty)")
    if q_est < q_true:
        print(f"  Note: Q(est) < Q(true) — ECF exploits finite-sample CF noise (expected for N={len(r1)}).")
    print(f"  Converged: {result.success}  ({result.message})")
    print(f"  Iterations: {result.n_iter}  |  best start: {result.best_start_index}")

    # ----- ECF objective by frequency group -----
    print(f"\nQ_N by frequency group (estimated params):")
    for grp, val in sorted(result.objective_by_group.items()):
        print(f"  {grp:<14} {val:.4e}")

    # ----- Jump cumulant comparison -----
    print(f"\nJump cumulants (annualized):")
    print(f"  {'':12}  {'λE[J]':>10}  {'λE[J²]':>10}  {'λE[J⁴]':>10}")
    print(f"  {'-'*46}")
    for label, p_obj in [("true A1", true), ("init A1", init), ("est  A1", est)]:
        k1, k2, k4 = jump_cumulants(p_obj.lam1, p_obj.p1, p_obj.eta1_pos, p_obj.eta1_neg)
        print(f"  {label:<12}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")
    print()
    for label, p_obj in [("true A2", true), ("init A2", init), ("est  A2", est)]:
        k1, k2, k4 = jump_cumulants(p_obj.lam2, p_obj.p2, p_obj.eta2_pos, p_obj.eta2_neg)
        print(f"  {label:<12}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")

    # ----- Spread Z diagnostics -----
    if result.diagnostics:
        d = result.diagnostics
        z_emp = d["moments_z"]
        sqc   = d["spread_quantile_comparison"]
        print(f"\nSpread Z = r1 - r2 (per period):")
        print(f"  {'':16}  {'std':>8}  {'skew':>8}  {'kurt':>8}  {'q1%':>8}  {'q5%':>8}")
        print(f"  {'-'*55}")
        print(f"  {'Empirical':<16}  "
              f"{z_emp['std']:8.5f}  {z_emp['skewness']:8.3f}  "
              f"{z_emp['kurtosis']:8.3f}  "
              f"{sqc['empirical']['q01']:8.5f}  {sqc['empirical']['q05']:8.5f}")
        print(f"  {'Model (est sim)':<16}  "
              f"{'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
              f"{sqc['model_sim']['q01']:8.5f}  {sqc['model_sim']['q05']:8.5f}")


def _plot_ecf_fit(r1: np.ndarray, r2: np.ndarray,
                  true_params: KouParams, est_params: KouParams,
                  dt: float) -> None:
    """
    Plot |phi_hat(s,-s)| and phase along the spread direction s in [0, 5/sz],
    comparing the empirical CF to the true and estimated model CFs.
    """
    title = (
        f"ECF fit along the spread direction $(s,-s)$\n"
        f"$N={len(r1)}$ daily obs., $\\Delta t={dt:.6f}$"
    )
    out = LATEX_DIR / "fig_ecf_fit.pdf"
    plot_ecf_spread_fit(
        r1,
        r2,
        dt,
        out,
        title,
        estimated_params=est_params,
        true_params=true_params,
    )
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
