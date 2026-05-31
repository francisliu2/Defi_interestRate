"""
Empirical ECF calibration on historical crypto/DeFi return data.

Loads bivariate daily log-returns from a CSV file, optionally applies
EWM smoothing, runs calibrate_ecf, prints a parameter summary and
diagnostics, saves the full result to results/ as JSON, and writes an
ECF fit plot to latex/.

Usage:  python jobs/calibrate_empirical.py
Output: results/calibration_<ASSET1>_<ASSET2>_<freq>_<timestamp>.json
        latex/fig_ecf_empirical.pdf

Configuration: edit the constants in the section below.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from optimal_long_short.calibration import (
    ECFCalibrationResult,
    calibrate_ecf,
    ewm_smooth,
)
from optimal_long_short.model_params import KouParams
from optimal_long_short.job_runners.common import (
    NpEncoder,
    REPO_ROOT,
    dt_from_freq,
    jump_cumulants,
    kou_params_to_dict,
    plot_ecf_spread_fit,
)

# ---------------------------------------------------------------------------
# Configuration — edit these to switch datasets or tuning options
# ---------------------------------------------------------------------------
ROOT        = REPO_ROOT
DATA_FILE   = ROOT / "results" / "returns" / "DAI_WETH_daily_log_returns.csv"
DATE_COL    = "snapped_at"
ASSET1_COL  = "DAI"
ASSET2_COL  = "WETH"
TARGET_FREQ = "1d"        # sets dt_years = 1/365

# EWM smoothing: set to a positive integer to smooth returns before fitting.
# None → no smoothing (raw returns passed directly to calibrate_ecf).
# Typical values: 3–10 for daily data.  Larger span → heavier smoothing.
EWM_SPAN: int | None = None

N_STARTS    = 30          # multi-start cloud size (6 ridge + 1 base + rest random)
SEED        = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(
    path: Path,
    date_col: str,
    a1_col: str,
    a2_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load bivariate log-return CSV.

    Expects a header row followed by rows of (date, r1, r2).
    Rows with non-finite values are dropped.

    Returns
    -------
    dates : (N,) array of date strings.
    r1    : (N,) float array of asset-1 log-returns.
    r2    : (N,) float array of asset-2 log-returns.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
        df = df[[date_col, a1_col, a2_col]].dropna()
        df[a1_col] = pd.to_numeric(df[a1_col], errors="coerce")
        df[a2_col] = pd.to_numeric(df[a2_col], errors="coerce")
        df = df.dropna().sort_values(date_col).reset_index(drop=True)
        dates = df[date_col].values.astype(str)
        r1 = df[a1_col].values.astype(float)
        r2 = df[a2_col].values.astype(float)
    except ImportError:
        # Fallback: plain numpy (no date parsing)
        data = np.genfromtxt(
            path, delimiter=",", skip_header=1,
            usecols=(0, 1, 2), dtype=None, encoding="utf-8",
        )
        dates = np.array([row[0] for row in data], dtype=str)
        r1 = np.array([float(row[1]) for row in data])
        r2 = np.array([float(row[2]) for row in data])

    ok = np.isfinite(r1) & np.isfinite(r2)
    return dates[ok], r1[ok], r2[ok]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_summary(
    result: ECFCalibrationResult,
    asset1: str,
    asset2: str,
    ewm_span: int | None,
    r1: np.ndarray,
    r2: np.ndarray,
) -> None:
    est  = result.params
    init = result.theta0

    print(f"\nAssets: {asset1} / {asset2}  |  N={result.n_obs}  |  "
          f"dt={result.dt_years:.6f} yr  |  freq={result.frequency_label}")
    if ewm_span is not None:
        print(f"EWM smoothing: span={ewm_span}  (alpha={2/(ewm_span+1):.3f})")
    else:
        print("EWM smoothing: none (raw returns)")

    # Parameter table
    rows = [
        ("mu1",       init.mu1,           est.mu1),
        ("muX1",   init.muX1, est.muX1),
        ("sigma1",    init.sigma1,        est.sigma1),
        ("lam1",      init.lam1,          est.lam1),
        ("p1",        init.p1,            est.p1),
        ("eta1_pos",  init.eta1_pos,      est.eta1_pos),
        ("eta1_neg",  init.eta1_neg,      est.eta1_neg),
        ("mu2",       init.mu2,           est.mu2),
        ("muX2",   init.muX2, est.muX2),
        ("sigma2",    init.sigma2,        est.sigma2),
        ("lam2",      init.lam2,          est.lam2),
        ("p2",        init.p2,            est.p2),
        ("eta2_pos",  init.eta2_pos,      est.eta2_pos),
        ("eta2_neg",  init.eta2_neg,      est.eta2_neg),
        ("rho",       init.rho,           est.rho),
    ]
    print(f"\n{'Parameter':<12} {'Init':>12} {'Estimated':>12}")
    print("-" * 40)
    for name, iv, ev in rows:
        print(f"{name:<12} {iv:12.4f} {ev:12.4f}")
    print("-" * 40)

    # Objective
    print(f"\nECF objective Q_N (normalized, anchored):")
    print(f"  Q(est)     = {result.objective:.6e}")
    print(f"  Converged  : {result.success}  ({result.message})")
    print(f"  Iterations : {result.n_iter}  |  best start: {result.best_start_index}")

    print(f"\nQ_N by frequency group:")
    for grp, val in sorted(result.objective_by_group.items()):
        print(f"  {grp:<14} {val:.4e}")

    # Jump cumulants
    print(f"\nJump cumulants (annualized):")
    print(f"  {'':12}  {'λE[J]':>10}  {'λE[J²]':>10}  {'λE[J⁴]':>10}")
    print(f"  {'-'*46}")
    for label, p_obj in [("init A1", init), ("est  A1", est)]:
        k1, k2, k4 = jump_cumulants(p_obj.lam1, p_obj.p1, p_obj.eta1_pos, p_obj.eta1_neg)
        print(f"  {label:<12}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")
    print()
    for label, p_obj in [("init A2", init), ("est  A2", est)]:
        k1, k2, k4 = jump_cumulants(p_obj.lam2, p_obj.p2, p_obj.eta2_pos, p_obj.eta2_neg)
        print(f"  {label:<12}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")

    # Spread Z diagnostics
    if result.diagnostics:
        d = result.diagnostics
        z_emp = d.get("moments_z", {})
        sqc   = d.get("spread_quantile_comparison", {})
        if z_emp and sqc:
            print(f"\nSpread Z = {asset1} - {asset2} (per period):")
            print(f"  {'':16}  {'std':>8}  {'skew':>8}  {'kurt':>8}  {'q1%':>8}  {'q5%':>8}")
            print(f"  {'-'*55}")
            print(f"  {'Empirical':<16}  "
                  f"{z_emp['std']:8.5f}  {z_emp['skewness']:8.3f}  "
                  f"{z_emp['kurtosis']:8.3f}  "
                  f"{sqc['empirical']['q01']:8.5f}  {sqc['empirical']['q05']:8.5f}")
            if "model_sim" in sqc:
                print(f"  {'Model (est sim)':<16}  "
                      f"{'N/A':>8}  {'N/A':>8}  {'N/A':>8}  "
                      f"{sqc['model_sim']['q01']:8.5f}  {sqc['model_sim']['q05']:8.5f}")

    # Return statistics
    print(f"\nReturn statistics (per period):")
    print(f"  {'':10}  {'mean':>10}  {'std':>10}  {'skew':>10}  {'kurt':>10}")
    print(f"  {'-'*46}")
    from scipy.stats import skew, kurtosis
    for label, r in [(asset1, r1), (asset2, r2)]:
        print(f"  {label:<10}  {np.mean(r):10.5f}  {np.std(r):10.5f}  "
              f"{skew(r):10.3f}  {kurtosis(r):10.3f}")


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def _save_result(
    result: ECFCalibrationResult,
    asset1: str,
    asset2: str,
    freq: str,
    data_file: Path,
    ewm_span: int | None,
    out_dir: Path,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"calibration_{asset1}_{asset2}_{freq}_{ts}.json"
    out_path = out_dir / fname

    payload = {
        "meta": {
            "run_at":       datetime.now(timezone.utc).isoformat(),
            "data_file":    str(data_file),
            "asset1":       asset1,
            "asset2":       asset2,
            "n_obs":        result.n_obs,
            "dt_years":     result.dt_years,
            "freq_label":   result.frequency_label,
            "ewm_span":     ewm_span,
            "n_starts":     N_STARTS,
            "seed":         SEED,
        },
        "params":           kou_params_to_dict(result.params),
        "init_params":      kou_params_to_dict(result.theta0),
        "objective":        result.objective,
        "objective_by_group": result.objective_by_group,
        "success":          result.success,
        "message":          result.message,
        "n_iter":           result.n_iter,
        "best_start_index": result.best_start_index,
        "starts_objectives": result.starts_objectives,
        "scale_info":       result.scale_info,
    }

    out_path.write_text(json.dumps(payload, indent=2, cls=NpEncoder))
    return out_path


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_ecf_fit(
    r1: np.ndarray,
    r2: np.ndarray,
    est_params: KouParams,
    dt: float,
    asset1: str,
    asset2: str,
    out_path: Path,
) -> None:
    title = (
        f"ECF fit — {asset1}/{asset2} spread direction $(s,-s)$\n"
        f"$N={len(r1)}$ daily obs., $\\Delta t={dt:.6f}$"
    )
    plot_ecf_spread_fit(r1, r2, dt, out_path, title, estimated_params=est_params)
    print(f"\nFigure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dt_years = dt_from_freq(TARGET_FREQ)

    print(f"Loading {DATA_FILE} ...")
    dates, r1, r2 = _load_csv(DATA_FILE, DATE_COL, ASSET1_COL, ASSET2_COL)
    print(f"Loaded {len(r1)} observations  ({dates[0]} → {dates[-1]})")

    if EWM_SPAN is not None:
        r1 = ewm_smooth(r1, EWM_SPAN)
        r2 = ewm_smooth(r2, EWM_SPAN)
        print(f"Applied EWM smoothing (span={EWM_SPAN})")

    result: ECFCalibrationResult = calibrate_ecf(
        r1, r2, dt_years,
        n_starts=N_STARTS,
        seed=SEED,
    )

    _print_summary(result, ASSET1_COL, ASSET2_COL, EWM_SPAN, r1, r2)

    out_dir = ROOT / "results"
    out_json = _save_result(result, ASSET1_COL, ASSET2_COL, TARGET_FREQ, DATA_FILE, EWM_SPAN, out_dir)
    print(f"\nResult saved to {out_json}")

    plot_path = ROOT / "latex" / "fig_ecf_empirical.pdf"
    _plot_ecf_fit(r1, r2, result.params, dt_years, ASSET1_COL, ASSET2_COL, plot_path)


if __name__ == "__main__":
    main()
