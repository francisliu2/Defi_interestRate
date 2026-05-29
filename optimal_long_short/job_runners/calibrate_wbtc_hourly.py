"""
Calibrate bivariate Kou parameters for WBTC (real) vs WBTC-synth (noise-augmented)
using AAVE v3 Ethereum hourly price data.

Asset convention
----------------
  Asset 1 = WBTC real   : aave-ts/data/AAVE/WBTC_AAVEv3_ethereum_hourly_rates_v4.parquet
  Asset 2 = WBTC synth  : aave-ts/data/AAVE/WBTC_synth_AAVEv3_ethereum_hourly_rates_v4.parquet
            (constructed by adding rho=0.60 correlated Gaussian noise to WBTC hourly returns)

Rate adjustment
---------------
  mu1 += supply_apr (hourly-mean, decimal)   — earn deposit yield on collateral
  mu2 += variable_borrow_apr (hourly-mean)   — borrow cost on synthetic leg

Usage : python jobs/calibrate_wbtc_hourly.py
Output: results/params_WBTC_hourly.json
        latex/fig_ecf_wbtc_hourly.pdf
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from optimal_long_short.calibration import (
    ECFCalibrationResult,
    ParameterBounds,
    calibrate_ecf,
)
from optimal_long_short.model_params import KouParams
from optimal_long_short.job_runners.common import (
    NpEncoder,
    REPO_ROOT,
    aave_constraint as make_aave_constraint,
    jump_cumulants,
    kou_params_to_dict,
    plot_ecf_spread_fit,
    rate_adjusted_params as adjust_kou_drifts,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT     = REPO_ROOT
AAVE_DIR = ROOT / "aave-ts" / "data" / "AAVE"

SRC_REAL  = AAVE_DIR / "WBTC_AAVEv3_ethereum_hourly_rates_v4.parquet"
SRC_SYNTH = AAVE_DIR / "WBTC_synth_AAVEv3_ethereum_hourly_rates_v4.parquet"

OUT_JSON = ROOT / "results" / "params_WBTC_hourly.json"
OUT_FIG  = ROOT / "latex" / "fig_ecf_wbtc_hourly.pdf"

DT_YEARS = 1.0 / (365.0 * 24.0)   # one hour in years

# AAVE v3 Ethereum WBTC risk parameters (governance-set)
AAVE_B         = 0.78
AAVE_LTV_MAX   = 0.73
AAVE_LIQ_BONUS = 0.05

N_STARTS = 30
SEED     = 42

CALIB_BOUNDS = ParameterBounds(
    max_moment_order=4,
    # lambda_max=5000.0,
    # eta_pos1_min=0.02,
    # eta_pos2_min=0.02,
    # eta_neg_min=0.02,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    real  = pd.read_parquet(SRC_REAL)
    synth = pd.read_parquet(SRC_SYNTH)
    real["datetime"]  = pd.to_datetime(real["datetime"])
    synth["datetime"] = pd.to_datetime(synth["datetime"])
    real  = real.sort_values("datetime").reset_index(drop=True)
    synth = synth.sort_values("datetime").reset_index(drop=True)
    return real, synth


def compute_returns(
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    r1 = np.diff(np.log(real["close"].values.astype(float)))
    r2 = np.diff(np.log(synth["close"].values.astype(float)))
    ok = np.isfinite(r1) & np.isfinite(r2)
    date0 = str(real["datetime"].iloc[1].date())
    date1 = str(real["datetime"].iloc[-1].date())
    return r1[ok], r2[ok], date0, date1


def compute_avg_rates(real: pd.DataFrame) -> dict[str, float]:
    """Mean annualised AAVE APRs from real WBTC data (already decimal fractions)."""
    return {
        "supply_wbtc":        float(real["supply_apr"].mean()),
        "borrow_wbtc":        float(real["variable_borrow_apr"].mean()),
    }


# ---------------------------------------------------------------------------
# Rate-adjusted KouParams
# ---------------------------------------------------------------------------

def aave_constraint() -> dict:
    return make_aave_constraint(AAVE_B, AAVE_LTV_MAX, AAVE_LIQ_BONUS)


def print_summary(
    result: ECFCalibrationResult,
    adj: KouParams,
    rates: dict,
    date0: str,
    date1: str,
    constraint: dict,
) -> None:
    raw  = result.params
    init = result.theta0

    print(f"\n{'='*70}")
    print(f"WBTC (real) / WBTC (synth)  |  AAVE v3 Ethereum hourly data")
    print(f"N={result.n_obs} obs  ({date0} → {date1})  dt={DT_YEARS:.8f} yr")
    print(f"{'='*70}")

    print(f"\nAAVE rate adjustment (annualised decimal):")
    print(f"  WBTC supply  r_s1 = {rates['supply_wbtc']:.6f}  "
          f"({rates['supply_wbtc']*100:.4f}%/yr)")
    print(f"  WBTC borrow  r_b2 = {rates['borrow_wbtc']:.6f}  "
          f"({rates['borrow_wbtc']*100:.4f}%/yr)")

    rows = [
        ("mu1",      init.mu1,           raw.mu1,           adj.mu1),
        ("eff_mu1",  init.effective_mu1, raw.effective_mu1, adj.effective_mu1),
        ("sigma1",   init.sigma1,        raw.sigma1,        adj.sigma1),
        ("lam1",     init.lam1,          raw.lam1,          adj.lam1),
        ("p1",       init.p1,            raw.p1,            adj.p1),
        ("eta1_pos", init.eta1_pos,      raw.eta1_pos,      adj.eta1_pos),
        ("eta1_neg", init.eta1_neg,      raw.eta1_neg,      adj.eta1_neg),
        ("mu2",      init.mu2,           raw.mu2,           adj.mu2),
        ("eff_mu2",  init.effective_mu2, raw.effective_mu2, adj.effective_mu2),
        ("sigma2",   init.sigma2,        raw.sigma2,        adj.sigma2),
        ("lam2",     init.lam2,          raw.lam2,          adj.lam2),
        ("p2",       init.p2,            raw.p2,            adj.p2),
        ("eta2_pos", init.eta2_pos,      raw.eta2_pos,      adj.eta2_pos),
        ("eta2_neg", init.eta2_neg,      raw.eta2_neg,      adj.eta2_neg),
        ("rho",      init.rho,           raw.rho,           adj.rho),
    ]
    print(f"\n{'Parameter':<12} {'Init':>10} {'ECF raw':>10} {'Rate-adj':>12}")
    print("-" * 50)
    for name, iv, rv, av in rows:
        print(f"{name:<12} {iv:10.4f} {rv:10.4f} {av:12.4f}")
    print("-" * 50)

    print(f"\nECF objective Q_N = {result.objective:.6e}  "
          f"(converged={result.success},  start={result.best_start_index})")
    print(f"\nQ_N by group:")
    for g, v in sorted(result.objective_by_group.items()):
        print(f"  {g:<14} {v:.4e}")

    print(f"\nJump cumulants (annualised, rate-adjusted params):")
    print(f"  {'':10}  {'λE[J]':>10}  {'λE[J²]':>10}  {'λE[J⁴]':>10}")
    print(f"  {'-'*44}")
    for lbl, lam, p_, ep, en in [
        ("WBTC  (A1)", adj.lam1, adj.p1, adj.eta1_pos, adj.eta1_neg),
        ("Synth (A2)", adj.lam2, adj.p2, adj.eta2_pos, adj.eta2_neg),
    ]:
        k1, k2, k4 = jump_cumulants(lam, p_, ep, en)
        print(f"  {lbl:<10}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")

    c = constraint
    print(f"\nAAVE v3 Ethereum WBTC risk parameters:")
    print(f"  b={c['b']:.2f}  LTV_max={c['ltv_max']:.2f}  "
          f"liq_bonus={c['liq_bonus']:.0%}")
    print(f"  h0_min = log(b/LTV_max) = {c['h0_min']:.4f}  "
          f"H0_min = {c['H0_min']:.4f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ecf_fit(
    r1: np.ndarray,
    r2: np.ndarray,
    params: KouParams,
    dt: float,
    out: Path,
) -> None:
    title = (
        r"ECF fit — WBTC/WBTC-synth spread $(s,-s)$, AAVE v3 Ethereum hourly data"
        f"\n$N={len(r1)}$ obs, $\\Delta t={dt:.2e}$ yr, rate-adjusted params"
    )
    plot_ecf_spread_fit(
        r1,
        r2,
        dt,
        out,
        title,
        estimated_params=params,
        estimated_label="ECF fit (rate-adj.)",
        fontsize=9,
    )
    print(f"Figure → {out}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_params(
    result: ECFCalibrationResult,
    adj: KouParams,
    rates: dict,
    constraint: dict,
    date0: str,
    date1: str,
) -> None:
    payload = {
        "_note": (
            "Generated by jobs/calibrate_wbtc_hourly.py. "
            "Asset 2 is a synthetic correlated copy of WBTC (rho=0.60 target). "
            "Rate-adjusted KouParams are the primary output."
        ),
        "meta": {
            "run_at":     datetime.now(timezone.utc).isoformat(),
            "data":       "AAVE v3 Ethereum WBTC hourly data + synthetic pair",
            "asset1":     "WBTC",
            "asset2":     "WBTC-synth",
            "date_range": [date0, date1],
            "n_obs":      result.n_obs,
            "dt_years":   DT_YEARS,
            "n_starts":   N_STARTS,
            "seed":       SEED,
        },
        "aave_rates":      {k: float(v) for k, v in rates.items()},
        "aave_constraint": constraint,
        "params":          kou_params_to_dict(adj),
        "params_raw_ecf":  kou_params_to_dict(result.params),
        "ecf": {
            "objective":          result.objective,
            "objective_by_group": result.objective_by_group,
            "success":            result.success,
            "message":            result.message,
            "n_iter":             result.n_iter,
            "best_start_index":   result.best_start_index,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, cls=NpEncoder))
    print(f"Params → {OUT_JSON}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading AAVE hourly data …")
    real, synth = load_data()
    r1, r2, date0, date1 = compute_returns(real, synth)
    rates      = compute_avg_rates(real)
    constraint = aave_constraint()

    print(f"N={len(r1)} hourly log-return observations  ({date0} → {date1})")
    print(f"WBTC  std(r1) = {np.std(r1):.6f}")
    print(f"Synth std(r2) = {np.std(r2):.6f}")
    print(f"Empirical rho(r1,r2) = {float(np.corrcoef(r1, r2)[0,1]):.4f}")

    result: ECFCalibrationResult = calibrate_ecf(
        r1, r2, DT_YEARS,
        bounds=CALIB_BOUNDS,
        n_starts=N_STARTS,
        seed=SEED,
    )

    adj = adjust_kou_drifts(result.params, rates["supply_wbtc"], rates["borrow_wbtc"])

    print_summary(result, adj, rates, date0, date1, constraint)
    save_params(result, adj, rates, constraint, date0, date1)
    plot_ecf_fit(r1, r2, adj, DT_YEARS, OUT_FIG)


if __name__ == "__main__":
    main()
