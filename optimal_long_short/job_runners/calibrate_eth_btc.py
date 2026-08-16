"""
Calibrate bivariate Kou parameters for WETH / WBTC from AAVE v3 Ethereum
on-chain price data, then adjust the drift parameters for AAVE supply
and borrow rates.

Strategy convention
-------------------
  Asset 1 = WETH  (long, posted as collateral) → earns supply_apr
  Asset 2 = WBTC  (short, variable borrow)     → costs variable_borrow_apr

Rate adjustment to mu_i (price-growth drift, annualised):
  mu1 += supply_apr_WETH / 100     (earn deposit yield on collateral)
  mu2 += borrow_apr_WBTC / 100     (borrow cost raises the required
                                    growth of the borrowed leg, shrinking the
                                    spread log-drift mu1^X - mu2^X)

AAVE v3 Ethereum WETH risk parameters at the empirical terminal block:
  Liquidation threshold  b       = 0.83   (83 %)
  Maximum LTV at origination     = 0.805  (80.5 %)
  → Minimum initial log-health   h0_min = log(b / LTV_max) ≈ 0.0306

Usage :  python jobs/calibrate_eth_btc.py
Output:  results/params_WETH_WBTC.json   (fixed name — deterministic)
         latex/fig_ecf_empirical.pdf
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
    ewm_smooth,
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
from optimal_long_short.drift import drift_summary, expected_log_return_drift

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT       = REPO_ROOT
AAVE_DIR   = ROOT / "aave-ts" / "data" / "AAVE"
OUT_JSON   = ROOT / "results" / "params_WETH_WBTC.json"   # fixed name
OUT_FIG    = ROOT / "latex" / "fig_ecf_empirical.pdf"

SECONDS_PER_YEAR = 365.0 * 24.0 * 3_600.0
FREQUENCY_SECONDS = {
    "hourly": 3_600.0,
    "4h": 4.0 * 3_600.0,
    "6h": 6.0 * 3_600.0,
    "8h": 8.0 * 3_600.0,
    "12h": 12.0 * 3_600.0,
    "daily": 24.0 * 3_600.0,
    "1d": 24.0 * 3_600.0,
}

# AAVE v3 Ethereum WETH protocol risk parameters at block 25,189,091.
# The on-chain liquidation-bonus factor is 1.05; this code stores its
# incremental 5% component.
AAVE_CONFIG_BLOCK = 25_189_091
AAVE_B           = 0.83    # liquidation threshold
AAVE_LTV_MAX     = 0.805   # maximum LTV at origination
AAVE_LIQ_BONUS   = 0.05    # liquidation bonus

N_STARTS = 30
SEED     = 42
EWM_DEMEAN_SPAN = 30

# Calibration bounds: use a broad search box for empirical 4h data.  The only
# hard economic/numerical constraint kept here is the moment-admissibility cap
# on both positive-jump means implied by max_moment_order.
CALIB_BOUNDS = ParameterBounds(
    max_moment_order=4,
    lambda_max=5000.0,
    p_min=0.001,
    p_max=0.999,
    eta_pos1_min=1e-5,
    eta_pos2_min=1e-5,
    eta_neg_min=1e-5,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _latest_manifest_row(manifest: pd.DataFrame, symbol: str) -> pd.Series:
    rows = manifest.loc[manifest["symbol"].str.upper() == symbol]
    if rows.empty:
        raise FileNotFoundError(f"No {symbol} row found in {AAVE_DIR / 'manifest.csv'}")
    return rows.sort_values("fetched_at").iloc[-1]


def _dt_from_frequency(freq: str) -> float:
    if freq not in FREQUENCY_SECONDS:
        raise ValueError(f"Unknown manifest frequency {freq!r}")
    return FREQUENCY_SECONDS[freq] / SECONDS_PER_YEAR


def load_aave_data() -> tuple[pd.DataFrame, dict]:
    """Load WBTC and WETH AAVE parquet files from the manifest, merge on block."""
    manifest_path = AAVE_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found. Run aave-ts history fetch first.")

    manifest = pd.read_csv(manifest_path)
    btc_meta = _latest_manifest_row(manifest, "WBTC")
    eth_meta = _latest_manifest_row(manifest, "WETH")

    if btc_meta["scheduled_latest_block"] != eth_meta["scheduled_latest_block"]:
        raise ValueError("WBTC and WETH manifest rows do not share a scheduled_latest_block")
    if btc_meta["frequency"] != eth_meta["frequency"]:
        raise ValueError("WBTC and WETH manifest rows do not share a frequency")

    btc = pd.read_parquet(AAVE_DIR / str(btc_meta["parquet_file"]))
    eth = pd.read_parquet(AAVE_DIR / str(eth_meta["parquet_file"]))

    merged = (
        pd.merge(btc, eth, on=["block", "datetime"], suffixes=("_btc", "_eth"))
        .sort_values("block")
        .reset_index(drop=True)
    )
    if len(merged) < 2:
        raise ValueError("Need at least two aligned WETH/WBTC rows to compute returns")

    meta = {
        "frequency": str(btc_meta["frequency"]),
        "dt_years": _dt_from_frequency(str(btc_meta["frequency"])),
        "btc_file": str(btc_meta["parquet_file"]),
        "eth_file": str(eth_meta["parquet_file"]),
        "scheduled_latest_block": int(btc_meta["scheduled_latest_block"]),
        "scheduled_sample_count": int(btc_meta["scheduled_sample_count"]),
        "initial_prices": {
            "WBTC": float(merged["close_btc"].iloc[-1]),
            "WETH": float(merged["close_eth"].iloc[-1]),
            "datetime": str(merged["datetime"].iloc[-1]),
            "block": int(merged["block"].iloc[-1]),
        },
    }
    return merged, meta


def compute_returns(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return aligned log returns in strategy order: WETH first, WBTC second."""
    r_eth = np.diff(np.log(df["close_eth"].values))
    r_btc = np.diff(np.log(df["close_btc"].values))
    date0 = str(df["datetime"].iloc[1])   # first return timestamp
    date1 = str(df["datetime"].iloc[-1])  # last return timestamp
    ok = np.isfinite(r_btc) & np.isfinite(r_eth)
    return r_eth[ok], r_btc[ok], date0, date1


def ewm_demean_returns(r: np.ndarray, span: float) -> np.ndarray:
    """Subtract an exponentially weighted mean from log returns."""
    return r - ewm_smooth(r, span)


def ewm_demean_with_mean(r: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray]:
    """Return EWM-demeaned returns and the EWM mean path."""
    mean = ewm_smooth(r, span)
    return r - mean, mean


def compute_avg_rates(df: pd.DataFrame) -> dict[str, float]:
    """
    Average annualised AAVE APRs (values in parquet are already in %).
    Returns decimal fractions (divide by 100).
    """
    return {
        "supply_btc":  float(df["supply_apr_btc"].mean())          / 100.0,
        "borrow_eth":  float(df["variable_borrow_apr_eth"].mean())  / 100.0,
        "supply_eth":  float(df["supply_apr_eth"].mean())           / 100.0,
        "borrow_btc":  float(df["variable_borrow_apr_btc"].mean())  / 100.0,
    }


# ---------------------------------------------------------------------------
# Rate-adjusted KouParams
# ---------------------------------------------------------------------------

def aave_constraint() -> dict:
    """Compute the WETH-collateral minimum h0 enforced at origination."""
    return make_aave_constraint(AAVE_B, AAVE_LTV_MAX, AAVE_LIQ_BONUS)


def print_summary(
    result: ECFCalibrationResult,
    adj: KouParams,
    rates: dict,
    date0: str,
    date1: str,
    constraint: dict,
    meta: dict,
) -> None:
    raw  = result.params
    init = result.theta0

    print(f"\n{'='*70}")
    print(f"WETH / WBTC  |  AAVE v3 Ethereum {meta['frequency']} data")
    print(f"N={result.n_obs} obs  ({date0} → {date1})  dt={meta['dt_years']:.8f} yr")
    print(f"Returns preprocessing: EWM demean span={meta['ewm_demean_span']}")
    print(
        "Initial prices for payoff analysis: "
        f"WETH S10={meta['initial_prices']['WETH']:.6f}, "
        f"WBTC S20={meta['initial_prices']['WBTC']:.6f} "
        f"at block {meta['initial_prices']['block']}"
    )
    print(f"{'='*70}")

    print(f"\nAdjustment to mu_i for AAVE rates (annual, decimal):")
    print(f"  WETH supply rate  r_s1 = {rates['supply_eth']:.6f}  "
          f"(+{rates['supply_eth']*100:.4f}%/yr on collateral)")
    print(f"  WBTC borrow rate  r_b2 = {rates['borrow_btc']:.6f}  "
          f"(+{rates['borrow_btc']*100:.4f}%/yr on borrowed leg)")
    print(f"  Net rate effect on spread drift = "
          f"{(rates['supply_eth']-rates['borrow_btc'])*100:+.4f}%/yr")

    rows = [
        ("mu1",      init.mu1,           raw.mu1,           adj.mu1),
        ("muX1",  init.muX1, raw.muX1, adj.muX1),
        ("sigma1",   init.sigma1,        raw.sigma1,        adj.sigma1),
        ("lam1",     init.lam1,          raw.lam1,          adj.lam1),
        ("p1",       init.p1,            raw.p1,            adj.p1),
        ("eta1_pos", init.eta1_pos,      raw.eta1_pos,      adj.eta1_pos),
        ("eta1_neg", init.eta1_neg,      raw.eta1_neg,      adj.eta1_neg),
        ("mu2",      init.mu2,           raw.mu2,           adj.mu2),
        ("muX2",  init.muX2, raw.muX2, adj.muX2),
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
    emp = meta.get("empirical_first_moments", {})
    if emp:
        log1, log2 = expected_log_return_drift(adj)
        print("\nFirst-moment diagnostics (annualized):")
        print(
            f"  empirical log drift after EWM demeaning: "
            f"WETH={emp['demeaned_log_mean_annualized']['WETH']:.4f}, "
            f"WBTC={emp['demeaned_log_mean_annualized']['WBTC']:.4f}"
        )
        print(
            f"  empirical simple-price growth after EWM demeaning: "
            f"WETH={emp['demeaned_price_growth_mu_annualized']['WETH']:.4f}, "
            f"WBTC={emp['demeaned_price_growth_mu_annualized']['WBTC']:.4f}"
        )
        print(
            f"  model log-return drift E[dX]/dt: "
            f"WETH={log1:.4f}, WBTC={log2:.4f}"
        )
        print(
            "  Note: ECF drift is weakly identified at 4h frequency; "
            "treat mu_i as a model normalizer, not a forecast."
        )
    print(f"\nQ_N by group:")
    for g, v in sorted(result.objective_by_group.items()):
        print(f"  {g:<14} {v:.4e}")

    print(f"\nJump cumulants (annualised, rate-adjusted params):")
    print(f"  {'':10}  {'λE[J]':>10}  {'λE[J²]':>10}  {'λE[J⁴]':>10}")
    print(f"  {'-'*44}")
    for lbl, lam, p_up, eta_pos, eta_neg in [
        ("WETH (A1)", adj.lam1, adj.p1, adj.eta1_pos, adj.eta1_neg),
        ("WBTC (A2)", adj.lam2, adj.p2, adj.eta2_pos, adj.eta2_neg),
    ]:
        k1, k2, k4 = jump_cumulants(lam, p_up, eta_pos, eta_neg)
        print(f"  {lbl:<10}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")

    c = constraint
    print(f"\nAAVE v3 Ethereum WETH collateral parameters:")
    print(f"  Liquidation threshold  b       = {c['b']:.2f}")
    print(f"  Max LTV at origination         = {c['ltv_max']:.2f}  "
          f"(largest initial LTV₀ = {c['ltv_max']:.1%})")
    print(f"  Liquidation bonus              = {c['liq_bonus']:.0%}")
    print(f"  Min initial log-health h₀_min  = log(b/LTV_max)"
          f" = log({c['b']}/{c['ltv_max']}) = {c['h0_min']:.4f}")
    print(f"  Min initial health factor H₀   = {c['H0_min']:.4f}")
    print(f"  → feasible h₀ range: [{c['h0_min']:.4f}, ∞)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ecf_fit(
    r1: np.ndarray,
    r2: np.ndarray,
    params: KouParams,
    dt: float,
    out: Path,
    frequency: str,
) -> None:
    title = (
        rf"ECF fit — WETH/WBTC spread $(s,-s)$, AAVE v3 Ethereum {frequency} data"
        f"\n$N={len(r1)}$ obs, $\\Delta t={dt:.6f}$, rate-adjusted params"
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
# Saving
# ---------------------------------------------------------------------------

def save_params(
    result: ECFCalibrationResult,
    adj: KouParams,
    rates: dict,
    constraint: dict,
    date0: str,
    date1: str,
    meta: dict,
) -> None:
    adj_log1, adj_log2 = expected_log_return_drift(adj)
    raw_log1, raw_log2 = expected_log_return_drift(result.params)
    payload = {
        "_note": (
            "Generated by jobs/calibrate_eth_btc.py. "
            "Rate-adjusted KouParams are the primary output for downstream analysis. "
            "Reproduce by re-running that script."
        ),
        "meta": {
            "run_at":     datetime.now(timezone.utc).isoformat(),
            "data":       f"AAVE v3 Ethereum WETH/WBTC {meta['frequency']} on-chain data",
            "asset1":     "WETH",
            "asset2":     "WBTC",
            "date_range": [date0, date1],
            "n_obs":      result.n_obs,
            "dt_years":   meta["dt_years"],
            "frequency":  meta["frequency"],
            "returns_preprocessing": {
                "method": "log_returns_minus_exponential_weighted_mean",
                "ewm_span": meta["ewm_demean_span"],
                "latest_ewm_mean_per_period": {
                    "WETH": meta["latest_ewm_mean_eth"],
                    "WBTC": meta["latest_ewm_mean_btc"],
                },
                "latest_ewm_mean_annualized": {
                    "WETH": meta["latest_ewm_mean_eth"] / meta["dt_years"],
                    "WBTC": meta["latest_ewm_mean_btc"] / meta["dt_years"],
                },
            },
            "drift_convention": {
                "params_mu": (
                    "Annualized model price-growth normalizer: "
                    "E[exp(X_i(t))] = exp(mu_i * t) for the fitted process. "
                    "At high frequency this is weakly identified and should "
                    "not be read as an out-of-sample price forecast."
                ),
                "muX": (
                    "Annualized log-process drift used by the characteristic "
                    "function, moments, and Monte Carlo."
                ),
                "muX_formula": (
                    "muX_i = mu_i - 0.5*sigma_i^2 "
                    "- lambda_i*(E[exp(J_i)] - 1)"
                ),
                "user_view_recommendation": (
                    "Apply user drift views to params.mu1/mu2 unless the view "
                    "is explicitly a log-process drift."
                ),
                "ewm_demean_note": (
                    "Calibration was fitted to log returns minus their EWM mean. "
                    "The latest EWM mean is reported but not automatically added "
                    "back to params; treat it as a drift view if desired."
                ),
            },
            "source_files": {
                "WBTC": meta["btc_file"],
                "WETH": meta["eth_file"],
            },
            "initial_prices": meta["initial_prices"],
            "scheduled_latest_block": meta["scheduled_latest_block"],
            "scheduled_sample_count": meta["scheduled_sample_count"],
            "n_starts":   N_STARTS,
            "seed":       SEED,
        },
        "aave_rates": {
            k: float(v) for k, v in rates.items()
        },
        "aave_constraint": constraint,
        "first_moment_diagnostics": {
            "empirical": meta.get("empirical_first_moments", {}),
            "model_rate_adjusted": {
                "mu_price_growth": {
                    "WETH": adj.mu1,
                    "WBTC": adj.mu2,
                },
                "muX_log_process": {
                    "WETH": adj.muX1,
                    "WBTC": adj.muX2,
                },
                "expected_log_return_drift": {
                    "WETH": adj_log1,
                    "WBTC": adj_log2,
                },
                "note": (
                    "mu_i is the model normalizer for E[exp(X_i(t))], not a "
                    "directional forecast. Compare against empirical first "
                    "moments before applying user drift views."
                ),
            },
            "model_raw_ecf": {
                "mu_price_growth": {
                    "WETH": result.params.mu1,
                    "WBTC": result.params.mu2,
                },
                "muX_log_process": {
                    "WETH": result.params.muX1,
                    "WBTC": result.params.muX2,
                },
                "expected_log_return_drift": {
                    "WETH": raw_log1,
                    "WBTC": raw_log2,
                },
            },
        },
        # Rate-adjusted params — use these for downstream analysis
        "params": kou_params_to_dict(adj),
        "drift_summary": drift_summary(adj),
        # Raw ECF output (before rate adjustment) for reference
        "params_raw_ecf": kou_params_to_dict(result.params),
        "drift_summary_raw_ecf": drift_summary(result.params),
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
    print("Loading AAVE on-chain data …")
    df, meta = load_aave_data()
    if meta["scheduled_latest_block"] != AAVE_CONFIG_BLOCK:
        raise ValueError(
            "The saved WETH contract constants are verified at block "
            f"{AAVE_CONFIG_BLOCK}, but the latest aligned sample ends at block "
            f"{meta['scheduled_latest_block']}. Re-query the AAVE configuration "
            "before calibrating a different terminal sample."
        )
    r_eth_raw, r_btc_raw, date0, date1 = compute_returns(df)
    r_eth, ewm_mean_eth = ewm_demean_with_mean(r_eth_raw, EWM_DEMEAN_SPAN)
    r_btc, ewm_mean_btc = ewm_demean_with_mean(r_btc_raw, EWM_DEMEAN_SPAN)
    meta["ewm_demean_span"] = EWM_DEMEAN_SPAN
    meta["latest_ewm_mean_btc"] = float(ewm_mean_btc[-1])
    meta["latest_ewm_mean_eth"] = float(ewm_mean_eth[-1])
    meta["empirical_first_moments"] = {
        "raw_log_mean_annualized": {
            "WBTC": float(np.mean(r_btc_raw) / meta["dt_years"]),
            "WETH": float(np.mean(r_eth_raw) / meta["dt_years"]),
        },
        "demeaned_log_mean_annualized": {
            "WBTC": float(np.mean(r_btc) / meta["dt_years"]),
            "WETH": float(np.mean(r_eth) / meta["dt_years"]),
        },
        "demeaned_price_growth_mu_annualized": {
            "WBTC": float(np.log(np.mean(np.exp(r_btc))) / meta["dt_years"]),
            "WETH": float(np.log(np.mean(np.exp(r_eth))) / meta["dt_years"]),
        },
        "demeaned_simple_return_mean_annualized": {
            "WBTC": float(np.mean(np.exp(r_btc) - 1.0) / meta["dt_years"]),
            "WETH": float(np.mean(np.exp(r_eth) - 1.0) / meta["dt_years"]),
        },
    }
    rates      = compute_avg_rates(df)
    constraint = aave_constraint()
    constraint.update(
        {
            "collateral_asset": "WETH",
            "debt_asset": "WBTC",
            "configuration_block": meta["initial_prices"]["block"],
            "configuration_datetime": meta["initial_prices"]["datetime"],
            "emode_applied": False,
        }
    )

    print(f"N={len(r_eth)} log-return observations  ({date0} → {date1})")
    print(f"Applied EWM demeaning to log returns (span={EWM_DEMEAN_SPAN})")

    result: ECFCalibrationResult = calibrate_ecf(
        r_eth, r_btc, meta["dt_years"],
        bounds=CALIB_BOUNDS,
        n_starts=N_STARTS,
        seed=SEED,
    )

    adj = adjust_kou_drifts(result.params, rates["supply_eth"], rates["borrow_btc"])

    print_summary(result, adj, rates, date0, date1, constraint, meta)
    save_params(result, adj, rates, constraint, date0, date1, meta)
    plot_ecf_fit(r_eth, r_btc, adj, meta["dt_years"], OUT_FIG, meta["frequency"])


if __name__ == "__main__":
    main()
