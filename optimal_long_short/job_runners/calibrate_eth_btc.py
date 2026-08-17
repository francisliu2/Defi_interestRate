"""Build the data-driven WETH/WBTC empirical showcase.

The ECF fit estimates only diffusion, jump, and dependence shape from centered
causal-EWM innovations. The residual log means are fixed at zero. Endpoint EWM
log-return trends are then combined with the shape-implied price-growth
corrections. Because AAVE carry depends on the assigned role, both possible
long/short assignments are evaluated after adding the applicable supply and
borrowing rates. The assignment with the larger positive empirical-mu spread is
used for the showcase.

Usage :  python jobs/calibrate_eth_btc.py
Output:  results/params_empirical_showcase.json
         results/params_<LONG>_<SHORT>.json
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
    CausalEWMResult,
    ECFCalibrationResult,
    ParameterBounds,
    calibrate_ecf,
    causal_ewm_detrend,
)
from optimal_long_short.model_params import KouParams
from optimal_long_short.job_runners.common import (
    NpEncoder,
    REPO_ROOT,
    aave_constraint as make_aave_constraint,
    jump_cumulants,
    kou_params_to_dict,
    plot_ecf_spread_fit,
)
from optimal_long_short.drift import (
    drift_summary,
    expected_log_return_drift,
    swap_asset_order,
    with_expected_log_return_drift,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT       = REPO_ROOT
AAVE_DIR   = ROOT / "aave-ts" / "data" / "AAVE"
OUT_JSON   = ROOT / "results" / "params_empirical_showcase.json"
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

# AAVE v3 Ethereum protocol risk parameters at block 25,189,091. The on-chain
# liquidation-bonus factor is 1.05; this code stores its incremental 5% part.
AAVE_CONFIG_BLOCK = 25_189_091
AAVE_RISK = {
    "WETH": {"b": 0.83, "ltv_max": 0.805, "liq_bonus": 0.05},
    "WBTC": {"b": 0.78, "ltv_max": 0.73, "liq_bonus": 0.05},
}
ASSET_RATE_SUFFIX = {"WETH": "eth", "WBTC": "btc"}

N_STARTS = 30
SEED     = 42
EMPIRICAL_HORIZON_YEARS = 1.0 / 12.0

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
    """Return aligned log returns in fixed market order: WETH, then WBTC."""
    r_eth = np.diff(np.log(df["close_eth"].values))
    r_btc = np.diff(np.log(df["close_btc"].values))
    date0 = str(df["datetime"].iloc[1])   # first return timestamp
    date1 = str(df["datetime"].iloc[-1])  # last return timestamp
    ok = np.isfinite(r_btc) & np.isfinite(r_eth)
    return r_eth[ok], r_btc[ok], date0, date1


def prepare_causal_innovations(
    r_eth: np.ndarray,
    r_btc: np.ndarray,
    *,
    dt_years: float,
    horizon_years: float = EMPIRICAL_HORIZON_YEARS,
) -> tuple[np.ndarray, np.ndarray, dict[str, CausalEWMResult | float]]:
    """Return centered causal-EWM innovations and preprocessing metadata."""
    if not np.isfinite(dt_years) or dt_years <= 0.0:
        raise ValueError("dt_years must be finite and positive")
    if not np.isfinite(horizon_years) or horizon_years <= 0.0:
        raise ValueError("horizon_years must be finite and positive")
    half_life_periods = horizon_years / dt_years
    eth = causal_ewm_detrend(r_eth, half_life_periods)
    btc = causal_ewm_detrend(r_btc, half_life_periods)
    if len(eth.centered_innovations) != len(btc.centered_innovations):
        raise ValueError("Causal WETH/WBTC innovations are not aligned")
    return (
        eth.centered_innovations,
        btc.centered_innovations,
        {
            "WETH": eth,
            "WBTC": btc,
            "half_life_periods": half_life_periods,
            "equivalent_span": 2.0 / (1.0 - eth.decay) - 1.0,
        },
    )


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

def aave_constraint(collateral_asset: str) -> dict:
    """Compute the selected collateral asset's origination constraint."""
    try:
        risk = AAVE_RISK[collateral_asset]
    except KeyError as exc:
        raise ValueError(f"Unsupported collateral asset {collateral_asset!r}") from exc
    return make_aave_constraint(risk["b"], risk["ltv_max"], risk["liq_bonus"])


def empirical_asset_exponents(
    residual_market_order: KouParams,
    annualized_ewm_log_drift: dict[str, float],
) -> dict[str, float]:
    """Carry-free empirical price-growth exponents used to select the legs."""
    return {
        "WETH": residual_market_order.mu1 + annualized_ewm_log_drift["WETH"],
        "WBTC": residual_market_order.mu2 + annualized_ewm_log_drift["WBTC"],
    }


def select_long_short(
    mu_emp_pre_carry: dict[str, float],
    rates: dict[str, float],
) -> tuple[str, str, dict[str, dict[str, float | str]]]:
    """Choose the role assignment with the largest carry-adjusted mu spread."""
    required = {"WETH", "WBTC"}
    if set(mu_emp_pre_carry) != required:
        raise ValueError(f"Expected empirical exponents for {sorted(required)}")
    candidates: dict[str, dict[str, float | str]] = {}
    for long_asset in sorted(required):
        short_asset = next(asset for asset in required if asset != long_asset)
        supply_rate = float(rates[f"supply_{ASSET_RATE_SUFFIX[long_asset]}"])
        borrow_rate = float(rates[f"borrow_{ASSET_RATE_SUFFIX[short_asset]}"])
        mu_long = float(mu_emp_pre_carry[long_asset] + supply_rate)
        mu_short = float(mu_emp_pre_carry[short_asset] + borrow_rate)
        candidates[long_asset] = {
            "long_asset": long_asset,
            "short_asset": short_asset,
            "long_supply_rate": supply_rate,
            "short_borrow_rate": borrow_rate,
            "mu_long": mu_long,
            "mu_short": mu_short,
            "mu_spread": mu_long - mu_short,
        }
    selected = max(candidates.values(), key=lambda row: float(row["mu_spread"]))
    if float(selected["mu_spread"]) <= 0.0:
        raise ValueError("Neither role assignment has a positive empirical-mu spread")
    return str(selected["long_asset"]), str(selected["short_asset"]), candidates


def orient_residual_params(
    residual_market_order: KouParams,
    long_asset: str,
) -> KouParams:
    """Return residual parameters with asset 1 equal to the selected long leg."""
    if long_asset == "WETH":
        return residual_market_order
    if long_asset == "WBTC":
        return swap_asset_order(residual_market_order)
    raise ValueError(f"Unsupported long asset {long_asset!r}")


def build_empirical_params(
    residual_market_order: KouParams,
    annualized_ewm_log_drift: dict[str, float],
    rates: dict[str, float],
) -> tuple[KouParams, KouParams, dict[str, object]]:
    """Select the legs and construct the final role-adjusted Kou parameters."""
    pre_carry = empirical_asset_exponents(
        residual_market_order,
        annualized_ewm_log_drift,
    )
    long_asset, short_asset, candidates = select_long_short(pre_carry, rates)
    residual_oriented = orient_residual_params(residual_market_order, long_asset)
    selected_candidate = candidates[long_asset]
    supply_rate = float(selected_candidate["long_supply_rate"])
    borrow_rate = float(selected_candidate["short_borrow_rate"])
    final_params = with_expected_log_return_drift(
        residual_oriented,
        drift1=annualized_ewm_log_drift[long_asset] + supply_rate,
        drift2=annualized_ewm_log_drift[short_asset] + borrow_rate,
    )
    if final_params.mu1 <= final_params.mu2:
        raise ValueError(
            "The selected long leg does not have the larger final empirical mu "
            "after role-specific carry"
        )
    selection = {
        "rule": "maximize the positive role-adjusted empirical-mu spread",
        "long_asset": long_asset,
        "short_asset": short_asset,
        "mu_emp_pre_carry": pre_carry,
        "candidate_assignments": candidates,
        "long_supply_rate": supply_rate,
        "short_borrow_rate": borrow_rate,
        "final_mu1": final_params.mu1,
        "final_mu2": final_params.mu2,
    }
    return final_params, residual_oriented, selection


def print_summary(
    result: ECFCalibrationResult,
    final_params: KouParams,
    residual_oriented: KouParams,
    selection: dict[str, object],
    rates: dict,
    date0: str,
    date1: str,
    constraint: dict,
    meta: dict,
) -> None:
    long_asset = str(selection["long_asset"])
    short_asset = str(selection["short_asset"])
    trends = meta["annualized_ewm_log_drift"]
    print(f"\n{'=' * 78}")
    print(f"AAVE v3 Ethereum {meta['frequency']} empirical showcase")
    print(
        f"Raw returns={meta['n_raw_returns']}, centered causal innovations={result.n_obs} "
        f"({date0} → {date1}), dt={meta['dt_years']:.8f} yr"
    )
    print(
        f"EWM half-life={meta['ewm_half_life_periods']:.3f} observations "
        f"({EMPIRICAL_HORIZON_YEARS * 365.0:.2f} days); "
        f"equivalent span={meta['ewm_equivalent_span']:.3f}"
    )
    print(f"Selected orientation: long {long_asset} / short {short_asset}")
    print(f"{'=' * 78}")

    print("\nEmpirical drift construction (annualized):")
    pre_carry = selection["mu_emp_pre_carry"]
    for asset in (long_asset, short_asset):
        residual_mu = (
            residual_oriented.mu1 if asset == long_asset else residual_oriented.mu2
        )
        role_rate = (
            float(selection["long_supply_rate"])
            if asset == long_asset
            else float(selection["short_borrow_rate"])
        )
        final_mu = final_params.mu1 if asset == long_asset else final_params.mu2
        role = "long/supply" if asset == long_asset else "short/borrow"
        print(
            f"  {asset} ({role}): EWM log drift {trends[asset]:+.6f} "
            f"+ residual correction {residual_mu:+.6f} "
            f"+ carry {role_rate:+.6f} = mu_emp {final_mu:+.6f} "
            f"(pre-carry {float(pre_carry[asset]):+.6f})"
        )

    rows = [
        ("mu", final_params.mu1, final_params.mu2),
        ("sigma", final_params.sigma1, final_params.sigma2),
        ("lambda", final_params.lam1, final_params.lam2),
        ("p", final_params.p1, final_params.p2),
        ("eta_pos", final_params.eta1_pos, final_params.eta2_pos),
        ("eta_neg", final_params.eta1_neg, final_params.eta2_neg),
    ]
    print(f"\n{'Parameter':<12} {long_asset:>14} {short_asset:>14}")
    print("-" * 42)
    for name, value1, value2 in rows:
        print(f"{name:<12} {value1:14.6f} {value2:14.6f}")
    print(f"{'rho':<12} {final_params.rho:14.6f} {final_params.rho:14.6f}")

    residual_log = expected_log_return_drift(residual_oriented)
    final_log = expected_log_return_drift(final_params)
    print(
        "\nResidual expected log drifts: "
        f"{residual_log[0]:+.3e}, {residual_log[1]:+.3e}"
    )
    print(
        "Final expected log drifts: "
        f"{long_asset}={final_log[0]:+.6f}, {short_asset}={final_log[1]:+.6f}"
    )
    print(
        f"ECF objective Q_N={result.objective:.6e} "
        f"(mode={result.drift_mode}, converged={result.success}, "
        f"start={result.best_start_index})"
    )
    print(f"\nQ_N by group:")
    for g, v in sorted(result.objective_by_group.items()):
        print(f"  {g:<14} {v:.4e}")

    print(f"\nJump cumulants (annualized, final params):")
    print(f"  {'':10}  {'λE[J]':>10}  {'λE[J²]':>10}  {'λE[J⁴]':>10}")
    print(f"  {'-'*44}")
    for lbl, lam, p_up, eta_pos, eta_neg in [
        (f"{long_asset} (A1)", final_params.lam1, final_params.p1, final_params.eta1_pos, final_params.eta1_neg),
        (f"{short_asset} (A2)", final_params.lam2, final_params.p2, final_params.eta2_pos, final_params.eta2_neg),
    ]:
        k1, k2, k4 = jump_cumulants(lam, p_up, eta_pos, eta_neg)
        print(f"  {lbl:<10}  {k1:10.5f}  {k2:10.5f}  {k4:10.5f}")

    c = constraint
    print(f"\nAAVE v3 Ethereum {long_asset} collateral parameters:")
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
        rf"Shape-only ECF fit — centered WETH/WBTC innovations $(s,-s)$, {frequency} data"
        f"\n$N={len(r1)}$, $\\Delta t={dt:.6f}$, zero residual log means"
    )
    plot_ecf_spread_fit(
        r1,
        r2,
        dt,
        out,
        title,
        estimated_params=params,
        estimated_label="Residual shape ECF fit",
        fontsize=9,
    )
    print(f"Figure → {out}")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_params(
    result: ECFCalibrationResult,
    final_params: KouParams,
    residual_oriented: KouParams,
    selection: dict[str, object],
    rates: dict,
    constraint: dict,
    date0: str,
    date1: str,
    meta: dict,
) -> None:
    long_asset = str(selection["long_asset"])
    short_asset = str(selection["short_asset"])
    final_log1, final_log2 = expected_log_return_drift(final_params)
    residual_log1, residual_log2 = expected_log_return_drift(residual_oriented)
    trends = meta["annualized_ewm_log_drift"]
    drift_components = {
        long_asset: {
            "role": "long_collateral",
            "ewm_expected_log_drift": trends[long_asset],
            "residual_price_growth_correction": residual_oriented.mu1,
            "mu_emp_pre_carry": selection["mu_emp_pre_carry"][long_asset],
            "role_carry": selection["long_supply_rate"],
            "final_mu_price_growth": final_params.mu1,
            "final_expected_log_return_drift": final_log1,
        },
        short_asset: {
            "role": "short_debt",
            "ewm_expected_log_drift": trends[short_asset],
            "residual_price_growth_correction": residual_oriented.mu2,
            "mu_emp_pre_carry": selection["mu_emp_pre_carry"][short_asset],
            "role_carry": selection["short_borrow_rate"],
            "final_mu_price_growth": final_params.mu2,
            "final_expected_log_return_drift": final_log2,
        },
    }
    payload = {
        "_note": (
            "Generated by jobs/calibrate_eth_btc.py. "
            "The primary params use data-selected long/short order. Reproduce by "
            "re-running that script; never infer asset order from the filename."
        ),
        "meta": {
            "run_at":     datetime.now(timezone.utc).isoformat(),
            "data":       f"AAVE v3 Ethereum WETH/WBTC {meta['frequency']} on-chain data",
            "asset1":     long_asset,
            "asset2":     short_asset,
            "market_calibration_order": ["WETH", "WBTC"],
            "date_range": [date0, date1],
            "n_obs":      result.n_obs,
            "n_raw_returns": meta["n_raw_returns"],
            "dt_years":   meta["dt_years"],
            "frequency":  meta["frequency"],
            "returns_preprocessing": {
                "method": "centered_lagged_normalized_ewm_innovations",
                "causal": True,
                "first_return_used_only_to_initialize_mean": True,
                "ewm_half_life_years": EMPIRICAL_HORIZON_YEARS,
                "ewm_half_life_periods": meta["ewm_half_life_periods"],
                "ewm_decay": meta["ewm_decay"],
                "equivalent_pandas_span": meta["ewm_equivalent_span"],
                "endpoint_ewm_mean_per_period": meta["endpoint_ewm_mean_per_period"],
                "endpoint_ewm_mean_annualized": trends,
                "uncentered_innovation_mean_per_period": meta["innovation_mean_per_period"],
                "centered_innovation_mean_per_period": meta["centered_innovation_mean_per_period"],
            },
            "drift_convention": {
                "params_mu": (
                    "Annualized final expected-price growth exponent satisfying "
                    "E[exp(X_i(t))] = exp(mu_i*t)."
                ),
                "construction": (
                    "mu_emp = endpoint EWM expected-log drift + residual "
                    "zero-log-mean price-growth correction + role-specific carry"
                ),
                "residual_ecf_mode": result.drift_mode,
                "orientation_rule": selection["rule"],
            },
            "aave_rate_aggregation": {
                "method": "arithmetic mean of aligned observation-level annual APRs",
                "date_range": [date0, date1],
                "units": "annual decimal fraction",
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
        "orientation_selection": selection,
        "drift_components_by_asset": drift_components,
        "params": kou_params_to_dict(final_params),
        "drift_summary": drift_summary(final_params),
        "params_residual_ecf_oriented": kou_params_to_dict(residual_oriented),
        "params_residual_ecf_market_order": kou_params_to_dict(result.params),
        "residual_expected_log_drift_oriented": [residual_log1, residual_log2],
        "drift_summary_residual_ecf_oriented": drift_summary(residual_oriented),
        "ecf": {
            "objective":          result.objective,
            "objective_by_group": result.objective_by_group,
            "success":            result.success,
            "message":            result.message,
            "n_iter":             result.n_iter,
            "best_start_index":   result.best_start_index,
            "drift_mode":         result.drift_mode,
            "n_frequency_points": len(result.freqs),
        },
    }
    serialized = json.dumps(payload, indent=2, cls=NpEncoder)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(serialized)
    oriented_path = OUT_JSON.parent / f"params_{long_asset}_{short_asset}.json"
    oriented_path.write_text(serialized)
    print(f"Params → {OUT_JSON}")
    print(f"Orientation copy → {oriented_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading AAVE on-chain data …")
    df, meta = load_aave_data()
    if meta["scheduled_latest_block"] != AAVE_CONFIG_BLOCK:
        raise ValueError(
            "The saved WETH/WBTC contract constants are verified at block "
            f"{AAVE_CONFIG_BLOCK}, but the latest aligned sample ends at block "
            f"{meta['scheduled_latest_block']}. Re-query the AAVE configuration "
            "before calibrating a different terminal sample."
        )
    r_eth_raw, r_btc_raw, date0, date1 = compute_returns(df)
    r_eth, r_btc, preprocessing = prepare_causal_innovations(
        r_eth_raw,
        r_btc_raw,
        dt_years=meta["dt_years"],
    )
    eth_pre = preprocessing["WETH"]
    btc_pre = preprocessing["WBTC"]
    assert isinstance(eth_pre, CausalEWMResult)
    assert isinstance(btc_pre, CausalEWMResult)
    meta.update(
        {
            "n_raw_returns": len(r_eth_raw),
            "ewm_half_life_periods": preprocessing["half_life_periods"],
            "ewm_equivalent_span": preprocessing["equivalent_span"],
            "ewm_decay": eth_pre.decay,
            "endpoint_ewm_mean_per_period": {
                "WETH": float(eth_pre.mean_path[-1]),
                "WBTC": float(btc_pre.mean_path[-1]),
            },
            "annualized_ewm_log_drift": {
                "WETH": float(eth_pre.mean_path[-1] / meta["dt_years"]),
                "WBTC": float(btc_pre.mean_path[-1] / meta["dt_years"]),
            },
            "innovation_mean_per_period": {
                "WETH": eth_pre.innovation_mean,
                "WBTC": btc_pre.innovation_mean,
            },
            "centered_innovation_mean_per_period": {
                "WETH": float(np.mean(r_eth)),
                "WBTC": float(np.mean(r_btc)),
            },
        }
    )
    rates = compute_avg_rates(df)

    print(
        f"N={len(r_eth_raw)} raw return pairs and {len(r_eth)} centered causal "
        f"innovations ({date0} → {date1})"
    )
    print(
        "Applied normalized causal EWM detrending with half-life "
        f"T={EMPIRICAL_HORIZON_YEARS:.8f} years "
        f"({float(preprocessing['half_life_periods']):.3f} observations)"
    )

    result: ECFCalibrationResult = calibrate_ecf(
        r_eth, r_btc, meta["dt_years"],
        bounds=CALIB_BOUNDS,
        n_starts=N_STARTS,
        drift_mode="zero_expected_log_return",
        seed=SEED,
    )
    final_params, residual_oriented, selection = build_empirical_params(
        result.params,
        meta["annualized_ewm_log_drift"],
        rates,
    )
    long_asset = str(selection["long_asset"])
    short_asset = str(selection["short_asset"])
    constraint = aave_constraint(long_asset)
    constraint.update(
        {
            "collateral_asset": long_asset,
            "debt_asset": short_asset,
            "configuration_block": meta["initial_prices"]["block"],
            "configuration_datetime": meta["initial_prices"]["datetime"],
            "emode_applied": False,
        }
    )

    print_summary(
        result,
        final_params,
        residual_oriented,
        selection,
        rates,
        date0,
        date1,
        constraint,
        meta,
    )
    save_params(
        result,
        final_params,
        residual_oriented,
        selection,
        rates,
        constraint,
        date0,
        date1,
        meta,
    )
    plot_ecf_fit(
        r_eth,
        r_btc,
        result.params,
        meta["dt_years"],
        OUT_FIG,
        meta["frequency"],
    )


if __name__ == "__main__":
    main()
