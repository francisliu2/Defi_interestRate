"""Full-history, multi-frequency robustness of the WBTC/WETH calibration.

Uses the full Aave v3 Ethereum history in data/AAVE_full (2023-01-27 onward,
fetched with ``jobs/fetch_universe.py --start 2023-01-27 --out-dir data/AAVE_full``)
and the stale-aware panels of :mod:`optimal_long_short.full_history`.

1. Calibrates the bivariate Kou model with the headline pipeline (EWM de-meaning,
   penalised ECF, CALIB_BOUNDS, APR drift adjustment) on
     * daily bars over the full 3.7 years, and
     * four-hourly bars from 2024-01 (the thin 2023 WBTC market excluded),
   next to the stored headline calibration.
2. Estimates the spread drift nu = mu1 - mu2 (price growth plus net carry) on the
   full daily sample with naive, Newey-West and block-bootstrap standard errors,
   by year.
3. Legacy: interior thresholds on the _BufferGrid / KellyGrid search, which caps
   H0 at 2 (L0 = 1.64) and so does not compare against the unlevered position.
   Kept under ``legacy_capped_thresholds`` for reference only; the paper's
   break-even drifts come from jobs/breakeven_drift.py.
4. Compares the empirical one-month barrier-crossing frequency (2024 onward,
   four-hourly path) with each calibration's model probability, with the fitted
   drift and with the price drift set to zero (carry only).
5. Recomputes the cross-section's realised spread volatility at daily sampling.

Writes results/horizon_analysis.json and results/pair_cross_section_daily.csv.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimal_long_short.calibration import calibrate_ecf  # noqa: E402
from optimal_long_short.model.drift_service import with_price_drift_view  # noqa: E402
from optimal_long_short.drift_bayes import _BufferGrid, optimal_buffer  # noqa: E402
from optimal_long_short.full_history import pair_panel  # noqa: E402
from jobs.calibrate_btc_eth import (  # noqa: E402
    CALIB_BOUNDS, EWM_DEMEAN_SPAN, ewm_demean_returns)
from optimal_long_short.utils.helpers import (  # noqa: E402
    load_calibrated_params, rate_adjusted_params)
from optimal_long_short.kelly import KellyGrid, kelly_buffer  # noqa: E402
from optimal_long_short.model.market_params import MarketParams  # noqa: E402
from optimal_long_short.model.model_params import KouParams  # noqa: E402
from optimal_long_short.model.moments import ConditionalMoments  # noqa: E402
from optimal_long_short.model.strategy import UnitExposureLongShortStrategy  # noqa: E402

HEADLINE_PARAMS = Path(__file__).resolve().parent.parent / "results" / "params_WBTC_WETH_headline.json"

T = 1.0 / 12.0
H0_GRID = (0.05, 0.07, 0.10, 0.15, 0.20)
KEYS = ["mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
        "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho"]


def calibrate(panel, n_starts=30, seed=42):
    r1 = ewm_demean_returns(panel.returns["r1"].to_numpy(), EWM_DEMEAN_SPAN)
    r2 = ewm_demean_returns(panel.returns["r2"].to_numpy(), EWM_DEMEAN_SPAN)
    res = calibrate_ecf(r1, r2, panel.dt_years, bounds=CALIB_BOUNDS,
                        n_starts=n_starts, seed=seed)
    window = panel.bars.loc[panel.returns.index[0]:panel.returns.index[-1]]
    s1, b2 = float(window["supply1"].mean()), float(window["borrow2"].mean())
    return rate_adjusted_params(res.params, s1, b2), {"supply_wbtc": s1, "borrow_weth": b2}


def sigma_z(p):
    return math.sqrt(p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2)


def sigma_z_total(p):
    def jv(lam, q, ep, en):
        return lam * 2.0 * (q * ep**2 + (1 - q) * en**2)
    return math.sqrt(sigma_z(p) ** 2 + jv(p.lam1, p.p1, p.eta1_pos, p.eta1_neg)
                     + jv(p.lam2, p.p2, p.eta2_pos, p.eta2_neg))


def p_cross(p, h0, b):
    s = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=b, S10=1.0, S20=1.0), T=T)
    return 1.0 - float(ConditionalMoments(params=p, strategy=s).p_surv())


def bisect_threshold(is_interior, lo=-0.2, hi=1.0, tol=0.005):
    if is_interior(lo):
        return None
    while not is_interior(hi):
        hi += 0.5
        if hi > 5:
            return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if is_interior(mid) else (hi, mid)
    return round(hi, 3)


def thresholds(p, b, liq):
    g = _BufferGrid(params=p, b=b, liq_bonus=liq, T=T, recovery="exact")
    out = {f"alpha={a}": bisect_threshold(lambda nu, a=a: optimal_buffer(g, nu, a)[1])
           for a in (0.5, 1.0, 2.0)}
    kg = KellyGrid(params=p, b=b, liq_bonus=liq, T=T)
    out["kelly"] = bisect_threshold(lambda nu: kelly_buffer(kg, nu)[1])
    return out


def newey_west_se(x, lags):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    v = np.dot(x, x) / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * np.dot(x[L:], x[:-L]) / n
    return math.sqrt(v / n)


def block_bootstrap_se(x, block, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    for i in range(B):
        starts = rng.integers(0, n, size=nb)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        means[i] = x[idx].mean()
    return float(means.std(ddof=1))


def empirical_crossing(bars, h0s, start="2024-01-01", step="1D"):
    """Frequency with which the realised spread falls by h0 within one month."""
    z = np.log(bars["price1"] / bars["price2"]).where(bars["fresh"]).dropna().loc[start:]
    starts = z.resample(step).first().dropna().index
    end = z.index[-1] - pd.Timedelta(days=30)
    hits = {h: [] for h in h0s}
    for t in starts[starts <= end]:
        path = z.loc[t:t + pd.Timedelta(days=30)]
        drop = path.min() - path.iloc[0]
        for h in h0s:
            hits[h].append(drop <= -h)
    return {str(h): float(np.mean(v)) for h, v in hits.items()}, len(hits[h0s[0]])


def main():
    t_start = time.time()
    out = {}
    head, market = load_calibrated_params(HEADLINE_PARAMS)
    b, liq = market["b"], market["liq_bonus"]

    # ---- panels --------------------------------------------------------------
    daily = pair_panel(freq="1D", start="2023-01-28")
    four = pair_panel(freq="4h", start="2024-01-01")
    four_all = pair_panel(freq="4h", start="2023-01-28")
    out["panels"] = {
        name: {"n_returns": len(P.returns), "dropped_stale": P.n_dropped,
               "span_years": P.span_years, "start": str(P.returns.index[0]),
               "end": str(P.returns.index[-1])}
        for name, P in (("daily_full", daily), ("4h_2024", four))
    }
    print(json.dumps(out["panels"], indent=1), flush=True)

    # ---- calibrations ------------------------------------------------------------
    cals = {"headline_4h": head}
    rates = {}
    for name, P in (("daily_full", daily), ("4h_2024", four)):
        t0 = time.time()
        cals[name], rates[name] = calibrate(P)
        print(f"calibrated {name} in {time.time() - t0:.0f}s", flush=True)
    out["calibrations"] = {
        name: {**{k: getattr(p, k) for k in KEYS}, "sigma_Z": sigma_z(p),
               "sigma_Z_total": sigma_z_total(p), "nu_ecf": p.mu1 - p.mu2,
               **rates.get(name, {})}
        for name, p in cals.items()
    }
    for name, c in out["calibrations"].items():
        print(f"{name:12s} sigma_Z={c['sigma_Z']:.3f} total={c['sigma_Z_total']:.3f} "
              f"rho={c['rho']:.3f} lam=({c['lam1']:.0f},{c['lam2']:.0f}) "
              f"eta2+={c['eta2_pos']:.4f} nu_ecf={c['nu_ecf']:+.3f}", flush=True)

    # ---- drift on the full daily sample --------------------------------------------
    r = daily.returns
    dt = daily.dt_years
    spread_ret = r["r1"] - r["r2"]
    carry = float((daily.bars["supply1"] - daily.bars["borrow2"]).mean())
    # price-growth spread: log E[e^{r1}] - log E[e^{r2}] per year, plus net carry
    nu_price = (math.log(np.mean(np.exp(r["r1"]))) - math.log(np.mean(np.exp(r["r2"])))) / dt
    nu_hat = nu_price + carry
    se_naive = spread_ret.std(ddof=1) / math.sqrt(len(r)) / dt
    se_nw = newey_west_se(spread_ret.values, 20) / dt
    se_bb = block_bootstrap_se(spread_ret.values, 20) / dt
    by_year = {}
    for y, g in r.groupby(r.index.year):
        s = g["r1"] - g["r2"]
        by_year[str(y)] = {
            "nu_price": (math.log(np.mean(np.exp(g["r1"]))) - math.log(np.mean(np.exp(g["r2"])))) / dt,
            "se_naive": s.std(ddof=1) / math.sqrt(len(s)) / dt, "n": len(s)}
    out["drift_full"] = {"nu_price": nu_price, "carry": carry, "nu_hat": nu_hat,
                         "se_naive": se_naive, "se_newey_west_20d": se_nw,
                         "se_block_bootstrap_20d": se_bb, "years": daily.span_years,
                         "by_year": by_year, "nu_ecf_daily": out["calibrations"]["daily_full"]["nu_ecf"]}
    print(f"full-sample drift nu = {nu_hat:+.3f} (price {nu_price:+.3f}, carry {carry:+.3f}); "
          f"SE naive {se_naive:.3f}, NW {se_nw:.3f}, block-bootstrap {se_bb:.3f}", flush=True)
    for y, d in by_year.items():
        print(f"   {y}: nu_price {d['nu_price']:+.3f} +- {d['se_naive']:.3f}", flush=True)

    # ---- legacy capped-grid thresholds (not the break-even drift; see breakeven_drift.py) ----
    out["legacy_capped_thresholds"] = {"note": "H0 searched only up to 2; not the break-even "
                                               "against the unlevered position. Use "
                                               "results/breakeven_drift.json."}
    for name, p in cals.items():
        t0 = time.time()
        out["legacy_capped_thresholds"][name] = thresholds(p, b, liq)
        print(f"legacy capped thresholds {name}: {out['legacy_capped_thresholds'][name]} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- tail risk under each calibration: bad debt and the overshoot correction ----
    from optimal_long_short.kelly import expected_log_wealth
    from optimal_long_short.overshoot_recovery import exact_wealth_moments
    out["tail_risk"] = {}
    for name, p in cals.items():
        rows = {}
        for H in (1.0685, 1.16, 1.30):
            h = math.log(H)
            k = expected_log_wealth(p, h, b=b, liq_bonus=liq, T=T)
            w = exact_wealth_moments(p, h, b=b, liq_bonus=liq, T=T)
            rows[str(H)] = {"p_liq": k["p_liq"], "p_bad_debt": k["p_bad_debt"],
                            "overshoot_corr": w["E_recovery"] / w["E_recovery_cc"] - 1,
                            "kelly_floor_term_1e-4": k["bad_debt_floor_term"]}
        out["tail_risk"][name] = rows
        print(f"tail {name}: " + "  ".join(
            f"H0={H}: p_liq {v['p_liq']:.3f} bad {v['p_bad_debt']:.1e} overshoot {v['overshoot_corr']:+.1%}"
            for H, v in rows.items()), flush=True)

    # ---- one-month barrier crossing: empirical vs model ------------------------------
    emp, n_win = empirical_crossing(four_all.bars, H0_GRID)
    out["crossing"] = {"empirical_2024on_4h": emp, "n_windows_daily_starts": n_win, "models": {}}
    print(f"empirical one-month crossing (2024+, {n_win} overlapping windows): {emp}", flush=True)
    for name, p in cals.items():
        carry_only = with_price_drift_view(p, delta_mu1=carry - (p.mu1 - p.mu2))
        row = {"fitted_drift": {str(h): p_cross(p, h, b) for h in H0_GRID},
               "carry_only_drift": {str(h): p_cross(carry_only, h, b) for h in H0_GRID}}
        out["crossing"]["models"][name] = row
        print(f"  {name:12s} fitted : " + "  ".join(f"{v:.3f}" for v in row["fitted_drift"].values()), flush=True)
        print(f"  {'':12s} carry  : " + "  ".join(f"{v:.3f}" for v in row["carry_only_drift"].values()), flush=True)

    # ---- cross-section sigma_Z at daily sampling ------------------------------------------
    try:
        from optimal_long_short.aave_hourly import DATA_DIR
        from optimal_long_short.screen import (ScreenConfig, apply_screen, load_panel,
                                               pair_cross_section)
        configs = pd.read_csv(DATA_DIR / "reserve_configs_v3_eth.csv", index_col=0)
        panel = load_panel(DATA_DIR)
        cfg_h, cfg_d = ScreenConfig(), ScreenConfig(vol_sampling_hours=24)
        screen = apply_screen(panel, configs, cfg_h)
        xs_h = pair_cross_section(panel, configs, screen, cfg_h)
        xs_d = pair_cross_section(panel, configs, screen, cfg_d)
        xs = xs_h[["collateral", "debt", "net_carry", "sigma_Z", "carry_over_barrier"]].copy()
        xs["sigma_Z_daily"] = xs_d["sigma_Z"].values
        xs["CB_daily"] = xs_d["carry_over_barrier"].values
        xs.to_csv(ROOT / "results/pair_cross_section_daily.csv", index=False)
        vol = xs[xs.sigma_Z >= 0.05]
        out["cross_section_daily"] = {
            "median_sigma_ratio_daily_over_hourly": float((xs.sigma_Z_daily / xs.sigma_Z).median()),
            "volatile_CB_daily_range": [float(vol.CB_daily.min()), float(vol.CB_daily.max())],
            "volatile_CB_hourly_range": [float(vol.carry_over_barrier.min()),
                                         float(vol.carry_over_barrier.max())],
            "WBTC_WETH": xs[(xs.collateral == "WBTC") & (xs.debt == "WETH")].iloc[0].to_dict(),
        }
        print("cross-section:", json.dumps(out["cross_section_daily"], default=float), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("cross-section step failed:", exc, flush=True)

    out["runtime_s"] = time.time() - t_start
    json.dump(out, open(ROOT / "results/horizon_analysis.json", "w"), indent=2, default=float)
    print("SAVED results/horizon_analysis.json", flush=True)


if __name__ == "__main__":
    main()
