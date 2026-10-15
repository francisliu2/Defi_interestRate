"""Out-of-sample backtest of one-month liquidation (barrier-crossing) probabilities.

For each calendar month from 2024 onward the WBTC/WETH spread model is
calibrated only on data before the month, predicts P(spread falls by h0 within
the next 30 days) for several buffers, and is scored against what happened.

Predictors
  daily_12m : Kou, daily bars, trailing 12 months (headline pipeline, 8 starts)
  4h_6m     : Kou, four-hourly bars, trailing 6 months
  gbm_30d   : driftless Brownian spread, trailing 30-day daily realised vol
Each Kou predictor is scored with its fitted drift and with the price drift set
to zero (window-average net carry only), separating the variance effect from
the drift effect.

Realised outcome: the four-hourly last-real-price path of log(WBTC/WETH) over the
30 days (discrete monitoring, so model probabilities, which monitor continuously,
are slightly conservative).

Writes results/rolling_backtest.json.
Usage: python jobs/rolling_backtest.py [workers=7]
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimal_long_short.calibration import calibrate_ecf  # noqa: E402
from optimal_long_short.drift import with_price_drift_view  # noqa: E402
from optimal_long_short.full_history import pair_panel  # noqa: E402
from optimal_long_short.job_runners.calibrate_btc_eth import (  # noqa: E402
    AAVE_B, CALIB_BOUNDS, EWM_DEMEAN_SPAN, ewm_demean_returns)
from optimal_long_short.job_runners.common import rate_adjusted_params  # noqa: E402
from optimal_long_short.market_params import MarketParams  # noqa: E402
from optimal_long_short.moments import ConditionalMoments  # noqa: E402
from optimal_long_short.strategy import UnitExposureLongShortStrategy  # noqa: E402

T = 1.0 / 12.0
H0_GRID = (0.03, 0.05, 0.07, 0.10, 0.15)
DAILY = pair_panel(freq="1D", start="2023-01-28")
FOUR = pair_panel(freq="4h", start="2023-01-28")
Z4 = np.log(FOUR.bars["price1"] / FOUR.bars["price2"]).where(FOUR.bars["fresh"]).dropna()
MONTHS = [m for m in pd.date_range("2024-01-01", "2026-12-01", freq="MS")
          if m + pd.Timedelta(days=30) <= Z4.index[-1]]


def _window(panel, start, end):
    r = panel.returns.loc[start:end - pd.Timedelta(seconds=1)]
    bars = panel.bars.loc[start:end - pd.Timedelta(seconds=1)]
    return r, bars


def _calibrate(r, bars, dt, seed):
    r1 = ewm_demean_returns(r["r1"].to_numpy(), EWM_DEMEAN_SPAN)
    r2 = ewm_demean_returns(r["r2"].to_numpy(), EWM_DEMEAN_SPAN)
    res = calibrate_ecf(r1, r2, dt, bounds=CALIB_BOUNDS, n_starts=8, seed=seed)
    s1, b2 = float(bars["supply1"].mean()), float(bars["borrow2"].mean())
    return rate_adjusted_params(res.params, s1, b2), s1 - b2


def _p_cross(p, h0):
    s = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=AAVE_B, S10=1.0, S20=1.0), T=T)
    return 1.0 - float(ConditionalMoments(params=p, strategy=s).p_surv())


def _brownian_cross(h, mu, sig, t):
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    st = sig * math.sqrt(t)
    surv = phi((h + mu * t) / st) - math.exp(-2.0 * mu * h / sig**2) * phi((-h + mu * t) / st)
    return 1.0 - surv


def realised(month):
    path = Z4.loc[month:month + pd.Timedelta(days=30)]
    drop = path.min() - path.iloc[0]
    return {str(h): bool(drop <= -h) for h in H0_GRID}


def task(args):
    scheme, month = args
    t0 = time.time()
    try:
        if scheme == "daily_12m":
            r, bars = _window(DAILY, month - pd.DateOffset(months=12), month)
            dt = DAILY.dt_years
        elif scheme == "4h_6m":
            r, bars = _window(FOUR, month - pd.DateOffset(months=6), month)
            dt = FOUR.dt_years
        else:  # gbm_30d
            r, _ = _window(DAILY, month - pd.Timedelta(days=30), month)
            sig = float((r["r1"] - r["r2"]).std(ddof=1) / math.sqrt(DAILY.dt_years))
            pred = {str(h): _brownian_cross(h, 0.0, sig, T) for h in H0_GRID}
            return {"scheme": scheme, "month": str(month.date()), "n_obs": len(r),
                    "sigma_spread": sig, "pred_fitted": pred, "pred_carry_only": pred,
                    "secs": time.time() - t0}
        p, carry = _calibrate(r, bars, dt, seed=int(month.strftime("%Y%m")))
        p0 = with_price_drift_view(p, delta_mu1=carry - (p.mu1 - p.mu2))
        sz = math.sqrt(p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2)
        return {"scheme": scheme, "month": str(month.date()), "n_obs": len(r),
                "nu_fitted": p.mu1 - p.mu2, "carry": carry, "sigma_Z": sz,
                "lam": [p.lam1, p.lam2],
                "pred_fitted": {str(h): _p_cross(p, h) for h in H0_GRID},
                "pred_carry_only": {str(h): _p_cross(p0, h) for h in H0_GRID},
                "secs": time.time() - t0}
    except Exception as exc:  # noqa: BLE001
        return {"scheme": scheme, "month": str(month.date()), "error": repr(exc)}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    jobs = [(s, m) for s in ("daily_12m", "4h_6m", "gbm_30d") for m in MONTHS
            if not (s == "4h_6m" and m < pd.Timestamp("2023-08-01"))]
    print(f"{len(MONTHS)} months {MONTHS[0].date()}..{MONTHS[-1].date()}, {len(jobs)} tasks", flush=True)
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(task, jobs), 1):
            rows.append(row)
            if "error" in row:
                print("  ERROR", row, flush=True)
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    outcomes = {str(m.date()): realised(m) for m in MONTHS}

    summary = {}
    for scheme in ("daily_12m", "4h_6m", "gbm_30d"):
        rs = [r for r in rows if r["scheme"] == scheme and "error" not in r]
        for var in ("pred_fitted", "pred_carry_only"):
            if scheme == "gbm_30d" and var == "pred_carry_only":
                continue
            tab = {}
            for h in map(str, H0_GRID):
                pred = np.array([r[var][h] for r in rs])
                obs = np.array([outcomes[r["month"]][h] for r in rs], float)
                tab[h] = {"mean_pred": float(pred.mean()), "realised_freq": float(obs.mean()),
                          "brier": float(np.mean((pred - obs) ** 2)),
                          "brier_climatology": float(np.mean((obs.mean() - obs) ** 2)),
                          "n_months": int(len(rs))}
            summary[f"{scheme}/{var}"] = tab
    for k, tab in summary.items():
        print(f"{k:28s} " + "  ".join(
            f"h0={h}: pred {v['mean_pred']:.3f} vs obs {v['realised_freq']:.3f} (Brier {v['brier']:.3f})"
            for h, v in tab.items() if h in ("0.05", "0.1")), flush=True)
    json.dump({"h0_grid": H0_GRID, "months": [str(m.date()) for m in MONTHS],
               "outcomes": outcomes, "rows": sorted(rows, key=lambda r: (r["scheme"], r["month"])),
               "summary": summary, "runtime_s": time.time() - t0},
              open(ROOT / "results/rolling_backtest.json", "w"), indent=2, default=float)
    print("SAVED results/rolling_backtest.json", flush=True)


if __name__ == "__main__":
    main()
