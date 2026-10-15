"""Realised (model-free) backtest of leveraged Aave v3 long-short books.

For every collateral/debt pair of the cross-section (results/pair_cross_section_daily.csv)
the book is replayed hour by hour on the full Aave v3 Ethereum history in data/AAVE_full:
oracle-implied prices, the collateral's supply APR and the debt's borrow APR. Equity is
in USD and starts at 1.

* At each roll date the book is reset to initial leverage L0 = collateral / equity
  (L0 = 1 is the unlevered benchmark: hold the collateral and earn its supply rate).
  ``max`` opens at the reserve's maximum origination LTV.  A cost of ``cost_bps`` is
  charged on the debt notional that the reset trades.
* Interest accrues every hour on both legs.  If the health factor b n1 S1 / (n2 S2)
  falls below one at an hourly mark, the book is liquidated:
    - ``aave``: a keeper repays 50% of the debt (100% if HF <= 0.95) and seizes
      repay * (1 + bonus) of collateral, repeated until HF >= 1; the remaining book
      runs on to the next roll (Aave v3 close-factor rules);
    - ``full``: the paper's convention -- the whole debt is repaid at the first
      crossing and the residual collateral is held unlevered to the next roll.
  If the collateral cannot cover the debt plus bonus the borrower is wiped out
  (non-recourse; equity 0, i.e. ruin).
* Roll horizons 1, 7, 30, 90, 365 days and buy-and-hold to the end of the window.
  Rolled books are averaged over four evenly spaced roll phases; buy-and-hold over
  four start dates 120 days apart (a single start date is one path).

Reported per pair, leverage and horizon: annualised log growth, excess over the
unlevered benchmark with a Newey--West standard error (daily excess log returns,
60-day Bartlett lag), number of liquidation events and ruin.  Also:
  * calendar-year excess growth (short-run vs long-run),
  * an ex-ante rule: lever to 2x for the next month only if the trailing 365-day
    realised spread growth (price plus carry) was positive.

Caveats: hourly marks miss intra-hour crossings; forward-filled (stale) hours carry
the last real oracle price; reserve parameters (b, LTV, bonus) are today's, not the
historical ones; gas and swap slippage enter only through ``cost_bps``.

Writes results/realised_backtest.json and results/realised_backtest_pairs.csv.
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from optimal_long_short.full_history import hourly_leg  # noqa: E402

DT = 1.0 / 8760.0
TOKENS = {"USDC": "usdc", "USDT": "usdt", "USDe": "usde", "WBTC": "wbtc", "WETH": "weth",
          "cbBTC": "cbbtc", "weETH": "weeth", "wstETH": "wsteth"}
CLASS = {"USDC": "USD", "USDT": "USD", "USDe": "USD", "WBTC": "BTC", "cbBTC": "BTC",
         "WETH": "ETH", "weETH": "ETH", "wstETH": "ETH"}
LEVERAGES = [1.25, 1.5, 2.0, 2.5, 3.0, "max"]
HORIZONS = [1, 7, 30, 90, 365, "hold"]
N_PHASES = 4
NW_LAG_DAYS = 60
HOLD_STAGGER_DAYS = 120
COSTS_BPS = [0.0, 10.0]


def group(c, d):
    if CLASS[c] == CLASS[d] == "USD":
        return "stablecoin"
    return "same-underlying" if CLASS[c] == CLASS[d] else "cross-asset"


@numba.njit(cache=True)
def simulate(S1, S2, s1, r2, roll, L, b, bonus, full_mode, cost):
    """Hourly replay; returns equity path, liquidation events, ruin flag."""
    n = S1.shape[0]
    eq = np.empty(n)
    n1 = 0.0
    n2 = 0.0
    E = 1.0
    nliq = 0
    ruined = False
    for t in range(n):
        if t > 0:
            n1 *= math.exp(s1[t] * DT)
            n2 *= math.exp(r2[t] * DT)
        if ruined:
            eq[t] = 0.0
            continue
        if roll[t]:
            E = n1 * S1[t] - n2 * S2[t] if t > 0 else 1.0
            d_new = (L - 1.0) * E
            E -= cost * abs(d_new - n2 * S2[t])
            n1 = L * E / S1[t]
            n2 = (L - 1.0) * E / S2[t]
        it = 0
        while n2 > 0.0 and b * n1 * S1[t] < n2 * S2[t] and it < 50:
            hf = b * n1 * S1[t] / (n2 * S2[t])
            frac = 1.0 if (full_mode or hf <= 0.95) else 0.5
            repay = frac * n2 * S2[t]
            seize = repay * (1.0 + bonus) / S1[t]
            nliq += 1
            if seize >= n1:          # collateral exhausted: bad debt, borrower wiped out
                n1 = 0.0
                n2 = 0.0
                ruined = True
                break
            n1 -= seize
            n2 -= repay / S2[t]
            if n2 * S2[t] < 1e-12:
                n2 = 0.0
            it += 1
        eq[t] = 0.0 if ruined else n1 * S1[t] - n2 * S2[t]
    return eq, nliq, ruined


def load_pair(c, d):
    a, z = hourly_leg(TOKENS[c]), hourly_leg(TOKENS[d])
    df = pd.DataFrame({"S1": a["price"].where(a["fresh"].astype(bool)),
                       "S2": z["price"].where(z["fresh"].astype(bool)),
                       "s1": a["supply_apr"], "r2": z["borrow_apr"]})
    df = df[~df.index.duplicated()].sort_index()
    last = min(df["S1"].last_valid_index(), df["S2"].last_valid_index())
    # forward-fill the last real oracle price before trimming, then start at the
    # first hour where both legs have one
    df = df.ffill().loc[:last].dropna(subset=["S1", "S2"])
    df = df.asfreq("1h").ffill().fillna({"s1": 0.0, "r2": 0.0})
    return df


def roll_mask(n, hours_per_roll, offset):
    m = np.zeros(n, dtype=np.bool_)
    m[0] = True
    if hours_per_roll:
        m[offset::hours_per_roll] = True
    return m


def daily_log(eq, index):
    s = pd.Series(eq, index=index).resample("1D").last()
    with np.errstate(divide="ignore"):
        return np.log(s.values)


def nw_se(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n
    return math.sqrt(max(v, 0.0) / n)


def run_pair(args):
    c, d, cfg = args
    df = load_pair(c, d)
    S1, S2 = df["S1"].values, df["S2"].values
    s1, r2 = df["s1"].values, df["r2"].values
    n = len(df)
    years = (n - 1) * DT
    lmax = (1.0 - 1e-6) / (1.0 - cfg["ltv"])
    out = {"collateral": c, "debt": d, "group": group(c, d), "start": str(df.index[0]),
           "end": str(df.index[-1]), "years": years, "b": cfg["b"], "ltv": cfg["ltv"],
           "bonus": cfg["bonus"], "L_max": lmax, "results": []}
    for cost_bps in COSTS_BPS:
        cost = cost_bps * 1e-4
        for mode in ("aave", "full"):
            if mode == "full" and cost_bps:
                continue
            for H in HORIZONS:
                hpr = 0 if H == "hold" else int(H) * 24
                if H == "hold":     # start dates staggered by HOLD_STAGGER_DAYS, held to the end
                    phases = [k * HOLD_STAGGER_DAYS * 24 for k in range(N_PHASES)]
                elif H == 1:
                    phases = [0]
                else:
                    phases = [int(k * hpr / N_PHASES) for k in range(N_PHASES)]
                base, yrs = {}, {}
                for ph in phases:
                    sl = slice(ph if H == "hold" else 0, n)
                    yrs[ph] = (n - sl.start - 1) * DT
                    m = roll_mask(n - sl.start, hpr, 0 if H == "hold" else ph)
                    base[ph] = simulate(S1[sl], S2[sl], s1[sl], r2[sl], m, 1.0, cfg["b"], cfg["bonus"],
                                        False, cost)[0]
                for Lspec in LEVERAGES:
                    L = lmax if Lspec == "max" else float(Lspec)
                    if L > lmax:
                        continue
                    g, gx, nl, ru, se = [], [], 0, 0, None
                    for ph in phases:
                        sl = slice(ph if H == "hold" else 0, n)
                        m = roll_mask(n - sl.start, hpr, 0 if H == "hold" else ph)
                        eq, nliq, ruined = simulate(S1[sl], S2[sl], s1[sl], r2[sl], m, L, cfg["b"],
                                                    cfg["bonus"], mode == "full", cost)
                        nl += nliq
                        ru += int(ruined)
                        gl = math.log(eq[-1]) / yrs[ph] if eq[-1] > 0 else -math.inf
                        gb = math.log(base[ph][-1]) / yrs[ph]
                        g.append(gl)
                        gx.append(gl - gb)
                        if ph == phases[0] and not ruined:
                            dl = np.diff(daily_log(eq, df.index[sl]) - daily_log(base[ph], df.index[sl]))
                            dl = dl[np.isfinite(dl)]
                            se = nw_se(dl, NW_LAG_DAYS) * 365.0
                    out["results"].append({
                        "mode": mode, "cost_bps": cost_bps, "horizon": H, "L0": Lspec, "L": L,
                        "growth": float(np.mean(g)), "excess": float(np.mean(gx)),
                        "excess_phases": [float(x) for x in gx], "se_nw": se,
                        "t": (float(np.mean(gx)) / se) if (se and np.all(np.isfinite(gx))) else None,
                        "liquidations": nl / len(phases), "ruined_phases": ru,
                        "unlevered_growth": float(np.mean([math.log(base[p][-1]) / yrs[p] for p in phases])),
                    })
    # calendar-year excess growth, L=2, 30-day rolls (phase 0), Aave mechanics, no cost
    m = roll_mask(n, 30 * 24, 0)
    eq2 = simulate(S1, S2, s1, r2, m, min(2.0, lmax), cfg["b"], cfg["bonus"], False, 0.0)[0]
    eq1 = simulate(S1, S2, s1, r2, m, 1.0, cfg["b"], cfg["bonus"], False, 0.0)[0]
    s2 = pd.Series(eq2, df.index)
    s1e = pd.Series(eq1, df.index)
    by_year = {}
    for y in sorted(set(df.index.year)):
        a2, a1 = s2[s2.index.year == y], s1e[s1e.index.year == y]
        span = (a2.index[-1] - a2.index[0]).total_seconds() / (365 * 86400)
        if span < 0.25 or a2.iloc[0] <= 0:
            continue
        x2 = math.log(a2.iloc[-1] / a2.iloc[0]) / span if a2.iloc[-1] > 0 else -math.inf
        by_year[str(y)] = {"span": span, "excess": x2 - math.log(a1.iloc[-1] / a1.iloc[0]) / span}
    out["by_year_L2_30d"] = by_year
    out["ex_ante_rule"] = ex_ante_rule(df, cfg, lmax)
    out["realised_breakeven"] = realised_breakeven(df)
    return out


def realised_breakeven(df):
    """Model-free counterpart of the Kelly break-even, from daily gross returns.

    With R1 the collateral's gross daily return (price times supply accrual) and R2
    the debt's (price times borrow accrual), the marginal log growth of the first unit
    of debt in a daily-rolled book is E[1 - R2/R1]; leverage pays at the margin iff
    E[R2/R1] < 1.  Annualised: margin = -365 log mean(R2/R1) = nu - nu_star with
    nu = 365 log(mean R1 / mean R2) and nu_star = margin's complement.
    """
    d = df.resample("1D").agg({"S1": "last", "S2": "last", "s1": "mean", "r2": "mean"}).dropna()
    R1 = (d["S1"] / d["S1"].shift()) * np.exp(d["s1"] / 365.0)
    R2 = (d["S2"] / d["S2"].shift()) * np.exp(d["r2"] / 365.0)
    R1, R2 = R1.iloc[1:].values, R2.iloc[1:].values
    margin = -365.0 * math.log(np.mean(R2 / R1))
    nu = 365.0 * math.log(np.mean(R1) / np.mean(R2))
    x1, x2 = np.log(R1), np.log(R2)
    return {"nu_hat": nu, "nu_star_hat": nu - margin, "margin": margin,
            "var1_minus_cov": float(365.0 * (np.var(x1) - np.cov(x1, x2)[0, 1])),
            "se_margin_nw": nw_se(1.0 - R2 / R1, NW_LAG_DAYS) * 365.0}


def ex_ante_rule(df, cfg, lmax, L=2.0, lookback_days=365, roll_days=30):
    """Lever to L for the next roll only if trailing spread growth (price + carry) > 0."""
    L = min(L, lmax)
    S1, S2, s1, r2 = (df[k].values for k in ("S1", "S2", "s1", "r2"))
    n, lb, hpr = len(df), lookback_days * 24, roll_days * 24
    if n < lb + 4 * hpr:
        return None
    spread = np.log(S1 / S2) + np.cumsum(s1 - r2) * DT
    sl = slice(lb, n)
    S1, S2, s1, r2, spread_t = S1[sl], S2[sl], s1[sl], r2[sl], spread
    m = roll_mask(len(S1), hpr, 0)
    rolls = np.flatnonzero(m)
    signal = np.array([spread_t[lb + k] - spread_t[k] > 0 for k in rolls])
    # replay: leverage is L on rolls with a positive signal, 1 otherwise
    eqs = {}
    for name, Ls in (("rule", np.where(signal, L, 1.0)), ("always", np.full(len(rolls), L)),
                     ("unlevered", np.ones(len(rolls)))):
        eq = np.empty(len(S1))
        E = 1.0
        for i, r0 in enumerate(rolls):
            r1 = rolls[i + 1] if i + 1 < len(rolls) else len(S1)
            seg = simulate(S1[r0:r1] * 1.0, S2[r0:r1] * 1.0, s1[r0:r1], r2[r0:r1],
                           roll_mask(r1 - r0, 0, 0), Ls[i], cfg["b"], cfg["bonus"], False, 0.0)[0]
            eq[r0:r1] = E * seg
            E = eq[r1 - 1]
        eqs[name] = eq
    years = (len(S1) - 1) * DT
    g = {k: (math.log(v[-1]) / years if v[-1] > 0 else -math.inf) for k, v in eqs.items()}
    return {"years": years, "share_levered": float(signal.mean()), "growth": g,
            "rule_minus_unlevered": g["rule"] - g["unlevered"],
            "always_minus_unlevered": g["always"] - g["unlevered"]}


def summarise(pairs):
    rows = []
    for p in pairs:
        for r in p["results"]:
            rows.append({"collateral": p["collateral"], "debt": p["debt"], "group": p["group"],
                         "years": p["years"], **{k: r[k] for k in ("mode", "cost_bps", "horizon",
                         "L0", "L", "growth", "excess", "se_nw", "t", "liquidations",
                         "ruined_phases", "unlevered_growth")}})
    df = pd.DataFrame(rows)
    df["horizon"] = df["horizon"].astype(str)
    df["L0"] = df["L0"].astype(str)
    base = df[(df["mode"] == "aave") & (df["cost_bps"] == 0)]
    summ = {}
    for (gname, H, L0), sub in base.groupby(["group", "horizon", "L0"]):
        ex = sub["excess"].replace(-np.inf, np.nan)
        summ[f"{gname}|{H}|{L0}"] = {
            "n_pairs": int(len(sub)), "median_excess": float(np.nanmedian(ex)) if ex.notna().any() else None,
            "share_positive": float((sub["excess"] > 0).mean()),
            "n_sig_pos": int((sub["t"] > 2).sum()), "n_sig_neg": int((sub["t"] < -2).sum()),
            "n_ruined": int((sub["ruined_phases"] > 0).sum())}
    return df, summ


def main():
    cs = pd.read_csv(ROOT / "results/pair_cross_section_daily.csv")
    cfgs = pd.read_csv(ROOT / "data/AAVE_full/reserve_configs_v3_eth.csv").set_index("symbol")
    tasks = []
    for _, r in cs.iterrows():
        c, d = r["collateral"], r["debt"]
        row = cfgs.loc[c]
        tasks.append((c, d, {"b": float(row["reserveLiquidationThreshold"]),
                             "ltv": float(row["baseLTVasCollateral"]),
                             "bonus": float(row["liquidationBonus"])}))
    t0 = time.time()
    simulate(np.ones(3), np.ones(3), np.zeros(3), np.zeros(3), np.ones(3, np.bool_),
             1.0, 0.8, 0.05, False, 0.0)          # compile once before forking
    with Pool() as pool:
        pairs = []
        for k, res in enumerate(pool.imap_unordered(run_pair, tasks), 1):
            pairs.append(res)
            print(f"{k:2d}/{len(tasks)} {res['collateral']}/{res['debt']} ({time.time() - t0:.0f}s)",
                  flush=True)
    pairs.sort(key=lambda p: (p["collateral"], p["debt"]))
    df, summ = summarise(pairs)
    df.to_csv(ROOT / "results/realised_backtest_pairs.csv", index=False)
    json.dump({"config": {"leverages": [str(x) for x in LEVERAGES], "horizons": [str(h) for h in HORIZONS],
                          "n_phases": N_PHASES, "nw_lag_days": NW_LAG_DAYS, "costs_bps": COSTS_BPS},
               "summary": summ, "pairs": pairs},
              open(ROOT / "results/realised_backtest.json", "w"), indent=1, default=float)
    print("SAVED results/realised_backtest.json, results/realised_backtest_pairs.csv")


if __name__ == "__main__":
    main()
