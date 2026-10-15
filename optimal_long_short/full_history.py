"""Pair panels from the full Aave v3 hourly history, with explicit staleness.

The Messari-schema hourly snapshots exist only for hours with market activity;
the fetcher forward-fills the gaps and records ``n_snapshots`` (0 = filled).  In
thin periods (WBTC in 2023: up to 44% of hours empty, gaps up to 17 h) a
forward-filled price produces a spurious zero return followed by a catch-up
jump, which biases volatility down and jump frequency up.  The panel builder
therefore marks a bar *fresh* only if both legs had at least one real snapshot
in it, and keeps a return only when both of its end bars are fresh.

Prices are oracle-implied, ``TVL_USD / (token_balance / 10**decimals)``, and each
bar takes the last hourly value in the bin (as ``jobs/build_merged_4h.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_long_short.aave_hourly import REPO_ROOT

FULL_DIR = REPO_ROOT / "data" / "AAVE_full"
DECIMALS = {"wbtc": 8, "weth": 18, "usdc": 6, "usdt": 6, "dai": 18, "wsteth": 18,
            "cbbtc": 8, "usde": 18, "weeth": 18, "link": 18, "gho": 18, "tbtc": 18}


def hourly_leg(token: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Hourly oracle-implied price, freshness flag and APRs for one reserve."""
    h = pd.read_parquet((data_dir or FULL_DIR) / f"hourly_{token}.parquet")
    h.index = h.index.tz_convert(None)
    bal = h["token_balance"].astype(float) / 10.0 ** DECIMALS[token]
    out = pd.DataFrame({
        "price": (h["TVL_USD"] / bal.where(bal > 0)).astype(float),
        "fresh": h["n_snapshots"] > 0 if "n_snapshots" in h else True,
        "supply_apr": h["lender_variable_apr"].astype(float) / 100.0,
        "borrow_apr": h["borrower_variable_apr"].astype(float) / 100.0,
    })
    return out.replace([np.inf, -np.inf], np.nan)


@dataclass
class PairPanel:
    """Bars of a collateral/debt pair at one sampling interval."""

    bars: pd.DataFrame          # price1, price2 (last real price), fresh, supply1, borrow2
    returns: pd.DataFrame       # r1, r2 log returns between consecutive fresh bars
    dt_years: float
    n_bars: int
    n_dropped: int              # returns discarded because an end bar was stale

    @property
    def span_years(self) -> float:
        idx = self.returns.index
        return (idx[-1] - idx[0]).total_seconds() / (365.0 * 24 * 3600)


def pair_panel(
    collateral: str = "wbtc",
    debt: str = "weth",
    *,
    freq: str = "1D",
    start: str | None = None,
    end: str | None = None,
    data_dir: Path | None = None,
) -> PairPanel:
    """Resample a pair to ``freq`` bars and keep only fresh-to-fresh returns."""
    a, b = hourly_leg(collateral, data_dir), hourly_leg(debt, data_dir)
    hourly = pd.DataFrame({
        # stale (forward-filled) hours are blanked per leg, so each bar takes the
        # last REAL price inside it
        "price1": a["price"].where(a["fresh"]), "price2": b["price"].where(b["fresh"]),
        "supply1": a["supply_apr"], "borrow2": b["borrow_apr"],
    }).loc[start:end]
    bars = pd.DataFrame({
        "price1": hourly["price1"].resample(freq).last(),
        "price2": hourly["price2"].resample(freq).last(),
        "supply1": hourly["supply1"].resample(freq).mean(),
        "borrow2": hourly["borrow2"].resample(freq).mean(),
    })
    # a bar is fresh if both legs had at least one real snapshot in it
    bars["fresh"] = bars["price1"].notna() & bars["price2"].notna()
    bars[["price1", "price2"]] = bars[["price1", "price2"]].ffill()
    bars = bars.dropna(subset=["price1", "price2"])
    lp1, lp2 = np.log(bars["price1"]), np.log(bars["price2"])
    rets = pd.DataFrame({"r1": lp1.diff(), "r2": lp2.diff()})
    ok = bars["fresh"] & bars["fresh"].shift(1, fill_value=False)
    n_drop = int((~ok).sum() - 1)
    rets = rets[ok].dropna()
    dt = pd.Timedelta(freq).total_seconds() / (365.0 * 24 * 3600)
    return PairPanel(bars=bars, returns=rets, dt_years=dt, n_bars=len(bars),
                     n_dropped=max(n_drop, 0))


def variance_ratios(log_spread: pd.Series, base: str, horizons: dict[str, str]) -> dict[str, float]:
    """``Var(h-period change) / (k * Var(base change))`` on a regular-grid series."""
    s = log_spread.dropna()
    v_base = s.resample(base).last().diff().var()
    out = {}
    for name, h in horizons.items():
        k = pd.Timedelta(h) / pd.Timedelta(base)
        sh = s.resample(base).last()
        out[name] = float(sh.diff(int(round(k))).var() / (k * v_base))
    return out
