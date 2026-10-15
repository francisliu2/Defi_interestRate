"""Build ``data/AAVE/merged_wbtc_weth_4h.parquet``, the headline WBTC/WETH dataset.

This file is the input to the headline calibration (``results/params_WBTC_WETH.json``),
the bootstrap (``jobs/bootstrap_se.py``) and the model comparison
(``jobs/model_comparison.py``).  It is derived from the hourly reserve snapshots
written by :mod:`optimal_long_short.aave_hourly`:

* prices are oracle-implied USD values, ``TVL_USD / (token_balance / 10**decimals)``,
  computed here from TVL and balance so the script works on hourly files written
  before the ``price_usd`` scaling fix;
* 4-hour bars take the **last** hourly value in each bin (pandas ``resample("4h").last()``);
* APR columns (percent) are 4-hour **means** of the Messari variable rates;
* ``block`` is a 0..N-1 row counter, kept for compatibility with the original file.

With the default window (2026-03-17 16:00 to 2026-08-04 16:00 UTC) the prices
reproduce the stored file exactly; APRs agree to within 1e-3 percentage points
(subgraph revisions since the original fetch).

Usage: python jobs/build_merged_4h.py [--start ...] [--end ...] [--out PATH]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "AAVE"

LEGS = [("wbtc", "btc", 8), ("weth", "eth", 18)]


def build(start: str, end: str) -> pd.DataFrame:
    cols = {}
    for fname, sfx, decimals in LEGS:
        h = pd.read_parquet(DATA / f"hourly_{fname}.parquet")
        h.index = h.index.tz_convert(None)
        h = h.loc[:end]
        price = h["TVL_USD"] / (h["token_balance"].astype(float) / 10.0**decimals)
        cols[f"close_{sfx}"] = price.resample("4h").last()
        cols[f"supply_apr_{sfx}"] = h["lender_variable_apr"].resample("4h").mean()
        cols[f"variable_borrow_apr_{sfx}"] = h["borrower_variable_apr"].resample("4h").mean()
    order = ["close_btc", "close_eth", "supply_apr_btc", "variable_borrow_apr_btc",
             "supply_apr_eth", "variable_borrow_apr_eth"]
    df = pd.DataFrame(cols)[order].loc[start:end].dropna()
    df["block"] = range(len(df))
    df.index.name = "datetime"
    return df.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2026-03-17 16:00:00")
    ap.add_argument("--end", default="2026-08-04 16:00:00")
    ap.add_argument("--out", default=str(DATA / "merged_wbtc_weth_4h.parquet"))
    args = ap.parse_args()
    df = build(args.start, args.end)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} rows, {df.datetime.iloc[0]} -> {df.datetime.iloc[-1]}")
