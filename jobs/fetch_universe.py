"""Fetch Aave v3 Ethereum hourly history and reserve configs for the universe.

Thin entry point.  Pulls generously (from 2025-01-01) so the canonical
estimation window 2026-03-17 -- 2026-08-04 has both a burn-in period and
out-of-window data available for validation.

    python jobs/fetch_universe.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimal_long_short.aave_hourly import (  # noqa: E402
    DATA_DIR,
    RESERVE_TOKENS,
    fetch_reserve_configs,
    fetch_universe,
)

HISTORY_START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch Aave v3 Ethereum hourly history.")
    ap.add_argument("--start", default=f"{HISTORY_START:%Y-%m-%d}",
                    help="first date to fetch (Aave v3 Ethereum launched 2023-01-27)")
    ap.add_argument("--out-dir", default=str(DATA_DIR),
                    help="output folder (default data/AAVE; use a new folder to keep the "
                         "headline data untouched)")
    args = ap.parse_args()
    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching reserve configs ...")
    configs = fetch_reserve_configs()
    configs.to_csv(out_dir / "reserve_configs_v3_eth.csv")
    print(f"  {len(configs)} reserves -> reserve_configs_v3_eth.csv")

    print(f"Fetching hourly history from {start:%Y-%m-%d} into {out_dir} ...")
    frames = fetch_universe(
        RESERVE_TOKENS,
        start_ts=int(start.timestamp()),
        out_dir=out_dir,
    )
    print(f"\nFetched {len(frames)}/{len(RESERVE_TOKENS)} reserves.")
    for token, frame in frames.items():
        print(
            f"  {token:>7} {len(frame):6d} hours  "
            f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}"
        )


if __name__ == "__main__":
    main()
