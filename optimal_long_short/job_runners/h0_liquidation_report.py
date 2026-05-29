"""CLI report for liquidation probabilities and moments over h0."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from optimal_long_short.drift import drift_summary
from optimal_long_short.job_runners.common import RESULTS_DIR, load_calibrated_params
from optimal_long_short.risk_report import h0_liquidation_moment_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an annualized price-drift view to calibrated WBTC/WETH "
            "parameters and report moments/liquidation probabilities over h0."
        )
    )
    parser.add_argument("--params", type=Path, default=RESULTS_DIR / "params_WBTC_WETH.json")
    parser.add_argument("--h0-min", type=float, default=None)
    parser.add_argument("--h0-max", type=float, default=2.0)
    parser.add_argument("--h0-count", type=int, default=20)
    parser.add_argument("--T", type=float, default=1.0 / 12.0)
    parser.add_argument("--mu1", type=float, default=None, help="Absolute annual price-growth drift for asset 1.")
    parser.add_argument("--mu2", type=float, default=None, help="Absolute annual price-growth drift for asset 2.")
    parser.add_argument("--delta-mu1", type=float, default=0.0, help="Additive annual price-growth drift view for asset 1.")
    parser.add_argument("--delta-mu2", type=float, default=0.0, help="Additive annual price-growth drift view for asset 2.")
    parser.add_argument("--moments", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "h0_liquidation_report.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    view = {
        "delta_mu1": args.delta_mu1,
        "delta_mu2": args.delta_mu2,
    }
    if args.mu1 is not None:
        view["mu1"] = args.mu1
    if args.mu2 is not None:
        view["mu2"] = args.mu2

    params, constraint = load_calibrated_params(args.params, price_drift_view=view)
    h0_min = constraint["h0_min"] if args.h0_min is None else args.h0_min
    h0_grid = np.linspace(h0_min, args.h0_max, args.h0_count)
    rows = h0_liquidation_moment_report(
        params,
        h0_grid,
        b=constraint["b"],
        T=args.T,
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
        max_moment_order=args.moments,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Drift convention: user views are annualized expected price-growth drifts.")
    print(json.dumps(drift_summary(params), indent=2))
    print(f"Rows: {len(rows)}")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
