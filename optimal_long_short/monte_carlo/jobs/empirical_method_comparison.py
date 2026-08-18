"""Compare semi-analytical killed moments with discretely monitored Monte Carlo."""

from __future__ import annotations

import argparse
import cmath
import csv
import math
import time
from pathlib import Path

from optimal_long_short.job_runners.common import (
    DEFAULT_EMPIRICAL_PARAMS,
    RESULTS_DIR,
    load_calibrated_params,
)
from optimal_long_short.model.kou_model import BivariateKouModel
from optimal_long_short.model.market_params import MarketParams
from optimal_long_short.model.moments import ConditionalMoments
from optimal_long_short.monte_carlo import MonteCarlo
from optimal_long_short.model.strategy import UnitExposureLongShortStrategy


DEFAULT_T = 1.0 / 12.0
DEFAULT_NO_SHORT_H0 = 100.0


def no_shorting_limit(params, T: float) -> tuple[float, float]:
    """Return the exact mean and variance of the limiting unit-long payoff."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    model = BivariateKouModel(params)
    raw1 = cmath.exp(T * model.levy_khintchine(-1j, 0.0))
    raw2 = cmath.exp(T * model.levy_khintchine(-2j, 0.0))
    if abs(raw1.imag) > 1e-10 or abs(raw2.imag) > 1e-10:
        raise ArithmeticError("No-shorting moments must be real")
    mean = float(raw1.real)
    variance = max(float(raw2.real) - mean**2, 0.0)
    return mean, variance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_EMPIRICAL_PARAMS)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "empirical_method_comparison.csv")
    parser.add_argument("--T", type=float, default=DEFAULT_T)
    parser.add_argument("--mc-paths", type=int, default=25_000)
    parser.add_argument("--mc-steps", type=int, default=1_008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-short-h0",
        type=float,
        default=DEFAULT_NO_SHORT_H0,
        help="Large finite log-health proxy used for SA/MC no-shorting validation.",
    )
    parser.add_argument(
        "--h0",
        type=float,
        action="append",
        default=None,
        help="Log-health value; repeat to override the default comparison grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.T <= 0.0:
        raise ValueError("--T must be positive")
    if args.mc_paths <= 0 or args.mc_steps <= 0:
        raise ValueError("--mc-paths and --mc-steps must be positive")

    params, constraint = load_calibrated_params(args.params)
    market = MarketParams(
        b=constraint["b"],
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
    )
    # Match the health-factor range and annotated locations used in the
    # empirical evaluation-map figure.
    h0_grid = args.h0 or [
        constraint["h0_min"],
        math.log(1.10),
        math.log(1.20),
        math.log(1.50),
        math.log(2.00),
    ]
    if args.no_short_h0 <= max(h0_grid):
        raise ValueError("--no-short-h0 must exceed every regular comparison h0")
    h0_grid = [*h0_grid, args.no_short_h0]
    if any(h0 < constraint["h0_min"] - 1e-12 for h0 in h0_grid):
        raise ValueError(f"Every h0 must be at least {constraint['h0_min']:.12f}")

    rows: list[dict[str, float | int | str]] = []
    for h0 in h0_grid:
        strategy = UnitExposureLongShortStrategy(
            h0=h0,
            market=market,
            T=args.T,
            ltv_max=constraint["ltv_max"],
        )

        moments = ConditionalMoments(params=params, strategy=strategy)
        started = time.perf_counter()
        p_surv = moments.p_surv()
        conditional_m1 = moments.killed_moment(1) / p_surv
        conditional_m2 = moments.killed_moment(2) / p_surv
        semi_runtime = time.perf_counter() - started
        semi_variance = max(conditional_m2 - conditional_m1**2, 0.0)
        rows.append(
            {
                "h0": h0,
                "H0": math.exp(h0),
                "method": "semi_analytical",
                "p_liq": min(max(1.0 - p_surv, 0.0), 1.0),
                "conditional_mean": conditional_m1,
                "conditional_variance": semi_variance,
                "runtime_seconds": semi_runtime,
                "n_paths": "",
                "n_intervals": "",
                "seed": "",
                "T": args.T,
                "talbot_M": 32,
            }
        )

        simulation = MonteCarlo(
            params=params,
            strategy=strategy,
            n_paths=args.mc_paths,
            n_steps=args.mc_steps,
            seed=args.seed,
        )
        started = time.perf_counter()
        result = simulation.run()
        mc_runtime = time.perf_counter() - started
        rows.append(
            {
                "h0": h0,
                "H0": math.exp(h0),
                "method": "monte_carlo",
                "p_liq": min(max(1.0 - result.p_surv, 0.0), 1.0),
                "conditional_mean": result.conditional_mean,
                "conditional_variance": result.conditional_variance,
                "runtime_seconds": mc_runtime,
                "n_paths": args.mc_paths,
                "n_intervals": args.mc_steps,
                "seed": args.seed,
                "T": args.T,
                "talbot_M": "",
            }
        )

        semi = rows[-2]
        mc = rows[-1]
        print(
            f"h0={h0:.6f} H0={math.exp(h0):.6f} | "
            f"semi p_liq={semi['p_liq']:.6f} mean={semi['conditional_mean']:.6f} "
            f"var={semi['conditional_variance']:.6f} t={semi_runtime:.3f}s | "
            f"MC p_liq={mc['p_liq']:.6f} mean={mc['conditional_mean']:.6f} "
            f"var={mc['conditional_variance']:.6f} t={mc_runtime:.3f}s"
        )

    limit_mean, limit_variance = no_shorting_limit(params, args.T)
    rows.append(
        {
            "h0": math.inf,
            "H0": math.inf,
            "method": "no_shorting_limit_exact",
            "p_liq": 0.0,
            "conditional_mean": limit_mean,
            "conditional_variance": limit_variance,
            "runtime_seconds": "",
            "n_paths": "",
            "n_intervals": "",
            "seed": "",
            "T": args.T,
            "talbot_M": "",
        }
    )
    print(
        "No-shorting limit | "
        f"p_liq=0 mean=exp(mu1*T)={limit_mean:.6f} "
        f"var={limit_variance:.6f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "h0",
        "H0",
        "method",
        "p_liq",
        "conditional_mean",
        "conditional_variance",
        "runtime_seconds",
        "n_paths",
        "n_intervals",
        "seed",
        "T",
        "talbot_M",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
