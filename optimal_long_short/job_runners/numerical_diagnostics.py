"""CLI implementation for baseline numerical robustness diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from optimal_long_short.job_runners.common import (
    DEFAULT_EMPIRICAL_PARAMS,
    RESULTS_DIR,
    load_calibrated_params,
)
from optimal_long_short.market_params import MarketParams
from optimal_long_short.numerical_diagnostics import (
    MonteCarloMonitoringDiagnostic,
    RootQualityDiagnostic,
    TalbotConvergenceDiagnostic,
    monte_carlo_monitoring_diagnostics,
    root_quality_diagnostics,
    talbot_convergence_diagnostics,
    write_numerical_diagnostics_csv,
)
from optimal_long_short.strategy import UnitExposureLongShortStrategy


def _integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item < 2 for item in parsed):
        raise argparse.ArgumentTypeError("all Talbot orders must be at least 2")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report Talbot-order convergence and root/barrier-system quality "
            "for the baseline calibrated Kou model."
        )
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_EMPIRICAL_PARAMS,
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=None,
        help="Initial log health buffer; defaults to the calibrated feasible minimum.",
    )
    parser.add_argument("--T", type=float, default=1.0 / 12.0)
    parser.add_argument(
        "--M-values",
        type=_integer_list,
        default=(12, 16, 24, 32, 40),
        help="Comma-separated Talbot orders for convergence (default: 12,16,24,32,40).",
    )
    parser.add_argument(
        "--root-M",
        type=int,
        default=32,
        help="Talbot order whose nodes are used for root diagnostics.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "numerical_diagnostics.csv",
    )
    parser.add_argument(
        "--mc-h0",
        type=float,
        default=0.10,
        help="Initial log health buffer for monitoring convergence (default: 0.10).",
    )
    parser.add_argument("--mc-paths", type=int, default=25_000)
    parser.add_argument(
        "--mc-intervals",
        type=_integer_list,
        default=(63, 252, 1008),
        help="Comma-separated monitoring intervals over T (default: 63,252,1008).",
    )
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--mc-laplace-M", type=int, default=32)
    return parser.parse_args()


def _print_convergence(rows: list[TalbotConvergenceDiagnostic]) -> None:
    by_M: dict[int, dict[tuple[str, int], TalbotConvergenceDiagnostic]] = {}
    for row in rows:
        by_M.setdefault(row.M, {})[(row.quantity, row.k)] = row

    print("\nTalbot-order convergence (killed, unconditional payoff moments)")
    print(f"{'M':>4} {'p_surv':>13} {'M1':>13} {'M2':>13} {'M3':>13} {'M4':>13}")
    for M, values in sorted(by_M.items()):
        output = [values[("p_surv", 0)].value]
        output.extend(values[("killed_moment", k)].value for k in range(1, 5))
        print(f"{M:4d}" + "".join(f" {value:13.8g}" for value in output))


def _print_root_quality(rows: list[RootQualityDiagnostic]) -> None:
    print("\nRoot and barrier-system diagnostics over Talbot nodes")
    print(
        f"{'k':>2} {'max rel root':>14} {'min root sep':>14} "
        f"{'min pole dist':>14} {'max cond':>12} {'max solve res':>14}"
    )
    for row in rows:
        print(
            f"{row.k:2d} {row.max_relative_root_residual:14.3e} "
            f"{row.min_root_separation:14.3e} "
            f"{row.min_root_pole_distance:14.3e} "
            f"{row.max_barrier_condition_number:12.3e} "
            f"{row.max_barrier_system_residual:14.3e}"
        )


def _print_mc_monitoring(rows: list[MonteCarloMonitoringDiagnostic]) -> None:
    print("\nDiscretely monitored Monte Carlo convergence")
    print("Intervals are counts over the full horizon T, not intervals per year.")
    print(
        f"{'intervals':>10} {'p_surv':>10} {'binom SE':>10} "
        f"{'95% CI':>23} {'MC-Lap':>10} {'seconds':>9}"
    )
    for row in rows:
        print(
            f"{row.n_intervals:10d} {row.p_surv:10.5f} "
            f"{row.binomial_standard_error:10.5f} "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] "
            f"{row.difference_from_laplace:10.5f} "
            f"{row.runtime_seconds:9.3f}"
        )
    if rows:
        print(
            f"Laplace reference: {rows[0].laplace_p_surv:.8f} "
            f"(Talbot M={rows[0].laplace_M}, h0={rows[0].h0:.4f})"
        )


def main() -> None:
    args = _parse_args()
    if args.root_M < 2:
        raise ValueError("--root-M must be at least 2")

    params, constraint = load_calibrated_params(args.params)
    h0 = constraint["h0_min"] if args.h0 is None else args.h0
    market = MarketParams(
        b=constraint["b"],
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
    )
    strategy = UnitExposureLongShortStrategy(
        h0=h0,
        market=market,
        T=args.T,
        ltv_max=constraint.get("ltv_max"),
    )

    convergence = talbot_convergence_diagnostics(
        params,
        strategy,
        M_values=args.M_values,
        moment_orders=(1, 2, 3, 4),
    )
    roots = root_quality_diagnostics(
        params,
        strategy,
        M=args.root_M,
        tilt_orders=(0, 1, 2, 3, 4),
    )
    mc_strategy = UnitExposureLongShortStrategy(
        h0=args.mc_h0,
        market=market,
        T=args.T,
        ltv_max=constraint.get("ltv_max"),
    )
    monte_carlo = monte_carlo_monitoring_diagnostics(
        params,
        mc_strategy,
        interval_counts=args.mc_intervals,
        n_paths=args.mc_paths,
        seed=args.mc_seed,
        laplace_M=args.mc_laplace_M,
    )
    write_numerical_diagnostics_csv(args.out, convergence, roots, monte_carlo)

    print(
        f"Baseline: h0={h0:.8f}, H0={strategy.H:.8f}, T={args.T:.8f}, "
        f"root-M={args.root_M}"
    )
    _print_convergence(convergence)
    _print_root_quality(roots)
    _print_mc_monitoring(monte_carlo)
    print(f"\nCSV -> {args.out}")
    print(
        "Scope: diagnostics use the solver's existing rank-based root selection; "
        "they measure it but do not implement root continuation."
    )


if __name__ == "__main__":
    main()
