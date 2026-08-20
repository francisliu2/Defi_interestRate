"""Sensitivity of liquidation risk and payoff moments to the log-return mean spread.

The calibrated midpoint of the two annual expected log-return means is held fixed
while their benchmark spread receives an additive annualized shock c. Thus c=0
is the calibrated benchmark, negative c narrows the spread, and positive c widens
it. The required expected-price-growth inputs are derived internally from the
unchanged Kou shape parameters. Outputs are machine-readable CSVs and
publication-ready PDF figures.

Usage
-----
python -m jobs.mu_spread_sensitivity
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from optimal_long_short.utils.helpers import (
    DEFAULT_EMPIRICAL_PARAMS,
    LATEX_DIR,
    RESULTS_DIR,
    load_calibrated_params,
)
from optimal_long_short.model.drift_service import (
    expected_log_return_drift,
    with_expected_log_return_drift,
)
from optimal_long_short.model.risk_report import h0_liquidation_moment_report


BASE_T = 1.0 / 12.0
DEFAULT_H0 = 1.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_EMPIRICAL_PARAMS)
    parser.add_argument("--H0", type=float, default=DEFAULT_H0)
    parser.add_argument("--T", type=float, default=BASE_T)
    parser.add_argument(
        "--c-min",
        type=float,
        default=-0.70,
        help="Minimum additive annualized log-mean-spread shock.",
    )
    parser.add_argument(
        "--c-max",
        type=float,
        default=0.70,
        help="Maximum additive annualized log-mean-spread shock.",
    )
    parser.add_argument("--c-count", type=int, default=57)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=RESULTS_DIR / "mu_spread_sensitivity.csv",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        default=LATEX_DIR / "fig_mu_spread_sensitivity.pdf",
    )
    parser.add_argument(
        "--killed-payoff-figure-out",
        type=Path,
        default=LATEX_DIR / "fig_expected_killed_payoff_sensitivity.pdf",
    )
    parser.add_argument("--optimal-H0-max", type=float, default=2.0)
    parser.add_argument(
        "--optimizer-xatol",
        type=float,
        default=1e-7,
        help="Absolute H0 tolerance for the bounded scalar optimizer.",
    )
    parser.add_argument(
        "--optimal-csv-out",
        type=Path,
        default=RESULTS_DIR / "optimal_h_killed_payoff_sensitivity.csv",
    )
    parser.add_argument(
        "--optimal-figure-out",
        type=Path,
        default=LATEX_DIR / "fig_optimal_h_killed_payoff_sensitivity.pdf",
    )
    return parser.parse_args()


def spread_params(params, c: float):
    """Apply an additive annualized spread shock at a fixed log-mean midpoint."""
    log_mean1, log_mean2 = expected_log_return_drift(params)
    return with_expected_log_return_drift(
        params,
        drift1=log_mean1 + 0.5 * float(c),
        drift2=log_mean2 - 0.5 * float(c),
    )


def expected_killed_payoff(conditional_mean: float, p_liq: float) -> float:
    """Return p_surv times conditional mean, equal to the killed first moment."""
    if not 0.0 <= p_liq <= 1.0:
        raise ValueError("p_liq must lie in [0, 1]")
    return (1.0 - p_liq) * conditional_mean


def compute_rows(args: argparse.Namespace) -> tuple[list[dict[str, float]], float, float]:
    params, constraint = load_calibrated_params(args.params)
    if args.H0 < constraint["H0_min"] - 1e-12:
        raise ValueError(
            f"H0={args.H0:.6f} is below the feasible minimum "
            f"{constraint['H0_min']:.6f}."
        )
    if args.c_count < 2 or args.c_max <= args.c_min:
        raise ValueError("The c grid must contain at least two increasing points.")
    log_mean1, log_mean2 = expected_log_return_drift(params)
    midpoint = 0.5 * (log_mean1 + log_mean2)
    benchmark_spread = log_mean1 - log_mean2
    rows: list[dict[str, float]] = []
    for c in np.linspace(args.c_min, args.c_max, args.c_count):
        varied = spread_params(params, float(c))
        varied_log_mean1, varied_log_mean2 = expected_log_return_drift(varied)
        report = h0_liquidation_moment_report(
            varied,
            [np.log(args.H0)],
            b=constraint["b"],
            T=args.T,
            S10=constraint.get("S10", 1.0),
            S20=constraint.get("S20", 1.0),
            ltv_max=constraint["ltv_max"],
            max_moment_order=4,
        )[0]
        p_surv = 1.0 - report["p_liq"]
        killed_payoff = expected_killed_payoff(
            report["conditional_mean"], report["p_liq"]
        )
        rows.append(
            {
                "spread_shock_c": float(c),
                "benchmark_log_mean_spread": benchmark_spread,
                "log_mean_midpoint": midpoint,
                "log_mean_spread": varied_log_mean1 - varied_log_mean2,
                "log_mean_long": varied_log_mean1,
                "log_mean_short": varied_log_mean2,
                "internal_price_growth_mu_long": varied.mu1,
                "internal_price_growth_mu_short": varied.mu2,
                "H0": args.H0,
                "T": args.T,
                "p_liq": report["p_liq"],
                "p_surv": p_surv,
                "conditional_mean": report["conditional_mean"],
                "conditional_variance": report["conditional_variance"],
                "expected_killed_payoff": killed_payoff,
                "conditional_skewness": report["conditional_skewness"],
                "conditional_excess_kurtosis": report[
                    "conditional_excess_kurtosis"
                ],
            }
        )
    return rows, midpoint, benchmark_spread


def write_csv(rows: list[dict[str, float]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def add_benchmark_line(axis) -> None:
    """Mark the calibrated benchmark c=0."""
    axis.axvline(0.0, color="0.35", lw=0.9, ls=":")


def select_optimal_health(
    reports: list[dict[str, float]],
) -> tuple[dict[str, float], float, int]:
    """Select the grid point maximizing the expected killed payoff."""
    if not reports:
        raise ValueError("reports must be non-empty")
    scores = [float(row["killed_moment_1"]) for row in reports]
    index = int(np.argmax(scores))
    return reports[index], float(scores[index]), index


def compute_optimal_health_rows(args: argparse.Namespace) -> list[dict[str, float]]:
    """Continuously optimize expected killed payoff over the feasible H0 interval."""
    params, constraint = load_calibrated_params(args.params)
    log_mean1, log_mean2 = expected_log_return_drift(params)
    benchmark_spread = log_mean1 - log_mean2
    H0_min = float(constraint["H0_min"])
    if args.optimal_H0_max <= H0_min:
        raise ValueError("--optimal-H0-max must exceed the feasible minimum H0")
    if args.optimizer_xatol <= 0.0:
        raise ValueError("--optimizer-xatol must be positive")

    def evaluate(varied, H0: float) -> dict[str, float]:
        return h0_liquidation_moment_report(
            varied,
            [np.log(H0)],
            b=constraint["b"],
            T=args.T,
            S10=constraint.get("S10", 1.0),
            S20=constraint.get("S20", 1.0),
            ltv_max=constraint["ltv_max"],
            max_moment_order=1,
        )[0]

    rows: list[dict[str, float]] = []
    for c in np.linspace(args.c_min, args.c_max, args.c_count):
        varied = spread_params(params, float(c))
        result = minimize_scalar(
            lambda H0: -evaluate(varied, float(H0))["killed_moment_1"],
            bounds=(H0_min, args.optimal_H0_max),
            method="bounded",
            options={"xatol": args.optimizer_xatol},
        )
        if not result.success:
            raise RuntimeError(
                f"Health-buffer optimization failed at c={float(c):.6g}: "
                f"{result.message}"
            )

        candidates = [
            evaluate(varied, H0_min),
            evaluate(varied, float(result.x)),
            evaluate(varied, args.optimal_H0_max),
        ]
        optimum, score, index = select_optimal_health(candidates)
        rows.append(
            {
                "spread_shock_c": float(c),
                "benchmark_log_mean_spread": benchmark_spread,
                "log_mean_spread": benchmark_spread + float(c),
                "optimal_h0": optimum["h0"],
                "optimal_H0": optimum["H0"],
                "optimal_initial_leverage": optimum["initial_leverage"],
                "max_expected_killed_payoff": score,
                "optimal_p_surv": optimum["p_surv"],
                "optimal_conditional_mean": optimum["conditional_mean"],
                "optimizer_at_lower_bound": float(index == 0),
                "optimizer_at_upper_bound": float(index == 2),
                "optimizer_success": float(result.success),
                "optimizer_nfev": float(result.nfev),
                "optimizer_xatol": float(args.optimizer_xatol),
                "H0_bound_min": H0_min,
                "H0_bound_max": float(args.optimal_H0_max),
            }
        )
    return rows


def plot(rows: list[dict[str, float]], out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 180,
        }
    )
    x = np.array([row["spread_shock_c"] for row in rows])
    series = {
        "p_liq": np.array([row["p_liq"] for row in rows]),
        "mean": np.array([row["conditional_mean"] for row in rows]),
        "variance": np.array([row["conditional_variance"] for row in rows]),
        "skewness": np.array([row["conditional_skewness"] for row in rows]),
        "kurtosis": np.array(
            [row["conditional_excess_kurtosis"] for row in rows]
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharex=True)
    color = "#1f4e79"
    axes[0, 0].plot(x, series["p_liq"], color=color, lw=1.8)
    axes[0, 0].set_ylabel(r"Liquidation probability $p_{\rm liq}$")
    axes[0, 0].set_title("(a) Default/liquidation risk")

    axes[0, 1].plot(x, series["mean"], color=color, lw=1.8)
    axes[0, 1].set_ylabel(r"$\mathrm{E}[\Pi_T\mid\tau>T]$")
    axes[0, 1].set_title("(b) Conditional mean")

    axes[1, 0].plot(x, series["variance"], color=color, lw=1.8)
    axes[1, 0].set_ylabel(r"$\mathrm{Var}(\Pi_T\mid\tau>T)$")
    axes[1, 0].set_title("(c) Conditional variance")

    ax = axes[1, 1]
    ax.plot(x, series["skewness"], color="#2c7fb8", lw=1.8, label="Skewness")
    ax.plot(
        x,
        series["kurtosis"],
        color="#b35806",
        lw=1.6,
        ls="--",
        label="Excess kurtosis",
    )
    ax.set_ylabel("Standardized conditional moment")
    ax.set_title("(d) Conditional shape")
    ax.legend(loc="best", fontsize=8)

    for axis in axes.flat:
        add_benchmark_line(axis)
        axis.grid(axis="y", color="0.85", lw=0.5)
        axis.set_xlim(x.min(), x.max())
    for axis in axes[1, :]:
        axis.set_xlabel(r"Annualized spread shock $c$")
    axes[0, 0].annotate(
        "benchmark",
        xy=(0.0, np.interp(0.0, x, series["p_liq"])),
        xytext=(0.10, 0.92 * series["p_liq"].max()),
        fontsize=7.5,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.35"},
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_expected_killed_payoff(rows: list[dict[str, float]], out: Path) -> None:
    """Plot expected killed payoff over signed spread c at fixed H0."""
    x = np.array([row["spread_shock_c"] for row in rows])
    y = np.array([row["expected_killed_payoff"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(x, y, color="#1f4e79", lw=1.8)
    add_benchmark_line(ax)
    benchmark = float(np.interp(0.0, x, y))
    ax.scatter([0.0], [benchmark], color="#1f4e79", s=18, zorder=3)
    ax.annotate(
        "benchmark",
        xy=(0.0, benchmark),
        xytext=(0.10, benchmark - 0.12 * (y.max() - y.min())),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "0.35"},
    )
    ax.set_xlabel(r"Annualized spread shock $c$")
    ax.set_ylabel(r"$p_{\rm surv}\,\mathrm{E}[\Pi_T\mid\tau>T]$")
    ax.set_xlim(x.min(), x.max())
    ax.grid(axis="y", color="0.85", lw=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_optimal_health_sensitivity(rows: list[dict[str, float]], out: Path) -> None:
    """Plot killed-payoff-optimal health and its objective value against c."""
    x = np.array([row["spread_shock_c"] for row in rows])
    H_star = np.array([row["optimal_H0"] for row in rows])
    score = np.array(
        [row["max_expected_killed_payoff"] for row in rows]
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
    axes[0].plot(x, H_star, color="#1f4e79", lw=1.8)
    axes[0].set_ylabel(r"Constrained optimum $H_0^*(c)$")
    axes[0].set_title("(a) Optimal health factor")
    axes[1].plot(x, score, color="#1f4e79", lw=1.8)
    axes[1].set_ylabel(r"$J_{\rm kill}(c,H_0^*)$")
    axes[1].set_title("(b) Maximized expected killed payoff")
    for axis in axes:
        add_benchmark_line(axis)
        axis.set_xlabel(r"Annualized spread shock $c$")
        axis.set_xlim(x.min(), x.max())
        axis.grid(axis="y", color="0.85", lw=0.5)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows, midpoint, benchmark_spread = compute_rows(args)
    write_csv(rows, args.csv_out)
    plot(rows, args.figure_out)
    plot_expected_killed_payoff(rows, args.killed_payoff_figure_out)
    optimal_rows = compute_optimal_health_rows(args)
    write_csv(optimal_rows, args.optimal_csv_out)
    plot_optimal_health_sensitivity(optimal_rows, args.optimal_figure_out)
    print(
        f"Benchmark log-mean midpoint={midpoint:.9f}, spread={benchmark_spread:.9f}; "
        f"evaluated {len(rows)} additive annualized spread shocks c at "
        f"H0={args.H0:.4f}, T={args.T:.6f}."
    )
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.figure_out}")
    print(f"Wrote {args.killed_payoff_figure_out}")
    print(f"Wrote {args.optimal_csv_out}")
    print(f"Wrote {args.optimal_figure_out}")


if __name__ == "__main__":
    main()
