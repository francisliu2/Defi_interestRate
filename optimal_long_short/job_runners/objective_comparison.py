"""Compare objective-specific initial health-buffer selections.

The moment engine is evaluated once on a configurable health-factor grid.
Several explicit sizing rules are then applied to the same objective-independent
risk report, and their selected buffers are written to a compact CSV table.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from optimal_long_short.job_runners.common import (
    DEFAULT_EMPIRICAL_PARAMS,
    RESULTS_DIR,
    load_calibrated_params,
)
from optimal_long_short.model.risk_report import h0_liquidation_moment_report
from optimal_long_short.model.sizing import (
    ObjectiveSpecificSelection,
    select_conditional_mean_variance_with_liquidation_penalty,
    select_liquidation_constrained,
    select_unconditional_killed_mean_variance,
)


DEFAULT_PBARS = (0.05, 0.10, 0.20)
DEFAULT_UNCONDITIONAL_ALPHAS = (0.5, 1.0, 5.0)
DEFAULT_CONDITIONAL_RULES = (
    (1.0, 0.0),
    (20.0, 0.0),
    (1.0, 0.25),
    (1.0, 0.5),
)

OUTPUT_FIELDS = (
    "objective_specific",
    "rule_id",
    "rule",
    "parameter_label",
    "pbar",
    "alpha",
    "delta",
    "score_name",
    "selected_H0",
    "selected_h0",
    "selected_initial_leverage",
    "selected_p_surv",
    "selected_p_liq",
    "selected_killed_moment_1",
    "selected_killed_moment_2",
    "selected_unconditional_mean",
    "selected_unconditional_variance",
    "selected_conditional_mean",
    "selected_conditional_variance",
    "objective_score",
)


def _conditional_rule(value: str) -> tuple[float, float]:
    """Parse an ``ALPHA:DELTA`` command-line value."""

    try:
        alpha_text, delta_text = value.split(":", maxsplit=1)
        alpha, delta = float(alpha_text), float(delta_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "conditional rules must use ALPHA:DELTA, for example 1.0:0.5"
        ) from exc
    if not math.isfinite(alpha) or alpha < 0.0:
        raise argparse.ArgumentTypeError("conditional-rule ALPHA must be non-negative")
    if not math.isfinite(delta) or delta < 0.0:
        raise argparse.ArgumentTypeError("conditional-rule DELTA must be non-negative")
    return alpha, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=DEFAULT_EMPIRICAL_PARAMS,
    )
    parser.add_argument("--H0-min", type=float, default=None)
    parser.add_argument("--H0-max", type=float, default=2.0)
    parser.add_argument("--H0-count", type=int, default=200)
    parser.add_argument("--T", type=float, default=1.0 / 12.0)
    parser.add_argument(
        "--pbar",
        type=float,
        action="append",
        default=None,
        help="Liquidation cap for a constrained rule; repeat for several caps.",
    )
    parser.add_argument(
        "--unconditional-alpha",
        type=float,
        action="append",
        default=None,
        help="Risk aversion for killed-payoff mean-variance; repeat as needed.",
    )
    parser.add_argument(
        "--conditional-rule",
        type=_conditional_rule,
        action="append",
        default=None,
        metavar="ALPHA:DELTA",
        help="Conditional mean-variance/liquidation parameters; repeat as needed.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "sizing_objective_comparison.csv",
    )
    return parser.parse_args()


def build_selections(
    rows: list[dict[str, float]],
    *,
    pbars: tuple[float, ...] | list[float] = DEFAULT_PBARS,
    unconditional_alphas: tuple[float, ...] | list[float] = (
        DEFAULT_UNCONDITIONAL_ALPHAS
    ),
    conditional_rules: tuple[tuple[float, float], ...] | list[tuple[float, float]] = (
        DEFAULT_CONDITIONAL_RULES
    ),
) -> list[tuple[str, str, ObjectiveSpecificSelection]]:
    """Apply several explicit decision rules to one shared risk-report grid."""

    selections: list[tuple[str, str, ObjectiveSpecificSelection]] = []
    for pbar in pbars:
        selection = select_liquidation_constrained(rows, pbar=pbar)
        rule_id = f"liquidation_constrained_pbar_{pbar:g}"
        parameter_label = f"pbar={pbar:g}; maximize initial_leverage"
        selections.append((rule_id, parameter_label, selection))

    for alpha in unconditional_alphas:
        selection = select_unconditional_killed_mean_variance(rows, alpha=alpha)
        rule_id = f"unconditional_killed_mean_variance_alpha_{alpha:g}"
        parameter_label = f"alpha={alpha:g}"
        selections.append((rule_id, parameter_label, selection))

    for alpha, delta in conditional_rules:
        selection = select_conditional_mean_variance_with_liquidation_penalty(
            rows,
            alpha=alpha,
            delta=delta,
        )
        rule_id = (
            "conditional_mean_variance_liquidation_penalty_"
            f"alpha_{alpha:g}_delta_{delta:g}"
        )
        parameter_label = f"alpha={alpha:g}; delta={delta:g}"
        selections.append((rule_id, parameter_label, selection))

    return selections


def comparison_rows(
    selections: list[tuple[str, str, ObjectiveSpecificSelection]],
) -> list[dict[str, object]]:
    """Flatten selections into stable CSV records."""

    output: list[dict[str, object]] = []
    for rule_id, parameter_label, selection in selections:
        selected = selection.selected
        parameters = selection.parameters
        output.append(
            {
                "objective_specific": True,
                "rule_id": rule_id,
                "rule": selection.objective_specific_rule,
                "parameter_label": parameter_label,
                "pbar": parameters.get("pbar"),
                "alpha": parameters.get("alpha"),
                "delta": parameters.get("delta"),
                "score_name": parameters.get("score_name"),
                "selected_H0": selected["H0"],
                "selected_h0": selected["h0"],
                "selected_initial_leverage": selected["initial_leverage"],
                "selected_p_surv": selected["p_surv"],
                "selected_p_liq": selected["p_liq"],
                "selected_killed_moment_1": selected["killed_moment_1"],
                "selected_killed_moment_2": selected["killed_moment_2"],
                "selected_unconditional_mean": selected["unconditional_mean"],
                "selected_unconditional_variance": selected[
                    "unconditional_variance"
                ],
                "selected_conditional_mean": selected["conditional_mean"],
                "selected_conditional_variance": selected[
                    "conditional_variance"
                ],
                "objective_score": selected["objective_score"],
            }
        )
    return output


def latex_ready_table(rows: list[dict[str, object]]) -> str:
    """Render the main comparison columns as a LaTeX-ready table."""

    rule_labels = {
        "liquidation_constrained": "Liquidation constrained",
        "unconditional_killed_mean_variance": "Unconditional killed MV",
        "conditional_mean_variance_liquidation_penalty": "Conditional MV + penalty",
    }
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Objective-specific rule & Parameters & Selected $H_0$ & "
            r"Initial leverage & $p_{\mathrm{liq}}$ \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        rule = str(row["rule"])
        label = rule_labels[rule]
        if rule == "liquidation_constrained":
            parameter_label = (
                f"$\\bar p={float(row['pbar']):g}$; max $L_0$"
            )
        elif rule == "unconditional_killed_mean_variance":
            parameter_label = f"$\\alpha={float(row['alpha']):g}$"
        else:
            parameter_label = (
                f"$\\alpha={float(row['alpha']):g}$, "
                f"$\\delta={float(row['delta']):g}$"
            )
        lines.append(
            f"{label} & {parameter_label} & "
            f"{float(row['selected_H0']):.3f} & "
            f"{float(row['selected_initial_leverage']):.3f} & "
            f"{float(row['selected_p_liq']):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    params, constraint = load_calibrated_params(args.params)

    protocol_H0_min = float(constraint["H0_min"])
    H0_min = protocol_H0_min if args.H0_min is None else float(args.H0_min)
    if not math.isfinite(H0_min) or H0_min < protocol_H0_min - 1e-12:
        raise ValueError(
            f"--H0-min must be at least the protocol minimum {protocol_H0_min:.6f}"
        )
    if not math.isfinite(args.H0_max) or args.H0_max <= H0_min:
        raise ValueError("--H0-max must be finite and greater than --H0-min")
    if args.H0_count < 2:
        raise ValueError("--H0-count must be at least 2")
    if not math.isfinite(args.T) or args.T <= 0.0:
        raise ValueError("--T must be finite and positive")

    health_grid = np.linspace(H0_min, args.H0_max, args.H0_count)
    rows = h0_liquidation_moment_report(
        params,
        np.log(health_grid),
        b=constraint["b"],
        T=args.T,
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
        ltv_max=constraint["ltv_max"],
        max_moment_order=2,
    )

    selections = build_selections(
        rows,
        pbars=tuple(args.pbar or DEFAULT_PBARS),
        unconditional_alphas=tuple(
            args.unconditional_alpha or DEFAULT_UNCONDITIONAL_ALPHAS
        ),
        conditional_rules=tuple(args.conditional_rule or DEFAULT_CONDITIONAL_RULES),
    )
    output_rows = comparison_rows(selections)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        f"Evaluated {len(rows)} candidate buffers on "
        f"H0=[{H0_min:.6f}, {args.H0_max:.6f}] at T={args.T:.6f}."
    )
    print(f"Objective-specific selections -> {args.out}")
    print()
    print(latex_ready_table(output_rows))


if __name__ == "__main__":
    main()
