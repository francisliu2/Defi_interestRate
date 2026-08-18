"""Objective-specific initial health-buffer selection from risk-report rows.

The semi-analytic engine produces an objective-independent mapping from an
initial health buffer to survival probabilities and payoff moments.  This
module applies explicit, caller-chosen decision rules to that mapping.  It
deliberately uses ``selected`` terminology: no rule is treated as a universal
portfolio optimum.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


RiskRow = Mapping[str, float]
ScoreFunction = Callable[[RiskRow], float]
EligibilityFunction = Callable[[RiskRow], bool]


@dataclass(frozen=True)
class ObjectiveSpecificSelection:
    """A selected buffer together with its rule and auditable scored grid."""

    objective_specific_rule: str
    parameters: dict[str, float | str | bool]
    selected: dict[str, Any]
    scored_rows: tuple[dict[str, Any], ...]


def _finite_parameter(name: str, value: float, *, nonnegative: bool) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _row_value(row: RiskRow, key: str) -> float:
    try:
        value = float(row[key])
    except KeyError as exc:
        raise KeyError(f"Risk-report row is missing required key {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Risk-report value {key!r} must be finite, got {value}")
    return value


def _nonnegative_row_value(row: RiskRow, key: str) -> float:
    value = _row_value(row, key)
    if value < -1e-10:
        raise ValueError(f"Risk-report value {key!r} must be non-negative, got {value}")
    return max(0.0, value)


def initial_leverage_score(row: RiskRow) -> float:
    """Default constrained-rule score: prefer more initial gross leverage."""

    return _row_value(row, "initial_leverage")


def unconditional_killed_mean_variance_score(
    row: RiskRow,
    *,
    alpha: float,
) -> float:
    """Return ``M1 - alpha/2 * (M2 - M1**2)`` for the killed payoff."""

    alpha = _finite_parameter("alpha", alpha, nonnegative=True)
    moment_1 = _row_value(row, "killed_moment_1")
    moment_2 = _row_value(row, "killed_moment_2")
    variance = moment_2 - moment_1**2
    scale = max(1.0, abs(moment_2), moment_1**2)
    if variance < -1e-10 * scale:
        raise ValueError(
            f"Killed moments imply a negative variance, got {variance}"
        )
    variance = max(0.0, variance)
    return moment_1 - 0.5 * alpha * variance


def conditional_mean_variance_liquidation_score(
    row: RiskRow,
    *,
    alpha: float,
    delta: float,
) -> float:
    """Return conditional mean-variance performance minus liquidation penalty."""

    alpha = _finite_parameter("alpha", alpha, nonnegative=True)
    delta = _finite_parameter("delta", delta, nonnegative=True)
    mean = _row_value(row, "conditional_mean")
    variance = _nonnegative_row_value(row, "conditional_variance")
    p_liq = _nonnegative_row_value(row, "p_liq")
    if p_liq > 1.0 + 1e-10:
        raise ValueError(f"Risk-report value 'p_liq' must not exceed 1, got {p_liq}")
    return mean - 0.5 * alpha * variance - delta * p_liq


def select_objective_specific(
    rows: Iterable[RiskRow],
    *,
    rule: str,
    score: ScoreFunction,
    parameters: Mapping[str, float | str | bool] | None = None,
    eligible: EligibilityFunction | None = None,
    maximize: bool = True,
) -> ObjectiveSpecificSelection:
    """Score a candidate grid and select one row under an explicit rule.

    Each returned scored row contains ``objective_score`` and
    ``objective_feasible``.  Ineligible rows remain in the result for audit
    purposes but cannot be selected.  Ties are resolved by input order.
    """

    if not rule or not rule.strip():
        raise ValueError("rule must be a non-empty objective-specific label")

    scored_rows: list[dict[str, Any]] = []
    selectable: list[tuple[float, int]] = []
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        is_eligible = True if eligible is None else bool(eligible(source_row))
        objective_score = float(score(source_row))
        if not math.isfinite(objective_score):
            raise ValueError(
                f"Objective score must be finite at candidate index {index}, "
                f"got {objective_score}"
            )
        row["objective_score"] = objective_score
        row["objective_feasible"] = is_eligible
        scored_rows.append(row)
        if is_eligible:
            selection_key = objective_score if maximize else -objective_score
            selectable.append((selection_key, index))

    if not scored_rows:
        raise ValueError("At least one risk-report row is required for selection")
    if not selectable:
        raise ValueError(f"No candidate is feasible under rule {rule!r}")

    _, selected_index = max(selectable, key=lambda item: item[0])
    selected = dict(scored_rows[selected_index])
    selected["selected"] = True
    selected["objective_specific"] = True

    return ObjectiveSpecificSelection(
        objective_specific_rule=rule,
        parameters=dict(parameters or {}),
        selected=selected,
        scored_rows=tuple(scored_rows),
    )


def select_liquidation_constrained(
    rows: Iterable[RiskRow],
    *,
    pbar: float,
    score: ScoreFunction | None = None,
    score_name: str = "initial_leverage",
    maximize: bool = True,
) -> ObjectiveSpecificSelection:
    """Select a buffer subject to ``p_liq <= pbar``.

    The score is caller supplied.  When omitted, the demonstrative rule
    maximizes initial leverage among buffers satisfying the liquidation cap.
    """

    pbar = _finite_parameter("pbar", pbar, nonnegative=True)
    if pbar > 1.0:
        raise ValueError(f"pbar must not exceed 1, got {pbar}")
    if not score_name or not score_name.strip():
        raise ValueError("score_name must be non-empty")
    scorer = initial_leverage_score if score is None else score

    return select_objective_specific(
        rows,
        rule="liquidation_constrained",
        score=scorer,
        parameters={
            "pbar": pbar,
            "score_name": score_name,
            "maximize": maximize,
        },
        eligible=lambda row: _row_value(row, "p_liq") <= pbar,
        maximize=maximize,
    )


def select_unconditional_killed_mean_variance(
    rows: Iterable[RiskRow],
    *,
    alpha: float,
) -> ObjectiveSpecificSelection:
    """Select using unconditional mean-variance of the zero-recovery payoff."""

    alpha = _finite_parameter("alpha", alpha, nonnegative=True)
    return select_objective_specific(
        rows,
        rule="unconditional_killed_mean_variance",
        score=lambda row: unconditional_killed_mean_variance_score(
            row,
            alpha=alpha,
        ),
        parameters={"alpha": alpha},
    )


def select_conditional_mean_variance_with_liquidation_penalty(
    rows: Iterable[RiskRow],
    *,
    alpha: float,
    delta: float,
) -> ObjectiveSpecificSelection:
    """Select using conditional mean-variance plus an explicit risk penalty."""

    alpha = _finite_parameter("alpha", alpha, nonnegative=True)
    delta = _finite_parameter("delta", delta, nonnegative=True)
    return select_objective_specific(
        rows,
        rule="conditional_mean_variance_liquidation_penalty",
        score=lambda row: conditional_mean_variance_liquidation_score(
            row,
            alpha=alpha,
            delta=delta,
        ),
        parameters={"alpha": alpha, "delta": delta},
    )
