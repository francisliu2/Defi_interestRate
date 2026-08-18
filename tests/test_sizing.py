import math

import pytest

from optimal_long_short.job_runners.objective_comparison import (
    build_selections,
    comparison_rows,
    latex_ready_table,
)
from optimal_long_short.model.sizing import (
    conditional_mean_variance_liquidation_score,
    select_conditional_mean_variance_with_liquidation_penalty,
    select_liquidation_constrained,
    select_unconditional_killed_mean_variance,
    unconditional_killed_mean_variance_score,
)


@pytest.fixture
def risk_rows() -> list[dict[str, float]]:
    def row(
        H0: float,
        leverage: float,
        p_liq: float,
        moment_1: float,
        moment_2: float,
        conditional_mean: float,
        conditional_variance: float,
    ) -> dict[str, float]:
        return {
            "h0": math.log(H0),
            "H0": H0,
            "initial_leverage": leverage,
            "p_surv": 1.0 - p_liq,
            "p_liq": p_liq,
            "killed_moment_1": moment_1,
            "killed_moment_2": moment_2,
            "unconditional_mean": moment_1,
            "unconditional_variance": moment_2 - moment_1**2,
            "conditional_mean": conditional_mean,
            "conditional_variance": conditional_variance,
        }

    return [
        row(1.10, 5.0, 0.20, 1.20, 2.00, 1.50, 0.50),
        row(1.20, 3.0, 0.08, 1.00, 1.10, 1.10, 0.10),
        row(1.50, 2.0, 0.01, 0.90, 0.82, 1.00, 0.02),
    ]


def test_liquidation_constrained_default_selects_leverage_within_cap(
    risk_rows: list[dict[str, float]],
) -> None:
    result = select_liquidation_constrained(risk_rows, pbar=0.10)

    assert result.objective_specific_rule == "liquidation_constrained"
    assert result.parameters == {
        "pbar": 0.10,
        "score_name": "initial_leverage",
        "maximize": True,
    }
    assert result.selected["H0"] == pytest.approx(1.20)
    assert result.selected["selected"] is True
    assert result.selected["objective_specific"] is True
    assert [row["objective_feasible"] for row in result.scored_rows] == [
        False,
        True,
        True,
    ]


def test_liquidation_constrained_accepts_caller_supplied_score(
    risk_rows: list[dict[str, float]],
) -> None:
    result = select_liquidation_constrained(
        risk_rows,
        pbar=0.10,
        score=lambda row: row["H0"],
        score_name="H0",
    )

    assert result.selected["H0"] == pytest.approx(1.50)
    assert result.parameters["score_name"] == "H0"


def test_unconditional_killed_mean_variance_rule_uses_killed_moments(
    risk_rows: list[dict[str, float]],
) -> None:
    result = select_unconditional_killed_mean_variance(risk_rows, alpha=1.0)

    expected_scores = [
        1.20 - 0.5 * (2.00 - 1.20**2),
        1.00 - 0.5 * (1.10 - 1.00**2),
        0.90 - 0.5 * (0.82 - 0.90**2),
    ]
    assert [row["objective_score"] for row in result.scored_rows] == pytest.approx(
        expected_scores
    )
    assert result.selected["H0"] == pytest.approx(1.20)
    assert unconditional_killed_mean_variance_score(
        risk_rows[0], alpha=1.0
    ) == pytest.approx(expected_scores[0])


def test_conditional_rule_applies_explicit_liquidation_penalty(
    risk_rows: list[dict[str, float]],
) -> None:
    result = select_conditional_mean_variance_with_liquidation_penalty(
        risk_rows,
        alpha=1.0,
        delta=3.0,
    )

    expected_scores = [
        1.50 - 0.5 * 0.50 - 3.0 * 0.20,
        1.10 - 0.5 * 0.10 - 3.0 * 0.08,
        1.00 - 0.5 * 0.02 - 3.0 * 0.01,
    ]
    assert [row["objective_score"] for row in result.scored_rows] == pytest.approx(
        expected_scores
    )
    assert result.selected["H0"] == pytest.approx(1.50)
    assert conditional_mean_variance_liquidation_score(
        risk_rows[0], alpha=1.0, delta=3.0
    ) == pytest.approx(expected_scores[0])


def test_rule_validation_is_explicit(risk_rows: list[dict[str, float]]) -> None:
    with pytest.raises(ValueError, match="No candidate is feasible"):
        select_liquidation_constrained(risk_rows, pbar=0.001)
    with pytest.raises(ValueError, match="pbar must not exceed 1"):
        select_liquidation_constrained(risk_rows, pbar=1.1)
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        select_unconditional_killed_mean_variance(risk_rows, alpha=-1.0)
    with pytest.raises(ValueError, match="delta must be non-negative"):
        select_conditional_mean_variance_with_liquidation_penalty(
            risk_rows,
            alpha=1.0,
            delta=-0.5,
        )

    invalid_variance = [dict(risk_rows[0], conditional_variance=-0.01)]
    with pytest.raises(ValueError, match="conditional_variance.*non-negative"):
        select_conditional_mean_variance_with_liquidation_penalty(
            invalid_variance,
            alpha=1.0,
            delta=0.5,
        )


def test_comparison_output_keeps_rule_parameters_and_selected_terminology(
    risk_rows: list[dict[str, float]],
) -> None:
    selections = build_selections(
        risk_rows,
        pbars=[0.10],
        unconditional_alphas=[1.0],
        conditional_rules=[(1.0, 3.0)],
    )
    rows = comparison_rows(selections)

    assert len(rows) == 3
    assert all(row["objective_specific"] is True for row in rows)
    assert rows[0]["pbar"] == pytest.approx(0.10)
    assert rows[1]["alpha"] == pytest.approx(1.0)
    assert rows[2]["alpha"] == pytest.approx(1.0)
    assert rows[2]["delta"] == pytest.approx(3.0)
    assert all("selected_H0" in row for row in rows)

    table = latex_ready_table(rows)
    assert "Objective-specific rule" in table
    assert "Selected $H_0$" in table
    assert "$\\bar p=0.1$" in table
    assert "initial_leverage" not in table
