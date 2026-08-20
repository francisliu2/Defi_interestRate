import numpy as np
import pandas as pd
import pytest

from optimal_long_short.calibration import construct_ewm_residual_increments
from optimal_long_short.calibration.transforms import (
    ParameterBounds,
    params_to_shape_unc,
    shape_unc_to_params,
)
from optimal_long_short.model.drift_service import (
    expected_log_return_drift,
    with_expected_log_return_drift,
    with_zero_expected_log_return,
)
from jobs.calibrate_eth_btc import (
    build_empirical_params,
    compute_avg_rates,
    empirical_asset_log_means,
    reconstruct_ewm_location_price,
)
from optimal_long_short.model.model_params import KouParams
from jobs.mu_spread_sensitivity import (
    expected_killed_payoff,
    spread_params,
    select_optimal_health,
)


def _params() -> KouParams:
    return KouParams(
        mu1=0.2,
        sigma1=0.3,
        lam1=4.0,
        p1=0.45,
        eta1_pos=0.03,
        eta1_neg=0.02,
        mu2=-0.1,
        sigma2=0.2,
        lam2=3.0,
        p2=0.55,
        eta2_pos=0.025,
        eta2_neg=0.03,
        rho=0.4,
    )


def test_causal_ewm_uses_only_lagged_mean_without_extra_demeaning():
    result = construct_ewm_residual_increments(
        np.array([1.0, 2.0, 3.0]), 1.0
    )

    assert result.ewm_mean_path_per_period == pytest.approx(
        [1.0, 5.0 / 3.0, 17.0 / 7.0]
    )
    assert result.residual_increments == pytest.approx([1.0, 4.0 / 3.0])
    assert result.residual_sample_mean == pytest.approx(7.0 / 6.0)


def test_ewm_location_price_compounds_only_removed_lagged_means():
    result = construct_ewm_residual_increments(
        np.array([0.1, 0.2, 0.3]), 1.0
    )
    location = reconstruct_ewm_location_price(
        np.array([100.0, 110.0, 121.0, 133.1]),
        result,
    )

    assert location == pytest.approx(
        [100.0, 100.0, 100.0 * np.exp(0.1), 100.0 * np.exp(0.1 + 1.0 / 6.0)]
    )


def test_aave_apr_percentages_are_converted_to_annual_decimal_rates():
    frame = pd.DataFrame(
        {
            "supply_apr_btc": [0.01, 0.03],
            "variable_borrow_apr_eth": [2.0, 4.0],
            "supply_apr_eth": [1.0, 3.0],
            "variable_borrow_apr_btc": [0.2, 0.4],
        }
    )
    rates = compute_avg_rates(frame)
    assert rates == pytest.approx(
        {
            "supply_btc": 0.0002,
            "borrow_eth": 0.03,
            "supply_eth": 0.02,
            "borrow_btc": 0.003,
        }
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01])
def test_aave_apr_conversion_rejects_invalid_annual_percentages(bad):
    frame = pd.DataFrame(
        {
            "supply_apr_btc": [bad],
            "variable_borrow_apr_eth": [2.0],
            "supply_apr_eth": [1.0],
            "variable_borrow_apr_btc": [0.2],
        }
    )
    with pytest.raises(ValueError, match="annual APR percentages"):
        compute_avg_rates(frame)


def test_zero_log_mean_shape_transform_is_eleven_dimensional():
    bounds = ParameterBounds(max_moment_order=4)
    normalized = with_zero_expected_log_return(_params())
    shape = params_to_shape_unc(normalized, bounds)
    restored = shape_unc_to_params(shape, bounds)

    assert shape.shape == (11,)
    assert expected_log_return_drift(restored) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert restored.sigma1 == pytest.approx(normalized.sigma1)
    assert restored.lam2 == pytest.approx(normalized.lam2)
    assert restored.rho == pytest.approx(normalized.rho)


def test_expected_log_drift_view_is_converted_to_price_growth_mu():
    adjusted = with_expected_log_return_drift(_params(), drift1=-0.4, drift2=0.25)
    assert expected_log_return_drift(adjusted) == pytest.approx((-0.4, 0.25))

    mean_jump1 = adjusted.p1 * adjusted.eta1_pos - (1.0 - adjusted.p1) * adjusted.eta1_neg
    chi1 = (
        adjusted.p1 / (1.0 - adjusted.eta1_pos)
        + (1.0 - adjusted.p1) / (1.0 + adjusted.eta1_neg)
        - 1.0
    )
    assert adjusted.mu1 == pytest.approx(
        -0.4 + 0.5 * adjusted.sigma1**2 + adjusted.lam1 * (chi1 - mean_jump1)
    )
    assert adjusted.muX1 == pytest.approx(-0.4 - adjusted.lam1 * mean_jump1)

    mean_jump2 = adjusted.p2 * adjusted.eta2_pos - (1.0 - adjusted.p2) * adjusted.eta2_neg
    assert adjusted.mean_jump_rate1 == pytest.approx(adjusted.lam1 * mean_jump1)
    assert adjusted.mean_jump_rate2 == pytest.approx(adjusted.lam2 * mean_jump2)
    assert adjusted.muX2 == pytest.approx(0.25 - adjusted.mean_jump_rate2)


def test_signed_spread_perturbation_is_centered_on_calibrated_benchmark():
    benchmark = with_expected_log_return_drift(_params(), drift1=0.4, drift2=-0.2)
    midpoint = 0.1
    benchmark_spread = 0.6

    narrowed = expected_log_return_drift(spread_params(benchmark, -0.3))
    centered = expected_log_return_drift(spread_params(benchmark, 0.0))
    widened = expected_log_return_drift(spread_params(benchmark, 0.3))

    assert narrowed == pytest.approx((0.25, -0.05))
    assert centered == pytest.approx((0.4, -0.2))
    assert widened == pytest.approx((0.55, -0.35))
    assert sum(narrowed) / 2 == pytest.approx(midpoint)
    assert sum(widened) / 2 == pytest.approx(midpoint)
    assert narrowed[0] - narrowed[1] == pytest.approx(benchmark_spread - 0.3)
    assert widened[0] - widened[1] == pytest.approx(benchmark_spread + 0.3)


def test_expected_killed_payoff_definition_and_validation():
    assert expected_killed_payoff(1.2, 0.25) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="p_liq"):
        expected_killed_payoff(1.2, 1.1)


def test_select_optimal_health_uses_expected_killed_payoff():
    reports = [
        {"H0": 1.1, "killed_moment_1": 0.9},
        {"H0": 1.2, "killed_moment_1": 0.99},
    ]
    optimum, score, index = select_optimal_health(reports)
    assert index == 1
    assert optimum["H0"] == pytest.approx(1.2)
    assert score == pytest.approx(0.99)


def test_empirical_orientation_maximizes_role_adjusted_log_mean_spread():
    residual = with_zero_expected_log_return(_params())
    trends = {"WETH": -0.50, "WBTC": 0.10}
    rates = {
        "supply_eth": 0.02,
        "borrow_eth": 0.03,
        "supply_btc": 0.001,
        "borrow_btc": 0.004,
    }

    pre_carry = empirical_asset_log_means(trends)
    final, residual_oriented, selection = build_empirical_params(residual, trends, rates)

    assert pre_carry["WBTC"] > pre_carry["WETH"]
    assert selection["long_asset"] == "WBTC"
    assert selection["short_asset"] == "WETH"
    assert final.mu1 == pytest.approx(residual_oriented.mu1 + trends["WBTC"] + rates["supply_btc"])
    assert final.mu2 == pytest.approx(residual_oriented.mu2 + trends["WETH"] + rates["borrow_eth"])
    assert final.mu1 > final.mu2


def test_role_carry_can_reverse_the_pre_carry_ranking():
    residual = with_zero_expected_log_return(_params())
    trends = {"WETH": 0.02, "WBTC": 0.01}
    rates = {
        "supply_eth": 0.0,
        "borrow_eth": 0.0,
        "supply_btc": 0.03,
        "borrow_btc": 0.02,
    }

    pre_carry = empirical_asset_log_means(trends)
    final, _, selection = build_empirical_params(residual, trends, rates)

    assert pre_carry["WETH"] > pre_carry["WBTC"]
    assert selection["long_asset"] == "WBTC"
    assert selection["short_asset"] == "WETH"
    assert selection["candidate_assignments"]["WBTC"]["g_spread"] > 0.0
    assert expected_log_return_drift(final)[0] > expected_log_return_drift(final)[1]
