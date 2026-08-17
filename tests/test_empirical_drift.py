import numpy as np
import pytest

from optimal_long_short.calibration import causal_ewm_detrend
from optimal_long_short.calibration.transforms import (
    ParameterBounds,
    params_to_shape_unc,
    shape_unc_to_params,
)
from optimal_long_short.drift import (
    expected_log_return_drift,
    with_expected_log_return_drift,
    with_zero_expected_log_return,
)
from optimal_long_short.job_runners.calibrate_eth_btc import (
    build_empirical_params,
    empirical_asset_exponents,
)
from optimal_long_short.model_params import KouParams


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


def test_causal_ewm_uses_only_lagged_mean_and_centers_innovations():
    result = causal_ewm_detrend(np.array([1.0, 2.0, 3.0]), 1.0)

    assert result.mean_path == pytest.approx([1.0, 5.0 / 3.0, 17.0 / 7.0])
    assert result.innovations == pytest.approx([1.0, 4.0 / 3.0])
    assert np.mean(result.centered_innovations) == pytest.approx(0.0, abs=1e-15)


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


def test_empirical_orientation_maximizes_role_adjusted_mu_spread():
    residual = with_zero_expected_log_return(_params())
    trends = {"WETH": -0.50, "WBTC": 0.10}
    rates = {
        "supply_eth": 0.02,
        "borrow_eth": 0.03,
        "supply_btc": 0.001,
        "borrow_btc": 0.004,
    }

    pre_carry = empirical_asset_exponents(residual, trends)
    final, residual_oriented, selection = build_empirical_params(residual, trends, rates)

    assert pre_carry["WBTC"] > pre_carry["WETH"]
    assert selection["long_asset"] == "WBTC"
    assert selection["short_asset"] == "WETH"
    assert final.mu1 == pytest.approx(residual_oriented.mu1 + trends["WBTC"] + rates["supply_btc"])
    assert final.mu2 == pytest.approx(residual_oriented.mu2 + trends["WETH"] + rates["borrow_eth"])
    assert final.mu1 > final.mu2


def test_role_carry_can_reverse_the_pre_carry_ranking():
    residual = with_zero_expected_log_return(_params())
    trends = {
        "WETH": 0.02 - residual.mu1,
        "WBTC": 0.01 - residual.mu2,
    }
    rates = {
        "supply_eth": 0.0,
        "borrow_eth": 0.0,
        "supply_btc": 0.03,
        "borrow_btc": 0.02,
    }

    pre_carry = empirical_asset_exponents(residual, trends)
    final, _, selection = build_empirical_params(residual, trends, rates)

    assert pre_carry["WETH"] > pre_carry["WBTC"]
    assert selection["long_asset"] == "WBTC"
    assert selection["short_asset"] == "WETH"
    assert selection["candidate_assignments"]["WBTC"]["mu_spread"] > 0.0
    assert final.mu1 > final.mu2
