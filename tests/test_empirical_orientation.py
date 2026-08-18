import json
import math

import pytest

from optimal_long_short.model.drift_service import expected_log_return_drift
from jobs.calibrate_eth_btc import (
    AAVE_CONFIG_BLOCK,
    AAVE_RISK,
    ASSET_RATE_SUFFIX,
)
from optimal_long_short.utils.helpers import (
    DEFAULT_EMPIRICAL_PARAMS,
    load_calibrated_params,
)
from jobs.empirical_method_comparison import no_shorting_limit
from optimal_long_short.model.kou_model import BivariateKouModel


def _payload() -> dict:
    return json.loads(DEFAULT_EMPIRICAL_PARAMS.read_text())


def test_primary_empirical_artifact_follows_data_selected_orientation():
    payload = _payload()
    meta = payload["meta"]
    selection = payload["orientation_selection"]
    constraint = payload["aave_constraint"]

    assert meta["asset1"] == selection["long_asset"] == constraint["collateral_asset"]
    assert meta["asset2"] == selection["short_asset"] == constraint["debt_asset"]
    candidates = selection["candidate_assignments"]
    selected_spread = candidates[meta["asset1"]]["g_spread"]
    assert selected_spread == max(row["g_spread"] for row in candidates.values())
    assert selected_spread > 0.0
    assert payload["params"]["mu1"] > payload["params"]["mu2"]
    assert constraint["emode_applied"] is False
    assert constraint["configuration_block"] == AAVE_CONFIG_BLOCK


def test_final_mu_decomposition_and_zero_mean_residuals():
    payload = _payload()
    params = payload["params"]
    residual = payload["params_residual_ecf_oriented"]
    components = payload["drift_components_by_asset"]
    asset1 = payload["meta"]["asset1"]
    asset2 = payload["meta"]["asset2"]

    assert params["mu1"] == pytest.approx(
        residual["mu1"]
        + components[asset1]["ewm_expected_log_drift"]
        + components[asset1]["role_carry"]
    )
    assert params["mu2"] == pytest.approx(
        residual["mu2"]
        + components[asset2]["ewm_expected_log_drift"]
        + components[asset2]["role_carry"]
    )
    assert payload["residual_expected_log_drift_oriented"] == pytest.approx(
        (0.0, 0.0), abs=1e-12
    )


def test_selected_role_rates_are_asset_specific():
    payload = _payload()
    selection = payload["orientation_selection"]
    rates = payload["aave_rates"]
    long_suffix = ASSET_RATE_SUFFIX[selection["long_asset"]]
    short_suffix = ASSET_RATE_SUFFIX[selection["short_asset"]]
    assert selection["long_supply_rate"] == pytest.approx(rates[f"supply_{long_suffix}"])
    assert selection["short_borrow_rate"] == pytest.approx(rates[f"borrow_{short_suffix}"])


def test_loader_maps_initial_prices_in_selected_asset_order():
    payload = _payload()
    _, constraint = load_calibrated_params(DEFAULT_EMPIRICAL_PARAMS)
    prices = payload["meta"]["initial_prices"]
    assert constraint["S10"] == pytest.approx(prices[payload["meta"]["asset1"]])
    assert constraint["S20"] == pytest.approx(prices[payload["meta"]["asset2"]])


def test_selected_collateral_terms_and_origination_boundary():
    payload = _payload()
    _, constraint = load_calibrated_params(DEFAULT_EMPIRICAL_PARAMS)
    collateral = payload["meta"]["asset1"]
    risk = AAVE_RISK[collateral]

    assert constraint["b"] == pytest.approx(risk["b"])
    assert constraint["ltv_max"] == pytest.approx(risk["ltv_max"])
    assert constraint["liq_bonus"] == pytest.approx(risk["liq_bonus"])
    assert constraint["h0_min"] == pytest.approx(math.log(risk["b"] / risk["ltv_max"]))
    assert constraint["H0_min"] == pytest.approx(risk["b"] / risk["ltv_max"])


def test_no_short_mean_uses_selected_long_asset_growth():
    params, _ = load_calibrated_params(DEFAULT_EMPIRICAL_PARAMS)
    horizon = 1.0 / 12.0
    model = BivariateKouModel(params)
    characteristic_mean = complex(
        math.e ** (horizon * model.levy_khintchine(-1j, 0))
    ).real
    assert characteristic_mean == pytest.approx(math.exp(params.mu1 * horizon))


def test_exact_no_shorting_limit_matches_first_two_long_asset_moments():
    params, _ = load_calibrated_params(DEFAULT_EMPIRICAL_PARAMS)
    horizon = 1.0 / 12.0
    mean, variance = no_shorting_limit(params, horizon)
    model = BivariateKouModel(params)
    raw2 = complex(math.e ** (horizon * model.levy_khintchine(-2j, 0))).real

    assert mean == pytest.approx(math.exp(params.mu1 * horizon))
    assert variance == pytest.approx(raw2 - mean**2)
    assert mean != pytest.approx(math.exp(expected_log_return_drift(params)[0] * horizon))
