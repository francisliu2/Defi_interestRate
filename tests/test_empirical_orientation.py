import json
import math

import pytest

from optimal_long_short.kou_model import BivariateKouModel
from optimal_long_short.job_runners.calibrate_eth_btc import (
    AAVE_B,
    AAVE_CONFIG_BLOCK,
    AAVE_LIQ_BONUS,
    AAVE_LTV_MAX,
)
from optimal_long_short.job_runners.common import RESULTS_DIR, load_calibrated_params


def test_primary_empirical_artifact_is_long_weth_short_wbtc():
    path = RESULTS_DIR / "params_WETH_WBTC.json"
    payload = json.loads(path.read_text())

    assert payload["meta"]["asset1"] == "WETH"
    assert payload["meta"]["asset2"] == "WBTC"
    assert payload["aave_constraint"]["collateral_asset"] == "WETH"
    assert payload["aave_constraint"]["debt_asset"] == "WBTC"
    assert payload["aave_constraint"]["emode_applied"] is False
    assert payload["aave_constraint"]["configuration_block"] == AAVE_CONFIG_BLOCK

    params = payload["params"]
    raw = payload["params_raw_ecf"]
    rates = payload["aave_rates"]
    assert params["mu1"] - raw["mu1"] == pytest.approx(rates["supply_eth"])
    assert params["mu2"] - raw["mu2"] == pytest.approx(rates["borrow_btc"])


def test_loader_maps_initial_prices_in_calibrated_asset_order():
    path = RESULTS_DIR / "params_WETH_WBTC.json"
    payload = json.loads(path.read_text())
    _, constraint = load_calibrated_params(path)

    prices = payload["meta"]["initial_prices"]
    assert constraint["asset1"] == "WETH"
    assert constraint["asset2"] == "WBTC"
    assert constraint["S10"] == pytest.approx(prices["WETH"])
    assert constraint["S20"] == pytest.approx(prices["WBTC"])


def test_weth_collateral_terms_and_origination_boundary():
    _, constraint = load_calibrated_params(RESULTS_DIR / "params_WETH_WBTC.json")

    assert constraint["b"] == pytest.approx(AAVE_B)
    assert constraint["ltv_max"] == pytest.approx(AAVE_LTV_MAX)
    assert constraint["liq_bonus"] == pytest.approx(AAVE_LIQ_BONUS)
    assert constraint["h0_min"] == pytest.approx(math.log(AAVE_B / AAVE_LTV_MAX))
    assert constraint["H0_min"] == pytest.approx(AAVE_B / AAVE_LTV_MAX)


def test_no_short_mean_is_long_weth_price_growth():
    params, _ = load_calibrated_params(RESULTS_DIR / "params_WETH_WBTC.json")
    horizon = 1.0 / 12.0
    model = BivariateKouModel(params)
    long_weth_mean = math.exp(params.mu1 * horizon)
    characteristic_mean = complex(
        math.e ** (horizon * model.levy_khintchine(-1j, 0))
    ).real
    assert characteristic_mean == pytest.approx(long_weth_mean)
