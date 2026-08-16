import math

import pytest

from optimal_long_short.market_params import MarketParams
from optimal_long_short.strategy import (
    UnitExposureLongShortStrategy,
    minimum_feasible_h0,
)


@pytest.fixture
def market():
    return MarketParams(b=0.78, S10=75_000.0, S20=2_000.0)


def test_unconstrained_strategy_requires_initial_solvency(market):
    with pytest.raises(ValueError, match="greater than 0"):
        UnitExposureLongShortStrategy(h0=0.0, market=market, T=1.0 / 12.0)
    with pytest.raises(ValueError, match="greater than 0"):
        UnitExposureLongShortStrategy(h0=-0.1, market=market, T=1.0 / 12.0)

    strategy = UnitExposureLongShortStrategy(
        h0=1e-6, market=market, T=1.0 / 12.0
    )
    assert strategy.H > 1.0


def test_origination_ltv_bound_is_inclusive(market):
    ltv_max = 0.73
    h0_min = minimum_feasible_h0(market.b, ltv_max)
    strategy = UnitExposureLongShortStrategy(
        h0=h0_min,
        market=market,
        T=1.0 / 12.0,
        ltv_max=ltv_max,
    )

    assert strategy.initial_ltv == pytest.approx(ltv_max)
    assert h0_min == pytest.approx(math.log(0.78 / 0.73))


def test_origination_ltv_bound_rejects_infeasible_buffer(market):
    ltv_max = 0.73
    h0_min = minimum_feasible_h0(market.b, ltv_max)
    with pytest.raises(ValueError, match="origination-LTV constraint"):
        UnitExposureLongShortStrategy(
            h0=h0_min - 1e-4,
            market=market,
            T=1.0 / 12.0,
            ltv_max=ltv_max,
        )


@pytest.mark.parametrize("ltv_max", [0.0, -0.1, 0.78, 0.9])
def test_invalid_origination_ltv_is_rejected(market, ltv_max):
    with pytest.raises(ValueError, match="ltv_max must be in"):
        UnitExposureLongShortStrategy(
            h0=0.1,
            market=market,
            T=1.0 / 12.0,
            ltv_max=ltv_max,
        )
