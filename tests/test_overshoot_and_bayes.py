"""Regression tests for the exact jump-overshoot recovery and the Bayesian
sizing distribution (paper Sections 6.3 and 8)."""
import math

import pytest

from optimal_long_short import recovery
from optimal_long_short.drift_bayes import (
    gaussian_posterior,
    interior_threshold,
    optimal_buffer,
    spread_drift_standard_error,
)
from optimal_long_short.drift_bayes import _BufferGrid
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.overshoot_recovery import (
    joint_hitting_moment,
    exact_wealth_moments,
)

# WBTC/WETH benchmark calibration (results/params_WBTC_WETH_headline.json).
PARAMS = KouParams(
    mu1=0.30248996722292226, sigma1=0.26420423265737336, lam1=358.2931592376608,
    p1=0.7697042317968746, eta1_pos=0.004131388862606639, eta1_neg=0.003673309370070836,
    mu2=1.1607226355960722, sigma2=0.3282370358321696, lam2=450.65232184597335,
    p2=0.45721640656481194, eta2_pos=0.011257598489179722, eta2_neg=0.006008320305675488,
    rho=0.8474913799853285,
)
B = 0.78
LIQ = 0.05
T = 1.0 / 12.0
H0 = math.log(1.16)


@pytest.mark.parametrize("k", [1, 2])
def test_bivariate_tilt_reduces_to_debt_leg(k):
    """Hit(0,k) must equal the existing (0,k) debt-leg transform."""
    mine = joint_hitting_moment(PARAMS, 0.0, float(k), H0, T)
    base = recovery.hitting_moment(PARAMS, k, H0, T)
    assert mine == pytest.approx(base, abs=1e-11)


def test_hit_zero_is_liquidation_probability():
    """Hit(0,0) must equal 1 - p_surv."""
    from optimal_long_short.model.market_params import MarketParams
    from optimal_long_short.model.moments import ConditionalMoments
    from optimal_long_short.model.strategy import UnitExposureLongShortStrategy

    strat = UnitExposureLongShortStrategy(h0=H0, market=MarketParams(b=B, S10=1, S20=1), T=T)
    p_liq = 1.0 - ConditionalMoments(params=PARAMS, strategy=strat).p_surv()
    assert joint_hitting_moment(PARAMS, 0.0, 0.0, H0, T) == pytest.approx(p_liq, abs=1e-9)


def test_overshoot_lowers_recovery_and_raises_variance():
    """Exact recovery must not exceed the continuous-crossing upper bound, and
    it must fatten the wealth left tail."""
    w = exact_wealth_moments(PARAMS, H0, b=B, liq_bonus=LIQ, T=T)
    assert w["E_recovery"] < w["E_recovery_cc"]        # overshoot reduces recovery
    assert w["var_wealth"] > w["var_wealth_cc"]        # fatter left tail
    assert w["E_recovery"] == pytest.approx(0.1568, abs=2e-3)


def test_drift_standard_error():
    se = spread_drift_standard_error(PARAMS, T_obs=0.3835616438356164)
    assert se["se_total"] == pytest.approx(0.5406, abs=1e-3)
    assert se["se_diffusion"] == pytest.approx(0.2822, abs=1e-3)


@pytest.mark.slow
def test_interior_optimum_anchor_and_threshold():
    """Reproduce the paper's Section 8 anchor (exact overshoot recovery): at
    delta=+0.2 the interior optima sit near H0 = 1.30, 1.38, 1.88 for
    alpha = 0.5, 1, 2; and the benchmark negative drift gives the unlevered
    boundary."""
    grid = _BufferGrid(params=PARAMS, b=B, liq_bonus=LIQ, T=T)
    anchors = {0.5: 1.30, 1.0: 1.38, 2.0: 1.88}
    for alpha, expected in anchors.items():
        h0star, interior = optimal_buffer(grid, 0.2, alpha)
        assert interior
        assert h0star == pytest.approx(expected, abs=0.02)
    # benchmark spread drift -> unlevered boundary, no interior optimum
    _, interior = optimal_buffer(grid, PARAMS.mu1 - PARAMS.mu2, 1.0)
    assert not interior
    assert interior_threshold(grid, 1.0) == pytest.approx(0.10, abs=0.02)


def test_posterior_probability_interior_is_small():
    """Under the carry-anchored prior and the adverse ECF estimate, the levered
    position carries low posterior probability."""
    se = spread_drift_standard_error(PARAMS, T_obs=0.3835616438356164)["se_total"]
    post_mean, post_sd = gaussian_posterior(-0.025, 0.20, -0.858, se)
    assert post_mean == pytest.approx(-0.125, abs=0.01)
    assert post_sd == pytest.approx(0.188, abs=0.01)
