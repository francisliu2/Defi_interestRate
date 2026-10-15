"""Checks on the semi-analytic Kelly objective (optimal_long_short.kelly)."""
import math

import pytest

from optimal_long_short.kelly import (
    expected_log_wealth,
    half_kelly_buffer,
    killed_joint_moment,
    overshoot_law,
)
from optimal_long_short.model.market_params import MarketParams
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.model.moments import ConditionalMoments
from optimal_long_short.model.strategy import UnitExposureLongShortStrategy

# WBTC/WETH benchmark calibration (results/params_WBTC_WETH_headline.json).
PARAMS = KouParams(
    mu1=0.30248996722292226, sigma1=0.26420423265737336, lam1=358.2931592376608,
    p1=0.7697042317968746, eta1_pos=0.004131388862606639, eta1_neg=0.003673309370070836,
    mu2=1.1607226355960722, sigma2=0.3282370358321696, lam2=450.65232184597335,
    p2=0.45721640656481194, eta2_pos=0.011257598489179722, eta2_neg=0.006008320305675488,
    rho=0.8474913799853285,
)
B, LIQ, T = 0.78, 0.05, 1.0 / 12.0
H0 = math.log(1.16)


def _cm(h0):
    s = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=B, S10=1.0, S20=1.0), T=T)
    return ConditionalMoments(params=PARAMS, strategy=s)


def test_untilted_killed_moment_is_survival():
    assert killed_joint_moment(PARAMS, 0.0, 0.0, H0, T) == pytest.approx(_cm(H0).p_surv(), abs=1e-10)


def test_tilted_moments_rebuild_killed_mean():
    """E[Pi_T] = (H0 S(1,0) - b S(0,1)) / (H0 - b) must match the integer-moment resolvent."""
    H = math.exp(H0)
    rebuilt = (H * killed_joint_moment(PARAMS, 1.0, 0.0, H0, T)
               - B * killed_joint_moment(PARAMS, 0.0, 1.0, H0, T)) / (H - B)
    assert rebuilt == pytest.approx(_cm(H0).killed_moment(1), abs=1e-10)


def test_overshoot_law_is_a_three_point_mixture():
    """Memorylessness: creep atom + two exponential phases reproduce every hitting moment."""
    law = overshoot_law(PARAMS, H0, T, check_orders=(4, 7, 15))
    assert law.max_check_error < 1e-9
    assert law.p_liq == pytest.approx(1.0 - _cm(H0).p_surv(), abs=1e-10)
    assert all(p >= -1e-12 for p in (law.p_creep, *law.p_jump))


def test_no_shorting_limit_is_expected_log_return_of_asset_1():
    """As h0 -> inf the book is a unit long in asset 1, so E[log W_T] -> E[X1_T]."""
    p = PARAMS
    expected = T * (p.muX1 + p.lam1 * (p.p1 * p.eta1_pos - (1 - p.p1) * p.eta1_neg))
    r = expected_log_wealth(PARAMS, 12.0, b=B, liq_bonus=LIQ, T=T)
    assert r["E_log_W"] == pytest.approx(expected, abs=1e-6)


def test_bad_debt_is_negligible_at_fitted_jumps():
    r = expected_log_wealth(PARAMS, H0, b=B, liq_bonus=LIQ, T=T)
    assert 0.0 < r["p_bad_debt"] < 1e-7
    assert r["series_tail_bound"] < 1e-5


def test_half_kelly_halves_the_debt():
    Hk = 1.3
    Hh = half_kelly_buffer(Hk, B)
    assert B / (Hh - B) == pytest.approx(0.5 * B / (Hk - B))


def test_heavy_jump_tails_cap_the_series_instead_of_failing():
    """With eta2_pos = 3.2% (full-history daily fit) the tilt order must stay below
    1/eta2_pos ~ 31; the series is capped and its truncation bound stays small."""
    heavy = KouParams(**{**{k: getattr(PARAMS, k) for k in (
        "mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
        "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho")},
        "eta2_pos": 0.0318, "lam2": 81.0, "lam1": 90.0})
    r = expected_log_wealth(heavy, H0, b=B, liq_bonus=LIQ, T=T)
    assert math.isfinite(r["E_log_W"])
    assert r["series_tail_bound"] < 1e-4
