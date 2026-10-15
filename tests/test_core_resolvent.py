"""Checks on the core Laplace-resolvent machinery (paper Sections 3--5).

These pin the building blocks of the evaluation map against quantities known
independently of the resolvent: the tilted Levy exponent (Lemma 3.1), the
no-shorting limit (Remark 4.1), the closed-form Brownian first-passage law in
the vanishing-jump limit, and the k=0 first-passage transform (eq. fp-transform).
"""
import math

import numpy as np
import pytest

from optimal_long_short import recovery
from optimal_long_short.kou_model import BivariateKouModel, KouZTiltedDynamics
from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.strategy import UnitExposureLongShortStrategy

# WBTC/WETH benchmark calibration (results/params_WBTC_WETH.json).
PARAMS = KouParams(
    mu1=0.30248996722292226, sigma1=0.26420423265737336, lam1=358.2931592376608,
    p1=0.7697042317968746, eta1_pos=0.004131388862606639, eta1_neg=0.003673309370070836,
    mu2=1.1607226355960722, sigma2=0.3282370358321696, lam2=450.65232184597335,
    p2=0.45721640656481194, eta2_pos=0.011257598489179722, eta2_neg=0.006008320305675488,
    rho=0.8474913799853285,
)
B = 0.78
T = 1.0 / 12.0


def _cm(params: KouParams, h0: float, T: float = T) -> ConditionalMoments:
    strategy = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=B, S10=1.0, S20=1.0), T=T)
    return ConditionalMoments(params=params, strategy=strategy)


@pytest.mark.parametrize("k", [0, 1, 2])
@pytest.mark.parametrize("s", [-3.0, -0.5, 0.7, 2.0])
def test_tilted_exponent_matches_joint_exponent(k, s):
    """Lemma 3.1: psi_Z^(k)(s) = Psi(-is, -i(k-s)) - Psi(0, -ik)."""
    joint = BivariateKouModel(params=PARAMS)
    expected = joint.levy_khintchine(-1j * s, -1j * (k - s)) - joint.levy_khintchine(0, -1j * k)
    got = KouZTiltedDynamics(params=PARAMS, k=k)(s)
    assert complex(got) == pytest.approx(complex(expected), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_no_shorting_limit(k):
    """Remark 4.1: E[Pi_T^k] -> E[exp(k X_{1,T})] as h0 -> infinity."""
    limit = np.exp(T * BivariateKouModel(params=PARAMS).levy_khintchine(-1j * k, 0)).real
    assert _cm(PARAMS, h0=20.0).killed_moment(k) == pytest.approx(limit, rel=1e-6)


def test_no_shorting_mean_is_price_growth():
    """k=1 limit equals exp(mu1 T) by the drift normalisation (eq. muX-drift)."""
    assert _cm(PARAMS, h0=20.0).killed_moment(1) == pytest.approx(math.exp(PARAMS.mu1 * T), rel=1e-6)


def _brownian_survival(h: float, mu: float, sigma: float, t: float) -> float:
    """P(inf_{s<=t} (mu s + sigma W_s) > -h) for drifted Brownian motion."""
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    st = sigma * math.sqrt(t)
    return phi((h + mu * t) / st) - math.exp(-2.0 * mu * h / sigma**2) * phi((-h + mu * t) / st)


@pytest.mark.parametrize("h0", [0.05, 0.10, 0.25])
def test_survival_matches_brownian_first_passage_as_jumps_vanish(h0):
    """With negligible jump intensity, Z is drifted Brownian motion and the
    resolvent survival must reproduce the reflection-principle formula."""
    p = KouParams(
        mu1=0.10, sigma1=0.50, lam1=1e-8, p1=0.5, eta1_pos=0.02, eta1_neg=0.03,
        mu2=0.40, sigma2=0.60, lam2=1e-8, p2=0.5, eta2_pos=0.04, eta2_neg=0.05,
        rho=0.3,
    )
    sigma_z = math.sqrt(p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2)
    mu_z = p.muX1 - p.muX2
    expected = _brownian_survival(h0, mu_z, sigma_z, T)
    assert _cm(p, h0).p_surv() == pytest.approx(expected, abs=1e-6)


def test_survival_is_a_probability_and_increases_with_buffer():
    ps = [_cm(PARAMS, h0).p_surv() for h0 in (0.07, 0.10, 0.25, 0.50)]
    assert all(0.0 <= x <= 1.0 + 1e-9 for x in ps)
    assert all(a < b for a, b in zip(ps, ps[1:]))


def test_k0_first_passage_transform_is_liquidation_probability():
    """eq. fp-transform at k=0 returns p_liq; it must agree with Proposition 5.2."""
    h0 = math.log(1.16)
    assert recovery.hitting_moment(PARAMS, 0, h0, T) == pytest.approx(1.0 - _cm(PARAMS, h0).p_surv(), abs=1e-9)


def test_conditional_moments_are_consistent():
    """Conditional variance positive; killed moments satisfy Cauchy--Schwarz."""
    cm = _cm(PARAMS, 0.10)
    assert cm.conditional_variance() > 0.0
    assert cm.killed_moment(1) ** 2 <= cm.killed_moment(2) * cm.p_surv() * (1 + 1e-9)
