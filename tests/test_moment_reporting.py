from dataclasses import replace

import pytest

from optimal_long_short.kou_model import KouZTiltedDynamics
from optimal_long_short.inversion import TalbotInverter
from optimal_long_short.laplace_resolvent import HomogeneousSolution, ParticularSolution
from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments, SurvivalResolvent
from optimal_long_short.risk_report import h0_liquidation_moment_report
from optimal_long_short.strategy import UnitExposureLongShortStrategy


@pytest.fixture
def params() -> KouParams:
    return KouParams(
        mu1=0.05,
        sigma1=0.20,
        lam1=1.0,
        p1=0.5,
        eta1_pos=0.10,
        eta1_neg=0.08,
        mu2=0.03,
        sigma2=0.15,
        lam2=0.8,
        p2=0.45,
        eta2_pos=0.10,
        eta2_neg=0.07,
        rho=0.3,
    )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("eta1_pos", "k \\* eta1_pos"),
        ("eta2_pos", "k \\* eta2_pos"),
    ],
)
def test_tilt_checks_both_positive_jump_moment_domains(
    params: KouParams,
    field: str,
    match: str,
) -> None:
    inadmissible = replace(params, **{field: 0.25})

    with pytest.raises(ValueError, match=match):
        KouZTiltedDynamics(params=inadmissible, k=4)


def test_survival_tilt_allows_zero_order(params: KouParams) -> None:
    dynamics = KouZTiltedDynamics(
        params=replace(params, eta1_pos=0.8, eta2_pos=0.8),
        k=0,
    )

    assert dynamics.k == 0


def test_survival_resolvent_reduces_coincident_phase_system(
    params: KouParams,
) -> None:
    coincident = replace(params, eta2_pos=params.eta1_neg)
    dynamics = KouZTiltedDynamics(params=coincident, k=0)
    strategy = UnitExposureLongShortStrategy(
        h0=0.2,
        market=MarketParams(b=0.78, S10=1.0, S20=1.0),
        T=1.0 / 12.0,
    )
    resolvent = SurvivalResolvent(dynamics=dynamics, strategy=strategy)
    q = 0.5

    roots = resolvent._genuine_negative_roots(q)
    matrix = resolvent._barrier_matrix(q)
    coefficients = resolvent.coefficients(q)

    assert len(roots) == 2
    assert matrix.shape == (2, 2)
    assert coefficients.shape == (2,)
    assert matrix @ coefficients == pytest.approx([-1.0 / q, -1.0 / q])
    assert resolvent.evaluate_at_origin(q).real > 0.0


def test_killed_moment_reduces_coincident_phase_system(
    params: KouParams,
) -> None:
    k = 2
    shared_rate = 1.0 / params.eta1_neg
    coincident = replace(params, eta2_pos=1.0 / (shared_rate + k))
    strategy = UnitExposureLongShortStrategy(
        h0=0.2,
        market=MarketParams(b=0.78, S10=1.0, S20=1.0),
        T=1.0 / 12.0,
    )
    particular = ParticularSolution(
        dynamics=KouZTiltedDynamics(params=coincident, k=k),
        strategy=strategy,
    )
    homogeneous = HomogeneousSolution(particular=particular)

    assert homogeneous.barrier_matrix(0.5).shape == (2, 2)
    assert homogeneous.coefficients(0.5).shape == (2,)

    killed = ConditionalMoments(
        params=coincident,
        strategy=strategy,
        inverter=TalbotInverter(M=16),
    ).killed_moment(k)
    assert killed > 0.0


def test_conditional_moment_rejects_inadmissible_long_leg_order(
    params: KouParams,
) -> None:
    strategy = UnitExposureLongShortStrategy(
        h0=0.2,
        market=MarketParams(b=0.78, S10=1.0, S20=1.0),
        T=1.0 / 12.0,
    )
    moments = ConditionalMoments(
        params=replace(params, eta1_pos=0.25),
        strategy=strategy,
    )

    with pytest.raises(ValueError, match="k \\* eta1_pos"):
        moments.killed_moment(4)


def test_report_exposes_killed_and_conditional_outputs_without_reinverting_survival(
    params: KouParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubConditionalMoments:
        survival_calls = 0
        killed_orders: list[int] = []

        def __init__(self, *, params, strategy) -> None:
            self.params = params
            self.strategy = strategy

        def p_surv(self) -> float:
            type(self).survival_calls += 1
            return 0.8

        def killed_moment(self, k: int) -> float:
            type(self).killed_orders.append(k)
            return {1: 0.8, 2: 1.6}[k]

    monkeypatch.setattr(
        "optimal_long_short.risk_report.ConditionalMoments",
        StubConditionalMoments,
    )

    rows = h0_liquidation_moment_report(
        params,
        [0.2],
        b=0.78,
        T=1.0 / 12.0,
        max_moment_order=2,
    )

    assert StubConditionalMoments.survival_calls == 1
    assert StubConditionalMoments.killed_orders == [1, 2]

    row = rows[0]
    assert row["p_surv"] == pytest.approx(0.8)
    assert row["p_liq"] == pytest.approx(0.2)
    assert row["killed_moment_1"] == pytest.approx(0.8)
    assert row["killed_moment_2"] == pytest.approx(1.6)
    assert row["unconditional_mean"] == pytest.approx(0.8)
    assert row["unconditional_variance"] == pytest.approx(0.96)

    # Existing conditional keys remain available and use the same survival
    # probability computed once for this row.
    assert row["conditional_moment_1"] == pytest.approx(1.0)
    assert row["conditional_moment_2"] == pytest.approx(2.0)
    assert row["conditional_mean"] == pytest.approx(1.0)
    assert row["conditional_variance"] == pytest.approx(1.0)


def test_report_validates_long_leg_domain_before_computation(params: KouParams) -> None:
    with pytest.raises(ValueError, match="k \\* eta1_pos"):
        h0_liquidation_moment_report(
            replace(params, eta1_pos=0.25),
            [0.2],
            b=0.78,
            T=1.0 / 12.0,
            max_moment_order=4,
        )


def test_report_rejects_materially_invalid_survival_probability(
    params: KouParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidSurvivalMoments:
        def __init__(self, *, params, strategy) -> None:
            pass

        def p_surv(self) -> float:
            return 1.01

        def killed_moment(self, k: int) -> float:
            return 1.0

    monkeypatch.setattr(
        "optimal_long_short.risk_report.ConditionalMoments",
        InvalidSurvivalMoments,
    )

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        h0_liquidation_moment_report(
            params,
            [0.2],
            b=0.78,
            T=1.0 / 12.0,
            max_moment_order=2,
        )


def test_report_uses_clipped_probability_consistently(
    params: KouParams,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RoundoffSurvivalMoments:
        def __init__(self, *, params, strategy) -> None:
            pass

        def p_surv(self) -> float:
            return 1.0 + 1e-10

        def killed_moment(self, k: int) -> float:
            return {1: 1.0, 2: 1.1}[k]

    monkeypatch.setattr(
        "optimal_long_short.risk_report.ConditionalMoments",
        RoundoffSurvivalMoments,
    )

    row = h0_liquidation_moment_report(
        params,
        [0.2],
        b=0.78,
        T=1.0 / 12.0,
        max_moment_order=2,
    )[0]

    assert row["p_surv"] == 1.0
    assert row["conditional_mean"] == pytest.approx(1.0)


def test_positive_laplace_abscissa_is_shifted_before_talbot_inversion(
    params: KouParams,
) -> None:
    growing = replace(params, mu1=5.0)
    strategy = UnitExposureLongShortStrategy(
        h0=10.0,
        market=MarketParams(b=0.78, S10=1.0, S20=1.0),
        T=1.0,
    )

    moment_32 = ConditionalMoments(
        params=growing,
        strategy=strategy,
        inverter=TalbotInverter(M=32),
    ).killed_moment(4)
    moment_48 = ConditionalMoments(
        params=growing,
        strategy=strategy,
        inverter=TalbotInverter(M=48),
    ).killed_moment(4)

    assert moment_32 > 0.0
    assert moment_32 == pytest.approx(moment_48, rel=2e-7)
