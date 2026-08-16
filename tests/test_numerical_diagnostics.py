import csv
import math

import numpy as np
import pytest

from optimal_long_short.inversion import TalbotInverter
from optimal_long_short.kou_model import KouZTiltedDynamics
from optimal_long_short.laplace_resolvent import ParticularSolution
from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.numerical_diagnostics import (
    killed_barrier_cancellation_residuals,
    root_quality_diagnostics,
    talbot_convergence_diagnostics,
    talbot_nodes,
    write_numerical_diagnostics_csv,
)
from optimal_long_short.strategy import UnitExposureLongShortStrategy


@pytest.fixture
def params() -> KouParams:
    return KouParams(
        mu1=0.05,
        sigma1=0.20,
        lam1=1.0,
        p1=0.52,
        eta1_pos=0.08,
        eta1_neg=0.09,
        mu2=0.03,
        sigma2=0.15,
        lam2=0.8,
        p2=0.45,
        eta2_pos=0.07,
        eta2_neg=0.06,
        rho=0.3,
    )


@pytest.fixture
def strategy() -> UnitExposureLongShortStrategy:
    return UnitExposureLongShortStrategy(
        h0=0.2,
        market=MarketParams(b=0.78, S10=100.0, S20=10.0),
        T=0.1,
    )


def test_talbot_inverts_constant_transform() -> None:
    value = TalbotInverter(M=16).invert(lambda q: 1.0 / q, T=0.7)

    assert value == pytest.approx(1.0, rel=1e-10, abs=1e-10)


def test_talbot_inverts_stable_exponential_transform() -> None:
    decay_rate = 2.0
    T = 0.7
    value = TalbotInverter(M=16).invert(
        lambda q: 1.0 / (q + decay_rate),
        T=T,
    )

    assert value == pytest.approx(math.exp(-decay_rate * T), rel=1e-9, abs=1e-10)


def test_root_quality_diagnostic_invariants(
    params: KouParams,
    strategy: UnitExposureLongShortStrategy,
) -> None:
    diagnostics = root_quality_diagnostics(
        params,
        strategy,
        M=8,
        tilt_orders=(0, 1, 2, 3, 4),
    )

    assert [row.k for row in diagnostics] == [0, 1, 2, 3, 4]
    for row in diagnostics:
        assert row.n_nodes == 8
        assert row.max_relative_root_residual < 1e-9
        assert row.min_root_separation > 0.0
        assert row.min_root_pole_distance > 0.0
        assert np.isfinite(row.max_barrier_condition_number)
        assert row.max_barrier_condition_number >= 1.0
        assert row.max_barrier_system_residual < 1e-10


def test_killed_coefficients_cancel_truncation_residuals(
    params: KouParams,
    strategy: UnitExposureLongShortStrategy,
) -> None:
    particular = ParticularSolution(
        dynamics=KouZTiltedDynamics(params=params, k=2),
        strategy=strategy,
    )

    residuals = killed_barrier_cancellation_residuals(
        particular,
        q=2.5 + 1.1j,
    )

    assert residuals.max_abs < 1e-11


def test_convergence_records_and_compact_csv(
    params: KouParams,
    strategy: UnitExposureLongShortStrategy,
    tmp_path,
) -> None:
    convergence = talbot_convergence_diagnostics(
        params,
        strategy,
        M_values=(8, 12),
        moment_orders=(1,),
    )
    roots = root_quality_diagnostics(
        params,
        strategy,
        M=6,
        tilt_orders=(0, 1),
    )

    assert len(convergence) == 4
    assert convergence[0].previous_value is None
    assert convergence[2].previous_value is not None
    assert convergence[2].relative_change is not None
    assert convergence[2].relative_change >= 0.0

    output = tmp_path / "diagnostics.csv"
    write_numerical_diagnostics_csv(output, convergence, roots)
    with output.open(newline="") as handle:
        records = list(csv.DictReader(handle))

    assert len(records) == len(convergence) + len(roots)
    assert {record["diagnostic"] for record in records} == {
        "talbot_convergence",
        "root_quality",
    }


def test_talbot_node_grid_matches_requested_order() -> None:
    nodes = talbot_nodes(M=10, T=0.25)

    assert len(nodes) == 10
    assert nodes[0].imag == 0.0
    assert nodes[0].real > 0.0
    assert all(np.isfinite(node.real) and np.isfinite(node.imag) for node in nodes)
