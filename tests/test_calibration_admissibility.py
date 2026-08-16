import numpy as np
import pytest

from optimal_long_short.calibration.transforms import (
    ParameterBounds,
    nat_to_unc,
    unc_to_nat,
)
from optimal_long_short.calibration.calibrate import calibrate_ecf
from optimal_long_short.model_params import KouParams


def _valid_params(**updates):
    values = {
        "mu1": 0.05,
        "sigma1": 0.20,
        "lam1": 1.0,
        "p1": 0.5,
        "eta1_pos": 0.08,
        "eta1_neg": 0.09,
        "mu2": 0.03,
        "sigma2": 0.15,
        "lam2": 0.8,
        "p2": 0.45,
        "eta2_pos": 0.07,
        "eta2_neg": 0.06,
        "rho": 0.3,
    }
    values.update(updates)
    return KouParams(**values)


def test_calibration_transform_enforces_both_positive_jump_moment_caps():
    bounds = ParameterBounds(max_moment_order=4, moment_eps=1e-4)
    natural = unc_to_nat(np.full(13, 40.0), bounds)

    assert 4 * natural[4] < 1.0
    assert 4 * natural[10] < 1.0
    assert natural[4] <= bounds.eta_pos1_admissible_max
    assert natural[10] <= bounds.eta_pos2_max


def test_admissible_natural_unconstrained_round_trip():
    bounds = ParameterBounds(max_moment_order=4)
    theta = np.array(
        [0.05, 0.2, 1.0, 0.5, 0.08, 0.09,
         0.03, 0.15, 0.8, 0.45, 0.07, 0.06, 0.3]
    )

    assert unc_to_nat(nat_to_unc(theta, bounds), bounds) == pytest.approx(theta)


def test_calibration_rejects_bounds_weaker_than_requested_moment_order():
    returns = np.linspace(-0.01, 0.01, 20)

    with pytest.raises(ValueError, match="do not enforce the requested moment order"):
        calibrate_ecf(
            returns,
            returns[::-1],
            dt_years=1.0 / 365.0,
            bounds=ParameterBounds(max_moment_order=2),
            max_moment_order=4,
            n_starts=1,
            run_diagnostics=False,
        )


@pytest.mark.parametrize("order", [0, 1.5, True])
def test_parameter_bounds_requires_integer_moment_order(order):
    with pytest.raises(ValueError, match="positive integer"):
        ParameterBounds(max_moment_order=order)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sigma1": 0.0}, "sigma1 must be positive"),
        ({"lam2": 0.0}, "lam2 must be positive"),
        ({"p1": 1.0}, "p1 must lie"),
        ({"eta1_pos": 1.0}, "eta1_pos must lie"),
        ({"eta2_neg": 0.0}, "eta2_neg must be positive"),
        ({"rho": -1.0}, "rho must lie"),
    ],
)
def test_kou_parameter_domain_is_explicit(updates, message):
    with pytest.raises(ValueError, match=message):
        _valid_params(**updates)
