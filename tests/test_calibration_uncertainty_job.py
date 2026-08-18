import dataclasses

import numpy as np
import pytest

from optimal_long_short.model.drift_service import (
    expected_log_return_drift,
    with_zero_expected_log_return,
)
from jobs.legacy_jobs import calibration_uncertainty as job
from optimal_long_short.model.model_params import KouParams


def _residual_market_params() -> KouParams:
    return with_zero_expected_log_return(
        KouParams(
            mu1=0.0,
            sigma1=0.30,
            lam1=2.0,
            p1=0.45,
            eta1_pos=0.08,
            eta1_neg=0.06,
            mu2=0.0,
            sigma2=0.20,
            lam2=1.5,
            p2=0.55,
            eta2_pos=0.07,
            eta2_neg=0.05,
            rho=0.4,
        )
    )


def test_fixed_showcase_inputs_orient_shape_and_add_expected_log_drifts():
    residual = _residual_market_params()
    final = job._apply_fixed_showcase_inputs(
        residual,
        long_asset="WBTC",
        short_asset="WETH",
        endpoint_trends={"WETH": -0.20, "WBTC": 0.10},
        long_carry=0.01,
        short_carry=0.03,
    )

    assert final.sigma1 == pytest.approx(residual.sigma2)
    assert final.sigma2 == pytest.approx(residual.sigma1)
    assert expected_log_return_drift(final) == pytest.approx((0.11, -0.17))


def test_replicate_worker_preserves_residual_location_and_uses_shape_only_fit(monkeypatch):
    residual = _residual_market_params()

    def fake_bootstrap_record(
        sample1,
        sample2,
        indices,
        *,
        calibration_kwargs,
        parameter_adjuster,
        replicate,
        **kwargs,
    ):
        assert np.array_equal(indices, np.arange(len(sample1)))
        assert np.mean(sample1) == pytest.approx(4.0)
        assert np.mean(sample2) == pytest.approx(8.0)
        assert sample2 == pytest.approx(2.0 * sample1)
        assert calibration_kwargs["drift_mode"] == "zero_expected_log_return"
        final = parameter_adjuster(residual)
        assert final.sigma1 == pytest.approx(residual.sigma2)
        assert expected_log_return_drift(final) == pytest.approx((0.11, -0.17))
        return {"replicate": replicate, "calibration_success": True}

    monkeypatch.setattr(job, "calibration_bootstrap_record", fake_bootstrap_record)
    values = np.arange(8.0)
    task = {
        "replicate": 3,
        "r1": values,
        "r2": 2.0 * values,
        "dt_years": 1.0 / 365.0,
        "block_length": 3,
        "bootstrap_seed": 19,
        "calibration_seed": 23,
        "n_starts": 1,
        "raw_point": dataclasses.asdict(residual),
        "long_asset": "WBTC",
        "short_asset": "WETH",
        "endpoint_trends": {"WETH": -0.20, "WBTC": 0.10},
        "long_carry": 0.01,
        "short_carry": 0.03,
        "health_grid": np.array([1.10, 1.20]),
        "reference_H0": 1.10,
        "pbar": 0.10,
        "b": 0.78,
        "T": 1.0 / 12.0,
        "S10": 1.0,
        "S20": 1.0,
        "ltv_max": 0.73,
    }

    record = job._replicate_worker(task)

    assert record == {"replicate": 3, "calibration_success": True}
