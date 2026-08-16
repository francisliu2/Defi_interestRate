from types import SimpleNamespace

import numpy as np
import pytest

import optimal_long_short.calibration_uncertainty as uncertainty
from optimal_long_short.model_params import KouParams


def _params(**updates) -> KouParams:
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


def test_circular_moving_blocks_have_exact_length_and_local_order():
    n_obs = 11
    block_length = 4
    indices = uncertainty.moving_block_bootstrap_indices(
        n_obs,
        block_length,
        np.random.default_rng(17),
    )

    assert indices.shape == (n_obs,)
    assert np.all((0 <= indices) & (indices < n_obs))
    for start in range(0, n_obs, block_length):
        block = indices[start : start + block_length]
        if len(block) > 1:
            assert np.all(np.diff(block) % n_obs == 1)


def test_bootstrap_record_applies_one_index_vector_to_both_return_legs():
    r1 = np.arange(8.0)
    r2 = r1 + 100.0
    indices = np.array([6, 7, 0, 1, 3, 4, 5, 6])

    def fake_calibrator(sample1, sample2, dt_years, **kwargs):
        assert np.array_equal(sample2 - sample1, np.full(8, 100.0))
        assert dt_years == pytest.approx(1.0 / 365.0)
        assert kwargs == {"marker": 12}
        return SimpleNamespace(
            params=_params(),
            objective=0.25,
            success=True,
            n_iter=7,
        )

    record = uncertainty.calibration_bootstrap_record(
        r1,
        r2,
        indices,
        dt_years=1.0 / 365.0,
        replicate=3,
        calibration_kwargs={"marker": 12},
        parameter_adjuster=lambda params: _params(mu1=params.mu1 + 0.01),
        downstream_evaluator=lambda params: {"p_surv": 0.9 + params.mu1},
        calibrator=fake_calibrator,
    )

    assert record["replicate"] == 3
    assert record["calibration_success"] is True
    assert record["mu1"] == pytest.approx(0.06)
    assert record["p_surv"] == pytest.approx(0.96)


def test_bootstrap_record_rejects_nonadmissible_fourth_moment_fit():
    def fake_calibrator(*args, **kwargs):
        return SimpleNamespace(
            params=_params(eta2_pos=0.26),
            objective=0.0,
            success=True,
            n_iter=1,
        )

    with pytest.raises(ValueError, match=r"k \* eta2_pos = 1.04"):
        uncertainty.calibration_bootstrap_record(
            np.arange(4.0),
            np.arange(4.0),
            np.arange(4),
            dt_years=1.0,
            replicate=0,
            calibrator=fake_calibrator,
        )


def test_liquidation_downstream_uses_explicit_constraint_and_reference(monkeypatch):
    class FakeMoments:
        def __init__(self, params, strategy):
            self.strategy = strategy

        def p_surv(self):
            return self.strategy.H - 0.35

    monkeypatch.setattr(uncertainty, "ConditionalMoments", FakeMoments)
    output = uncertainty.liquidation_constrained_downstream(
        _params(),
        np.array([1.10, 1.20, 1.30]),
        reference_H0=1.10,
        pbar=0.10,
        b=0.78,
        T=1.0 / 12.0,
    )

    assert output["p_surv_at_reference_H0"] == pytest.approx(0.75)
    assert output["selection_feasible"] is True
    assert output["selected_H0"] == pytest.approx(1.30)
    assert output["selected_p_liq"] == pytest.approx(0.05)


def test_survival_grid_rejects_materially_invalid_inversion(monkeypatch):
    class FakeMoments:
        def __init__(self, params, strategy):
            pass

        def p_surv(self):
            return 1.01

    monkeypatch.setattr(uncertainty, "ConditionalMoments", FakeMoments)
    with pytest.raises(ValueError, match="Invalid survival inversion"):
        uncertainty.survival_grid_report(
            _params(),
            np.array([1.10]),
            b=0.78,
            T=1.0 / 12.0,
        )


def test_percentile_summary_filters_nonconverged_and_nonfinite_records():
    records = [
        {"calibration_success": True, "theta": 1.0},
        {"calibration_success": False, "theta": 100.0},
        {"calibration_success": True, "theta": 3.0},
        {"calibration_success": True, "theta": np.nan},
    ]
    [summary] = uncertainty.summarize_bootstrap_records(
        records,
        ["theta"],
        point_estimates={"theta": 2.0},
        confidence_level=0.50,
    )

    assert summary["point_estimate"] == pytest.approx(2.0)
    assert summary["bootstrap_mean"] == pytest.approx(2.0)
    assert summary["bootstrap_std"] == pytest.approx(np.sqrt(2.0))
    assert summary["ci_lower"] == pytest.approx(1.5)
    assert summary["bootstrap_median"] == pytest.approx(2.0)
    assert summary["ci_upper"] == pytest.approx(2.5)
    assert summary["n_finite"] == 2
