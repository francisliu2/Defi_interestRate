"""Moving-block bootstrap uncertainty for Kou calibration and sizing outputs.

The routines in this module keep the two return legs paired and resample
short contiguous stretches, rather than treating observations or assets as
independent.  They are intentionally lightweight: the bootstrap reuses the
existing ECF calibrator, moment-admissibility check, survival engine, and
objective-specific sizing selector.

The resulting percentile intervals quantify sampling variability conditional
on the chosen Kou specification, return preprocessing, block length, and
sizing grid.  They are not model-selection or out-of-sample forecast bands.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from optimal_long_short.calibration import calibrate_ecf
from optimal_long_short.kou_model import validate_moment_admissibility
from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.sizing import select_liquidation_constrained
from optimal_long_short.strategy import UnitExposureLongShortStrategy


PARAMETER_NAMES = tuple(
    field.name for field in dataclasses.fields(KouParams) if field.init
)


def moving_block_bootstrap_indices(
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one circular moving-block bootstrap index vector.

    Blocks start uniformly over all observations and wrap around the sample.
    Circular wrapping gives every observation the same chance of appearing in
    every within-block position.  The returned vector has exactly ``n_obs``
    entries and should be applied to both return legs.
    """

    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    if block_length < 1 or block_length > n_obs:
        raise ValueError("block_length must lie between 1 and n_obs")

    n_blocks = math.ceil(n_obs / block_length)
    starts = rng.integers(0, n_obs, size=n_blocks)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % n_obs
    return indices.ravel()[:n_obs]


def survival_grid_report(
    params: KouParams,
    H0_grid: Sequence[float] | np.ndarray,
    *,
    b: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    ltv_max: float | None = None,
) -> list[dict[str, float]]:
    """Evaluate survival on a health-factor grid for a sizing selector.

    Only objective-independent survival quantities and initial leverage are
    computed.  This avoids recalculating killed payoff moments when the
    downstream rule is solely a liquidation-probability constraint.
    """

    health = np.asarray(H0_grid, dtype=float)
    if health.ndim != 1 or len(health) == 0:
        raise ValueError("H0_grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(health)) or np.any(health <= 1.0):
        raise ValueError("Every H0_grid value must be finite and greater than 1")

    market = MarketParams(b=b, S10=S10, S20=S20)
    rows: list[dict[str, float]] = []
    for H0 in health:
        h0 = float(np.log(H0))
        strategy = UnitExposureLongShortStrategy(
            h0=h0,
            market=market,
            T=T,
            ltv_max=ltv_max,
        )
        raw_p_surv = float(
            ConditionalMoments(params=params, strategy=strategy).p_surv()
        )
        if (
            not math.isfinite(raw_p_surv)
            or raw_p_surv < -1e-8
            or raw_p_surv > 1.0 + 1e-8
        ):
            raise ValueError(
                f"Invalid survival inversion at H0={H0}: {raw_p_surv}"
            )
        p_surv = float(np.clip(raw_p_surv, 0.0, 1.0))
        rows.append(
            {
                "h0": h0,
                "H0": float(H0),
                "initial_leverage": float(H0 / (H0 - b)),
                "p_surv": p_surv,
                "p_liq": 1.0 - p_surv,
            }
        )
    return rows


def liquidation_constrained_downstream(
    params: KouParams,
    H0_grid: Sequence[float] | np.ndarray,
    *,
    reference_H0: float,
    pbar: float,
    b: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    ltv_max: float | None = None,
) -> dict[str, float | bool]:
    """Return reference survival and a constrained selected buffer.

    The rule is explicit: among grid points with ``p_liq <= pbar``, select the
    one with the greatest initial leverage.  If no point is feasible, selected
    fields are NaN and ``selection_feasible`` is false.
    """

    rows = survival_grid_report(
        params,
        H0_grid,
        b=b,
        T=T,
        S10=S10,
        S20=S20,
        ltv_max=ltv_max,
    )
    reference_index = int(
        np.argmin(np.abs(np.asarray([row["H0"] for row in rows]) - reference_H0))
    )
    reference = rows[reference_index]
    output: dict[str, float | bool] = {
        "reference_H0": float(reference["H0"]),
        "p_surv_at_reference_H0": float(reference["p_surv"]),
        "p_liq_at_reference_H0": float(reference["p_liq"]),
    }

    try:
        selection = select_liquidation_constrained(rows, pbar=pbar)
    except ValueError:
        output.update(
            {
                "selection_feasible": False,
                "selected_H0": math.nan,
                "selected_h0": math.nan,
                "selected_initial_leverage": math.nan,
                "selected_p_surv": math.nan,
                "selected_p_liq": math.nan,
            }
        )
        return output

    selected = selection.selected
    output.update(
        {
            "selection_feasible": True,
            "selected_H0": float(selected["H0"]),
            "selected_h0": float(selected["h0"]),
            "selected_initial_leverage": float(selected["initial_leverage"]),
            "selected_p_surv": float(selected["p_surv"]),
            "selected_p_liq": float(selected["p_liq"]),
        }
    )
    return output


def calibration_bootstrap_record(
    r1: np.ndarray,
    r2: np.ndarray,
    indices: np.ndarray,
    *,
    dt_years: float,
    replicate: int,
    calibration_kwargs: Mapping[str, Any] | None = None,
    parameter_adjuster: Callable[[KouParams], KouParams] | None = None,
    downstream_evaluator: Callable[[KouParams], Mapping[str, Any]] | None = None,
    max_moment_order: int = 4,
    calibrator: Callable[..., Any] = calibrate_ecf,
) -> dict[str, Any]:
    """Calibrate and evaluate one pre-indexed paired bootstrap sample.

    ``calibrator`` and the two evaluator hooks make the orchestration easy to
    test without running numerical optimization.  The default path uses the
    repository ECF calibration directly.
    """

    returns1 = np.asarray(r1, dtype=float)
    returns2 = np.asarray(r2, dtype=float)
    sample_indices = np.asarray(indices, dtype=int)
    if returns1.ndim != 1 or returns1.shape != returns2.shape:
        raise ValueError("r1 and r2 must be one-dimensional arrays of equal length")
    if sample_indices.shape != returns1.shape:
        raise ValueError("indices must have the same one-dimensional shape as returns")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(returns1)):
        raise ValueError("indices contain an out-of-range observation")

    result = calibrator(
        returns1[sample_indices],
        returns2[sample_indices],
        dt_years,
        **dict(calibration_kwargs or {}),
    )
    params = result.params
    if parameter_adjuster is not None:
        params = parameter_adjuster(params)
    validate_moment_admissibility(params, max_moment_order)

    record: dict[str, Any] = {
        "replicate": int(replicate),
        "calibration_success": bool(result.success),
        "objective": float(result.objective),
        "n_iter": int(result.n_iter),
    }
    record.update({name: float(getattr(params, name)) for name in PARAMETER_NAMES})
    if downstream_evaluator is not None:
        record.update(dict(downstream_evaluator(params)))
    return record


def summarize_bootstrap_records(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    point_estimates: Mapping[str, float] | None = None,
    confidence_level: float = 0.90,
    require_convergence: bool = True,
) -> list[dict[str, float | int | str]]:
    """Compute percentile intervals and dispersion for named record fields."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not records:
        raise ValueError("At least one bootstrap record is required")

    alpha = (1.0 - confidence_level) / 2.0
    point = dict(point_estimates or {})
    output: list[dict[str, float | int | str]] = []
    for metric in metrics:
        values = []
        for record in records:
            if require_convergence and not bool(record.get("calibration_success", False)):
                continue
            try:
                value = float(record[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if not values:
            raise ValueError(f"No finite bootstrap estimates for metric {metric!r}")

        sample = np.asarray(values)
        lower, median, upper = np.quantile(sample, [alpha, 0.5, 1.0 - alpha])
        output.append(
            {
                "metric": metric,
                "point_estimate": float(point.get(metric, math.nan)),
                "bootstrap_mean": float(np.mean(sample)),
                "bootstrap_std": (
                    float(np.std(sample, ddof=1)) if len(sample) > 1 else math.nan
                ),
                "ci_lower": float(lower),
                "bootstrap_median": float(median),
                "ci_upper": float(upper),
                "confidence_level": float(confidence_level),
                "n_finite": int(len(sample)),
            }
        )
    return output
