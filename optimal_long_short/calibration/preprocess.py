"""
Return-series preprocessing utilities for ECF calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EWMResidualResult:
    """EWM mean path and the residual increments supplied to the ECF fit."""

    ewm_mean_path_per_period: np.ndarray
    residual_increments: np.ndarray
    residual_sample_mean: float
    decay: float
    half_life_periods: float


def ewm_smooth(r: np.ndarray, span: float) -> np.ndarray:
    """
    Exponential weighted mean smoother for a return series.

    Applies the recursive filter:
        out[0] = r[0]
        out[i] = alpha * r[i] + (1 - alpha) * out[i-1]

    where alpha = 2 / (span + 1).  Larger span = more smoothing.
    The output has the same length as the input.

    Parameters
    ----------
    r    : (N,) array of log-returns.
    span : EWM span (equivalent to pandas ewm(span=span)).
           span=1 leaves returns unchanged (alpha=1); span->inf converges
           to a cumulative mean.

    Returns
    -------
    (N,) array of smoothed returns.
    """
    if span <= 0:
        raise ValueError(f"span must be positive, got {span!r}.")
    r = np.asarray(r, dtype=float)
    alpha = 2.0 / (span + 1.0)
    beta = 1.0 - alpha
    out = np.empty_like(r)
    out[0] = r[0]
    for i in range(1, len(r)):
        out[i] = alpha * r[i] + beta * out[i - 1]
    return out


def normalized_ewm_mean(r: np.ndarray, half_life_periods: float) -> np.ndarray:
    """Return the finite-sample normalized EWM mean path.

    The geometric decay is ``beta = 2**(-1 / half_life_periods)``. At time
    ``t`` the mean uses observations through ``r[t]`` with weights normalized
    to sum to one, avoiding dependence on an arbitrary recursive initial
    state.
    """
    if not np.isfinite(half_life_periods) or half_life_periods <= 0.0:
        raise ValueError("half_life_periods must be finite and positive")
    values = np.asarray(r, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("r must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("r must contain only finite values")

    decay = float(2.0 ** (-1.0 / half_life_periods))
    numerator = 0.0
    denominator = 0.0
    means = np.empty_like(values)
    for index, value in enumerate(values):
        numerator = float(value) + decay * numerator
        denominator = 1.0 + decay * denominator
        means[index] = numerator / denominator
    return means


def construct_ewm_residual_increments(
    r: np.ndarray,
    half_life_periods: float,
) -> EWMResidualResult:
    """Subtract the lagged normalized EWM mean from each observed increment.

    For ``t >= 1``, the residual is ``r[t] - ewm_mean[t-1]``. The lag makes the
    transformation causal: the contemporaneous increment is not used in its
    own conditional-mean estimate. The first increment initializes the EWM path
    and is not itself supplied to the ECF fit. No additional sample demeaning is
    performed; the realized residual mean is retained as a model diagnostic.
    """
    values = np.asarray(r, dtype=float)
    ewm_mean_path = normalized_ewm_mean(values, half_life_periods)
    if len(values) < 2:
        raise ValueError("At least two increments are required for EWM subtraction")
    residual_increments = values[1:] - ewm_mean_path[:-1]
    decay = float(2.0 ** (-1.0 / half_life_periods))
    return EWMResidualResult(
        ewm_mean_path_per_period=ewm_mean_path,
        residual_increments=residual_increments,
        residual_sample_mean=float(np.mean(residual_increments)),
        decay=decay,
        half_life_periods=float(half_life_periods),
    )
