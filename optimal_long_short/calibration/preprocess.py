"""
Return-series preprocessing utilities for ECF calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausalEWMResult:
    """Outputs from causal EWM trend removal."""

    mean_path: np.ndarray
    innovations: np.ndarray
    centered_innovations: np.ndarray
    innovation_mean: float
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


def causal_ewm_detrend(
    r: np.ndarray,
    half_life_periods: float,
) -> CausalEWMResult:
    """Construct lagged-mean innovations for shape-only calibration.

    The innovation at index ``t >= 1`` is ``r[t] - m[t-1]``; hence its trend
    estimate contains no part of the contemporaneous return. The first return
    initializes the normalized EWM and is not itself used as an innovation.
    Innovations are centered exactly before ECF shape estimation, deliberately
    keeping directional location outside the residual-law calibration.
    """
    values = np.asarray(r, dtype=float)
    mean_path = normalized_ewm_mean(values, half_life_periods)
    if len(values) < 2:
        raise ValueError("At least two returns are required for causal detrending")
    innovations = values[1:] - mean_path[:-1]
    innovation_mean = float(np.mean(innovations))
    centered = innovations - innovation_mean
    decay = float(2.0 ** (-1.0 / half_life_periods))
    return CausalEWMResult(
        mean_path=mean_path,
        innovations=innovations,
        centered_innovations=centered,
        innovation_mean=innovation_mean,
        decay=decay,
        half_life_periods=float(half_life_periods),
    )
