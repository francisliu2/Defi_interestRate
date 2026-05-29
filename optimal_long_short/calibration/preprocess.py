"""
Return-series preprocessing utilities for ECF calibration.
"""
from __future__ import annotations

import numpy as np


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
