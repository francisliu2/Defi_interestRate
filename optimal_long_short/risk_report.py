"""Liquidation and moment reports over admissible h0 grids."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

import numpy as np

from optimal_long_short.market_params import MarketParams
from optimal_long_short.kou_model import validate_moment_admissibility
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.strategy import UnitExposureLongShortStrategy


_NUMERICAL_TOL = 1e-8


def _variance_from_moments(second: float, mean: float, *, label: str) -> float:
    """Return a non-negative variance, tolerating only roundoff-scale negatives."""
    variance = float(second - mean**2)
    scale = max(1.0, abs(second), mean**2)
    if variance < -_NUMERICAL_TOL * scale:
        raise ValueError(
            f"{label} variance is materially negative ({variance}); "
            "the moment inversion is numerically inconsistent"
        )
    return max(0.0, variance)


def _standardized_from_raw(raw: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(raw) >= 1:
        out["conditional_mean"] = raw[0]
    if len(raw) >= 2:
        var = _variance_from_moments(
            raw[1],
            raw[0],
            label="Conditional",
        )
        out["conditional_variance"] = var
    if len(raw) >= 3 and out.get("conditional_variance", 0.0) > 0.0:
        std = math.sqrt(out["conditional_variance"])
        out["conditional_skewness"] = (
            raw[2] - 3.0 * raw[0] * raw[1] + 2.0 * raw[0] ** 3
        ) / std ** 3
    if len(raw) >= 4 and out.get("conditional_variance", 0.0) > 0.0:
        var = out["conditional_variance"]
        out["conditional_excess_kurtosis"] = (
            raw[3] - 4.0 * raw[0] * raw[2] + 6.0 * raw[0] ** 2 * raw[1] - 3.0 * raw[0] ** 4
        ) / var ** 2 - 3.0
    return out


def h0_liquidation_moment_report(
    params: KouParams,
    h0_grid: Iterable[float],
    *,
    b: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    ltv_max: Optional[float] = None,
    max_moment_order: int = 4,
    clip_probabilities: bool = True,
) -> list[dict[str, float]]:
    """
    Compute objective-independent killed-payoff outputs and conditional moments.

    Parameters
    ----------
    params : KouParams
        Calibrated parameters after applying any user drift view.
    h0_grid : iterable of float
        Initial log-health values. Values must satisfy ``h0 > 0`` without an
        origination constraint, or ``h0 >= log(b / ltv_max)`` when
        ``ltv_max`` is supplied.
    b, T, S10, S20 : float
        Market and horizon inputs for the strategy.
    ltv_max : float, optional
        Maximum LTV at origination. If supplied, every grid point is checked
        against the corresponding feasible lower bound.
    max_moment_order : int
        Number of killed and conditional raw moments to compute. Uses 1..K.
        Both ``K * eta1_pos`` and ``K * eta2_pos`` must be less than one.

    Returns
    -------
    list[dict[str, float]]
        One row per h0 with ``p_surv``, ``p_liq``, ``killed_moment_k``, raw
        conditional moments, unconditional killed-payoff mean/variance,
        conditional variance/skew/kurtosis when available, and leverage.
    """
    if max_moment_order < 1:
        raise ValueError("max_moment_order must be at least 1.")
    validate_moment_admissibility(params, max_moment_order)

    market = MarketParams(b=b, S10=S10, S20=S20)
    rows: list[dict[str, float]] = []
    for h0 in h0_grid:
        h0 = float(h0)
        strategy = UnitExposureLongShortStrategy(
            h0=h0, market=market, T=T, ltv_max=ltv_max
        )
        cm = ConditionalMoments(params=params, strategy=strategy)
        raw_p_surv = float(cm.p_surv())
        if not math.isfinite(raw_p_surv):
            raise ValueError(
                f"Survival inversion must be finite at h0={h0}, got {raw_p_surv}"
            )
        if raw_p_surv < -_NUMERICAL_TOL or raw_p_surv > 1.0 + _NUMERICAL_TOL:
            raise ValueError(
                f"Survival inversion lies outside [0, 1] at h0={h0}: "
                f"{raw_p_surv}"
            )
        p_surv = raw_p_surv
        if clip_probabilities:
            p_surv = float(np.clip(p_surv, 0.0, 1.0))
        if p_surv <= 0.0:
            raise ValueError(
                f"Conditional moments are undefined with zero survival at h0={h0}"
            )
        killed = [
            float(cm.killed_moment(k))
            for k in range(1, max_moment_order + 1)
        ]
        if not all(math.isfinite(value) for value in killed):
            raise ValueError(f"Killed-moment inversion is non-finite at h0={h0}")
        conditional = [value / p_surv for value in killed]
        row = {
            "h0": h0,
            "H0": math.exp(h0),
            "initial_leverage": math.exp(h0) / (math.exp(h0) - b),
            "p_surv": p_surv,
            "p_liq": 1.0 - p_surv,
        }
        for k, value in enumerate(killed, start=1):
            row[f"killed_moment_{k}"] = value
        for k, value in enumerate(conditional, start=1):
            row[f"conditional_moment_{k}"] = value
        row["unconditional_mean"] = killed[0]
        if len(killed) >= 2:
            row["unconditional_variance"] = _variance_from_moments(
                killed[1],
                killed[0],
                label="Unconditional",
            )
        row.update(_standardized_from_raw(conditional))
        rows.append(row)
    return rows
