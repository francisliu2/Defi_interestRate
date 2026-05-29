"""Liquidation and moment reports over admissible h0 grids."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.strategy import UnitExposureLongShortStrategy


def _standardized_from_raw(raw: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(raw) >= 1:
        out["conditional_mean"] = raw[0]
    if len(raw) >= 2:
        var = raw[1] - raw[0] ** 2
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
    max_moment_order: int = 4,
    clip_probabilities: bool = True,
) -> list[dict[str, float]]:
    """
    Compute survival/liquidation probabilities and conditional moments by h0.

    Parameters
    ----------
    params : KouParams
        Calibrated parameters after applying any user drift view.
    h0_grid : iterable of float
        Initial log-health values.  Values must satisfy ``h0 > log(b)``.
    b, T, S10, S20 : float
        Market and horizon inputs for the strategy.
    max_moment_order : int
        Number of conditional raw moments to compute.  Uses 1..K.

    Returns
    -------
    list[dict[str, float]]
        One row per h0 with ``p_surv``, ``p_liq``, raw conditional moments,
        variance/skew/kurtosis when available, and initial leverage.
    """
    if max_moment_order < 1:
        raise ValueError("max_moment_order must be at least 1.")
    if max_moment_order * params.eta2_pos >= 1.0:
        raise ValueError(
            "Moment tilt is not admissible: "
            f"max_moment_order * eta2_pos = {max_moment_order * params.eta2_pos:.6g} >= 1."
        )

    market = MarketParams(b=b, S10=S10, S20=S20)
    rows: list[dict[str, float]] = []
    for h0 in h0_grid:
        h0 = float(h0)
        strategy = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
        cm = ConditionalMoments(params=params, strategy=strategy)
        p_surv = float(cm.p_surv())
        if clip_probabilities:
            p_surv = float(np.clip(p_surv, 0.0, 1.0))
        raw = [float(cm.conditional_moment(k)) for k in range(1, max_moment_order + 1)]
        row = {
            "h0": h0,
            "H0": math.exp(h0),
            "initial_leverage": math.exp(h0) / (math.exp(h0) - b),
            "p_surv": p_surv,
            "p_liq": 1.0 - p_surv,
        }
        for k, value in enumerate(raw, start=1):
            row[f"conditional_moment_{k}"] = value
        row.update(_standardized_from_raw(raw))
        rows.append(row)
    return rows

