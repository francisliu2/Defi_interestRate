"""Drift-convention helpers for calibrated Kou parameters.

The package uses one convention throughout:

``KouParams.mu1`` and ``KouParams.mu2`` are annualized expected price-growth
rates, meaning ``E[exp(X_i(t))] = exp(mu_i t)``.

The characteristic function, moment resolvent, and Monte Carlo simulator use
the derived log-process drifts ``effective_mu1`` and ``effective_mu2``:

``effective_mu_i = mu_i - 0.5*sigma_i^2 - lambda_i*E[exp(J_i)-1]``.

User views should normally be expressed on the price-growth drift ``mu``.
If a view is naturally stated as a log-process drift, use
``with_effective_drift_view`` to convert it safely.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

from optimal_long_short.model_params import KouParams


PRICE_GROWTH_DRIFT = "price_growth_mu"
EFFECTIVE_LOG_DRIFT = "effective_log_process_mu"


def _copy_with(params: KouParams, **updates: float) -> KouParams:
    """Return a KouParams copy, letting __post_init__ refresh derived drifts."""
    base = {
        k: v
        for k, v in dataclasses.asdict(params).items()
        if k not in {"jump_compensator1", "jump_compensator2", "effective_mu1", "effective_mu2"}
    }
    base.update(updates)
    return KouParams(**base)


def with_price_drift_view(
    params: KouParams,
    *,
    mu1: float | None = None,
    mu2: float | None = None,
    delta_mu1: float = 0.0,
    delta_mu2: float = 0.0,
) -> KouParams:
    """
    Apply a user view to annualized expected price-growth drifts.

    Parameters
    ----------
    params : KouParams
        Base calibrated parameters.
    mu1, mu2 : float, optional
        Absolute annualized price-growth views.  If omitted, the base value is
        used.
    delta_mu1, delta_mu2 : float
        Annualized additive views applied after any absolute view.

    Returns
    -------
    KouParams
        A copy with only ``mu1`` and/or ``mu2`` changed.  Effective drifts are
        recomputed automatically.
    """
    next_mu1 = (params.mu1 if mu1 is None else float(mu1)) + float(delta_mu1)
    next_mu2 = (params.mu2 if mu2 is None else float(mu2)) + float(delta_mu2)
    return _copy_with(params, mu1=next_mu1, mu2=next_mu2)


def with_effective_drift_view(
    params: KouParams,
    *,
    effective_mu1: float | None = None,
    effective_mu2: float | None = None,
    delta_effective_mu1: float = 0.0,
    delta_effective_mu2: float = 0.0,
) -> KouParams:
    """
    Apply a user view stated on the log-process drift.

    This converts the requested effective drift back into the saved
    price-growth ``mu`` convention by adding the existing compensator.
    """
    next_eff1 = (
        params.effective_mu1 if effective_mu1 is None else float(effective_mu1)
    ) + float(delta_effective_mu1)
    next_eff2 = (
        params.effective_mu2 if effective_mu2 is None else float(effective_mu2)
    ) + float(delta_effective_mu2)
    return _copy_with(
        params,
        mu1=next_eff1 + params.jump_compensator1,
        mu2=next_eff2 + params.jump_compensator2,
    )


def apply_price_drift_view(
    params: KouParams,
    view: Mapping[str, float] | None = None,
) -> KouParams:
    """
    Convenience wrapper accepting keys ``mu1``, ``mu2``, ``delta_mu1``,
    and ``delta_mu2``.
    """
    if not view:
        return params
    allowed = {"mu1", "mu2", "delta_mu1", "delta_mu2"}
    extra = set(view) - allowed
    if extra:
        raise ValueError(f"Unsupported price drift view keys: {sorted(extra)}")
    return with_price_drift_view(
        params,
        mu1=view.get("mu1"),
        mu2=view.get("mu2"),
        delta_mu1=view.get("delta_mu1", 0.0),
        delta_mu2=view.get("delta_mu2", 0.0),
    )


def expected_log_return_drift(params: KouParams) -> tuple[float, float]:
    """
    Annualized drift of E[X_i(t)]/t, including mean jump sizes.

    This differs from both the saved price-growth drift ``mu_i`` and the
    log-process drift ``effective_mu_i``.
    """
    mean_jump1 = params.p1 * params.eta1_pos - (1.0 - params.p1) * params.eta1_neg
    mean_jump2 = params.p2 * params.eta2_pos - (1.0 - params.p2) * params.eta2_neg
    return (
        params.effective_mu1 + params.lam1 * mean_jump1,
        params.effective_mu2 + params.lam2 * mean_jump2,
    )


def drift_summary(params: KouParams) -> dict[str, dict[str, float] | str]:
    """Return a machine-readable summary of all drift conventions."""
    log1, log2 = expected_log_return_drift(params)
    return {
        "saved_mu_convention": PRICE_GROWTH_DRIFT,
        "effective_mu_convention": EFFECTIVE_LOG_DRIFT,
        "asset1": {
            "mu_price_growth": params.mu1,
            "jump_price_compensator": params.jump_compensator1,
            "effective_mu_log_process": params.effective_mu1,
            "expected_log_return_drift": log1,
        },
        "asset2": {
            "mu_price_growth": params.mu2,
            "jump_price_compensator": params.jump_compensator2,
            "effective_mu_log_process": params.effective_mu2,
            "expected_log_return_drift": log2,
        },
        "spread": {
            "mu_price_growth_1_minus_2": params.mu1 - params.mu2,
            "effective_mu_1_minus_2": params.effective_mu1 - params.effective_mu2,
            "expected_log_return_drift_1_minus_2": log1 - log2,
        },
    }

