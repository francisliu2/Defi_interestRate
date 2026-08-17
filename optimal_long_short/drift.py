"""Drift-convention helpers for calibrated Kou parameters.

The package uses one convention throughout:

``KouParams.mu1`` and ``KouParams.mu2`` are annualized expected price-growth
rates, meaning ``E[exp(X_i(t))] = exp(mu_i t)``.

The characteristic function, moment resolvent, and Monte Carlo simulator use
the derived log-process drifts ``muX1`` and ``muX2``:

``muX_i = mu_i - 0.5*sigma_i^2 - lambda_i*E[exp(J_i)-1]``.

User views should normally be expressed on the price-growth drift ``mu``.
If a view is naturally stated as a log-process drift, use
``with_muX_drift_view`` to convert it safely.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping

from optimal_long_short.model_params import KouParams


PRICE_GROWTH_DRIFT = "price_growth_mu"
MUX_LOG_DRIFT = "log_process_muX"


def _copy_with(params: KouParams, **updates: float) -> KouParams:
    """Return a KouParams copy, letting __post_init__ refresh derived drifts."""
    base = {
        k: v
        for k, v in dataclasses.asdict(params).items()
        if k not in {"jump_compensator1", "jump_compensator2", "muX1", "muX2"}
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
        A copy with only ``mu1`` and/or ``mu2`` changed.  muX drifts are
        recomputed automatically.
    """
    next_mu1 = (params.mu1 if mu1 is None else float(mu1)) + float(delta_mu1)
    next_mu2 = (params.mu2 if mu2 is None else float(mu2)) + float(delta_mu2)
    return _copy_with(params, mu1=next_mu1, mu2=next_mu2)


def with_muX_drift_view(
    params: KouParams,
    *,
    muX1: float | None = None,
    muX2: float | None = None,
    delta_muX1: float = 0.0,
    delta_muX2: float = 0.0,
) -> KouParams:
    """
    Apply a user view stated on the log-process drift.

    This converts the requested muX drift back into the saved
    price-growth ``mu`` convention by adding the existing compensator.
    """
    next_muX1 = (
        params.muX1 if muX1 is None else float(muX1)
    ) + float(delta_muX1)
    next_muX2 = (
        params.muX2 if muX2 is None else float(muX2)
    ) + float(delta_muX2)
    return _copy_with(
        params,
        mu1=next_muX1 + params.jump_compensator1,
        mu2=next_muX2 + params.jump_compensator2,
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
    log-process drift ``muX_i``.
    """
    mean_jump1 = params.p1 * params.eta1_pos - (1.0 - params.p1) * params.eta1_neg
    mean_jump2 = params.p2 * params.eta2_pos - (1.0 - params.p2) * params.eta2_neg
    return (
        params.muX1 + params.lam1 * mean_jump1,
        params.muX2 + params.lam2 * mean_jump2,
    )


def residual_price_growth_correction(
    sigma: float,
    lam: float,
    p: float,
    eta_pos: float,
    eta_neg: float,
) -> float:
    """Price-growth exponent of a Kou residual with zero expected log drift.

    If ``E[X(t)]/t = 0``, the paper's expected-price-growth convention implies

    ``mu = 0.5*sigma**2 + lam*(chi - E[J])``.

    This is a Jensen/jump correction determined by the residual distribution's
    shape; it is not a directional return forecast.
    """
    mean_jump = p * eta_pos - (1.0 - p) * eta_neg
    chi = p / (1.0 - eta_pos) + (1.0 - p) / (1.0 + eta_neg) - 1.0
    return 0.5 * sigma**2 + lam * (chi - mean_jump)


def with_expected_log_return_drift(
    params: KouParams,
    *,
    drift1: float,
    drift2: float,
) -> KouParams:
    """Set annualized expected log-return drifts while preserving shape.

    ``drift1`` and ``drift2`` are rates of ``E[X_i(t)]/t``. They are
    converted to the stored expected-price-growth convention by adding the
    diffusion/jump correction implied by each fitted marginal law.
    """
    correction1 = residual_price_growth_correction(
        params.sigma1,
        params.lam1,
        params.p1,
        params.eta1_pos,
        params.eta1_neg,
    )
    correction2 = residual_price_growth_correction(
        params.sigma2,
        params.lam2,
        params.p2,
        params.eta2_pos,
        params.eta2_neg,
    )
    return _copy_with(
        params,
        mu1=float(drift1) + correction1,
        mu2=float(drift2) + correction2,
    )


def with_zero_expected_log_return(params: KouParams) -> KouParams:
    """Normalize both residual marginals to zero expected log drift."""
    return with_expected_log_return_drift(params, drift1=0.0, drift2=0.0)


def swap_asset_order(params: KouParams) -> KouParams:
    """Exchange asset-1 and asset-2 marginal parameters; preserve ``rho``."""
    return KouParams(
        mu1=params.mu2,
        sigma1=params.sigma2,
        lam1=params.lam2,
        p1=params.p2,
        eta1_pos=params.eta2_pos,
        eta1_neg=params.eta2_neg,
        mu2=params.mu1,
        sigma2=params.sigma1,
        lam2=params.lam1,
        p2=params.p1,
        eta2_pos=params.eta1_pos,
        eta2_neg=params.eta1_neg,
        rho=params.rho,
    )


def drift_summary(params: KouParams) -> dict[str, dict[str, float] | str]:
    """Return a machine-readable summary of all drift conventions."""
    log1, log2 = expected_log_return_drift(params)
    return {
        "saved_mu_convention": PRICE_GROWTH_DRIFT,
        "muX_convention": MUX_LOG_DRIFT,
        "asset1": {
            "mu_price_growth": params.mu1,
            "jump_price_compensator": params.jump_compensator1,
            "muX_log_process": params.muX1,
            "expected_log_return_drift": log1,
        },
        "asset2": {
            "mu_price_growth": params.mu2,
            "jump_price_compensator": params.jump_compensator2,
            "muX_log_process": params.muX2,
            "expected_log_return_drift": log2,
        },
        "spread": {
            "mu_price_growth_1_minus_2": params.mu1 - params.mu2,
            "muX_1_minus_2": params.muX1 - params.muX2,
            "expected_log_return_drift_1_minus_2": log1 - log2,
        },
    }
