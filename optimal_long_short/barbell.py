"""Barbell construction: long the two convexities the levered book is short.

The empirical cross-section in ``results/pair_cross_section.csv`` says the
leveraged long-short book on Aave is a bad trade in a specific, structural way.
Every ETH- and BTC-collateral pair carries *negative* net carry (-1.8% to -4.5%:
BTC wrappers pay 0.00-0.01% as collateral while debt costs 2.5-4.5%), so the
position collects no premium.  It nevertheless holds two short-convexity
positions:

1. **Short the barrier.**  The payoff is killed at liquidation -- a small,
   bounded upside against a rare near-total loss.
2. **Short the utilization kink.**  The borrower pays ``r_b(U)``, which is
   piecewise linear with a slope that jumps by a factor of 25 or more at
   ``U_opt`` (published ``slope2`` is 0.60-0.80 for WETH and the stables, 3.00
   for the BTC wrappers).  The borrower is short a call on utilization.

The barbell inverts both.  A large safe allocation earns the supply rate with a
buffer so deep that liquidation is negligible -- in the framework this is the
``h0 -> inf`` limit of Remark 4, where the payoff converges to a unit long and
the barrier becomes unreachable.  A small allocation buys the two convexities:

*Supplier leg.*  A pure supplier (no borrow, hence no barrier) earns
``(1 - reserve_factor) * U * r_b(U)``, a convex function of ``U``.  Convexity is
paid for by Jensen: the supplier's expected yield strictly exceeds the yield at
expected utilization, and the gap is the value of the kink.

*Keeper leg.*  The liquidator collects the bonus on seized collateral, a payoff
that is zero except on ``{tau <= T}``.  This is the same first-passage transform
that :mod:`optimal_long_short.recovery` uses for ``R_tau`` -- what the borrower
forfeits is what the keeper collects -- so the two legs of the trade are priced
by one object with opposite signs.

Both legs are on-protocol and need no options market.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats

from optimal_long_short.aave_hourly import RateCurve
from optimal_long_short.model_params import KouParams
from optimal_long_short.rate_curve import JacobiParams
from optimal_long_short.recovery import hitting_moment

__all__ = [
    "SupplierLeg",
    "supplier_leg",
    "keeper_leg",
    "BarbellResult",
    "barbell",
]


@dataclass(frozen=True)
class SupplierLeg:
    """Expected yield of a pure supplier, and the value of the kink.

    Attributes
    ----------
    expected_yield : float
        ``E[(1 - rf) U r_b(U)]`` under the stationary law of utilization.
    yield_at_mean : float
        ``(1 - rf) E[U] r_b(E[U])`` -- what a model that ignores the kink and
        uses average utilization would report.
    jensen_gap : float
        ``expected_yield - yield_at_mean``.  Strictly positive because the
        supplier's rate is convex in ``U``; this is the premium earned for being
        long the kink, and it is invisible to any deterministic-rate model.
    yield_stressed, yield_normal : float
        Conditional expected yields above and below ``U_opt``.
    convexity_ratio : float
        ``yield_stressed / yield_normal``.
    """

    expected_yield: float
    yield_at_mean: float
    jensen_gap: float
    yield_stressed: float
    yield_normal: float
    convexity_ratio: float


def _beta_law(jacobi: JacobiParams) -> stats.rv_continuous:
    """Stationary Beta law of the Jacobi diffusion."""
    s2 = jacobi.sigma**2
    a = 2.0 * jacobi.kappa * jacobi.theta / s2
    b = 2.0 * jacobi.kappa * (1.0 - jacobi.theta) / s2
    if a <= 0.0 or b <= 0.0:
        raise ValueError(
            f"Non-stationary Jacobi parameters: Beta shape ({a:.4g}, {b:.4g})."
        )
    return stats.beta(a, b)


def supplier_leg(
    curve: RateCurve,
    jacobi: JacobiParams,
    reserve_factor: float,
) -> SupplierLeg:
    """Price the supplier's long position in the utilization kink.

    Expectations are taken under the stationary Beta law implied by the Jacobi
    fit, which is the honest stationary description of a bounded utilization
    process -- unlike a CIR fit, which has no mechanism to respect ``U <= 1``.
    """
    law = _beta_law(jacobi)
    scale = 1.0 - reserve_factor

    def yield_at(u):
        return scale * u * curve.borrow_rate(u)

    expected, _ = integrate.quad(
        lambda u: yield_at(u) * law.pdf(u), 0.0, 1.0, limit=200
    )
    mean_u = float(law.mean())
    at_mean = yield_at(mean_u)

    mass_stressed = float(law.sf(curve.u_opt))
    mass_normal = 1.0 - mass_stressed
    if mass_stressed > 1e-12:
        num, _ = integrate.quad(
            lambda u: yield_at(u) * law.pdf(u), curve.u_opt, 1.0, limit=200
        )
        y_stressed = num / mass_stressed
    else:
        y_stressed = float("nan")
    if mass_normal > 1e-12:
        num, _ = integrate.quad(
            lambda u: yield_at(u) * law.pdf(u), 0.0, curve.u_opt, limit=200
        )
        y_normal = num / mass_normal
    else:
        y_normal = float("nan")

    return SupplierLeg(
        expected_yield=float(expected),
        yield_at_mean=float(at_mean),
        jensen_gap=float(expected - at_mean),
        yield_stressed=float(y_stressed),
        yield_normal=float(y_normal),
        convexity_ratio=float(y_stressed / y_normal) if y_normal else float("nan"),
    )


def keeper_leg(
    params: KouParams,
    h0: float,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    close_factor: float = 0.5,
    inverter=None,
) -> dict[str, float]:
    """Expected keeper revenue from liquidating one position of buffer ``h0``.

    The liquidator repays up to ``close_factor`` of the outstanding debt and
    seizes collateral worth ``(1 + liq_bonus)`` times the repayment, so revenue
    is ``liq_bonus * close_factor * V_{2,tau}`` on ``{tau <= T}``.  Aave's close
    factor is 0.5 for a position at the boundary and 1.0 once the health factor
    falls below 0.95; 0.5 is the conservative default here.

    Normalized per unit of the *borrower's* initial equity, so it is directly
    comparable with the levered book's payoff: the keeper and the borrower are
    two sides of the same barrier.
    """
    scale = b / (math.exp(h0) - b)  # = n_2 S_{2,0}
    expected = (
        liq_bonus * close_factor * scale * hitting_moment(params, 1, h0, T, inverter)
    )
    return {
        "h0": h0,
        "H0": math.exp(h0),
        "debt_per_equity": scale,
        "expected_revenue": float(expected),
    }


@dataclass(frozen=True)
class BarbellResult:
    """A barbell allocation and the levered book it is compared against."""

    w_safe: float
    w_keeper: float
    safe_yield: float
    keeper_revenue: float
    barbell_mean: float
    levered_mean: float
    levered_var: float
    levered_p_liq: float
    jensen_gap: float


def barbell(
    params: KouParams,
    *,
    curve: RateCurve,
    jacobi: JacobiParams,
    reserve_factor: float,
    h0_levered: float,
    h0_safe: float,
    b: float,
    liq_bonus: float,
    T: float,
    w_keeper: float = 0.10,
    close_factor: float = 0.5,
    inverter=None,
) -> BarbellResult:
    """Compare a barbell against the levered book at ``h0_levered``.

    The safe leg supplies at ``h0_safe``, a buffer deep enough that liquidation
    is negligible, and earns the kink-convex supply yield.  The convex leg puts
    ``w_keeper`` into keeper revenue on positions of buffer ``h0_levered`` --
    that is, it is long exactly the barrier the levered book is short.
    """
    from optimal_long_short.recovery import wealth_moments

    supply = supplier_leg(curve, jacobi, reserve_factor)
    keep = keeper_leg(
        params,
        h0_levered,
        b=b,
        liq_bonus=liq_bonus,
        T=T,
        close_factor=close_factor,
        inverter=inverter,
    )
    levered = wealth_moments(
        params, h0_levered, b=b, liq_bonus=liq_bonus, T=T, inverter=inverter
    )

    w_safe = 1.0 - w_keeper
    safe_payoff = w_safe * (1.0 + supply.expected_yield * T)
    keeper_payoff = w_keeper * keep["expected_revenue"]

    return BarbellResult(
        w_safe=w_safe,
        w_keeper=w_keeper,
        safe_yield=supply.expected_yield,
        keeper_revenue=keep["expected_revenue"],
        barbell_mean=float(safe_payoff + keeper_payoff),
        levered_mean=levered["mean_wealth"],
        levered_var=levered["var_wealth"],
        levered_p_liq=levered["p_liq"],
        jensen_gap=supply.jensen_gap,
    )
