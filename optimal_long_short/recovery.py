"""Liquidation recovery: the ``R_tau`` term of Remark 1.

The base framework kills the payoff at the barrier, ``Pi_T = (n1 S1,T - n2 S2,T)
1{tau > T}``, setting terminal wealth to zero on liquidated paths.  That
convention is deliberately severe: it is not what a borrower actually retains.
Investor terminal wealth is

    W_T = Pi_T + R_tau 1{tau <= T},

and this module supplies the second term, which the repository has always had
the data for -- ``liquidationBonus`` is stored in every calibration JSON and used
by nothing.

Recovery at the barrier
-----------------------
A liquidator repays debt and seizes collateral worth ``(1 + lambda)`` times the
repayment, ``lambda`` being the protocol's liquidation bonus.  Repaying the full
debt leaves the borrower

    R_tau = V_{1,tau} - (1 + lambda) V_{2,tau}.

On a *continuous* crossing the health factor is exactly one, so
``V_{2,tau} = b V_{1,tau}`` and

    R_tau = V_{2,tau} (1 - b(1 + lambda)) / b,

which is positive whenever ``lambda < 1/b - 1`` (0.282 at ``b = 0.78``).
Expressing recovery through the *debt* leg rather than the collateral leg is
what keeps it inside the existing machinery: it makes ``R_tau`` proportional to
``exp(X_2)`` evaluated at ``tau``, and ``X_2`` is the leg the framework already
tilts by.

.. warning::

   This is the continuous-crossing approximation.  A downward jump can carry the
   health factor strictly below one, leaving ``V_2 > b V_1`` and less recovery
   than the formula returns -- possibly negative (bad debt).  Recovery computed
   here is therefore an **upper bound**, and the sizing results built on it are
   optimistic about liquidation.  The exact treatment is implemented in
   :mod:`optimal_long_short.overshoot_recovery`: Kou's double-exponential jumps
   make the overshoot below the barrier *exactly* exponentially distributed by
   memorylessness, and the exact residual equity
   ``R_tau = V_{1,tau} - (1+lambda) V_{2,tau}`` has moments given by the joint
   hitting moments of the two legs under a bivariate tilt, so the overshoot is
   captured with no density to discretize.  The correction is ~3% at the fitted
   jump sizes and flips recovery into bad debt at flash-crash jump magnitudes.

The first-passage transform
---------------------------
The object needed is ``E[exp(k X_{2,tau}) 1{tau <= T}]`` for ``k = 1`` (mean) and
``k = 2`` (variance).  Because ``M_t = exp(k X_{2,t} - t Psi(0,-ik))`` is a
martingale and ``tau`` is a stopping time, changing measure on ``F_tau`` gives

    E[exp(k X_{2,tau}) 1{tau<=T}] = E^{(k)}[exp(alpha tau) 1{tau<=T}],
    alpha := Psi(0, -ik),

whose Laplace transform in ``T`` is

    (1/q) (1 - (q - alpha) phat^{(k)}_surv(q - alpha)),                    (FP)

with ``phat^{(k)}_surv`` the survival transform evaluated under the *k-tilted*
dynamics.  So recovery costs one extra Laplace inversion per moment order and
reuses the same 3x3 barrier system -- no new linear algebra.

The same transform (FP) is the keeper's revenue in
:mod:`optimal_long_short.barbell`: what the borrower forfeits at liquidation is
what the liquidator collects.  Recovery and the barbell's convex leg are the
same object with different coefficients.
"""
from __future__ import annotations

import numpy as np

from optimal_long_short.model.kou_model import BivariateKouModel, KouZTiltedDynamics
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.laplace.root_finder import CharacteristicRootFinder

__all__ = [
    "survival_transform_tilted",
    "tilt_exponent",
    "hitting_moment_transform",
    "recovery_coefficient",
    "hitting_moment",
    "wealth_moments",
]

_DEGEN_TOL = 1e-8


def survival_transform_tilted(
    dynamics: KouZTiltedDynamics, h0: float, q: complex
) -> complex:
    """Laplace transform of ``P(tau > t)`` under the tilted dynamics supplied.

    This is equation (29) of the paper -- forcing ``G_0 = 1`` and the constant
    particular solution ``1/q`` -- but evaluated with the *k-tilted* exponent
    rather than the untilted one, which is what (FP) requires.  The tilt order
    enters only through the roots and phase rates; the barrier system is the
    same 3x3.
    """
    r1, r2 = dynamics.r1_neg, dynamics.r2_neg
    if abs(r1 - r2) < _DEGEN_TOL:
        raise NotImplementedError(
            "Coincident downward phase rates: the 2x2 reduction of Appendix A, "
            "Remark 5 is not implemented for the tilted survival transform."
        )
    roots = CharacteristicRootFinder(dynamics).find(q).negative

    M = np.empty((3, 3), dtype=complex)
    for m, gamma in enumerate(roots):
        M[0, m] = 1.0
        M[1, m] = r1 / (r1 + gamma)
        M[2, m] = r2 / (r2 + gamma)

    rhs = -np.full(3, 1.0 / q, dtype=complex)
    B = np.linalg.solve(M, rhs)
    return 1.0 / q + sum(B[m] * np.exp(roots[m] * h0) for m in range(3))


def tilt_exponent(params: KouParams, k: int) -> float:
    """``Psi(0, -ik) = log E[exp(k X_2)]`` per unit time."""
    return float(np.real(BivariateKouModel(params).levy_khintchine(0.0, -1j * k)))


def hitting_moment_transform(
    params: KouParams, k: int, h0: float, q: complex
) -> complex:
    """Laplace transform in ``T`` of ``E[exp(k X_{2,tau}) 1{tau <= T}]``.

    Implements (FP).  For *real* ``q`` the identity requires
    ``q > Psi(0,-ik)`` so that the shifted survival transform sits right of its
    abscissa of convergence, and that is checked.  Complex arguments are not
    checked: the Talbot algorithm deforms its contour into the left half-plane,
    where the formula remains valid by analytic continuation -- the same
    reasoning under which
    :class:`optimal_long_short.laplace.root_finder.CharacteristicRootFinder` classifies
    roots by real-part rank rather than by strict sign.
    """
    alpha = tilt_exponent(params, k)
    shifted = q - alpha
    if np.isreal(q) and np.real(shifted) <= 0.0:
        raise ValueError(
            f"Real Laplace argument q={np.real(q)} is not to the right of the "
            f"tilt exponent Psi(0,-i{k})={alpha:.4f}; (FP) does not converge."
        )
    dynamics = KouZTiltedDynamics(params=params, k=k)
    surv = survival_transform_tilted(dynamics, h0, shifted)
    return (1.0 - shifted * surv) / q


def recovery_coefficient(b: float, liq_bonus: float) -> float:
    """The factor ``(1 - b(1 + lambda)) / b`` multiplying ``V_{2,tau}``.

    Returns a negative number when ``b (1 + lambda) > 1``, i.e. when the
    liquidation bonus is large enough that seizure exceeds the collateral
    backing the debt.  Callers should treat that as bad debt rather than
    negative recovery.
    """
    return (1.0 - b * (1.0 + liq_bonus)) / b


def hitting_moment(
    params: KouParams,
    k: int,
    h0: float,
    T: float,
    inverter=None,
) -> float:
    """``E[exp(k X_{2,tau}) 1{tau <= T}]`` in the time domain.

    At ``k = 0`` this is the liquidation probability, which makes it a free
    consistency check against ``1 - p_surv`` from the base model.
    """
    from optimal_long_short.laplace.inversion import TalbotInverter

    inverter = inverter or TalbotInverter(M=32)
    return float(
        inverter.invert(lambda q: hitting_moment_transform(params, k, h0, q), T)
    )


def wealth_moments(
    params: KouParams,
    h0: float,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    inverter=None,
) -> dict[str, float]:
    """Moments of investor terminal wealth ``W_T = Pi_T + R_tau 1{tau<=T}``.

    The two terms live on disjoint events -- ``Pi_T`` is supported on
    ``{tau > T}`` and ``R_tau`` on ``{tau <= T}`` -- so the cross term vanishes
    and

        E[W_T]   = E[Pi_T] + E[R_tau 1],
        E[W_T^2] = E[Pi_T^2] + E[R_tau^2 1].

    Under the unit-equity normalization ``n_2 S_{2,0} = b / (e^{h0} - b)``, so
    the recovery scale is ``(1 - b(1+lambda)) / (e^{h0} - b)`` per unit of
    initial equity: it grows as the buffer shrinks, because a thinner buffer is
    a larger book.

    Returns both the recovery-adjusted moments and the zero-recovery baseline,
    so a caller can price the difference the convention makes.
    """
    import math

    from optimal_long_short.laplace.inversion import TalbotInverter
    from optimal_long_short.model.market_params import MarketParams
    from optimal_long_short.model.moments import ConditionalMoments
    from optimal_long_short.model.strategy import UnitExposureLongShortStrategy

    inverter = inverter or TalbotInverter(M=32)
    strategy = UnitExposureLongShortStrategy(
        h0=h0, market=MarketParams(b=b, S10=S10, S20=S20), T=T
    )
    cm = ConditionalMoments(params=params, strategy=strategy)

    p_surv = float(np.clip(cm.p_surv(), 0.0, 1.0))
    killed_1 = float(cm.killed_moment(1))
    killed_2 = float(cm.killed_moment(2))

    scale = recovery_coefficient(b, liq_bonus) * b / (math.exp(h0) - b)
    rec_1 = scale * hitting_moment(params, 1, h0, T, inverter)
    rec_2 = scale**2 * hitting_moment(params, 2, h0, T, inverter)

    mean_w = killed_1 + rec_1
    var_w = killed_2 + rec_2 - mean_w**2
    return {
        "h0": h0,
        "H0": math.exp(h0),
        "initial_leverage": math.exp(h0) / (math.exp(h0) - b),
        "p_surv": p_surv,
        "p_liq": 1.0 - p_surv,
        "recovery_scale": scale,
        "E_recovery": rec_1,
        "mean_wealth": mean_w,
        "var_wealth": var_w,
        # Zero-recovery baseline, the convention of the base paper.
        "mean_killed": killed_1,
        "var_killed": killed_2 - killed_1**2,
    }
