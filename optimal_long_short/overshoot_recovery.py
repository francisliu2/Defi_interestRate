r"""Exact jump-overshoot liquidation recovery via a bivariate exponential tilt.

The base recovery module :mod:`optimal_long_short.recovery` models the borrower's
residual equity at liquidation under the *continuous-crossing* approximation: it
assumes the health factor equals one exactly at ``tau``, so that
``V_{1,tau} = V_{2,tau} / H_0`` and recovery collapses onto the debt leg ``e^{X_2}``
alone.  Under a jump-diffusion the barrier is overwhelmingly crossed *by a downward
jump*, which overshoots: the health factor lands strictly below one, so the true
collateral leg ``V_{1,tau}`` is smaller than the continuous formula assumes and
recovery is smaller -- possibly negative (bad debt).

The exact residual equity needs no approximation:

    R_tau = V_{1,tau} - (1 + lambda) V_{2,tau},           (definition, any crossing)

with ``V_{i,tau} = w_i S_{i,0} e^{X_{i,tau}}`` the *actual* leg values wherever the
jump lands.  Its moments are therefore linear combinations of the *joint* hitting
moments

    Hit(a, c) := E[ exp(a X_{1,tau} + c X_{2,tau}) 1_{tau <= T} ],

for ``(a, c)`` in ``{(1,0),(0,1)}`` (mean) and ``{(2,0),(1,1),(0,2)}`` (second
moment).  Each ``Hit(a, c)`` is obtained by the same first-passage transform the
base paper uses for the ``(0, k)`` debt-leg tilt, generalised to a bivariate tilt
of ``(X_1, X_2)``.  The overshoot is captured *exactly and automatically*, because
the tilted first-passage law of ``Z = X_1 - X_2`` already integrates over where the
process lands below the barrier -- there is no separate overshoot density to
approximate.

Bivariate tilt
--------------
Under ``dP^{(a,c)}/dP|_{F_t} = exp(a X_{1,t} + c X_{2,t} - t Psi_MGF(a, c))`` the
log-ratio ``Z`` is Levy with moment exponent

    psi_Z^{(a,c)}(s) = Psi_MGF(a + s, c - s) - Psi_MGF(a, c)
                     = mu_Z^{(a,c)} s + 0.5 sigma_Z^2 s^2
                       + lam1 (M1(a + s) - M1(a))
                       + lam2 (M2(c - s) - M2(c)),

    sigma_Z^2      = sigma1^2 + sigma2^2 - 2 rho sigma1 sigma2,
    mu_Z^{(a,c)}   = (muX1 - muX2)
                     + a (sigma1^2 - rho sigma1 sigma2)
                     - c (sigma2^2 - rho sigma1 sigma2),

and the k-dependent phase rates of the base module generalise to

    r1_pos = 1/eta1_pos - a     r1_neg = 1/eta1_neg + a
    r2_pos = 1/eta2_neg + c     r2_neg = 1/eta2_pos - c.

Setting ``(a, c) = (0, k)`` recovers :class:`KouZTiltedDynamics` exactly, which is
used as a regression check in the test suite.  The degree-6 root polynomial has the
*same* residue coefficients as the base finder -- ``lam1 p1 / eta1_pos`` etc. --
which do not depend on ``(a, c)``; the tilt enters only through the phase rates, the
drift ``mu_Z``, and the constants ``M1(a)``, ``M2(c)``.

Killing rate
------------
Changing measure on ``F_tau`` gives

    Hit(a, c) = E^{(a,c)}[ exp(alpha tau) 1_{tau <= T} ],   alpha = Psi_MGF(a, c),

whose Laplace transform in ``T`` is the base first-passage transform evaluated under
the ``(a, c)``-tilted survival dynamics:

    (1/q) (1 - (q - alpha) phat^{(a,c)}_surv(q - alpha)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optimal_long_short.inversion import LaplaceInverter, TalbotInverter
from optimal_long_short.model_params import KouParams

__all__ = [
    "BivariateTiltedZDynamics",
    "tilt_exponent_bivariate",
    "survival_transform_bivariate",
    "joint_hitting_moment_transform",
    "joint_hitting_moment",
    "exact_recovery_moments",
    "exact_wealth_moments",
]

_DEGEN_TOL = 1e-8


@dataclass(frozen=True)
class BivariateTiltedZDynamics:
    """Levy exponent and phase rates of ``Z = X1 - X2`` under ``P^{(a,c)}``.

    Parameters
    ----------
    params : KouParams
        Model parameters.
    a, c : float
        Tilt orders on ``X1`` and ``X2`` respectively.  Requires
        ``a < 1/eta1_pos`` and ``c < 1/eta2_pos`` for the tilt to be
        well defined (both hold comfortably for calibrated jump means).
    """

    params: KouParams
    a: float
    c: float

    def __post_init__(self) -> None:
        p = self.params
        if self.a >= 1.0 / p.eta1_pos:
            raise ValueError(
                f"a={self.a} must be < 1/eta1_pos={1.0 / p.eta1_pos:.4f} "
                "for the X1 tilt to be integrable."
            )
        if self.c >= 1.0 / p.eta2_pos:
            raise ValueError(
                f"c={self.c} must be < 1/eta2_pos={1.0 / p.eta2_pos:.4f} "
                "for the X2 tilt to be integrable."
            )

    # -- Brownian part ---------------------------------------------------
    @property
    def sigma_Z_sq(self) -> float:
        p = self.params
        return p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2

    @property
    def mu_Z(self) -> float:
        p = self.params
        rss = p.rho * p.sigma1 * p.sigma2
        return (
            (p.muX1 - p.muX2)
            + self.a * (p.sigma1**2 - rss)
            - self.c * (p.sigma2**2 - rss)
        )

    # -- phase rates -----------------------------------------------------
    @property
    def r1_pos(self) -> float:
        return 1.0 / self.params.eta1_pos - self.a

    @property
    def r1_neg(self) -> float:
        return 1.0 / self.params.eta1_neg + self.a

    @property
    def r2_pos(self) -> float:
        return 1.0 / self.params.eta2_neg + self.c

    @property
    def r2_neg(self) -> float:
        return 1.0 / self.params.eta2_pos - self.c

    # -- jump MGFs of the individual assets ------------------------------
    @staticmethod
    def _mgf_kou(s: complex, pi: float, ep: float, en: float) -> complex:
        return pi / (1.0 - s * ep) + (1 - pi) / (1.0 + s * en)

    def mgf_J1(self, s: complex) -> complex:
        p = self.params
        return self._mgf_kou(s, p.p1, p.eta1_pos, p.eta1_neg)

    def mgf_J2(self, s: complex) -> complex:
        p = self.params
        return self._mgf_kou(s, p.p2, p.eta2_pos, p.eta2_neg)

    def __call__(self, s: complex) -> complex:
        """Evaluate ``psi_Z^{(a,c)}(s)``."""
        p = self.params
        return (
            self.mu_Z * s
            + 0.5 * self.sigma_Z_sq * s**2
            + p.lam1 * (self.mgf_J1(self.a + s) - self.mgf_J1(self.a))
            + p.lam2 * (self.mgf_J2(self.c - s) - self.mgf_J2(self.c))
        )

    # -- characteristic roots -------------------------------------------
    def _root_polynomial(self, q: complex) -> np.ndarray:
        """Degree-6 polynomial whose roots solve ``psi_Z^{(a,c)}(s) = q``."""
        p = self.params
        r1p, r1n = self.r1_pos, self.r1_neg
        r2p, r2n = self.r2_pos, self.r2_neg

        fa = np.array([-1.0, r1p], dtype=complex)  # (r1p - s)
        fb = np.array([1.0, r1n], dtype=complex)    # (r1n + s)
        fc = np.array([-1.0, r2p], dtype=complex)  # (r2p - s)
        fd = np.array([1.0, r2n], dtype=complex)    # (r2n + s)

        D = np.polymul(np.polymul(fa, fb), np.polymul(fc, fd))
        D_no_r1p = np.polymul(np.polymul(fb, fc), fd)
        D_no_r1n = np.polymul(np.polymul(fa, fc), fd)
        D_no_r2p = np.polymul(np.polymul(fa, fb), fd)
        D_no_r2n = np.polymul(np.polymul(fa, fb), fc)

        M1a = self.mgf_J1(self.a)
        M2c = self.mgf_J2(self.c)

        quad = np.array(
            [0.5 * self.sigma_Z_sq, self.mu_Z, -q - p.lam1 * M1a - p.lam2 * M2c],
            dtype=complex,
        )
        poly = np.polymul(quad, D)
        poly = np.polyadd(poly, p.lam1 * p.p1 / p.eta1_pos * D_no_r1p)
        poly = np.polyadd(poly, p.lam1 * (1 - p.p1) / p.eta1_neg * D_no_r1n)
        poly = np.polyadd(poly, p.lam2 * p.p2 / p.eta2_pos * D_no_r2n)
        poly = np.polyadd(poly, p.lam2 * (1 - p.p2) / p.eta2_neg * D_no_r2p)
        return poly

    def negative_roots(self, q: complex, tol: float = 1e-8) -> tuple:
        """The three roots of ``psi_Z^{(a,c)}(s) = q`` with smallest real part.

        Classified by real-part rank, matching
        :class:`optimal_long_short.root_finder.CharacteristicRootFinder`: for
        ``Re(q) > 0`` this is the strict-sign split, and on the Talbot contour it
        is the analytic continuation.
        """
        roots = np.roots(self._root_polynomial(q))
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                if abs(roots[i] - roots[j]) < tol:
                    raise ValueError(
                        f"Root collision {abs(roots[i] - roots[j]):.2e} at q={q}."
                    )
        ordered = sorted(roots, key=lambda r: r.real, reverse=True)
        return tuple(sorted(ordered[3:], key=lambda r: r.real))


def tilt_exponent_bivariate(params: KouParams, a: float, c: float) -> float:
    """``Psi_MGF(a, c) = log E[exp(a X1 + c X2)]`` per unit time."""
    p = params
    rss = p.rho * p.sigma1 * p.sigma2
    diffusion = (
        a * p.muX1
        + c * p.muX2
        + 0.5 * (p.sigma1**2 * a**2 + p.sigma2**2 * c**2 + 2 * rss * a * c)
    )
    m1 = p.p1 / (1.0 - a * p.eta1_pos) + (1 - p.p1) / (1.0 + a * p.eta1_neg)
    m2 = p.p2 / (1.0 - c * p.eta2_pos) + (1 - p.p2) / (1.0 + c * p.eta2_neg)
    return float(diffusion + p.lam1 * (m1 - 1.0) + p.lam2 * (m2 - 1.0))


def survival_transform_bivariate(
    dynamics: BivariateTiltedZDynamics, h0: float, q: complex
) -> complex:
    """Laplace transform of ``P(tau > t)`` under the ``(a,c)``-tilted dynamics.

    Same 3x3 barrier system as :func:`optimal_long_short.recovery.survival_transform_tilted`,
    with the generalised downward phase rates ``r1_neg``, ``r2_neg``.
    """
    r1, r2 = dynamics.r1_neg, dynamics.r2_neg
    if abs(r1 - r2) < _DEGEN_TOL:
        raise NotImplementedError(
            "Coincident downward phase rates under the bivariate tilt; the 2x2 "
            "degenerate reduction is not implemented."
        )
    roots = dynamics.negative_roots(q)
    M = np.empty((3, 3), dtype=complex)
    for m, gamma in enumerate(roots):
        M[0, m] = 1.0
        M[1, m] = r1 / (r1 + gamma)
        M[2, m] = r2 / (r2 + gamma)
    B = np.linalg.solve(M, -np.full(3, 1.0 / q, dtype=complex))
    return 1.0 / q + sum(B[m] * np.exp(roots[m] * h0) for m in range(3))


def joint_hitting_moment_transform(
    params: KouParams, a: float, c: float, h0: float, q: complex
) -> complex:
    """Laplace transform in ``T`` of ``Hit(a, c) = E[exp(a X1_tau + c X2_tau) 1_{tau<=T}]``."""
    alpha = tilt_exponent_bivariate(params, a, c)
    shifted = q - alpha
    if np.isreal(q) and np.real(shifted) <= 0.0:
        raise ValueError(
            f"Real Laplace argument q={np.real(q)} is not right of the tilt "
            f"exponent Psi_MGF({a},{c})={alpha:.4f}; the transform does not converge."
        )
    dyn = BivariateTiltedZDynamics(params=params, a=a, c=c)
    surv = survival_transform_bivariate(dyn, h0, shifted)
    return (1.0 - shifted * surv) / q


def joint_hitting_moment(
    params: KouParams,
    a: float,
    c: float,
    h0: float,
    T: float,
    inverter: LaplaceInverter | None = None,
) -> float:
    """``Hit(a, c) = E[exp(a X1_tau + c X2_tau) 1_{tau <= T}]`` in the time domain.

    At ``(a, c) = (0, 0)`` this returns the liquidation probability
    ``1 - p_surv``, a free consistency check.
    """
    inverter = inverter or TalbotInverter(M=32)
    return float(
        inverter.invert(
            lambda q: joint_hitting_moment_transform(params, a, c, h0, q), T
        )
    )


def exact_recovery_moments(
    params: KouParams,
    h0: float,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    inverter: LaplaceInverter | None = None,
) -> dict[str, float]:
    """First and second moments of the exact residual equity ``R_tau 1_{tau<=T}``.

    ``R_tau = A e^{X1_tau} - B e^{X2_tau}`` per unit initial equity, with
    ``A = H0/(H0-b)`` and ``B = (1+lambda) b/(H0-b)``.  No continuous-crossing
    assumption: the leg values are the actual (overshot) values at ``tau``.

    Returns the exact recovery moments alongside the continuous-crossing values,
    so a caller can price the overshoot correction directly.
    """
    inverter = inverter or TalbotInverter(M=32)
    H0 = math.exp(h0)
    A = H0 / (H0 - b)
    B = (1.0 + liq_bonus) * b / (H0 - b)

    def hit(a: float, c: float) -> float:
        return joint_hitting_moment(params, a, c, h0, T, inverter)

    hit_10, hit_01 = hit(1.0, 0.0), hit(0.0, 1.0)
    hit_20, hit_11, hit_02 = hit(2.0, 0.0), hit(1.0, 1.0), hit(0.0, 2.0)

    rec_1 = A * hit_10 - B * hit_01
    rec_2 = A**2 * hit_20 - 2.0 * A * B * hit_11 + B**2 * hit_02

    # Continuous-crossing counterpart: substitute e^{X1_tau} = e^{X2_tau}/H0,
    # i.e. Hit(1,0) -> Hit(0,1)/H0, Hit(2,0) -> Hit(0,2)/H0^2, Hit(1,1) -> Hit(0,2)/H0.
    cc_scale = (1.0 - b * (1.0 + liq_bonus)) / (H0 - b)
    cc_1 = cc_scale * hit_01
    cc_2 = cc_scale**2 * hit_02

    return {
        "E_recovery": rec_1,
        "E_recovery_sq": rec_2,
        "E_recovery_cc": cc_1,
        "E_recovery_sq_cc": cc_2,
        "hit_10": hit_10,
        "hit_01": hit_01,
        "hit_20": hit_20,
        "hit_11": hit_11,
        "hit_02": hit_02,
        "p_liq": hit(0.0, 0.0),
    }


def exact_wealth_moments(
    params: KouParams,
    h0: float,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    inverter: LaplaceInverter | None = None,
) -> dict[str, float]:
    """Terminal-wealth moments ``W_T = Pi_T + R_tau 1_{tau<=T}`` with exact overshoot.

    Mirrors :func:`optimal_long_short.recovery.wealth_moments` but uses the exact
    residual equity in place of the continuous-crossing approximation, and reports
    both so the overshoot correction is explicit.
    """
    from optimal_long_short.market_params import MarketParams
    from optimal_long_short.moments import ConditionalMoments

    inverter = inverter or TalbotInverter(M=32)
    strategy = UnitExposureLongShortStrategy(
        h0=h0, market=MarketParams(b=b, S10=S10, S20=S20), T=T
    )
    cm = ConditionalMoments(params=params, strategy=strategy, inverter=inverter)

    p_surv = float(np.clip(cm.p_surv(), 0.0, 1.0))
    killed_1 = float(cm.killed_moment(1))
    killed_2 = float(cm.killed_moment(2))

    rec = exact_recovery_moments(
        params, h0, b=b, liq_bonus=liq_bonus, T=T, inverter=inverter
    )

    mean_w = killed_1 + rec["E_recovery"]
    var_w = killed_2 + rec["E_recovery_sq"] - mean_w**2

    mean_w_cc = killed_1 + rec["E_recovery_cc"]
    var_w_cc = killed_2 + rec["E_recovery_sq_cc"] - mean_w_cc**2

    return {
        "h0": h0,
        "H0": math.exp(h0),
        "p_surv": p_surv,
        "p_liq": 1.0 - p_surv,
        # exact overshoot
        "E_recovery": rec["E_recovery"],
        "mean_wealth": mean_w,
        "var_wealth": var_w,
        # continuous-crossing approximation (base paper)
        "E_recovery_cc": rec["E_recovery_cc"],
        "mean_wealth_cc": mean_w_cc,
        "var_wealth_cc": var_w_cc,
        # zero-recovery baseline (killed convention)
        "mean_killed": killed_1,
        "var_killed": killed_2 - killed_1**2,
    }


from optimal_long_short.strategy import UnitExposureLongShortStrategy  # noqa: E402
