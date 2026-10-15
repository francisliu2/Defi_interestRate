r"""Kelly (log-utility) sizing on the health-buffer evaluation map.

The Kelly investor maximises the expected log of terminal wealth,

    K(h0) = E[log W_T],   W_T = Pi_T + R_tau 1{tau <= T},

per unit of initial equity.  ``log`` is not a sum of exponentials, so the
polynomial-forcing trick behind the integer moments does not apply directly.
This module computes ``K(h0)`` semi-analytically from two families of objects the
package already evaluates exactly, both on the same 3x3 barrier system.

**Survival paths.**  Per unit equity ``Pi_T = e^{X2_T} (e^{h_T} - b) / (H0 - b)`` with
``h_T = h0 + Z_T >= 0`` on ``{tau > T}``, so

    log Pi_T = X2_T - log(H0 - b) + h_T - sum_{n>=1} (b^n / n) e^{-n h_T},

and the series converges geometrically because ``b e^{-h_T} <= b < 1``.  Every
term is a killed joint moment

    S(a, c) = E[e^{a X1_T + c X2_T} 1{tau > T}],

whose Laplace transform in ``T`` is the survival transform under the
``(a, c)``-tilted dynamics at the shifted argument ``q - Psi(a, c)``.  The linear
terms ``E[X2_T 1]`` and ``E[Z_T 1]`` are central differences of ``S`` in the tilt.
The series is truncated at ``n_terms`` and the remainder bounded with the
monotone tail ``E[e^{-n h} 1] <= E[e^{-N h} 1]``.

**Liquidation paths.**  The exact residual equity is
``R_tau = e^{X2_tau} (e^{h_tau} - c) / (H0 - b)`` with ``c = b (1 + lambda)``, so

    log R_tau = X2_tau - log(H0 - b) + log(e^{-u} - c),   u = -h_tau >= 0.

By memorylessness of the exponential jump sizes, the overshoot ``u`` is zero on a
continuous crossing and exponential with rate ``r1 = 1/eta1_neg`` or
``r2 = 1/eta2_pos`` when the barrier is crossed by an ``X1`` down-jump or an
``X2`` up-jump, independently of ``tau``.  Hence

    E[e^{-j u} 1{tau <= T}] = P0 + P1 r1/(r1 + j) + P2 r2/(r2 + j),

and the three weights ``(P0, P1, P2)`` follow from the hitting moments
``e^{j h0} Hit(j, -j)`` at ``j = 0, 1, 2``; further ``j`` are a consistency check.
``E[log(e^{-u} - c) 1]`` is then a one-dimensional integral against this
mixture, and the **bad-debt probability** ``P(u >= -log c)`` is exact:
``P1 e^{-r1 u*} + P2 e^{-r2 u*}``.

**Bad debt and the log.**  On bad-debt paths a non-recourse borrower ends with
``W_T = 0`` and ``log W_T = -inf``.  Because exponential jump tails make that
probability positive (if astronomically small at fitted parameters), strict
Kelly is ``-inf`` at every levered buffer.  The objective is therefore reported
with wealth floored at ``w_floor`` on those paths, together with the bad-debt
probability itself, so the reader can see that the floor is immaterial unless
jumps are large.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import integrate

from optimal_long_short.drift import with_price_drift_view
from optimal_long_short.inversion import LaplaceInverter, TalbotInverter
from optimal_long_short.model_params import KouParams
from optimal_long_short.overshoot_recovery import (
    BivariateTiltedZDynamics,
    joint_hitting_moment,
    survival_transform_bivariate,
    tilt_exponent_bivariate,
)

__all__ = [
    "killed_joint_moment",
    "overshoot_law",
    "expected_log_wealth",
    "KellyGrid",
    "kelly_buffer",
    "kelly_threshold",
    "half_kelly_buffer",
]

_FD_STEP = 1e-3
_NEGLIGIBLE_LIQ = 1e-10


def _check_tilt(params: KouParams, a: float, c: float) -> None:
    """Integrability of e^{a X1 + c X2}: both jump MGFs must be finite."""
    p = params
    if not (-1.0 / p.eta1_neg < a < 1.0 / p.eta1_pos):
        raise ValueError(f"tilt a={a} outside (-1/eta1_neg, 1/eta1_pos)")
    if not (-1.0 / p.eta2_neg < c < 1.0 / p.eta2_pos):
        raise ValueError(f"tilt c={c} outside (-1/eta2_neg, 1/eta2_pos)")


def killed_joint_moment(
    params: KouParams,
    a: float,
    c: float,
    h0: float,
    T: float,
    inverter: LaplaceInverter | None = None,
) -> float:
    """``S(a, c) = E[exp(a X1_T + c X2_T) 1{tau > T}]``.

    Changing measure on ``F_T`` gives ``S = e^{T Psi(a,c)} P^{(a,c)}(tau > T)``,
    whose Laplace transform is the tilted survival transform at ``q - Psi(a, c)``.
    ``S(0, 0)`` is the survival probability.
    """
    _check_tilt(params, a, c)
    inverter = inverter or TalbotInverter(M=32)
    psi = tilt_exponent_bivariate(params, a, c)
    dyn = BivariateTiltedZDynamics(params=params, a=a, c=c)
    return float(inverter.invert(lambda q: survival_transform_bivariate(dyn, h0, q - psi), T))


@dataclass(frozen=True)
class OvershootLaw:
    """Law of the overshoot ``u = -h_tau`` on ``{tau <= T}``.

    ``p_creep`` is the zero-overshoot atom; ``p_jump[i]`` the probability of
    liquidation by a jump of phase ``i`` with exponential rate ``rates[i]``.
    ``max_check_error`` is the worst mismatch, over the check orders, between
    the mixture and the directly computed hitting moments.
    """

    p_creep: float
    p_jump: tuple[float, float]
    rates: tuple[float, float]
    max_check_error: float

    @property
    def p_liq(self) -> float:
        return self.p_creep + sum(self.p_jump)

    def tail(self, u: float) -> float:
        """``P(overshoot >= u, tau <= T)`` for ``u > 0``."""
        return sum(p * math.exp(-r * u) for p, r in zip(self.p_jump, self.rates))


def overshoot_law(
    params: KouParams,
    h0: float,
    T: float,
    inverter: LaplaceInverter | None = None,
    check_orders: tuple[int, ...] = (5, 10),
) -> OvershootLaw:
    """Recover ``(P0, P1, P2)`` from ``E[e^{-j u} 1] = e^{j h0} Hit(j, -j)``, j = 0, 1, 2."""
    inverter = inverter or TalbotInverter(M=32)
    r1, r2 = 1.0 / params.eta1_neg, 1.0 / params.eta2_pos

    def lhs(j: int) -> float:
        return math.exp(j * h0) * joint_hitting_moment(params, float(j), float(-j), h0, T, inverter)

    rows = [[1.0, r1 / (r1 + j), r2 / (r2 + j)] for j in (0, 1, 2)]
    P = np.linalg.solve(np.array(rows), np.array([lhs(j) for j in (0, 1, 2)]))
    err = 0.0
    for j in check_orders:
        model = P[0] + P[1] * r1 / (r1 + j) + P[2] * r2 / (r2 + j)
        err = max(err, abs(model - lhs(j)))
    return OvershootLaw(float(P[0]), (float(P[1]), float(P[2])), (r1, r2), err)


def expected_log_wealth(
    params: KouParams,
    h0: float,
    *,
    b: float,
    liq_bonus: float,
    T: float,
    n_terms: int = 40,
    w_floor: float = 1e-4,
    inverter: LaplaceInverter | None = None,
) -> dict[str, float]:
    """``E[log W_T]`` per unit initial equity, with exact jump-overshoot recovery.

    Returns the objective and its parts: the survival and liquidation
    contributions, the bad-debt probability and its floored contribution, the
    series truncation bound, and the overshoot-law consistency error.
    """
    inverter = inverter or TalbotInverter(M=32)
    H0 = math.exp(h0)
    c = b * (1.0 + liq_bonus)
    if c >= 1.0:
        raise ValueError("b(1+lambda) >= 1: recovery is never positive")
    log_scale = math.log(H0 - b)
    eps = _FD_STEP

    def S(a: float, cc: float) -> float:
        return killed_joint_moment(params, a, cc, h0, T, inverter)

    # ---- survival paths ---------------------------------------------------
    p_surv = S(0.0, 0.0)
    E_X2_surv = (S(0.0, eps) - S(0.0, -eps)) / (2 * eps)
    E_Z_surv = (S(eps, -eps) - S(-eps, eps)) / (2 * eps)
    # E[e^{-n h_T} 1] = e^{-n h0} S(-n, n); monotone decreasing in n.  The tilt
    # (-n, n) is admissible only for n < min(1/eta2_pos, 1/eta1_neg), so heavy
    # jump tails cap the series length; the tail bound below accounts for it.
    n_max = int(0.9 * min(1.0 / params.eta2_pos, 1.0 / params.eta1_neg))
    n_terms = max(1, min(n_terms, n_max))
    exp_moments = [math.exp(-n * h0) * S(-float(n), float(n)) for n in range(1, n_terms + 1)]
    series = sum((b**n / n) * m for n, m in zip(range(1, n_terms + 1), exp_moments))
    tail_bound = exp_moments[-1] * (b ** (n_terms + 1)) / ((n_terms + 1) * (1.0 - b))
    E_log_pi = E_X2_surv - log_scale * p_surv + (h0 * p_surv + E_Z_surv) - series

    # ---- liquidation paths ------------------------------------------------
    if 1.0 - p_surv < _NEGLIGIBLE_LIQ:
        # The overshoot weights come from e^{j h0} Hit(j,-j); when liquidation is
        # essentially impossible Hit is ~e^{-j h0} and the inverter's absolute
        # error is amplified by e^{j h0}.  The true contribution is bounded by
        # p_liq * (log(H0 - b) + |E[X2]| + |log(1 - c)|), i.e. negligible.
        return {
            "h0": h0, "H0": H0, "initial_leverage": H0 / (H0 - b),
            "p_surv": p_surv, "p_liq": 1.0 - p_surv,
            "E_log_W": E_log_pi, "E_log_survival_part": E_log_pi,
            "E_log_liquidation_part": 0.0, "p_bad_debt": 0.0, "bad_debt_floor_term": 0.0,
            "series_tail_bound": tail_bound, "overshoot_check_error": 0.0, "p_creep": 0.0,
        }
    law = overshoot_law(params, h0, T, inverter)
    E_X2_liq = (
        joint_hitting_moment(params, 0.0, eps, h0, T, inverter)
        - joint_hitting_moment(params, 0.0, -eps, h0, T, inverter)
    ) / (2 * eps)
    u_star = -math.log(c)  # overshoot at which recovery hits zero

    def g(u: float) -> float:
        return math.log(math.exp(-u) - c)

    E_g = law.p_creep * g(0.0)
    for p_i, r_i in zip(law.p_jump, law.rates):
        val, _ = integrate.quad(lambda u: g(u) * r_i * math.exp(-r_i * u), 0.0, u_star,
                                limit=200)
        E_g += p_i * val
    p_bad = law.tail(u_star)
    E_log_r = E_X2_liq - log_scale * law.p_liq + E_g
    floor_term = p_bad * math.log(w_floor)

    return {
        "h0": h0,
        "H0": H0,
        "initial_leverage": H0 / (H0 - b),
        "p_surv": p_surv,
        "p_liq": law.p_liq,
        "E_log_W": E_log_pi + E_log_r + floor_term,
        "E_log_survival_part": E_log_pi,
        "E_log_liquidation_part": E_log_r,
        "p_bad_debt": p_bad,
        "bad_debt_floor_term": floor_term,
        "series_tail_bound": tail_bound,
        "overshoot_check_error": law.max_check_error,
        "p_creep": law.p_creep,
    }


@dataclass
class KellyGrid:
    """Cache of ``E[log W_T]`` along the paper's buffer grid, per drift view.

    Mirrors :class:`optimal_long_short.drift_bayes._BufferGrid`: the spread drift
    ``nu = mu1 - mu2`` is moved through ``mu1`` with a price-growth drift view.
    """

    params: KouParams
    b: float
    liq_bonus: float
    T: float
    H0_lo: float = 1.0684931506849316
    H0_hi: float = 2.0
    n_grid: int = 40
    n_terms: int = 40

    def __post_init__(self) -> None:
        self.h_grid = np.linspace(math.log(self.H0_lo), math.log(self.H0_hi), self.n_grid)
        self._cache: dict[float, np.ndarray] = {}

    def curve(self, nu: float) -> np.ndarray:
        key = round(float(nu), 4)
        if key not in self._cache:
            base = self.params.mu1 - self.params.mu2
            pv = with_price_drift_view(self.params, delta_mu1=key - base)
            self._cache[key] = np.array([
                expected_log_wealth(pv, h, b=self.b, liq_bonus=self.liq_bonus, T=self.T,
                                    n_terms=self.n_terms)["E_log_W"]
                for h in self.h_grid
            ])
        return self._cache[key]


def kelly_buffer(grid: KellyGrid, nu: float) -> tuple[float, bool]:
    """Kelly-optimal ``H0`` on the grid for drift view ``nu``; ``(H0*, interior?)``."""
    vals = grid.curve(nu)
    i = int(np.argmax(vals))
    return math.exp(grid.h_grid[i]), 0 < i < len(vals) - 1


def kelly_threshold(grid: KellyGrid, lo: float, hi: float, tol: float = 0.005) -> float:
    """Smallest spread drift giving an interior Kelly optimum, by bisection on ``[lo, hi]``.

    Requires a corner at ``lo`` and an interior optimum at ``hi``.
    """
    if kelly_buffer(grid, lo)[1] or not kelly_buffer(grid, hi)[1]:
        raise ValueError("bracket must have a corner at lo and an interior optimum at hi")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if kelly_buffer(grid, mid)[1]:
            hi = mid
        else:
            lo = mid
    return hi


def half_kelly_buffer(H0_kelly: float, b: float) -> float:
    """Buffer carrying half the Kelly debt: ``n2 S20 = b/(H0 - b)`` halves, so
    ``H0_half - b = 2 (H0_kelly - b)``."""
    return b + 2.0 * (H0_kelly - b)
