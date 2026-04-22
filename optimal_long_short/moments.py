from dataclasses import dataclass
from typing import Callable

import numpy as np

from optimal_long_short.inversion import LaplaceInverter, TalbotInverter
from optimal_long_short.kou_model import BivariateKouModel, KouZTiltedDynamics
from optimal_long_short.laplace_resolvent import (
    GeneralSolution,
    HomogeneousSolution,
    ParticularSolution,
)
from optimal_long_short.model_params import KouParams
from optimal_long_short.root_finder import CharacteristicRootFinder
from optimal_long_short.strategy import UnitExposureLongShortStrategy


# ---------------------------------------------------------------------------
# Survival resolvent  (k = 0, constant forcing f ≡ 1)
# ---------------------------------------------------------------------------

@dataclass
class SurvivalResolvent:
    """
    Laplace-domain resolvent for the survival probability under the untilted
    measure P = P^{(2,0)}.

    The survival probability p_surv(t, z; h0) = P_z(tau(h0) > t) satisfies

        (q - L^(0)) U_surv(q, z) = 1,   z > -h0,
        U_surv(q, z) = 0,                z <= -h0.

    Particular solution:  U_part = 1/q  (since psi_Z^(0)(0) = 0).
    Homogeneous solution: same three negative-root modes as the payoff case.
    Boundary system:      same M_bar structure with k=0 phase rates, and
                          forcing vector b = (1/q, 1/q, 1/q) because
                          K_{j,-}^(0)[1/q](-h0) = 1/q.

    Parameters
    ----------
    dynamics : KouZTiltedDynamics
        Must be constructed with k=0.
    strategy : UnitExposureLongShortStrategy
        Supplies h0.
    """
    dynamics: KouZTiltedDynamics
    strategy: UnitExposureLongShortStrategy

    def __post_init__(self) -> None:
        if self.dynamics.k != 0:
            raise ValueError(
                f"SurvivalResolvent requires k=0, got k={self.dynamics.k}"
            )

    def _barrier_matrix(self, q: complex) -> np.ndarray:
        neg_roots = CharacteristicRootFinder(self.dynamics).find(q).negative
        dyn = self.dynamics
        M = np.zeros((3, 3), dtype=complex)
        for m, gamma in enumerate(neg_roots):
            M[0, m] = 1.0
            M[1, m] = dyn.r1_neg / (dyn.r1_neg + gamma)
            M[2, m] = dyn.r2_neg / (dyn.r2_neg + gamma)
        return M

    def coefficients(self, q: complex) -> np.ndarray:
        """Solve M_bar * C = -(1/q, 1/q, 1/q) and return C = (C1, C2, C3)."""
        M = self._barrier_matrix(q)
        b = np.array([1.0 / q, 1.0 / q, 1.0 / q], dtype=complex)
        return np.linalg.solve(M, -b)

    def evaluate_at_origin(self, q: complex) -> complex:
        """
        U_surv(q, 0; h0) = 1/q + sum_{m=1}^{3} C_m * exp(gamma_{m+3} * h0).
        """
        C = self.coefficients(q)
        neg_roots = CharacteristicRootFinder(self.dynamics).find(q).negative
        h0 = self.strategy.h0
        result = 1.0 / q
        for Cm, gamma in zip(C, neg_roots):
            result += Cm * np.exp(gamma * h0)
        return result


# ---------------------------------------------------------------------------
# Killed moments  m_k(T; h0)  via Laplace inversion
# ---------------------------------------------------------------------------

@dataclass
class KilledMoments:
    """
    Computes the killed moments m_k(T; h0) = E^{(2,k)}[g_k(Z_T) 1_{tau > T}]
    for k = 1, 2 and the survival probability p_surv(T, 0; h0) via Laplace
    inversion of the respective resolvents.

    Parameters
    ----------
    params : KouParams
        Kou model parameters.
    strategy : UnitExposureLongShortStrategy
        Portfolio strategy, supplying h0, T, and market params.
    inverter : LaplaceInverter
        Numerical Laplace inverter. Defaults to TalbotInverter(M=32).
    """
    params: KouParams
    strategy: UnitExposureLongShortStrategy
    inverter: LaplaceInverter = None

    def __post_init__(self) -> None:
        if self.inverter is None:
            self.inverter = TalbotInverter(M=32)

    def _general_solution(self, k: int) -> GeneralSolution:
        dyn = KouZTiltedDynamics(params=self.params, k=k)
        part = ParticularSolution(dynamics=dyn, strategy=self.strategy)
        hom = HomogeneousSolution(particular=part)
        return GeneralSolution(homogeneous=hom)

    def m(self, k: int) -> float:
        """
        Compute m_k(T; h0) = L^{-1}_{q->T}[ U_hat^{(k)}(q, 0; h0) ].

        Parameters
        ----------
        k : int
            Moment order, 1 or 2.

        Returns
        -------
        float
        """
        sol = self._general_solution(k)
        F: Callable[[complex], complex] = lambda q: sol.evaluate_at_origin(q)
        return self.inverter.invert(F, self.strategy.T)

    def p_surv(self) -> float:
        """
        Compute p_surv(T, 0; h0) = L^{-1}_{q->T}[ U_surv(q, 0; h0) ].

        Returns
        -------
        float
        """
        dyn0 = KouZTiltedDynamics(params=self.params, k=0)
        surv = SurvivalResolvent(dynamics=dyn0, strategy=self.strategy)
        F: Callable[[complex], complex] = lambda q: surv.evaluate_at_origin(q)
        return self.inverter.invert(F, self.strategy.T)


# ---------------------------------------------------------------------------
# Conditional mean and variance
# ---------------------------------------------------------------------------

@dataclass
class ConditionalMoments:
    """
    Conditional mean and variance of the terminal payoff Pi_T, conditioned on
    the position surviving to maturity (tau > T).

    From the moment reduction (Proposition 3.1 of the paper):

        E[Pi_T^k] = (w2/b)^k * S20^k * exp(T * Psi(0, -ik)) * m_k(T; h0),

    so the killed (unconditional) moments are:

        E[Pi_T^k * 1_{tau > T}] = (w2/b)^k * S20^k * A_k * m_k(T; h0),

    where A_k = exp(T * Psi(0, -ik)) is the k-th moment of exp(X2_T) under P.

    Conditioning on survival:

        E[Pi_T | tau > T]       = E[Pi_T * 1_{tau>T}] / p_surv,
        E[Pi_T^2 | tau > T]     = E[Pi_T^2 * 1_{tau>T}] / p_surv,
        Var(Pi_T | tau > T)     = E[Pi_T^2 | tau > T] - (E[Pi_T | tau > T])^2.

    Parameters
    ----------
    params : KouParams
        Kou model parameters.
    strategy : UnitExposureLongShortStrategy
        Portfolio strategy.
    inverter : LaplaceInverter, optional
        Numerical Laplace inverter. Defaults to TalbotInverter(M=32).
    """
    params: KouParams
    strategy: UnitExposureLongShortStrategy
    inverter: LaplaceInverter = None

    def __post_init__(self) -> None:
        if self.inverter is None:
            self.inverter = TalbotInverter(M=32)
        self._km = KilledMoments(
            params=self.params,
            strategy=self.strategy,
            inverter=self.inverter,
        )
        self._model = BivariateKouModel(params=self.params)

    def _A(self, k: int) -> float:
        """A_k = exp(T * Psi(0, -ik)) = E[exp(k * X2_T)] under P."""
        T = self.strategy.T
        return np.exp(T * self._model.levy_khintchine(0, -1j * k)).real

    def _scale(self, k: int) -> float:
        """(w2 / b)^k * S20^k."""
        s = self.strategy
        return (s.w2 / s.market.b) ** k * s.market.S20 ** k

    def killed_moment(self, k: int) -> float:
        """
        E[Pi_T^k * 1_{tau > T}] = scale_k * A_k * m_k(T; h0).
        """
        return self._scale(k) * self._A(k) * self._km.m(k)

    def p_surv(self) -> float:
        """Survival probability p_surv(T, 0; h0)."""
        return self._km.p_surv()

    def conditional_mean(self) -> float:
        """E[Pi_T | tau > T]."""
        return self.killed_moment(1) / self.p_surv()

    def conditional_second_moment(self) -> float:
        """E[Pi_T^2 | tau > T]."""
        return self.killed_moment(2) / self.p_surv()

    def conditional_variance(self) -> float:
        """Var(Pi_T | tau > T) = E[Pi_T^2 | tau>T] - (E[Pi_T | tau>T])^2."""
        mu = self.conditional_mean()
        return self.conditional_second_moment() - mu ** 2
