import math
from dataclasses import dataclass
from math import comb

import numpy as np

from optimal_long_short.kou_model import KouZTiltedDynamics
from optimal_long_short.root_finder import CharacteristicRootFinder, SixRoots
from optimal_long_short.strategy import UnitExposureLongShortStrategy


@dataclass
class ParticularSolution:
    """
    Particular solution to the resolvent equation (q - L^(k)) U = g_k(z; h0)
    on the survival domain z > -h0.

    Because b in (0,1) and z > -h0 imply e^{h0+z} > 1 > b, the payoff kernel

        g_k(z; h0) = (e^{h0+z} - b)^k

    expands as a polynomial in e^z with no kink on the domain:

        g_k(z; h0) = sum_{j=0}^{k} c_{k,j}(h0) * e^{j*z},
        c_{k,j}(h0) = C(k,j) * (-1)^{k-j} * b^{k-j} * e^{j*h0}.

    By the eigenrelation L^(k) e^{j*z} = psi_Z^(k)(j) e^{j*z}, the particular
    solution is (Lemma 4.2 in the paper):

        U_part(q, z; h0) = sum_{j=0}^{k} c_{k,j}(h0) / (q - psi_Z^(k)(j)) * e^{j*z}.

    Applying the downward phase operator K_{j,-}^(k) at z = -h0
    (using eq. (6): K_{j,-}^(k)[e^{gamma*z}](z) = r/(r+gamma) * e^{gamma*z})
    gives the entries of the forcing vector b^(k) in the barrier linear system:

        K_{r,-}[U_part](-h0) = sum_{j=0}^{k} c_{k,j} / (q - psi(j))
                                              * r / (r + j) * e^{-j*h0}.

    Parameters
    ----------
    dynamics : KouZTiltedDynamics
        Lévy exponent and phase rates of Z under P^(2,k).
    strategy : UnitExposureLongShortStrategy
        Supplies h0 and the collateral factor b.
    """
    dynamics: KouZTiltedDynamics
    strategy: UnitExposureLongShortStrategy

    @property
    def k(self) -> int:
        """Tilting order, also the degree of the payoff polynomial."""
        return self.dynamics.k

    @property
    def h0(self) -> float:
        return self.strategy.h0

    @property
    def b(self) -> float:
        return self.strategy.market.b

    def coefficients(self) -> list[float]:
        """
        Forcing coefficients c_{k,j}(h0) for j = 0, ..., k.

            c_{k,j}(h0) = C(k,j) * (-1)^{k-j} * b^{k-j} * exp(j * h0)
        """
        k, h0, b = self.k, self.h0, self.b
        return [
            comb(k, j) * ((-1) ** (k - j)) * (b ** (k - j)) * math.exp(j * h0)
            for j in range(k + 1)
        ]

    def evaluate(self, q: complex, z: float) -> complex:
        """
        Evaluate U_part(q, z; h0) = sum_j c_{k,j} / (q - psi(j)) * exp(j*z).

        Parameters
        ----------
        q : complex
            Laplace parameter.
        z : float
            Spatial argument.

        Returns
        -------
        complex
        """
        coeffs = self.coefficients()
        result = 0.0 + 0j
        for j, c in enumerate(coeffs):
            result += c / (q - self.dynamics(j)) * math.exp(j * z)
        return result

    def evaluate_at_barrier(self, q: complex) -> complex:
        """
        U_part(q, -h0; h0) = sum_j c_{k,j} / (q - psi(j)) * exp(-j*h0).
        """
        return self.evaluate(q, -self.h0)

    def phase_op_neg_at_barrier(self, q: complex, phase_rate: float) -> complex:
        """
        Apply the downward phase operator K_{-}^(k) with rate r to U_part at z = -h0.

            K_{r,-}[U_part](-h0) = sum_j c_{k,j} / (q - psi(j))
                                           * r / (r + j) * exp(-j * h0).

        Parameters
        ----------
        q : complex
            Laplace parameter.
        phase_rate : float
            The rate r of the downward exponential density (r1_neg or r2_neg).

        Returns
        -------
        complex
        """
        coeffs = self.coefficients()
        result = 0.0 + 0j
        for j, c in enumerate(coeffs):
            eigen_factor = phase_rate / (phase_rate + j) if j > 0 else 1.0
            result += c / (q - self.dynamics(j)) * eigen_factor * math.exp(-j * self.h0)
        return result

    def forcing_vector(self, q: complex) -> list[complex]:
        """
        The three-entry forcing vector b^(k)(q; h0) for the barrier linear system
        M_bar * B = -b^(k):

            b^(k) = [U_part(-h0),
                     K_{r1_neg,-}[U_part](-h0),
                     K_{r2_neg,-}[U_part](-h0)].

        Returns
        -------
        list of three complex numbers.
        """
        dyn = self.dynamics
        return [
            self.evaluate_at_barrier(q),
            self.phase_op_neg_at_barrier(q, dyn.r1_neg),
            self.phase_op_neg_at_barrier(q, dyn.r2_neg),
        ]


@dataclass
class HomogeneousSolution:
    """
    Homogeneous solution to (q - L^(k)) U = 0 on the survival domain z > -h0,
    admissible at +infinity (Lemma 4.3 in the paper).

    The three admissible modes are the negative-real-part roots gamma_4, gamma_5,
    gamma_6 of the characteristic equation psi_Z^(k)(gamma) = q, centered at the
    barrier z = -h0:

        U_hom(q, z; h0) = sum_{m=1}^{3} B_m * exp(gamma_{m+3} * (z + h0)).

    The coefficients B = (B1, B2, B3) are determined by the three barrier
    conditions from killing (Proposition 4.4):

        M_bar * B = -b^(k),

    where M_bar is the 3x3 barrier matrix built from the negative roots and the
    downward phase rates, and b^(k) is the forcing vector from ParticularSolution.

    Parameters
    ----------
    particular : ParticularSolution
        The particular solution, supplying the forcing vector b^(k).
    """
    particular: ParticularSolution

    @property
    def dynamics(self) -> KouZTiltedDynamics:
        return self.particular.dynamics

    @property
    def h0(self) -> float:
        return self.particular.h0

    def _roots(self, q: complex) -> SixRoots:
        return CharacteristicRootFinder(self.dynamics).find(q)

    def barrier_matrix(self, q: complex) -> np.ndarray:
        """
        Build the 3x3 barrier matrix M_bar(q):

            M_bar[0, m] = 1
            M_bar[1, m] = r1_neg / (r1_neg + gamma_{m+3})
            M_bar[2, m] = r2_neg / (r2_neg + gamma_{m+3})

        for m = 0, 1, 2 (columns correspond to gamma_4, gamma_5, gamma_6).
        """
        neg_roots = self._roots(q).negative   # (gamma_4, gamma_5, gamma_6)
        dyn = self.dynamics
        M = np.zeros((3, 3), dtype=complex)
        for m, gamma in enumerate(neg_roots):
            M[0, m] = 1.0
            M[1, m] = dyn.r1_neg / (dyn.r1_neg + gamma)
            M[2, m] = dyn.r2_neg / (dyn.r2_neg + gamma)
        return M

    def coefficients(self, q: complex) -> np.ndarray:
        """
        Solve M_bar * B = -b^(k) and return B = (B1, B2, B3).

        Parameters
        ----------
        q : complex
            Laplace parameter.

        Returns
        -------
        np.ndarray of shape (3,), dtype complex
        """
        M = self.barrier_matrix(q)
        b = np.array(self.particular.forcing_vector(q), dtype=complex)
        return np.linalg.solve(M, -b)

    def evaluate(self, q: complex, z: float) -> complex:
        """
        Evaluate U_hom(q, z; h0) = sum_{m=1}^{3} B_m * exp(gamma_{m+3} * (z + h0)).

        Parameters
        ----------
        q : complex
            Laplace parameter.
        z : float
            Spatial argument. Must satisfy z > -h0.

        Returns
        -------
        complex
        """
        B = self.coefficients(q)
        neg_roots = self._roots(q).negative
        result = 0.0 + 0j
        for Bm, gamma in zip(B, neg_roots):
            result += Bm * np.exp(gamma * (z + self.h0))
        return result


@dataclass
class GeneralSolution:
    """
    Full resolvent solution on the survival domain z > -h0:

        U_hat(q, z; h0) = U_part(q, z; h0) + U_hom(q, z; h0).

    Evaluated at the initial state Z_0 = 0 this gives (eq. Uhat-zero):

        U_hat(q, 0; h0) = sum_{j=0}^{k} c_{k,j} / (q - psi(j))
                        + sum_{m=1}^{3} B_m * exp(gamma_{m+3} * h0),

    which is the Laplace-domain killed moment. Time-domain inversion then yields

        m_k(T; h0) = L^{-1}_{q -> T} [ U_hat(q, 0; h0) ].

    Parameters
    ----------
    homogeneous : HomogeneousSolution
        The homogeneous solution, which already holds the particular solution.
    """
    homogeneous: HomogeneousSolution

    @property
    def particular(self) -> ParticularSolution:
        return self.homogeneous.particular

    def evaluate(self, q: complex, z: float) -> complex:
        """
        Evaluate U_hat(q, z; h0) = U_part(q, z; h0) + U_hom(q, z; h0).

        Parameters
        ----------
        q : complex
            Laplace parameter.
        z : float
            Spatial argument. Must satisfy z > -h0.

        Returns
        -------
        complex
        """
        return self.particular.evaluate(q, z) + self.homogeneous.evaluate(q, z)

    def evaluate_at_origin(self, q: complex) -> complex:
        """
        Evaluate U_hat(q, 0; h0), the Laplace-domain killed moment at Z_0 = 0.

        Parameters
        ----------
        q : complex
            Laplace parameter.

        Returns
        -------
        complex
        """
        return self.evaluate(q, 0.0)
