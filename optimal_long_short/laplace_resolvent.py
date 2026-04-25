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


_DEGEN_TOL = 1e-8   # threshold for declaring r1_neg == r2_neg


@dataclass
class HomogeneousSolution:
    """
    Homogeneous solution to (q - L^(k)) U = 0 on the survival domain z > -h0,
    admissible at +infinity (Lemma 4.3 in the paper).

    Generic case (r1_neg != r2_neg)
    --------------------------------
    Three admissible modes gamma_4, gamma_5, gamma_6 (negative-real-part roots
    of psi_Z^(k)(gamma) = q).  Coefficients determined by the 3x3 barrier system.

    Degenerate case (r1_neg == r2_neg = r)
    ----------------------------------------
    When the two downward phase rates coincide, the denominator in
    _build_polynomial gains a repeated factor (r + s)^2.  This introduces a
    spurious root at s = -r into the degree-6 polynomial; it is NOT a genuine
    root of psi_Z^(k)(gamma) = q (psi has a pole at s = -r).  Consequently:

      - Only 2 genuine negative roots exist (gamma_4, gamma_5).
      - Rows 1 and 2 of the 3x3 barrier matrix are identical, making it singular.

    In this case we:
      1. Identify and discard the spurious root (the negative root nearest -r).
      2. Build a 2x2 barrier system using only the continuity condition and the
         single (merged) downward-jump condition.

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

    @property
    def _degenerate(self) -> bool:
        """True when r1_neg and r2_neg coincide (barrier matrix would be singular)."""
        dyn = self.dynamics
        return abs(dyn.r1_neg - dyn.r2_neg) < _DEGEN_TOL

    def _genuine_negative_roots(self, q: complex) -> tuple:
        """
        Return the genuine negative roots of psi_Z^(k)(gamma) = q.

        Generic case : all 3 negative roots from CharacteristicRootFinder.
        Degenerate   : remove the spurious root (closest to -r1_neg) and return
                       the remaining 2.
        """
        neg = list(self._roots(q).negative)
        if not self._degenerate:
            return tuple(neg)
        r = self.dynamics.r1_neg
        spurious_idx = int(np.argmin([abs(gamma + r) for gamma in neg]))
        return tuple(neg[i] for i in range(len(neg)) if i != spurious_idx)

    def barrier_matrix(self, q: complex) -> np.ndarray:
        """
        Build the n x n barrier matrix M_bar(q), where n = 3 (generic) or 2
        (degenerate).

        Generic (n=3):
            M_bar[0, m] = 1
            M_bar[1, m] = r1_neg / (r1_neg + gamma_m)
            M_bar[2, m] = r2_neg / (r2_neg + gamma_m)

        Degenerate (n=2, r1_neg == r2_neg == r):
            M_bar[0, m] = 1
            M_bar[1, m] = r / (r + gamma_m)
        """
        genuine_neg = self._genuine_negative_roots(q)
        n = len(genuine_neg)
        dyn = self.dynamics
        M = np.zeros((n, n), dtype=complex)
        for m, gamma in enumerate(genuine_neg):
            M[0, m] = 1.0
            M[1, m] = dyn.r1_neg / (dyn.r1_neg + gamma)
            if n == 3:
                M[2, m] = dyn.r2_neg / (dyn.r2_neg + gamma)
        return M

    def coefficients(self, q: complex) -> np.ndarray:
        """
        Solve M_bar * B = -b^(k) and return B.

        Returns shape (3,) in the generic case, shape (2,) in the degenerate case.
        """
        genuine_neg = self._genuine_negative_roots(q)
        n = len(genuine_neg)
        M = self.barrier_matrix(q)
        bvec = np.array(self.particular.forcing_vector(q)[:n], dtype=complex)
        return np.linalg.solve(M, -bvec)

    def evaluate(self, q: complex, z: float) -> complex:
        """
        Evaluate U_hom(q, z; h0) = sum_m B_m * exp(gamma_m * (z + h0)).
        """
        B = self.coefficients(q)
        genuine_neg = self._genuine_negative_roots(q)
        result = 0.0 + 0j
        for Bm, gamma in zip(B, genuine_neg):
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
