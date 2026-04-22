from abc import ABC, abstractmethod
from math import factorial, comb
from typing import Callable

import numpy as np


class LaplaceInverter(ABC):
    """
    Abstract base class for numerical Laplace inversion.

    Subclasses implement the inversion of a function F(q) in the Laplace
    domain to recover f(T) in the time domain via

        f(T) = L^{-1}_{q -> T} [F(q)].

    All implementations must expose a single `invert` method.
    """

    @abstractmethod
    def invert(self, F: Callable[[complex], complex], T: float) -> float:
        """
        Numerically invert the Laplace transform F at time T.

        Parameters
        ----------
        F : callable
            Laplace-domain function F(q) -> complex, accepting a complex
            argument q.
        T : float
            Time at which the inverse is evaluated. Must be positive.

        Returns
        -------
        float
            Approximation of f(T) = L^{-1}[F](T).
        """


class TalbotInverter(LaplaceInverter):
    """
    Fixed-Talbot algorithm for numerical Laplace inversion.

    The method deforms the Bromwich contour to a parabolic contour that
    hugs the negative real axis, concentrating the quadrature nodes near
    the singularities of F. This gives super-geometric convergence for
    functions that are analytic in a half-plane.

    The contour parametrisation follows Talbot (1979) as implemented by
    Abate & Whitt (2006):

        theta in (-pi, pi],
        delta = 2*M / (5*T),
        sigma(theta) = delta * theta * (cot(theta) + i),
        sigma'(theta) = delta * (i - theta / sin^2(theta)),

    and the inversion formula is approximated by

        f(T) ≈ (2 / (5*T)) * sum_{k=0}^{M-1} Re[ w_k * F(sigma(theta_k)) ]

    where theta_k = k*pi/M and

        w_0  = 0.5 * exp(delta*T),
        w_k  = exp(T * sigma(theta_k)) * (1 + i*theta_k*(1 + cot^2(theta_k)) - i*cot(theta_k))
               for k >= 1.

    Reference
    ---------
    Abate, J. & Whitt, W. (2006). A unified framework for numerically inverting
    Laplace transforms. INFORMS Journal on Computing, 18(4), 408-421.

    Parameters
    ----------
    M : int
        Number of quadrature nodes. Accuracy improves with M; M=16 to 64
        is typical for financial applications.
    """

    def __init__(self, M: int = 32) -> None:
        if M < 2:
            raise ValueError(f"M must be at least 2, got {M}")
        self.M = M

    def invert(self, F: Callable[[complex], complex], T: float) -> float:
        """
        Approximate f(T) = L^{-1}[F](T) via the fixed-Talbot method.

        Parameters
        ----------
        F : callable
            Laplace-domain function F(q) -> complex.
        T : float
            Evaluation time. Must be strictly positive.

        Returns
        -------
        float
            Real part of the approximation (imaginary residual is a
            numerical artefact and is discarded).
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        M = self.M
        delta = 2.0 * M / (5.0 * T)

        # k=0 term
        result = 0.5 * np.exp(delta * T) * F(delta)

        # k=1, ..., M-1
        for k in range(1, M):
            theta = k * np.pi / M
            cot_theta = np.cos(theta) / np.sin(theta)
            sigma = delta * theta * (cot_theta + 1j)
            sigma_prime = delta * (1j - theta / np.sin(theta) ** 2)
            weight = np.exp(T * sigma) * (1.0 + 1j * theta * (1.0 + cot_theta**2) - 1j * cot_theta)
            result += weight * F(sigma)

        return float((2.0 / (5.0 * T) * result).real)


class StehfestInverter(LaplaceInverter):
    """
    Gaver-Stehfest algorithm for numerical Laplace inversion.

    The method approximates f(T) by evaluating F at N real points along
    the positive real axis:

        f(T) ≈ (ln2 / T) * sum_{k=1}^{N} V_k * F(k * ln2 / T),

    where the Stehfest weights V_k depend only on N:

        V_k = (-1)^{k + N/2} * sum_{j=floor((k+1)/2)}^{min(k, N/2)}
              j^{N/2} * (2j)! / ( (N/2 - j)! * j! * (j-1)! * (k-j)! * (2j-k)! ).

    The algorithm requires only real evaluations of F, making it simple and
    fast. It converges well for smooth f but may struggle with discontinuities
    or oscillatory behaviour. N must be even; N=12 to 20 is typical.

    Reference
    ---------
    Stehfest, H. (1970). Algorithm 368: Numerical inversion of Laplace
    transforms. Communications of the ACM, 13(1), 47-49.

    Parameters
    ----------
    N : int
        Number of terms (must be even). Larger N gives higher accuracy
        but accumulates floating-point cancellation errors; N=16 is a
        practical upper limit in double precision.
    """

    def __init__(self, N: int = 12) -> None:
        if N < 2 or N % 2 != 0:
            raise ValueError(f"N must be a positive even integer, got {N}")
        self.N = N
        self._weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        N = self.N
        half = N // 2
        V = np.zeros(N)
        for k in range(1, N + 1):
            total = 0.0
            for j in range((k + 1) // 2, min(k, half) + 1):
                num = j**half * factorial(2 * j)
                den = (
                    factorial(half - j)
                    * factorial(j)
                    * factorial(j - 1)
                    * factorial(k - j)
                    * factorial(2 * j - k)
                )
                total += num / den
            V[k - 1] = (-1) ** (k + half) * total
        return V

    def invert(self, F: Callable[[complex], complex], T: float) -> float:
        """
        Approximate f(T) = L^{-1}[F](T) via the Gaver-Stehfest method.

        Parameters
        ----------
        F : callable
            Laplace-domain function F(q) -> complex. Only real q values
            are passed; F may return real or complex (imaginary part ignored).
        T : float
            Evaluation time. Must be strictly positive.

        Returns
        -------
        float
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        ln2_over_T = np.log(2.0) / T
        result = sum(
            self._weights[k] * F((k + 1) * ln2_over_T).real
            for k in range(self.N)
        )
        return float(ln2_over_T * result)


class DeHoogInverter(LaplaceInverter):
    """
    de Hoog, Knight & Stokes (1982) algorithm for numerical Laplace inversion.

    The method approximates the Bromwich integral by evaluating F on a
    vertical line Re(q) = sigma and fitting a Padé approximant to the
    resulting Fourier series. It uses 2M+1 evaluations per inversion and
    achieves high accuracy for a wide class of functions, including those
    with oscillatory or discontinuous inverses.

    The implementation follows the recurrence in de Hoog et al. (1982):
    the 2M+1 samples F(sigma + i*k*pi/T2) are assembled into a quotient-
    difference table and the Padé value is extracted via a back-substitution.

    Reference
    ---------
    de Hoog, F. R., Knight, J. H. & Stokes, A. N. (1982). An improved method
    for numerical inversion of Laplace transforms. SIAM Journal on Scientific
    and Statistical Computing, 3(3), 357-366.

    Parameters
    ----------
    M : int
        Half the number of Fourier terms; 2M+1 evaluations of F are used.
        M=16 to 32 is typical.
    alpha : float
        Bromwich shift. A small positive value (e.g. 1e-6) is sufficient
        when F has no singularities on the positive real axis.
    tol : float
        Tolerance for the series-acceleration convergence test.
    """

    def __init__(self, M: int = 24, alpha: float = 1e-6, tol: float = 1e-9) -> None:
        if M < 1:
            raise ValueError(f"M must be at least 1, got {M}")
        self.M = M
        self.alpha = alpha
        self.tol = tol

    def invert(self, F: Callable[[complex], complex], T: float) -> float:
        """
        Approximate f(T) = L^{-1}[F](T) via the de Hoog algorithm.

        Parameters
        ----------
        F : callable
            Laplace-domain function F(q) -> complex.
        T : float
            Evaluation time. Must be strictly positive.

        Returns
        -------
        float
        """
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")

        M = self.M
        T2 = 2.0 * T   # period of the Fourier series
        gamma = self.alpha - np.log(self.tol) / T2

        # --- sample F at the 2M+1 nodes ---
        # q_k = gamma + i*k*pi/T2,  k = 0, ..., 2M
        n = 2 * M + 1
        F_vals = np.array(
            [F(gamma + 1j * k * np.pi / T2) for k in range(n)],
            dtype=complex,
        )

        # --- Fourier coefficients ---
        c = np.empty(n, dtype=complex)
        c[0] = 0.5 * F_vals[0]
        c[1:] = F_vals[1:]

        # --- quotient-difference table (Padé via continued fractions) ---
        # Build the e and q arrays of the QD algorithm
        e = np.zeros((n, M + 1), dtype=complex)
        q = np.zeros((n, M + 1), dtype=complex)
        q[:, 0] = c

        for r in range(1, M + 1):
            e[:-1, r] = q[1:, r - 1] - q[:-1, r - 1] + (e[1:, r - 1] if r > 1 else 0)
            if r < M:
                q[:-2, r] = q[1:-1, r - 1] * e[1:-1, r] / e[:-2, r]

        # --- evaluate the continued fraction at z = exp(i*pi*T/T2) ---
        z = np.exp(1j * np.pi * T / T2)

        # Back-substitution for the Padé value
        A = np.zeros(n + 1, dtype=complex)
        B = np.zeros(n + 1, dtype=complex)
        A[0] = 0.0
        A[1] = q[0, 0]
        B[0] = 1.0
        B[1] = 1.0

        for r in range(1, M + 1):
            A[r + 1] = A[r] + (-z) * e[0, r] * A[r - 1]
            B[r + 1] = B[r] + (-z) * e[0, r] * B[r - 1]
            if r < M:
                A[r + 1] = (A[r + 1] + (-z) * q[0, r] * A[r]) / (1.0 + (-z) * q[0, r] / B[r + 1] * B[r])
                B[r + 1] = B[r + 1] + (-z) * q[0, r] * B[r]

        result = np.exp(gamma * T) / T2 * (A[M + 1] / B[M + 1]).real
        return float(result)
