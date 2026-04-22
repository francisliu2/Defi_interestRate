from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from optimal_long_short.kou_model import KouZTiltedDynamics


class SixRoots(NamedTuple):
    """
    The six roots of the characteristic equation psi_Z^(k)(gamma) = q,
    split by sign of real part in accordance with Assumption 4.1(ii).

    positive : tuple of 3 complex
        Roots with Re(gamma) > 0, i.e. gamma_1, gamma_2, gamma_3.
    negative : tuple of 3 complex
        Roots with Re(gamma) < 0, i.e. gamma_4, gamma_5, gamma_6.
    """
    positive: tuple
    negative: tuple


@dataclass
class CharacteristicRootFinder:
    """
    Finds the six roots of the characteristic equation

        psi_Z^(k)(gamma) = q

    for the log-ratio process Z under the k-tilted measure P^(2,k).

    Derivation of the degree-6 polynomial
    --------------------------------------
    The poles of psi_Z^(k)(s) come from the four MGF denominators:

        (r1_pos - s),  (r1_neg + s),  (r2_pos - s),  (r2_neg + s)

    where the phase rates are those of KouZTiltedDynamics.  Multiplying
    both sides of psi_Z^(k)(gamma) = q by

        D(s) = (r1_pos - s)(r1_neg + s)(r2_pos - s)(r2_neg + s)

    clears all denominators and yields:

        P(s) = [0.5*sigma_Z^2 * s^2 + mu_Z * s - q - lam1 - lam2*M2(k)] * D(s)
             + lam1 * p1  * r1_pos * D_no_r1p(s)
             + lam1 * (1-p1) * r1_neg * D_no_r1n(s)
             + lam2 * p2  * eta2_pos * D_no_r2n(s)
             + lam2 * (1-p2) * eta2_neg * D_no_r2p(s)  = 0,

    where D_no_X denotes D(s) with the factor containing X removed.

    Parameters
    ----------
    dynamics : KouZTiltedDynamics
        Lévy exponent and phase rates of Z under P^(2,k).
    """
    dynamics: KouZTiltedDynamics

    def _build_polynomial(self, q: complex) -> np.ndarray:
        """
        Return coefficients of the degree-6 polynomial P(s) in numpy.poly1d
        convention (highest power first).
        """
        dyn = self.dynamics
        p = dyn.params
        k = dyn.k

        a = dyn.r1_pos   # eta1_pos
        b = dyn.r1_neg   # eta1_neg
        c = dyn.r2_pos   # eta2_neg + k
        d = dyn.r2_neg   # eta2_pos - k

        # Linear factors: numpy.poly1d high-to-low convention
        fa = np.array([-1.0, a], dtype=complex)   # (a - s)
        fb = np.array([1.0,  b], dtype=complex)   # (b + s)
        fc = np.array([-1.0, c], dtype=complex)   # (c - s)
        fd = np.array([1.0,  d], dtype=complex)   # (d + s)

        D       = np.polymul(np.polymul(fa, fb), np.polymul(fc, fd))
        D_no_a  = np.polymul(np.polymul(fb, fc), fd)   # (b+s)(c-s)(d+s)
        D_no_b  = np.polymul(np.polymul(fa, fc), fd)   # (a-s)(c-s)(d+s)
        D_no_c  = np.polymul(np.polymul(fa, fb), fd)   # (a-s)(b+s)(d+s)
        D_no_d  = np.polymul(np.polymul(fa, fb), fc)   # (a-s)(b+s)(c-s)

        # M2(k) = p2*(d+k)/d + (1-p2)*(c-k)/c
        #       = p2*eta2_pos/(eta2_pos-k) + (1-p2)*eta2_neg/(eta2_neg+k)
        M2k = p.p2 * (d + k) / d + (1 - p.p2) * (c - k) / c

        # Quadratic polynomial: 0.5*sigma_Z^2*s^2 + mu_Z*s + (-q - lam1 - lam2*M2k)
        quad = np.array(
            [0.5 * dyn.sigma_Z_sq, dyn.mu_Z, -q - p.lam1 - p.lam2 * M2k],
            dtype=complex,
        )

        poly = np.polymul(quad, D)
        poly = np.polyadd(poly, p.lam1 * p.p1 * a       * D_no_a)
        poly = np.polyadd(poly, p.lam1 * (1 - p.p1) * b * D_no_b)
        poly = np.polyadd(poly, p.lam2 * p.p2 * (d + k)       * D_no_d)
        poly = np.polyadd(poly, p.lam2 * (1 - p.p2) * (c - k) * D_no_c)

        return poly

    def find(self, q: complex, tol: float = 1e-8) -> SixRoots:
        """
        Compute and classify the six roots of psi_Z^(k)(gamma) = q.

        Parameters
        ----------
        q : complex
            Laplace parameter. Must be positive real for standard usage.
        tol : float
            Minimum separation between any two roots; raises ValueError if
            any pair is closer than this (root collision / Assumption 4.1 violated).

        Returns
        -------
        SixRoots
            Named tuple with fields `positive` (Re > 0) and `negative` (Re < 0),
            each a tuple of 3 complex numbers sorted by descending and ascending
            real part respectively.

        Raises
        ------
        ValueError
            If fewer or more than 3 roots fall on each side of the imaginary axis,
            or if any two roots are closer than `tol`.
        """
        poly = self._build_polynomial(q)
        roots = np.roots(poly)

        # --- collision check ---
        for i in range(len(roots)):
            for j in range(i + 1, len(roots)):
                dist = abs(roots[i] - roots[j])
                if dist < tol:
                    raise ValueError(
                        f"Root collision: roots {i} and {j} are {dist:.2e} apart "
                        f"(tol={tol}). Assumption 4.1(ii) is violated at q={q}."
                    )

        # --- split by sign of real part ---
        pos = [r for r in roots if r.real > 0]
        neg = [r for r in roots if r.real < 0]

        if len(pos) != 3 or len(neg) != 3:
            raise ValueError(
                f"Expected 3 roots with Re>0 and 3 with Re<0, "
                f"got {len(pos)} positive and {len(neg)} negative "
                f"(roots: {roots}). Check q={q} and model parameters."
            )

        pos = tuple(sorted(pos, key=lambda r: r.real, reverse=True))
        neg = tuple(sorted(neg, key=lambda r: r.real))

        return SixRoots(positive=pos, negative=neg)


if __name__ == "__main__":
    from optimal_long_short.model_params import KouParams

    params = KouParams(
        mu1=0.05,  sigma1=0.20, lam1=1.0, p1=0.5, eta1_pos=10.0, eta1_neg=8.0,
        mu2=0.03,  sigma2=0.15, lam2=0.8, p2=0.5, eta2_pos=12.0, eta2_neg=9.0,
        rho=0.3,
    )

    for k in (0, 1, 2):
        dyn = KouZTiltedDynamics(params=params, k=k)
        finder = CharacteristicRootFinder(dynamics=dyn)
        roots = finder.find(q=0.05)

        print(f"\n=== k={k} ===")
        print("  Positive roots (Re > 0):")
        for i, r in enumerate(roots.positive, 1):
            print(f"    gamma_{i}   = {r.real:+.6f} {r.imag:+.6f}j")
        print("  Negative roots (Re < 0):")
        for i, r in enumerate(roots.negative, 4):
            print(f"    gamma_{i}   = {r.real:+.6f} {r.imag:+.6f}j")
