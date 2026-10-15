"""Markov-modulated carry: the regime-switching killed resolvent.

Extends the single-regime Laplace-resolvent construction of
``optimal_long_short.laplace.laplace_resolvent`` to a net carry that switches with the
lending pool's utilization regime.

Motivation
----------
Deterministic carry is inert for volatile pairs: across the 37 admissible pairs
in ``results/pair_cross_section.csv`` the carry drift is worth between -0.03 and
+0.02 barrier standard deviations over a one-month horizon.  What is *not* inert
is the fact that Aave's borrow rate is convex in utilization.  Above the kink
``U_opt`` the rate climbs a slope of 0.60 to 3.00, so a debt leg that normally
costs 2.5% can cost tens of percent while the pool is stressed -- and the
stressed regime is exactly the deleveraging state that also moves the price
ratio ``Z``.  Modelling that requires the carry to be a two-state process, not
a constant.

Construction
------------
Let ``J_t`` in ``{1, 2}`` be a continuous-time Markov chain (1 = normal,
2 = stressed) with generator

    Q = [[-q1,  q1],
         [ q2, -q2]],

and let the net carry in regime ``i`` be ``c_i``.  The log-ratio process picks up
a regime-dependent drift, so its Levy exponent in regime ``i`` is

    psi^i(s) = psi_Z^(k)(s) + c_i * s.

Only the drift switches; diffusion and jump structure are shared, so the phase
rates ``r_{1,-}``, ``r_{2,-}`` and the moment-admissibility bounds
``k*eta_{1,+} < 1``, ``k*eta_{2,+} < 1`` are untouched.

Writing ``Uhat^i(q, .)`` for the killed resolvent started in regime ``i``, the
backward equations couple through ``Q``:

    (q + q1 - L^1) Uhat^1 - q1 Uhat^2 = G_k,
    -q2 Uhat^1 + (q + q2 - L^2) Uhat^2 = G_k.

An exponential mode ``B v(gamma) e^{gamma x}`` solves the homogeneous system
exactly when ``v`` spans the kernel of

    A(gamma) = [[q + q1 - psi^1(gamma),          -q1],
                [         -q2,          q + q2 - psi^2(gamma)]],

that is, when ``det A(gamma) = 0``.  This is the regime characteristic equation

    Delta(gamma) = (q + q1 - psi^1)(q + q2 - psi^2) - q1 q2 = 0.       (RS-char)

Root count and system size
--------------------------
Each ``psi^i`` has the same four simple poles, so ``psi^i(s) = N^i(s)/D(s)`` with
``deg D = 4`` and ``deg N^i = 6``.  Clearing denominators in (RS-char),

    Delta(s) D(s)^2 = Ntilde^1(s) Ntilde^2(s) - q1 q2 D(s)^2,
    Ntilde^i(s) := (q + q_i) D(s) - N^i(s),

which has degree 12: **twelve** roots, six with negative real part.  The
boundary conditions count to six as well -- absorption ``Uhat^i(q, 0) = 0`` for
each regime, plus the two downward-phase cancellation conditions of Appendix A
for each regime, since the zero-extension residual is generated separately in
each equation (the coupling term carries no generator and contributes none).
The resulting system is therefore ``6 x 6``, and the single-regime ``3 x 3``
is recovered exactly as ``q1 = q2 = 0``.

Invertibility
-------------
``det M_bar`` is analytic in ``(q1, q2)`` and at ``q1 = q2 = 0`` factorizes into
the product of the two single-regime barrier determinants, each nonzero by
Lemma 5.2 of the paper.  A nonzero analytic function vanishes only on a set of
measure zero, so the ``6 x 6`` system is invertible for all but a null set of
switching rates.  This is the regime analogue of Assumption 5.1 and is checked
numerically by :func:`RegimeResolvent.condition_number`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb

import numpy as np

from optimal_long_short.model.kou_model import KouZTiltedDynamics
from optimal_long_short.laplace.root_finder import CharacteristicRootFinder

__all__ = [
    "RegimeSpec",
    "RegimeSwitchingDynamics",
    "RegimeRootFinder",
    "RegimeResolvent",
]


@dataclass(frozen=True)
class RegimeSpec:
    """Two-state carry regime.

    Attributes
    ----------
    q1, q2 : float
        Switching intensities (per year): ``q1`` out of the normal state,
        ``q2`` out of the stressed state.  The stationary probability of the
        stressed state is ``q1 / (q1 + q2)``, which is matched to the observed
        occupation fraction above ``U_opt``.
    c1, c2 : float
        Net carry ``r_s - r_b`` (annualized decimal) in the normal and stressed
        regimes.  Typically ``c1`` is small and positive while ``c2`` is large
        and negative, because the stressed regime is defined by a borrow rate
        that has climbed the steep segment of the rate curve.
    """

    q1: float
    q2: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        if self.q1 < 0.0 or self.q2 < 0.0:
            raise ValueError("Switching intensities must be non-negative.")

    @property
    def stationary(self) -> tuple[float, float]:
        """Stationary distribution ``(pi_normal, pi_stressed)``."""
        total = self.q1 + self.q2
        if total <= 0.0:
            return (1.0, 0.0)
        return (self.q2 / total, self.q1 / total)

    @property
    def mean_carry(self) -> float:
        """Stationary mean net carry -- what a deterministic model would use."""
        pi1, pi2 = self.stationary
        return pi1 * self.c1 + pi2 * self.c2

    @classmethod
    def from_occupation(
        cls,
        *,
        frac_stressed: float,
        mean_duration_stressed: float,
        c_normal: float,
        c_stressed: float,
    ) -> "RegimeSpec":
        """Build a spec from an occupation fraction and a stress duration.

        ``mean_duration_stressed`` is in years and equals ``1 / q2``; ``q1`` is
        then pinned by matching the stationary occupation fraction.  These are
        the two quantities the utilization series identifies directly.
        """
        if not 0.0 < frac_stressed < 1.0:
            raise ValueError(f"frac_stressed must be in (0,1), got {frac_stressed}")
        if mean_duration_stressed <= 0.0:
            raise ValueError("mean_duration_stressed must be positive.")
        q2 = 1.0 / mean_duration_stressed
        q1 = q2 * frac_stressed / (1.0 - frac_stressed)
        return cls(q1=q1, q2=q2, c1=c_normal, c2=c_stressed)


@dataclass
class RegimeSwitchingDynamics:
    """Levy exponents of ``Z`` in each carry regime under the k-tilted measure.

    Wraps a single-regime :class:`KouZTiltedDynamics` and adds the per-regime
    drift shift.  Because carry enters only the drift, ``psi^i(s)`` differs from
    the base exponent by the linear term ``c_i * s`` -- this is exactly the
    deterministic-carry proposition applied state by state.
    """

    base: KouZTiltedDynamics
    spec: RegimeSpec

    @property
    def k(self) -> int:
        return self.base.k

    def psi(self, s: complex, regime: int) -> complex:
        """Levy exponent of ``Z`` in regime ``regime`` (1 or 2)."""
        carry = self.spec.c1 if regime == 1 else self.spec.c2
        return self.base(s) + carry * s

    def kernel_matrix(self, q: complex, gamma: complex) -> np.ndarray:
        """The 2x2 matrix ``A(gamma)`` whose kernel gives the mode's regime split."""
        sp = self.spec
        return np.array(
            [
                [q + sp.q1 - self.psi(gamma, 1), -sp.q1],
                [-sp.q2, q + sp.q2 - self.psi(gamma, 2)],
            ],
            dtype=complex,
        )

    def eigenvector(self, q: complex, gamma: complex) -> np.ndarray:
        """Kernel vector of ``A(gamma)``, taken from the adjugate.

        Using the adjugate rather than an explicit formula keeps the decoupled
        limit ``q1 = q2 = 0`` well-defined, where one adjugate column vanishes
        identically and the other selects the regime the root belongs to.
        """
        A = self.kernel_matrix(q, gamma)
        adj = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], dtype=complex)
        col0, col1 = adj[:, 0], adj[:, 1]
        v = col0 if np.linalg.norm(col0) >= np.linalg.norm(col1) else col1
        norm = np.linalg.norm(v)
        if norm < 1e-300:
            raise ValueError(f"Degenerate eigenvector at gamma={gamma}, q={q}.")
        return v / norm


@dataclass
class RegimeRootFinder:
    """Roots of the regime characteristic equation ``Delta(gamma) = 0``.

    Builds the degree-12 polynomial by reusing the single-regime degree-6
    construction of :class:`CharacteristicRootFinder`, which returns
    ``(psi(s) - q) D(s)``; evaluating it at ``q = 0`` recovers ``psi(s) D(s)``
    and hence ``D(s)`` itself, from which both regimes' numerators follow.
    """

    dynamics: RegimeSwitchingDynamics

    def _base_polys(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(psi*D, D)`` as polynomial coefficient arrays."""
        finder = CharacteristicRootFinder(self.dynamics.base)
        psi_D = finder._build_polynomial(0.0)
        # (psi - q)D at q=0 and q=1 differ by exactly -D.
        psi_D_minus_one = finder._build_polynomial(1.0)
        D = np.polysub(psi_D, psi_D_minus_one)
        return psi_D, D

    def polynomial(self, q: complex) -> np.ndarray:
        """Coefficients of ``Delta(s) D(s)^2``, a degree-12 polynomial."""
        sp = self.dynamics.spec
        psi_D, D = self._base_polys()
        s_D = np.polymul(np.array([1.0, 0.0], dtype=complex), D)  # s * D(s)

        # Ntilde^i(s) = (q + q_i) D - psi^i D,  psi^i D = psi D + c_i s D
        n1 = np.polysub(np.polysub((q + sp.q1) * D, psi_D), sp.c1 * s_D)
        n2 = np.polysub(np.polysub((q + sp.q2) * D, psi_D), sp.c2 * s_D)

        return np.polysub(np.polymul(n1, n2), sp.q1 * sp.q2 * np.polymul(D, D))

    def find(self, q: complex, *, tol: float = 1e-10) -> tuple[complex, ...]:
        """Return the six roots with negative real part.

        Classification is by real-part rank rather than strict sign, matching
        :class:`CharacteristicRootFinder`: on the Talbot contour ``q`` moves into
        the left half-plane and roots cross the imaginary axis continuously,
        but the six that started decaying remain the six smallest in real part.
        """
        roots = np.roots(self.polynomial(q))
        if roots.size < 6:
            raise ValueError(
                f"Expected 12 roots of the regime characteristic equation, "
                f"got {roots.size} at q={q}."
            )
        ordered = sorted(roots, key=lambda r: r.real)
        decaying = ordered[:6]
        for i in range(len(decaying)):
            for j in range(i + 1, len(decaying)):
                if abs(decaying[i] - decaying[j]) < tol:
                    raise ValueError(
                        f"Root collision at q={q}: {decaying[i]} and {decaying[j]} "
                        f"are {abs(decaying[i] - decaying[j]):.2e} apart."
                    )
        return tuple(sorted(decaying, key=lambda r: r.real))


@dataclass
class RegimeResolvent:
    """The 6x6 regime-switching killed resolvent.

    Coordinates follow ``optimal_long_short.laplace.laplace_resolvent``: the barrier
    sits at ``z = -h0`` and the resolvent is reported at ``z = 0``.

    Parameters
    ----------
    dynamics : RegimeSwitchingDynamics
    h0, b : float
        Initial log health buffer and the collateral factor.
    """

    dynamics: RegimeSwitchingDynamics
    h0: float
    b: float

    @property
    def k(self) -> int:
        return self.dynamics.k

    def forcing_coefficients(self) -> list[float]:
        """``c_{k,j}(h0) = C(k,j) (-1)^{k-j} b^{k-j} e^{j h0}``, as in the base model."""
        k, h0, b = self.k, self.h0, self.b
        return [
            comb(k, j) * ((-1) ** (k - j)) * (b ** (k - j)) * math.exp(j * h0)
            for j in range(k + 1)
        ]

    def particular(self, q: complex) -> np.ndarray:
        """Particular-solution coefficients ``P[j, i]`` for ``j = 0..k``, regime ``i``.

        For each forcing exponent ``e^{jz}`` the two regimes solve a 2x2 system
        ``A(j) P_j = (c_{k,j}, c_{k,j})``.  Since ``det A(j) = Delta(j)``, the
        single-regime non-resonance condition ``q != psi(j)`` generalizes to
        ``Delta(j) != 0``.
        """
        coeffs = self.forcing_coefficients()
        out = np.zeros((len(coeffs), 2), dtype=complex)
        for j, c_kj in enumerate(coeffs):
            A = self.dynamics.kernel_matrix(q, float(j))
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-14:
                raise ValueError(
                    f"Regime resonance: Delta({j}) = {det:.3e} at q={q}. "
                    "Perturb q or the switching rates."
                )
            out[j] = np.linalg.solve(A, np.array([c_kj, c_kj], dtype=complex))
        return out

    def barrier_system(self, q: complex) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the 6x6 matrix and right-hand side.

        Rows are ordered ``(regime 1: Dirichlet, phase r1-, phase r2-,
        regime 2: Dirichlet, phase r1-, phase r2-)``; columns index the six
        decaying modes.
        """
        dyn = self.dynamics
        base = dyn.base
        roots = RegimeRootFinder(dyn).find(q)
        P = self.particular(q)
        rates = (base.r1_neg, base.r2_neg)

        vectors = [dyn.eigenvector(q, g) for g in roots]

        M = np.zeros((6, 6), dtype=complex)
        rhs = np.zeros(6, dtype=complex)
        for i in range(2):  # regime index
            # Absorption: Uhat^i(q, -h0) = 0.
            row = 3 * i
            for m, (gamma, v) in enumerate(zip(roots, vectors)):
                M[row, m] = v[i]
            rhs[row] = -sum(
                P[j, i] * math.exp(-j * self.h0) for j in range(P.shape[0])
            )
            # Downward-phase cancellation, one per phase rate.
            for d, r in enumerate(rates):
                row = 3 * i + 1 + d
                for m, (gamma, v) in enumerate(zip(roots, vectors)):
                    M[row, m] = v[i] * r / (r + gamma)
                rhs[row] = -sum(
                    P[j, i] * (r / (r + j) if j > 0 else 1.0) * math.exp(-j * self.h0)
                    for j in range(P.shape[0])
                )
        return M, rhs

    def condition_number(self, q: complex) -> float:
        """Condition number of the 6x6 barrier matrix -- the numerical check on
        the invertibility argument in the module docstring."""
        M, _ = self.barrier_system(q)
        return float(np.linalg.cond(M))

    def transform(self, q: complex) -> np.ndarray:
        """Laplace-domain killed moment at ``z = 0``, per starting regime.

        Returns
        -------
        ndarray of shape (2,)
            ``[Uhat^1(q, 0), Uhat^2(q, 0)]``.
        """
        dyn = self.dynamics
        roots = RegimeRootFinder(dyn).find(q)
        M, rhs = self.barrier_system(q)
        B = np.linalg.solve(M, rhs)
        P = self.particular(q)

        out = np.zeros(2, dtype=complex)
        for i in range(2):
            particular = sum(P[j, i] for j in range(P.shape[0]))
            homogeneous = sum(
                B[m] * dyn.eigenvector(q, g)[i] * np.exp(g * self.h0)
                for m, g in enumerate(roots)
            )
            out[i] = particular + homogeneous
        return out

    def transform_stationary(self, q: complex) -> complex:
        """Laplace-domain killed moment averaged over the stationary regime law."""
        pi1, pi2 = self.dynamics.spec.stationary
        values = self.transform(q)
        return pi1 * values[0] + pi2 * values[1]
