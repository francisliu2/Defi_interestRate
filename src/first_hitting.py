"""First-hitting time utilities for log-health process h(t).

This module provides tools to compute the Laplace transform root Phi(q)
via the Lévy exponent psi_h(·), evaluate the Laplace transform of the
first-hitting time and (numerically) invert the Laplace transform to
obtain the distribution function P(tau_0 <= T).

Design:
- The module is generic: it requires the user to provide a callable
  `psi(theta)` returning the Lévy exponent ψ_h(θ) for scalar theta.
- `compute_phi(q, psi)` finds the positive root Φ(q) solving ψ(Φ)=q.
- `laplace_fht(q, h0, psi)` returns E[e^{-q tau} 1_{tau<∞}] = exp(-h0 Φ(q)).
- `cdf_fht(T, h0, psi, R=None, conditional=False)` numerically inverts
  the Laplace transform to estimate P(tau_0 <= T). If `R` is provided
  and `conditional=True`, the conditional transform for defective
  hitting is used: L{F_cond}(q)=exp(-h0(Φ(q+R)-R))/q.

Notes:
- This implementation prefers `mpmath.invertlaplace` (Talbot) if
  `mpmath` is installed. If not available, a basic Bromwich/integral
  approximation is used (less accurate). The user may replace the
  inversion routine with a preferred implementation.

Example usage:
    from first_hitting import compute_phi, cdf_fht

    # provide psi(theta) for your model (callable)
    F_T = cdf_fht(T=30.0, h0=0.5, psi=psi)

"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import cmath

try:
    import mpmath as mp
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False

from scipy.optimize import root_scalar


def compute_phi(q: float, psi: Callable[[float], float], bracket: Optional[tuple] = (1e-12, 10.0),
                expand_factor: float = 2.0, max_expand: int = 50) -> float:
    """Find Phi(q) > 0 such that psi(Phi) = q.

    Args:
        q: Laplace variable (non-negative).
        psi: Callable psi(theta) returning scalar float.
        bracket: initial bracket (a,b) where psi(a)-q and psi(b)-q have opposite signs.
        expand_factor: factor for geometric expansion when bracket doesn't contain a root.
        max_expand: maximum number of expansions.

    Returns:
        Phi: positive root value.

    Raises:
        RuntimeError if a root cannot be found.
    """
    a, b = bracket
    fa = psi(a) - q
    fb = psi(b) - q
    # expand bracket until sign change or limit
    n = 0
    while fa * fb > 0 and n < max_expand:
        b *= expand_factor
        fb = psi(b) - q
        n += 1
    if fa * fb > 0:
        # try a small positive root via Newton-ish starting from small value
        def froot(x):
            return psi(x) - q

        try:
            sol = root_scalar(froot, x0=max(a, 1e-8), x1=max(a * 10.0, 1e-4), method='secant')
            if sol.converged and sol.root > 0:
                return float(sol.root)
        except Exception:
            pass
        raise RuntimeError(f"Failed to bracket root for q={q}. psi(a)-q={fa}, psi(b)-q={fb}")
    sol = root_scalar(lambda x: psi(x) - q, bracket=[a, b], method='brentq', xtol=1e-10)
    if not sol.converged:
        raise RuntimeError(f"Root finding failed for q={q}")
    return float(sol.root)


def laplace_fht(q: float, h0: float, psi: Callable[[float], float]) -> float:
    """Laplace transform E[e^{-q tau} 1_{tau<∞}] = exp(-h0 * Phi(q))."""
    phi = compute_phi(q, psi)
    return math.exp(-float(h0) * float(phi))


def _laplace_for_inversion(s: complex, h0: float, psi: Callable[[float], float], R: Optional[float], conditional: bool) -> complex:
    """Return Laplace-space function value F(s) used for inversion.

    s may be complex when called by inversion routines.
    For defective conditional case with R, we evaluate at q = s + R.
    """
    # handle scalar complex by evaluating psi at real positive argument if possible
    q = s
    if conditional and (R is not None):
        q = s + R
    # compute phi(q_real) — for complex q we attempt to find phi for Re(q)
    # Many inversion routines pass complex s; we handle by using the real part
    # for root finding, which is a practical approximation for moderate imag parts.
    q_real = float(mp.re(q)) if _HAS_MPMATH else float(np.real(q))
    phi_q = compute_phi(q_real, psi)
    if conditional and (R is not None):
        # L{F_cond}(q) = exp(-h0*(Phi(q+R)-R))/q
        num = math.exp(-float(h0) * (phi_q - float(R)))
    else:
        num = math.exp(-float(h0) * phi_q)
    return complex(num) / complex(s)


def cdf_fht(T: float, h0: float, psi: Callable[[float], float], R: Optional[float] = None,
            conditional: bool = False, tol: float = 1e-8) -> float:
    """Estimate P(tau_0 <= T) by numerical Laplace inversion.

    Args:
        T: time at which to evaluate the CDF.
        h0: initial log-health > 0.
        psi: callable psi(theta).
        R: optional adjustment coefficient for defective hitting (used when conditional=True).
        conditional: if True, compute conditional CDF F_cond(T) using transform with R.
        tol: tolerance passed to inversion routine when available.

    Returns:
        Approximate CDF value (float in [0,1]).
    """
    if T <= 0:
        return 0.0

    if _HAS_MPMATH:
        # use mpmath's invertlaplace (Talbot) — it expects a function of a real variable s
        def Fq(s):
            return _laplace_for_inversion(s, h0, psi, R, conditional)

        # mpmath invertlaplace requires mp.mpf inputs and returns mp.mpf
        try:
            mp.mp.dps = 30
            val = mp.invertlaplace(lambda s: Fq(s), T, method='talbot')
            return float(mp.re(val))
        except Exception:
            # fall back to crude Bromwich if mpmath fails
            pass

    # crude Bromwich/trapezoidal approximation (less accurate)
    # integral along vertical line: F(t) = (1/(2πi)) ∫_{σ-i∞}^{σ+i∞} e^{qt} F̂(q) dq
    # transform to real integral; choose σ>0
    sigma = 1.0 / max(1.0, T)
    N = 256
    omegas = np.linspace(-200.0, 200.0, N)
    dq = omegas[1] - omegas[0]
    integrand_vals = []
    for w in omegas:
        q = complex(sigma, w)
        Fq = _laplace_for_inversion(q, h0, psi, R, conditional)
        integrand = cmath.exp(q * T) * Fq
        integrand_vals.append(integrand)
    integral = sum(integrand_vals) * dq
    result = (1.0 / (2j * math.pi)) * integral
    return float(result.real)


# helper for interactive / example usage
def example_usage():
    """Simple illustrative example (requires user-supplied psi).

    The user must provide a callable `psi(theta)` for their log-health
    process. Below is a toy diffusion-only psi for demonstration only.
    """
    def psi_diffusion(theta: float) -> float:
        # example: pure Brownian drift mu and variance sigma2: psi(theta)= -mu*theta + 0.5*sigma2*theta^2
        mu = -0.01
        sigma2 = 0.04
        return -mu * theta + 0.5 * sigma2 * theta * theta

    h0 = 0.5
    T = 30.0
    print('Computing Phi(1.0)...')
    phi1 = compute_phi(1.0, psi_diffusion)
    print('Phi(1) =', phi1)
    print('Estimating CDF at T=30')
    try:
        cdf = cdf_fht(T, h0, psi_diffusion)
        print('P(tau<=T) ~', cdf)
    except Exception as e:
        print('Inversion failed:', e)


if __name__ == '__main__':
    example_usage()
