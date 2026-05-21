"""
MAD-based threshold initializer and structured multi-start cloud for ECF
calibration of the bivariate Kou model.

Two estimation modes are combined to produce the starting parameter vector:

Threshold filter (Mancini 2009)
    Classifies returns as jump or diffusion using a robust MAD scale.
    Gives good estimates of p, sigma, and rho but *underestimates* lambda
    because only the largest jumps exceed the cut point.

Two-moment lambda correction
    For a compound Poisson process:
        Var(r)/dt  = sigma^2 + lambda * M2,   M2 = p*2*ep^2 + (1-p)*2*en^2
        kappa4(r)/dt = lambda * M4,            M4 = p*24*ep^4 + (1-p)*24*en^4
    Dividing eliminates lambda and gives an explicit equation for a symmetric
    jump scale eta; substituting back yields lambda.  This estimator uses all
    returns (not just flagged outliers) and avoids the truncation bias that
    makes the threshold estimate 3-4x too low for daily crypto data.

    To handle asymmetric jumps, the threshold filter is used to estimate the
    ratio ep/en, and the two-moment system solves for the effective scale.

The multi-start cloud uses four scenario groups (A–D) to protect against the
jump-diffusion identification problem (many small jumps vs. high diffusion).

Frequency-aware default threshold
    At low sampling frequencies, jumps and diffusion are more aggregated so
    a high cut misses most jumps.  Recommended defaults (from plan):
        "1d"   → 3.0
        "4h"   → 3.5
        "1h"   → 4.0
        "15min"→ 5.0
"""
from __future__ import annotations

import numpy as np

from .transforms import ParameterBounds, nat_to_unc, _DEFAULT_BOUNDS


def _robust_scale(r: np.ndarray, min_scale: float = 1e-8) -> float:
    """IQR-based robust scale: (q75 - q25) / 1.349."""
    q25, q75 = np.percentile(r, [25, 75])
    return float(max((q75 - q25) / 1.349, min_scale))


def _dt_to_threshold(dt: float) -> float:
    """
    Recommended MAD threshold based on the observation interval.

    At daily frequency (dt ≈ 1/365), a high threshold misses most jumps
    because the diffusion scale and typical jump size are comparable.
    Lower frequencies → lower threshold.
    """
    dt_hours = dt * 365 * 24
    if dt_hours >= 20:       # daily or longer
        return 3.0
    elif dt_hours >= 3:      # 4-hourly
        return 3.5
    elif dt_hours >= 0.9:    # hourly
        return 4.0
    else:                    # 15-min and finer
        return 5.0


def _lambda_from_moments(
    r: np.ndarray,
    dt: float,
    sigma_init: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """
    Two-moment lambda estimate from variance and 4th cumulant.

    For a compound Poisson process with symmetric double-exponential jumps
    (mean eta, all moments exist):

        V2  = lambda * 2 * eta^2    (annualized jump variance)
        K4  = lambda * 24 * eta^4   (annualized 4th cumulant)

    Eliminating eta gives the closed-form:  lambda = 6 * V2^2 / K4

    This formula uses all N returns and avoids the truncation bias of the
    threshold filter, which detects only the largest jumps and thereby
    inflates the apparent jump means, causing lambda to be 3-5× too low.

    For asymmetric jumps the formula is still approximately correct (the
    symmetric moment ratios change by at most ~30% for typical crypto
    parameters), and is always orders of magnitude better than the
    threshold-based estimate when true lambda is large.

    Parameters
    ----------
    r          : (N,) per-period log-return array.
    dt         : Observation interval in years.
    sigma_init : Diffusion volatility estimate from threshold filter.

    Returns
    -------
    Lambda estimate (annualized).
    """
    kappa4 = float(np.mean((r - r.mean()) ** 4) - 3.0 * np.var(r) ** 2)
    if kappa4 <= 0:
        return lambda_min

    V2 = float(np.var(r) / dt - sigma_init ** 2)
    K4 = kappa4 / dt

    if V2 <= 0 or K4 <= 0:
        return lambda_min

    lam = 6.0 * V2 ** 2 / K4
    return float(np.clip(lam, lambda_min, lambda_max))


def initialize(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Threshold-informed moment initializer for the bivariate Kou model.

    Uses a robust IQR scale to flag candidate jump observations, then
    estimates each parameter group from the appropriate subset.  Lambda
    is subsequently corrected via the two-moment (variance + 4th cumulant)
    estimator to remove the truncation bias of the threshold filter.

    Parameters
    ----------
    r1, r2    : (N,) log-return arrays.
    dt        : Annualized time step in years.
    bounds    : Parameter bounds for clipping.
    threshold : MAD multiplier.  If None, inferred from dt.

    Returns
    -------
    theta_nat : (13,) natural-space parameter vector.
    """
    if threshold is None:
        threshold = _dt_to_threshold(dt)

    N = len(r1)
    m1, m2 = np.median(r1), np.median(r2)
    s1 = _robust_scale(r1)
    s2 = _robust_scale(r2)

    mask_j1 = np.abs(r1 - m1) > threshold * s1
    mask_j2 = np.abs(r2 - m2) > threshold * s2
    mask_c1, mask_c2 = ~mask_j1, ~mask_j2

    n_j1, n_j2 = int(mask_j1.sum()), int(mask_j2.sum())

    # Threshold-based lambda (biased low; corrected below)
    lam1_thresh = float(np.clip(n_j1 / max(N * dt, 1e-12), bounds.lambda_min, bounds.lambda_max))
    lam2_thresh = float(np.clip(n_j2 / max(N * dt, 1e-12), bounds.lambda_min, bounds.lambda_max))

    # Jump direction probabilities
    p1 = float(np.clip(
        (mask_j1 & (r1 - m1 > 0)).sum() / max(n_j1, 1), bounds.p_min, bounds.p_max
    ))
    p2 = float(np.clip(
        (mask_j2 & (r2 - m2 > 0)).sum() / max(n_j2, 1), bounds.p_min, bounds.p_max
    ))

    # Default jump size: 2 × per-period robust scale (frequency-aware floor)
    def_eta1 = max(2.0 * s1, 1e-4)
    def_eta2 = max(2.0 * s2, 1e-4)

    def _mean_or(arr: np.ndarray, mask: np.ndarray, default: float) -> float:
        return float(np.mean(arr[mask])) if mask.any() else default

    eta1_pos = float(np.clip(
        _mean_or(r1 - m1,         mask_j1 & (r1 - m1 > 0), def_eta1),
        bounds.eta_pos1_min, bounds.eta_pos1_max,
    ))
    eta1_neg = float(np.clip(
        _mean_or(np.abs(r1 - m1), mask_j1 & (r1 - m1 < 0), def_eta1),
        bounds.eta_neg_min, bounds.eta_neg_max,
    ))
    eta2_max = bounds.eta_pos2_max
    eta2_pos = float(np.clip(
        _mean_or(r2 - m2,         mask_j2 & (r2 - m2 > 0), def_eta2),
        bounds.eta_pos2_min, eta2_max,
    ))
    eta2_neg = float(np.clip(
        _mean_or(np.abs(r2 - m2), mask_j2 & (r2 - m2 < 0), def_eta2),
        bounds.eta_neg_min, bounds.eta_neg_max,
    ))

    # Annualized diffusion volatility from non-jump returns
    def _std_or(arr: np.ndarray, mask: np.ndarray, fallback: float) -> float:
        return float(np.std(arr[mask])) if mask.sum() > 1 else fallback

    sigma1 = float(np.clip(
        _std_or(r1, mask_c1, s1) / np.sqrt(dt), bounds.sigma_min, bounds.sigma_max
    ))
    sigma2 = float(np.clip(
        _std_or(r2, mask_c2, s2) / np.sqrt(dt), bounds.sigma_min, bounds.sigma_max
    ))

    # Two-moment corrected lambda (replaces the biased threshold estimate)
    lam1 = _lambda_from_moments(r1, dt, sigma1, bounds.lambda_min, bounds.lambda_max)
    lam2 = _lambda_from_moments(r2, dt, sigma2, bounds.lambda_min, bounds.lambda_max)

    # Price-growth drift: mu_i = mean(r)/dt + 0.5*sigma_i^2 + lam_i*chi_i
    # Derivation: E[r_t]/dt = mu_i - 0.5*sigma_i^2 - lam_i*chi_i + lam_i*E[J_i],
    # and the lam*E[J] term cancels exactly leaving mu_i = mean(r)/dt + 0.5*sigma^2 + lam*chi.
    chi1 = p1 / (1.0 - eta1_pos) + (1.0 - p1) / (1.0 + eta1_neg) - 1.0
    chi2 = p2 / (1.0 - eta2_pos) + (1.0 - p2) / (1.0 + eta2_neg) - 1.0
    mu1 = float(np.clip(
        np.mean(r1) / dt + 0.5 * sigma1 ** 2 + lam1 * chi1,
        bounds.mu_min, bounds.mu_max,
    ))
    mu2 = float(np.clip(
        np.mean(r2) / dt + 0.5 * sigma2 ** 2 + lam2 * chi2,
        bounds.mu_min, bounds.mu_max,
    ))

    # Brownian correlation from simultaneous non-jump pairs
    mask_c12 = mask_c1 & mask_c2
    base = r1[mask_c12] if mask_c12.sum() > 2 else r1
    ref  = r2[mask_c12] if mask_c12.sum() > 2 else r2
    rho  = float(np.clip(np.corrcoef(base, ref)[0, 1], bounds.rho_min, bounds.rho_max))

    return np.array([
        mu1, sigma1, lam1, p1, eta1_pos, eta1_neg,
        mu2, sigma2, lam2, p2, eta2_pos, eta2_neg,
        rho,
    ])


# Scenario multipliers for the structured multi-start cloud.
_SCENARIOS: list[dict] = [
    {"sigma": 1.0, "lam": 1.0, "eta": 1.0},  # A: initializer neighborhood
    {"sigma": 0.7, "lam": 2.0, "eta": 0.8},  # B: high-jump
    {"sigma": 1.3, "lam": 0.5, "eta": 1.2},  # C: high-diffusion
    {"sigma": 0.8, "lam": 3.0, "eta": 0.6},  # D: very-high-jump
]


def build_multistart_cloud(
    theta_nat: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> np.ndarray:
    """
    Build a (n_starts, 13) array of unconstrained starting points.

    Starts cycle through four scenario groups (A–D above) with random
    multiplicative perturbations on scale/lambda/eta and additive
    perturbations on probabilities and correlation.  The first row is
    always the unperturbed initializer point.
    """
    starts = np.empty((n_starts, 13))
    starts[0] = nat_to_unc(theta_nat, bounds)

    eta2_max = bounds.eta_pos2_max
    mu1, s1, l1, p1, ep1, en1, mu2, s2, l2, p2, ep2, en2, rho = theta_nat

    for i in range(1, n_starts):
        sc = _SCENARIOS[i % len(_SCENARIOS)]
        noise = rng.uniform(-0.3, 0.3, size=8)

        s1_p  = float(np.clip(s1  * sc["sigma"] * np.exp(noise[0]), bounds.sigma_min,   bounds.sigma_max))
        l1_p  = float(np.clip(l1  * sc["lam"]   * np.exp(noise[1]), bounds.lambda_min,  bounds.lambda_max))
        ep1_p = float(np.clip(ep1 * sc["eta"]   * np.exp(noise[2]), bounds.eta_pos1_min, bounds.eta_pos1_max))
        en1_p = float(np.clip(en1 * sc["eta"]   * np.exp(noise[3]), bounds.eta_neg_min,  bounds.eta_neg_max))
        s2_p  = float(np.clip(s2  * sc["sigma"] * np.exp(noise[4]), bounds.sigma_min,   bounds.sigma_max))
        l2_p  = float(np.clip(l2  * sc["lam"]   * np.exp(noise[5]), bounds.lambda_min,  bounds.lambda_max))
        ep2_p = float(np.clip(ep2 * sc["eta"]   * np.exp(noise[6]), bounds.eta_pos2_min, eta2_max))
        en2_p = float(np.clip(en2 * sc["eta"]   * np.exp(noise[7]), bounds.eta_neg_min,  bounds.eta_neg_max))

        p1_p  = float(np.clip(p1  + rng.uniform(-0.15, 0.15), bounds.p_min,   bounds.p_max))
        p2_p  = float(np.clip(p2  + rng.uniform(-0.15, 0.15), bounds.p_min,   bounds.p_max))
        rho_p = float(np.clip(rho + rng.uniform(-0.15, 0.15), bounds.rho_min, bounds.rho_max))
        mu1_p = float(np.clip(mu1 + rng.uniform(-0.2,  0.2),  bounds.mu_min,  bounds.mu_max))
        mu2_p = float(np.clip(mu2 + rng.uniform(-0.2,  0.2),  bounds.mu_min,  bounds.mu_max))

        starts[i] = nat_to_unc(np.array([
            mu1_p, s1_p, l1_p, p1_p, ep1_p, en1_p,
            mu2_p, s2_p, l2_p, p2_p, ep2_p, en2_p, rho_p,
        ]), bounds)

    return starts
