"""
Threshold/POT initializer and structured multi-start cloud for ECF
calibration of the bivariate Kou model.

Two estimation modes are combined to produce the starting parameter vector:

Threshold filter (Mancini 2009)
    Classifies returns as jump or diffusion using a robust IQR scale.
    Used exclusively to identify the diffusion subset for estimating sigma
    and rho; not used for jump parameters.

POT-based jump estimation (Peak-Over-Threshold)
    Sets a high quantile of the positive/negative return pool as the
    exceedance threshold u.  By the memoryless property of the exponential
    distribution, mean(X - u | X > u) = eta regardless of u, giving an
    unbiased estimate of the jump-size mean.  Exceedance intensity is
    extrapolated to full jump intensity via P(J > u) = exp(-u / eta),
    avoiding the truncation bias of simple jump-counting methods.

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

    For a compound Poisson process: V2 = lambda * 2*eta^2, K4 = lambda * 24*eta^4.
    Dividing: lambda = 6 * V2^2 / K4.  Uses all N returns, avoiding the truncation
    bias of threshold counting.
    """
    kappa4 = float(np.mean((r - r.mean()) ** 4) - 3.0 * np.var(r) ** 2)
    if kappa4 <= 0:
        return lambda_min
    V2 = float(np.var(r) / dt - sigma_init ** 2)
    K4 = kappa4 / dt
    if V2 <= 0 or K4 <= 0:
        return lambda_min
    return float(np.clip(6.0 * V2 ** 2 / K4, lambda_min, lambda_max))


def pot_init_one_asset(
    r: np.ndarray,
    dt: float,
    q_pos: float = 0.99,
    q_neg: float = 0.99,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> tuple[float, float, float, float]:
    """
    POT-style jump parameter estimator for one marginal Kou process.

    Computes the exceedance threshold at quantile q_pos / q_neg of the
    positive and negative return pools respectively, estimates jump-size
    means from the mean excess above each threshold (unbiased for
    exponential by the memoryless property), then extrapolates the
    exceedance rate to the full jump intensity component using
    P(J > u) = exp(-u / eta).

    Returns
    -------
    lam, p, eta_pos, eta_neg  — all clipped to bounds.
    """
    r = np.asarray(r, dtype=float)
    N = len(r)

    x = r - np.median(r)
    pos = x[x > 0]
    neg = -x[x < 0]

    if len(pos) < 10 or len(neg) < 10:
        return 2.0, 0.5, 0.05, 0.05

    u_pos = float(np.quantile(pos, q_pos))
    u_neg = float(np.quantile(neg, q_neg))

    exc_pos = pos[pos > u_pos] - u_pos
    exc_neg = neg[neg > u_neg] - u_neg

    n_pos = len(exc_pos)
    n_neg = len(exc_neg)

    eta_pos = float(np.mean(exc_pos)) if n_pos > 0 else 0.05
    eta_neg = float(np.mean(exc_neg)) if n_neg > 0 else 0.05
    eta_pos = float(np.clip(eta_pos, bounds.eta_pos1_min, bounds.eta_pos1_max))
    eta_neg = float(np.clip(eta_neg, bounds.eta_neg_min, bounds.eta_neg_max))

    lam_pos_exc = n_pos / (N * dt)
    lam_neg_exc = n_neg / (N * dt)
    lam_pos = lam_pos_exc * float(np.exp(min(u_pos / eta_pos, 20.0))) if n_pos > 0 else 0.0
    lam_neg = lam_neg_exc * float(np.exp(min(u_neg / eta_neg, 20.0))) if n_neg > 0 else 0.0

    lam = float(np.clip(lam_pos + lam_neg, bounds.lambda_min, bounds.lambda_max))
    p = lam_pos / (lam_pos + lam_neg) if (lam_pos + lam_neg) > 0 else 0.5
    p = float(np.clip(p, bounds.p_min, bounds.p_max))

    return lam, p, eta_pos, eta_neg


def pot_anchors(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    q_pos: float = 0.99,
    q_neg: float = 0.99,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> dict[str, float]:
    """
    POT-based log-scale anchors for jump-size means (both assets).

    The excess-mean estimator is unbiased for the exponential distribution
    regardless of threshold.  Used as soft priors in the ECF objective to
    prevent the high-lambda / small-eta identification degeneracy.

    Note: lambda anchors are computed separately via the two-moment estimator,
    which is more reliable than POT extrapolation when lambda*dt is small.

    Returns
    -------
    dict with keys:
      "log_ep1", "log_en1"   log(eta_pos/neg) anchors for asset 1
      "log_ep2", "log_en2"   log(eta_pos/neg) anchors for asset 2
    """
    _, _, ep1, en1 = pot_init_one_asset(r1, dt, q_pos, q_neg, bounds)
    _, _, ep2, en2 = pot_init_one_asset(r2, dt, q_pos, q_neg, bounds)
    return {
        "log_ep1": float(np.log(max(ep1, 1e-12))),
        "log_en1": float(np.log(max(en1, 1e-12))),
        "log_ep2": float(np.log(max(ep2, 1e-12))),
        "log_en2": float(np.log(max(en2, 1e-12))),
    }


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

    m1, m2 = np.median(r1), np.median(r2)
    s1 = _robust_scale(r1)
    s2 = _robust_scale(r2)

    mask_j1 = np.abs(r1 - m1) > threshold * s1
    mask_j2 = np.abs(r2 - m2) > threshold * s2
    mask_c1, mask_c2 = ~mask_j1, ~mask_j2
    n_j1, n_j2 = int(mask_j1.sum()), int(mask_j2.sum())

    # Jump direction probability from threshold filter
    p1 = float(np.clip(
        (mask_j1 & (r1 - m1 > 0)).sum() / max(n_j1, 1), bounds.p_min, bounds.p_max
    ))
    p2 = float(np.clip(
        (mask_j2 & (r2 - m2 > 0)).sum() / max(n_j2, 1), bounds.p_min, bounds.p_max
    ))

    # POT excess means for jump-size parameters (unbiased exponential estimator)
    _, _, eta1_pos, eta1_neg = pot_init_one_asset(r1, dt, bounds=bounds)
    _, _, eta2_pos, eta2_neg = pot_init_one_asset(r2, dt, bounds=bounds)
    eta2_pos = float(np.clip(eta2_pos, bounds.eta_pos2_min, bounds.eta_pos2_max))

    # Annualized diffusion volatility from non-jump returns
    def _std_or(arr: np.ndarray, mask: np.ndarray, fallback: float) -> float:
        return float(np.std(arr[mask])) if mask.sum() > 1 else fallback

    sigma1 = float(np.clip(
        _std_or(r1, mask_c1, s1) / np.sqrt(dt), bounds.sigma_min, bounds.sigma_max
    ))
    sigma2 = float(np.clip(
        _std_or(r2, mask_c2, s2) / np.sqrt(dt), bounds.sigma_min, bounds.sigma_max
    ))

    # Two-moment corrected lambda (avoids truncation bias of threshold counting)
    lam1 = _lambda_from_moments(r1, dt, sigma1, bounds.lambda_min, bounds.lambda_max)
    lam2 = _lambda_from_moments(r2, dt, sigma2, bounds.lambda_min, bounds.lambda_max)

    # Drift: shrink the jump-mean correction by 0.25 to avoid amplifying eta bias.
    Ej1 = p1 * eta1_pos - (1.0 - p1) * eta1_neg
    Ej2 = p2 * eta2_pos - (1.0 - p2) * eta2_neg
    mu1 = float(np.clip(np.mean(r1) / dt - 0.25 * lam1 * Ej1, bounds.mu_min, bounds.mu_max))
    mu2 = float(np.clip(np.mean(r2) / dt - 0.25 * lam2 * Ej2, bounds.mu_min, bounds.mu_max))

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

# Ridge scales for lambda-eta traversal: scale lambda by c, eta by 1/sqrt(c).
# This preserves lambda*E[J^2] (jump variance) while exploring the identification ridge.
_RIDGE_SCALES: np.ndarray = np.array([0.1, 0.25, 0.5, 2.0, 4.0, 8.0])


def build_multistart_cloud(
    theta_nat: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> np.ndarray:
    """
    Build a (n_starts, 13) array of unconstrained starting points.

    Layout:
      [0]       unperturbed initializer point
      [1..R]    ridge-traversing starts (scale lambda by c, eta by 1/sqrt(c))
      [R+1..]   scenario-based random starts cycling through A–D

    The ridge starts systematically explore the lambda-eta identification
    valley along which lambda*E[J^2] is approximately constant, preventing
    the optimizer from being trapped near the initializer's lambda value.
    """
    starts = np.empty((n_starts, 13))
    starts[0] = nat_to_unc(theta_nat, bounds)

    eta2_max = bounds.eta_pos2_max
    mu1, s1, l1, p1, ep1, en1, mu2, s2, l2, p2, ep2, en2, rho = theta_nat

    # Ridge-traversing starts
    n_ridge = min(len(_RIDGE_SCALES), n_starts - 1)
    for i, c in enumerate(_RIDGE_SCALES[:n_ridge]):
        sq = float(np.sqrt(c))
        l1_r  = float(np.clip(l1  * c,   bounds.lambda_min,  bounds.lambda_max))
        l2_r  = float(np.clip(l2  * c,   bounds.lambda_min,  bounds.lambda_max))
        ep1_r = float(np.clip(ep1 / sq,  bounds.eta_pos1_min, bounds.eta_pos1_max))
        en1_r = float(np.clip(en1 / sq,  bounds.eta_neg_min,  bounds.eta_neg_max))
        ep2_r = float(np.clip(ep2 / sq,  bounds.eta_pos2_min, eta2_max))
        en2_r = float(np.clip(en2 / sq,  bounds.eta_neg_min,  bounds.eta_neg_max))
        starts[i + 1] = nat_to_unc(np.array([
            mu1, s1, l1_r, p1, ep1_r, en1_r,
            mu2, s2, l2_r, p2, ep2_r, en2_r, rho,
        ]), bounds)

    # Scenario-based random starts for the remainder
    for i in range(n_ridge + 1, n_starts):
        sc = _SCENARIOS[(i - n_ridge - 1) % len(_SCENARIOS)]
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
