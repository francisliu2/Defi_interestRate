"""
Singleton-style empirical characteristic function (ECF) calibration
for the bivariate Kou double-exponential jump-diffusion model.

Estimates theta by minimising the weighted squared distance between
the empirical CF of observed bivariate log-returns and the model-implied
CF exp(dt * Psi_theta(u, v)):

    Q_N(theta) = sum_{(u,v) in G} w(u,v) * |phi_hat_N(u,v) - exp(dt*Psi_theta(u,v))|^2

The frequency grid G = G1 ∪ G2 ∪ GZ consists of three groups:
  G1 (marginal):  (u, 0) and (0, v)  — identifies each asset's marginal Kou law
  G2 (joint):     (u, v) grid        — identifies Brownian correlation rho
  GZ (spread):    (s, -s)            — identifies the spread Z = X1 - X2 that
                                       governs DeFi liquidation

Constrained parameters are handled via an unconstrained reparameterisation:
  sigma_i  = exp(a_i)
  lambda_i = exp(b_i)
  p_i      = sigmoid(c_i)
  eta_i+   = sigmoid(d_i)      # guarantees (0, 1)
  eta_i-   = exp(e_i)
  rho      = tanh(f)

The initialiser uses a robust MAD-based threshold filter following the logic
of Mancini (2009) to seed a multi-start cloud around the filtered values.

References:
  Singleton (2001), J. Econometrics 102(1):111-141.
  Mancini (2009), Scand. J. Statist. 36(2):270-296.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from optimal_long_short.kou_model import BivariateKouModel
from optimal_long_short.model_params import KouParams


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------

@dataclass
class CalibrationGrid:
    """
    Three-group frequency grid for ECF calibration.

    G1 (marginal):  (u, 0) and (0, v) for u, v in marginal_freqs
    G2 (joint):     (u, v) for all u in joint_freqs, v in joint_freqs
    GZ (spread):    (s, -s) for s in spread_freqs

    The spread direction directly identifies the characteristic function of
    Z = X1 - X2, the log-ratio process governing DeFi liquidation.
    Weights use Gaussian decay: w(u,v) = exp(-a*(u^2+v^2)).
    """
    marginal_freqs: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    )
    joint_freqs: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 1.0, 2.0])
    )
    spread_freqs: np.ndarray = field(
        default_factory=lambda: np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    )
    weight_decay: float = 1.0  # a in w(u,v) = exp(-a*(u^2+v^2))

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (freqs, weights): freqs is (M, 2), weights is (M,)."""
        pairs: list[tuple[float, float]] = []
        for u in self.marginal_freqs:
            pairs.append((float(u), 0.0))
        for v in self.marginal_freqs:
            pairs.append((0.0, float(v)))
        for u in self.joint_freqs:
            for v in self.joint_freqs:
                pairs.append((float(u), float(v)))
        for s in self.spread_freqs:
            pairs.append((float(s), -float(s)))
        freqs = np.array(pairs)  # (M, 2)
        weights = np.exp(-self.weight_decay * (freqs[:, 0] ** 2 + freqs[:, 1] ** 2))
        return freqs, weights


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ECFCalibrationResult:
    """Result of ECF/GMM calibration."""
    params: KouParams
    objective: float
    success: bool
    message: str
    n_iter: int


# ---------------------------------------------------------------------------
# Natural <-> unconstrained reparameterisation
# ---------------------------------------------------------------------------
# Natural (constrained):
#   theta = [mu1, sigma1, lam1, p1, eta1_pos, eta1_neg,
#            mu2, sigma2, lam2, p2, eta2_pos, eta2_neg, rho]
#
# Unconstrained:
#   tau = [mu1, log(sigma1), log(lam1), logit(p1), logit(eta1_pos), log(eta1_neg),
#          mu2, log(sigma2), log(lam2), logit(p2), logit(eta2_pos), log(eta2_neg), atanh(rho)]
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return float(np.log(p) - np.log1p(-p))


def _nat_to_unc(theta: np.ndarray) -> np.ndarray:
    """Natural -> unconstrained transformation."""
    mu1, s1, l1, p1, ep1, en1, mu2, s2, l2, p2, ep2, en2, rho = theta
    return np.array([
        mu1,
        np.log(float(np.clip(s1,  1e-8, None))),
        np.log(float(np.clip(l1,  1e-8, None))),
        _logit(p1),
        _logit(float(np.clip(ep1, 1e-8, 1.0 - 1e-8))),
        np.log(float(np.clip(en1, 1e-8, None))),
        mu2,
        np.log(float(np.clip(s2,  1e-8, None))),
        np.log(float(np.clip(l2,  1e-8, None))),
        _logit(p2),
        _logit(float(np.clip(ep2, 1e-8, 1.0 - 1e-8))),
        np.log(float(np.clip(en2, 1e-8, None))),
        float(np.arctanh(np.clip(rho, -1.0 + 1e-8, 1.0 - 1e-8))),
    ])


def _unc_to_nat(tau: np.ndarray) -> np.ndarray:
    """Unconstrained -> natural transformation."""
    mu1, a1, b1, c1, d1, e1, mu2, a2, b2, c2, d2, e2, f = tau
    return np.array([
        mu1,
        float(np.exp(a1)),
        float(np.exp(b1)),
        float(_sigmoid(c1)),
        float(_sigmoid(d1)),
        float(np.exp(e1)),
        mu2,
        float(np.exp(a2)),
        float(np.exp(b2)),
        float(_sigmoid(c2)),
        float(_sigmoid(d2)),
        float(np.exp(e2)),
        float(np.tanh(f)),
    ])


def _unc_to_params(tau: np.ndarray) -> KouParams:
    return _theta_to_params(_unc_to_nat(tau))


def _theta_to_params(theta: np.ndarray) -> KouParams:
    """Natural-space vector -> KouParams."""
    return KouParams(
        mu1=theta[0],  sigma1=theta[1],  lam1=theta[2],
        p1=theta[3],   eta1_pos=theta[4], eta1_neg=theta[5],
        mu2=theta[6],  sigma2=theta[7],  lam2=theta[8],
        p2=theta[9],   eta2_pos=theta[10], eta2_neg=theta[11],
        rho=theta[12],
    )


def params_to_theta(p: KouParams) -> np.ndarray:
    """Pack KouParams into the flat 13-element natural-space vector."""
    return np.array([
        p.mu1, p.sigma1, p.lam1, p.p1, p.eta1_pos, p.eta1_neg,
        p.mu2, p.sigma2, p.lam2, p.p2, p.eta2_pos, p.eta2_neg,
        p.rho,
    ])


# ---------------------------------------------------------------------------
# Empirical CF and ECF objective
# ---------------------------------------------------------------------------

def empirical_cf(r1: np.ndarray, r2: np.ndarray,
                 freqs: np.ndarray) -> np.ndarray:
    """
    Empirical characteristic function at all frequency pairs in ``freqs``.

    phi_hat(u, v) = (1/N) * sum_n exp(i*(u*r1_n + v*r2_n))

    Parameters
    ----------
    r1, r2 : (N,) arrays of log-returns.
    freqs : (M, 2) array of frequency pairs.

    Returns
    -------
    phi_hat : (M,) complex array.
    """
    angles = freqs[:, 0:1] * r1[None, :] + freqs[:, 1:2] * r2[None, :]  # (M, N)
    return np.mean(np.exp(1j * angles), axis=1)  # (M,)


def _model_cf_unc(tau: np.ndarray, dt: float, freqs: np.ndarray) -> np.ndarray:
    """Model CF exp(dt * Psi(u,v)) evaluated at unconstrained params tau."""
    try:
        params = _unc_to_params(tau)
    except Exception:
        return np.full(len(freqs), 1e10 + 0j)
    model = BivariateKouModel(params)
    out = np.empty(len(freqs), dtype=complex)
    for m, (u, v) in enumerate(freqs):
        out[m] = np.exp(dt * model.levy_khintchine(u, v))
    return out


def _objective_unc(tau: np.ndarray, phi_hat: np.ndarray, dt: float,
                   freqs: np.ndarray, weights: np.ndarray) -> float:
    """
    Q_N in unconstrained parameter space:
    Q_N(tau) = sum_{(u,v)} w(u,v) * |phi_hat(u,v) - exp(dt*Psi_tau(u,v))|^2
    """
    diff = phi_hat - _model_cf_unc(tau, dt, freqs)
    return float(np.real(np.dot(weights, diff * np.conj(diff))))


# ---------------------------------------------------------------------------
# Robust MAD-based threshold initialiser (Mancini 2009 logic)
# ---------------------------------------------------------------------------

def _initialize(r1: np.ndarray, r2: np.ndarray, dt: float,
                threshold: float = 4.0) -> np.ndarray:
    """
    Threshold-informed moment initialiser for the bivariate Kou model.

    Uses a robust MAD scale (Mancini 2009) to classify returns into
    candidate continuous (diffusion) and jump observations, then estimates
    each parameter group from the appropriate subset.

    Parameters
    ----------
    r1, r2 : (N,) log-return arrays.
    dt : annualised time step.
    threshold : multiplier c; |r - median| > c * MAD_scale flags a jump.

    Returns
    -------
    theta_nat : (13,) natural-scale parameter vector.
    """
    N = len(r1)

    # Step 1: robust center and scale via MAD
    m1, m2 = np.median(r1), np.median(r2)
    s1 = 1.4826 * np.median(np.abs(r1 - m1))
    s2 = 1.4826 * np.median(np.abs(r2 - m2))
    # fallback if data is very uniform
    s1 = max(s1, 1e-8)
    s2 = max(s2, 1e-8)

    # Step 2: detect candidate jump observations
    mask_j1 = np.abs(r1 - m1) > threshold * s1
    mask_j2 = np.abs(r2 - m2) > threshold * s2
    mask_c1, mask_c2 = ~mask_j1, ~mask_j2

    # Step 3: jump intensity from jump-day fraction
    n_j1, n_j2 = mask_j1.sum(), mask_j2.sum()
    lam1 = float(np.clip(n_j1 / (N * dt), 0.5, 99.0))
    lam2 = float(np.clip(n_j2 / (N * dt), 0.5, 99.0))

    # Step 4: upward jump probability
    j1_pos = mask_j1 & (r1 - m1 > 0)
    j1_neg = mask_j1 & (r1 - m1 < 0)
    j2_pos = mask_j2 & (r2 - m2 > 0)
    j2_neg = mask_j2 & (r2 - m2 < 0)
    p1 = float(np.clip(j1_pos.sum() / max(n_j1, 1), 0.05, 0.95))
    p2 = float(np.clip(j2_pos.sum() / max(n_j2, 1), 0.05, 0.95))

    # Step 5: jump-size means from candidate jumps
    def _mean_or(arr: np.ndarray, mask: np.ndarray, default: float) -> float:
        return float(np.mean(arr[mask])) if mask.any() else default

    eta1_pos = float(np.clip(_mean_or(r1 - m1,         j1_pos, 0.05), 1e-4, 0.95))
    eta1_neg = float(np.clip(_mean_or(np.abs(r1 - m1), j1_neg, 0.05), 1e-4, 9.0))
    eta2_pos = float(np.clip(_mean_or(r2 - m2,         j2_pos, 0.05), 1e-4, 0.95))
    eta2_neg = float(np.clip(_mean_or(np.abs(r2 - m2), j2_neg, 0.05), 1e-4, 9.0))

    # Step 6: diffusion volatility from non-jump returns
    def _std_or(arr: np.ndarray, mask: np.ndarray, fallback: float) -> float:
        return float(np.std(arr[mask])) if mask.sum() > 1 else fallback

    sigma1 = float(np.clip(_std_or(r1, mask_c1, s1) / np.sqrt(dt), 0.01, 4.9))
    sigma2 = float(np.clip(_std_or(r2, mask_c2, s2) / np.sqrt(dt), 0.01, 4.9))

    # Step 7: initialize log-price drift muX, then convert to the saved
    # price-growth convention mu = muX + 0.5*sigma^2 + lambda*chi.
    Ej1 = p1 * eta1_pos - (1.0 - p1) * eta1_neg
    Ej2 = p2 * eta2_pos - (1.0 - p2) * eta2_neg
    chi1 = p1 / (1.0 - eta1_pos) + (1.0 - p1) / (1.0 + eta1_neg) - 1.0
    chi2 = p2 / (1.0 - eta2_pos) + (1.0 - p2) / (1.0 + eta2_neg) - 1.0
    muX1 = np.mean(r1) / dt - lam1 * Ej1
    muX2 = np.mean(r2) / dt - lam2 * Ej2
    mu1 = float(np.clip(muX1 + 0.5 * sigma1**2 + lam1 * chi1, -2.5, 2.5))
    mu2 = float(np.clip(muX2 + 0.5 * sigma2**2 + lam2 * chi2, -2.5, 2.5))

    # Step 8: Brownian correlation from simultaneous non-jump observations
    mask_c12 = mask_c1 & mask_c2
    if mask_c12.sum() > 2:
        rho = float(np.clip(np.corrcoef(r1[mask_c12], r2[mask_c12])[0, 1], -0.99, 0.99))
    else:
        rho = float(np.clip(np.corrcoef(r1, r2)[0, 1], -0.99, 0.99))

    return np.array([mu1, sigma1, lam1, p1, eta1_pos, eta1_neg,
                     mu2, sigma2, lam2, p2, eta2_pos, eta2_neg, rho])


# ---------------------------------------------------------------------------
# Multi-start cloud
# ---------------------------------------------------------------------------

def _build_multistart_cloud(theta_nat: np.ndarray, n_starts: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Build a (n_starts, 13) array of unconstrained starting points by
    randomly perturbing theta_nat.

    Scale parameters (sigma, lam, eta) are perturbed multiplicatively on
    the log scale; probability parameters (p, rho) additively.
    The first row is the unperturbed point.
    """
    starts = np.empty((n_starts, 13))
    tau0 = _nat_to_unc(theta_nat)
    starts[0] = tau0

    for i in range(1, n_starts):
        mu1, s1, l1, p1, ep1, en1, mu2, s2, l2, p2, ep2, en2, rho = theta_nat

        # multiplicative perturbation on log scale for positive params
        fac = rng.uniform(-0.4, 0.4, size=8)
        s1_p  = float(np.clip(s1  * np.exp(fac[0]), 0.01, 4.9))
        l1_p  = float(np.clip(l1  * np.exp(fac[1]), 0.1,  99.0))
        ep1_p = float(np.clip(ep1 * np.exp(fac[2]), 1e-4, 0.95))
        en1_p = float(np.clip(en1 * np.exp(fac[3]), 1e-4, 9.0))
        s2_p  = float(np.clip(s2  * np.exp(fac[4]), 0.01, 4.9))
        l2_p  = float(np.clip(l2  * np.exp(fac[5]), 0.1,  99.0))
        ep2_p = float(np.clip(ep2 * np.exp(fac[6]), 1e-4, 0.95))
        en2_p = float(np.clip(en2 * np.exp(fac[7]), 1e-4, 9.0))

        # additive for probabilities and correlation
        p1_p  = float(np.clip(p1  + rng.uniform(-0.15, 0.15), 0.05, 0.95))
        p2_p  = float(np.clip(p2  + rng.uniform(-0.15, 0.15), 0.05, 0.95))
        rho_p = float(np.clip(rho + rng.uniform(-0.15, 0.15), -0.99, 0.99))

        # small drift perturbation
        mu1_p = float(np.clip(mu1 + rng.uniform(-0.1, 0.1), -2.5, 2.5))
        mu2_p = float(np.clip(mu2 + rng.uniform(-0.1, 0.1), -2.5, 2.5))

        starts[i] = _nat_to_unc(np.array([
            mu1_p, s1_p, l1_p, p1_p, ep1_p, en1_p,
            mu2_p, s2_p, l2_p, p2_p, ep2_p, en2_p, rho_p,
        ]))

    return starts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calibrate_ecf(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    grid: CalibrationGrid | None = None,
    theta0: np.ndarray | None = None,
    n_starts: int = 20,
    threshold: float = 4.0,
    seed: int | None = None,
) -> ECFCalibrationResult:
    """
    Estimate bivariate Kou parameters by minimising the ECF distance.

    The estimator matches the empirical characteristic function of (r1, r2)
    to exp(dt * Psi_theta(u, v)) over the three-group frequency grid
    G = G1 ∪ G2 ∪ GZ using a transformed-parameter L-BFGS-B optimizer run
    from a multi-start cloud seeded by a robust threshold initialiser.

    Parameters
    ----------
    r1, r2 : 1-D arrays of observed bivariate log-returns.
    dt : Annualised time step (e.g. 1/252 for daily, 1/(365*24) for hourly).
    grid : Frequency grid; uses CalibrationGrid defaults if None.
    theta0 : 13-element natural-space starting vector; auto-initialised if None.
    n_starts : Number of starting points in the multi-start cloud.
    threshold : MAD multiplier c for jump detection (default 4.0).
    seed : Random seed for the perturbation cloud.

    Returns
    -------
    ECFCalibrationResult with the best-objective KouParams.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.ndim != 1 or r1.shape != r2.shape:
        raise ValueError("r1 and r2 must be 1-D arrays of equal length.")

    if grid is None:
        grid = CalibrationGrid()
    freqs, weights = grid.build()
    phi_hat = empirical_cf(r1, r2, freqs)

    theta_nat0 = theta0 if theta0 is not None else _initialize(r1, r2, dt, threshold)
    rng = np.random.default_rng(seed)
    starts_unc = _build_multistart_cloud(theta_nat0, n_starts, rng)

    best_fun = np.inf
    best_result: OptimizeResult | None = None

    for tau0 in starts_unc:
        res = minimize(
            _objective_unc,
            x0=tau0,
            args=(phi_hat, dt, freqs, weights),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-9},
        )
        if res.fun < best_fun:
            best_fun = res.fun
            best_result = res

    return ECFCalibrationResult(
        params=_unc_to_params(best_result.x),
        objective=float(best_result.fun),
        success=best_result.success,
        message=best_result.message,
        n_iter=best_result.nit,
    )
