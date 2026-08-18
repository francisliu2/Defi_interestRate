"""
Parameter bounds, constraints, and natural <-> unconstrained transformations.

Natural-space vector (13 elements):
  theta = [mu1, sigma1, lam1, p1, eta1_pos, eta1_neg,
           mu2, sigma2, lam2, p2, eta2_pos, eta2_neg, rho]

Unconstrained vector (13 elements):
  tau = [mu1,
         log(sigma1),  log(lam1),  logit(p1),  logit(eta1_pos / eta1_max),  log(eta1_neg),
         mu2,
         log(sigma2),  log(lam2),  logit(p2),  logit(eta2_pos / eta2_max),  log(eta2_neg),
         atanh(rho)]

Both positive-jump means use scaled sigmoids so that
eta_i_pos < eta_i_max <= (1-eps)/K automatically, satisfying the two
K-th payoff-moment admissibility conditions from the paper.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from numbers import Integral

from optimal_long_short.model.model_params import KouParams
from optimal_long_short.model.drift_service import with_zero_expected_log_return


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

@dataclass
class ParameterBounds:
    """
    Box constraints for the bivariate Kou model parameters.

    ``max_moment_order`` sets the admissibility constraint: the killed-moment
    resolvent requires both K*eta1_pos < 1 and K*eta2_pos < 1 for moment
    orders k = 1, ..., K. The bounds are enforced hard through the
    unconstrained parameterisation.
    """
    mu_min: float = -5.0
    mu_max: float = 5.0
    sigma_min: float = 0.005
    sigma_max: float = 6.0
    lambda_min: float = 1e-4
    lambda_max: float = 500.0
    p_min: float = 0.01
    p_max: float = 0.99
    eta_pos1_min: float = 1e-5
    eta_pos1_max: float = 0.99    # additional user cap before the K-th moment cap
    eta_pos2_min: float = 1e-5
    eta_neg_min: float = 1e-5
    eta_neg_max: float = 5.0
    rho_min: float = -0.995
    rho_max: float = 0.995
    max_moment_order: int = 4     # K: requires K*eta_i_pos < 1 for i=1,2
    moment_eps: float = 1e-4      # safety margin

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_moment_order, bool)
            or not isinstance(self.max_moment_order, Integral)
            or self.max_moment_order < 1
        ):
            raise ValueError("max_moment_order must be a positive integer")
        if not 0.0 < self.moment_eps < 1.0:
            raise ValueError("moment_eps must lie in (0, 1)")
        ordered_bounds = (
            ("mu", self.mu_min, self.mu_max),
            ("sigma", self.sigma_min, self.sigma_max),
            ("lambda", self.lambda_min, self.lambda_max),
            ("p", self.p_min, self.p_max),
            ("eta_neg", self.eta_neg_min, self.eta_neg_max),
            ("rho", self.rho_min, self.rho_max),
        )
        for name, lower, upper in ordered_bounds:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name}_min must be finite and below {name}_max")
        if self.sigma_min <= 0.0 or self.lambda_min <= 0.0:
            raise ValueError("sigma_min and lambda_min must be positive")
        if self.eta_neg_min <= 0.0:
            raise ValueError("eta_neg_min must be positive")
        if not 0.0 < self.p_min < self.p_max < 1.0:
            raise ValueError("p bounds must lie strictly inside (0, 1)")
        if not -1.0 < self.rho_min < self.rho_max < 1.0:
            raise ValueError("rho bounds must lie strictly inside (-1, 1)")
        if not 0.0 < self.eta_pos1_min < self.eta_pos1_max:
            raise ValueError("eta_pos1 bounds must be positive and ordered")
        if self.eta_pos2_min <= 0.0:
            raise ValueError("eta_pos2_min must be positive")
        if self.eta_pos1_min >= self.eta_pos1_admissible_max:
            raise ValueError("eta_pos1_min exceeds the moment-admissible maximum")
        if self.eta_pos2_min >= self.eta_pos2_max:
            raise ValueError("eta_pos2_min exceeds the moment-admissible maximum")

    @property
    def eta_pos1_admissible_max(self) -> float:
        """Effective upper bound on eta1_pos for moments through order K."""
        return min(
            self.eta_pos1_max,
            (1.0 - self.moment_eps) / self.max_moment_order,
        )

    @property
    def eta_pos2_max(self) -> float:
        """Hard upper bound on eta2_pos: (1 - eps) / K."""
        return (1.0 - self.moment_eps) / self.max_moment_order

    def unc_bounds(self) -> list[tuple[float, float]]:
        """
        L-BFGS-B bounds in unconstrained tau space.
        Wide enough not to bind under normal conditions.
        """
        eta1m = self.eta_pos1_admissible_max
        eta2m = self.eta_pos2_max
        # eta_i_pos uses a scaled logit: tau = logit(ep_i / eta_i_max)
        e1lo = _logit(max(self.eta_pos1_min, 1e-12) / eta1m)
        e1hi = _logit(1.0 - self.moment_eps)
        # eta2_pos uses scaled logit: tau = logit(ep2 / eta2m), ep2 = eta2m * sigmoid(tau)
        e2lo = _logit(max(self.eta_pos2_min, 1e-12) / eta2m)
        e2hi = _logit(1.0 - self.moment_eps)  # logit((1-eps)) ≈ 9.2 for eps=1e-4
        return [
            (self.mu_min,  self.mu_max),
            (np.log(self.sigma_min), np.log(self.sigma_max)),
            (np.log(self.lambda_min), np.log(self.lambda_max)),
            (_logit(self.p_min), _logit(self.p_max)),
            (e1lo, e1hi),                  # eta1_pos: logit(ep1/eta1m)
            (np.log(self.eta_neg_min), np.log(self.eta_neg_max)),
            (self.mu_min,  self.mu_max),
            (np.log(self.sigma_min), np.log(self.sigma_max)),
            (np.log(self.lambda_min), np.log(self.lambda_max)),
            (_logit(self.p_min), _logit(self.p_max)),
            (e2lo, e2hi),                  # eta2_pos: logit(ep2/eta2m)
            (np.log(self.eta_neg_min), np.log(self.eta_neg_max)),
            (np.arctanh(self.rho_min), np.arctanh(self.rho_max)),
        ]


_DEFAULT_BOUNDS = ParameterBounds()


# ---------------------------------------------------------------------------
# Elementary transforms
# ---------------------------------------------------------------------------

def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p) - np.log1p(-p))


# ---------------------------------------------------------------------------
# Natural <-> unconstrained
# ---------------------------------------------------------------------------

def nat_to_unc(theta: np.ndarray,
               bounds: ParameterBounds = _DEFAULT_BOUNDS) -> np.ndarray:
    """Natural-space vector -> unconstrained vector."""
    mu1, s1, l1, p1, ep1, en1, mu2, s2, l2, p2, ep2, en2, rho = theta
    eta1m = bounds.eta_pos1_admissible_max
    eta2m = bounds.eta_pos2_max
    ep1_sc = float(np.clip(ep1 / eta1m, 1e-10, 1.0 - 1e-10))
    ep2_sc = float(np.clip(ep2 / eta2m, 1e-10, 1.0 - 1e-10))
    return np.array([
        float(mu1),
        float(np.log(np.clip(s1,  bounds.sigma_min,  bounds.sigma_max))),
        float(np.log(np.clip(l1,  bounds.lambda_min, bounds.lambda_max))),
        _logit(float(np.clip(p1,  bounds.p_min,      bounds.p_max))),
        _logit(ep1_sc),
        float(np.log(np.clip(en1, bounds.eta_neg_min, bounds.eta_neg_max))),
        float(mu2),
        float(np.log(np.clip(s2,  bounds.sigma_min,  bounds.sigma_max))),
        float(np.log(np.clip(l2,  bounds.lambda_min, bounds.lambda_max))),
        _logit(float(np.clip(p2,  bounds.p_min,      bounds.p_max))),
        _logit(ep2_sc),
        float(np.log(np.clip(en2, bounds.eta_neg_min, bounds.eta_neg_max))),
        float(np.arctanh(np.clip(rho, bounds.rho_min, bounds.rho_max))),
    ])


def unc_to_nat(tau: np.ndarray,
               bounds: ParameterBounds = _DEFAULT_BOUNDS) -> np.ndarray:
    """Unconstrained vector -> natural-space vector."""
    mu1, a1, b1, c1, d1, e1, mu2, a2, b2, c2, d2, e2, f = tau
    eta1m = bounds.eta_pos1_admissible_max
    eta2m = bounds.eta_pos2_max
    return np.array([
        float(mu1),
        float(np.exp(a1)),
        float(np.exp(b1)),
        float(_sigmoid(c1)),
        eta1m * float(_sigmoid(d1)),    # eta1_pos in (0, eta1_max) < 1/K
        float(np.exp(e1)),
        float(mu2),
        float(np.exp(a2)),
        float(np.exp(b2)),
        float(_sigmoid(c2)),
        eta2m * float(_sigmoid(d2)),    # eta2_pos in (0, eta2_max) < 1/K
        float(np.exp(e2)),
        float(np.tanh(f)),
    ])


# ---------------------------------------------------------------------------
# KouParams <-> flat vector
# ---------------------------------------------------------------------------

def theta_to_params(theta: np.ndarray) -> KouParams:
    """Natural-space vector -> KouParams."""
    return KouParams(
        mu1=theta[0],   sigma1=theta[1],  lam1=theta[2],
        p1=theta[3],    eta1_pos=theta[4], eta1_neg=theta[5],
        mu2=theta[6],   sigma2=theta[7],  lam2=theta[8],
        p2=theta[9],    eta2_pos=theta[10], eta2_neg=theta[11],
        rho=theta[12],
    )


def params_to_theta(p: KouParams) -> np.ndarray:
    """KouParams -> natural-space flat vector (13 elements)."""
    return np.array([
        p.mu1, p.sigma1, p.lam1, p.p1, p.eta1_pos, p.eta1_neg,
        p.mu2, p.sigma2, p.lam2, p.p2, p.eta2_pos, p.eta2_neg,
        p.rho,
    ])


def unc_to_params(tau: np.ndarray,
                  bounds: ParameterBounds = _DEFAULT_BOUNDS) -> KouParams:
    """Unconstrained vector -> KouParams."""
    return theta_to_params(unc_to_nat(tau, bounds))


# The two free drift coordinates (indices 0 and 6) are omitted when the ECF
# estimator is used only for residual-law shape. The corresponding price-growth
# exponents are reconstructed at every objective evaluation so that each
# residual marginal has zero expected log-return drift.
SHAPE_UNC_INDICES = np.array([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=int)


def params_to_shape_unc(
    params: KouParams,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> np.ndarray:
    """Kou parameters -> 11-dimensional unconstrained shape vector."""
    full = nat_to_unc(params_to_theta(params), bounds)
    return full[SHAPE_UNC_INDICES]


def shape_unc_to_params(
    tau_shape: np.ndarray,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> KouParams:
    """11-dimensional shape vector -> zero-log-mean residual parameters."""
    values = np.asarray(tau_shape, dtype=float)
    if values.shape != (len(SHAPE_UNC_INDICES),):
        raise ValueError(
            f"tau_shape must have length {len(SHAPE_UNC_INDICES)}, got {values.shape}"
        )
    full = np.zeros(13, dtype=float)
    full[SHAPE_UNC_INDICES] = values
    return with_zero_expected_log_return(unc_to_params(full, bounds))


def shape_unc_bounds(bounds: ParameterBounds = _DEFAULT_BOUNDS) -> list[tuple[float, float]]:
    """L-BFGS-B bounds for the 11-dimensional residual-shape vector."""
    full_bounds = bounds.unc_bounds()
    return [full_bounds[index] for index in SHAPE_UNC_INDICES]
