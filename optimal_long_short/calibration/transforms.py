"""
Parameter bounds, constraints, and natural <-> unconstrained transformations.

Natural-space vector (13 elements):
  theta = [mu1, sigma1, lam1, p1, eta1_pos, eta1_neg,
           mu2, sigma2, lam2, p2, eta2_pos, eta2_neg, rho]

Unconstrained vector (13 elements):
  tau = [mu1,
         log(sigma1),  log(lam1),  logit(p1),  logit(eta1_pos),  log(eta1_neg),
         mu2,
         log(sigma2),  log(lam2),  logit(p2),  logit(eta2_pos / eta2_max),  log(eta2_neg),
         atanh(rho)]

eta2_pos uses a *scaled* sigmoid so that eta2_pos < eta2_max = (1-eps)/K
automatically, satisfying the k-tilted moment-admissibility condition K*eta2_pos < 1
from the paper.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from optimal_long_short.model_params import KouParams


# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------

@dataclass
class ParameterBounds:
    """
    Box constraints for the bivariate Kou model parameters.

    ``max_moment_order`` sets the admissibility constraint: the Laplace
    resolvent requires K*eta2_pos < 1 for moment orders k = 0, ..., K.
    The bound is enforced hard via the unconstrained parameterisation.
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
    eta_pos1_max: float = 0.99    # eta1_pos only needs < 1 (compensator)
    eta_pos2_min: float = 1e-5
    eta_neg_min: float = 1e-5
    eta_neg_max: float = 5.0
    rho_min: float = -0.995
    rho_max: float = 0.995
    max_moment_order: int = 4     # K: requires K*eta2_pos < 1
    moment_eps: float = 1e-4      # safety margin

    @property
    def eta_pos2_max(self) -> float:
        """Hard upper bound on eta2_pos: (1 - eps) / K."""
        return (1.0 - self.moment_eps) / self.max_moment_order

    def unc_bounds(self) -> list[tuple[float, float]]:
        """
        L-BFGS-B bounds in unconstrained tau space.
        Wide enough not to bind under normal conditions.
        """
        eta2m = self.eta_pos2_max
        # eta1_pos uses logit transform: tau = logit(ep1), ep1 = sigmoid(tau)
        logit_e1min = _logit(self.eta_pos1_min)
        logit_e1max = _logit(self.eta_pos1_max)
        # eta2_pos uses scaled logit: tau = logit(ep2 / eta2m), ep2 = eta2m * sigmoid(tau)
        e2lo = _logit(max(self.eta_pos2_min, 1e-12) / eta2m)
        e2hi = _logit(1.0 - self.moment_eps)  # logit((1-eps)) ≈ 9.2 for eps=1e-4
        return [
            (self.mu_min,  self.mu_max),
            (np.log(self.sigma_min), np.log(self.sigma_max)),
            (np.log(self.lambda_min), np.log(self.lambda_max)),
            (_logit(self.p_min), _logit(self.p_max)),
            (logit_e1min, logit_e1max),    # eta1_pos: logit(ep1)
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
    eta2m = bounds.eta_pos2_max
    ep2_sc = float(np.clip(ep2 / eta2m, 1e-10, 1.0 - 1e-10))
    return np.array([
        float(mu1),
        float(np.log(np.clip(s1,  bounds.sigma_min,  bounds.sigma_max))),
        float(np.log(np.clip(l1,  bounds.lambda_min, bounds.lambda_max))),
        _logit(float(np.clip(p1,  bounds.p_min,      bounds.p_max))),
        _logit(float(np.clip(ep1, bounds.eta_pos1_min, bounds.eta_pos1_max))),
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
    eta2m = bounds.eta_pos2_max
    return np.array([
        float(mu1),
        float(np.exp(a1)),
        float(np.exp(b1)),
        float(_sigmoid(c1)),
        float(_sigmoid(d1)),            # eta1_pos in (0, 1)
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
