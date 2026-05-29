"""
Empirical characteristic function and ECF objective for bivariate Kou
calibration.

The estimator minimises the normalized weighted squared distance:

    Q_N(theta) = [ sum_m w_m * |phi_hat_m - phi_model_m|^2 ] / sum_m w_m

Normalization by sum(w) makes the objective comparable across grids and
frequencies.  Non-finite model CF values are guarded: any NaN/Inf in the
model CF returns a large penalty (1e100).
"""
from __future__ import annotations

import numpy as np

from optimal_long_short.kou_model import BivariateKouModel
from .transforms import ParameterBounds, unc_to_params, unc_to_nat, _DEFAULT_BOUNDS


def empirical_cf(r1: np.ndarray, r2: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Empirical characteristic function at all frequency pairs in freqs.

        phi_hat(u, v) = (1/N) * sum_n exp(i*(u*r1_n + v*r2_n))

    Parameters
    ----------
    r1, r2 : (N,) log-return arrays.
    freqs  : (M, 2) frequency pairs.

    Returns
    -------
    phi_hat : (M,) complex array.
    """
    angles = freqs[:, 0:1] * r1[None, :] + freqs[:, 1:2] * r2[None, :]  # (M, N)
    return np.mean(np.exp(1j * angles), axis=1)


def model_cf(
    tau: np.ndarray,
    dt: float,
    freqs: np.ndarray,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> np.ndarray:
    """
    Model CF exp(dt * Psi(u, v)) evaluated in unconstrained parameter space.

    Returns an array filled with 1e10+0j on any numerical failure so the
    optimizer sees a large but finite penalty rather than a crash.
    """
    try:
        params = unc_to_params(tau, bounds)
    except Exception:
        return np.full(len(freqs), 1e10 + 0j)

    m = BivariateKouModel(params)
    out = np.empty(len(freqs), dtype=complex)
    for k, (u, v) in enumerate(freqs):
        out[k] = np.exp(dt * m.levy_khintchine(u, v))

    if not (np.all(np.isfinite(out.real)) and np.all(np.isfinite(out.imag))):
        return np.full(len(freqs), 1e10 + 0j)
    return out


def objective_unc(
    tau: np.ndarray,
    phi_hat: np.ndarray,
    dt: float,
    freqs: np.ndarray,
    weights: np.ndarray,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> float:
    """
    Normalized ECF objective in unconstrained parameter space.

        Q_N = sum_m w_m * |phi_hat_m - phi_model_m|^2  /  sum_m w_m
    """
    diff = phi_hat - model_cf(tau, dt, freqs, bounds)
    val = float(np.real(np.dot(weights, diff * np.conj(diff))))
    if not np.isfinite(val):
        return 1e100
    w_sum = float(np.sum(weights))
    return val / w_sum if w_sum > 0 else 1e100


def objective_unc_pot_anchored(
    tau: np.ndarray,
    phi_hat: np.ndarray,
    dt: float,
    freqs: np.ndarray,
    weights: np.ndarray,
    log_lam1_anch: float,
    log_lam2_anch: float,
    lam_anchor_weight: float,
    log_ep1_anch: float,
    log_en1_anch: float,
    log_ep2_anch: float,
    log_en2_anch: float,
    eta_anchor_weight: float,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> float:
    """
    ECF objective with soft log-scale penalties on lambda and eta from POT.

    Anchors lambda and the jump-size means near their POT estimates in
    log-scale, blocking the high-lambda / small-eta identification degeneracy
    while still allowing the ECF to move parameters away from the prior when
    the CF evidence is strong.

    Penalty = lam_anchor_weight * [(log lam1 - log_lam1_anch)^2
                                  + (log lam2 - log_lam2_anch)^2]
            + eta_anchor_weight * [(log ep1 - log_ep1_anch)^2
                                  + (log en1 - log_en1_anch)^2
                                  + (log ep2 - log_ep2_anch)^2
                                  + (log en2 - log_en2_anch)^2]
    """
    ecf = objective_unc(tau, phi_hat, dt, freqs, weights, bounds)
    try:
        theta = unc_to_nat(tau, bounds)
        log_lam1 = np.log(max(theta[2],  1e-12))
        log_lam2 = np.log(max(theta[8],  1e-12))
        log_ep1  = np.log(max(theta[4],  1e-12))
        log_en1  = np.log(max(theta[5],  1e-12))
        log_ep2  = np.log(max(theta[10], 1e-12))
        log_en2  = np.log(max(theta[11], 1e-12))
    except Exception:
        return ecf + 1e100
    lam_pen = lam_anchor_weight * (
        (log_lam1 - log_lam1_anch) ** 2 + (log_lam2 - log_lam2_anch) ** 2
    )
    eta_pen = eta_anchor_weight * (
        (log_ep1 - log_ep1_anch) ** 2 + (log_en1 - log_en1_anch) ** 2
        + (log_ep2 - log_ep2_anch) ** 2 + (log_en2 - log_en2_anch) ** 2
    )
    val = ecf + float(lam_pen) + float(eta_pen)
    return val if np.isfinite(val) else 1e100


def objective_by_group(
    tau: np.ndarray,
    phi_hat: np.ndarray,
    dt: float,
    freqs: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    bounds: ParameterBounds = _DEFAULT_BOUNDS,
) -> dict[str, float]:
    """
    Group-decomposed ECF objective for calibration diagnostics.

    Returns a dict mapping each group label to its normalized sub-objective,
    plus a "total" key for the full normalized objective.
    """
    phi_m = model_cf(tau, dt, freqs, bounds)
    sq = weights * np.abs(phi_hat - phi_m) ** 2
    w_total = float(np.sum(weights))

    result: dict[str, float] = {}
    for label in sorted(np.unique(groups)):
        mask = groups == label
        w_g = float(np.sum(weights[mask]))
        result[label] = float(np.sum(sq[mask])) / w_g if w_g > 0 else float("nan")

    result["total"] = float(np.sum(sq)) / w_total if w_total > 0 else float("nan")
    return result
