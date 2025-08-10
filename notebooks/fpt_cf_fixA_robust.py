
# fpt_cf_fixA_robust.py
# First-hitting-time transform via Fix A (mode superposition with constant u's)
# - Robust weight fitting (ridge + conditioning)
# - Diverse mode selection
# - Easy to increase K

from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Optional

@dataclass
class Params:
    # Diffusion of h (already includes rho effect inside sigma_h)
    mu_h: float
    sigma_h: float

    # Hawkes baselines and decays
    nu_X: float
    nu_Y: float
    beta_X: float
    beta_Y: float

    # Hawkes excitation
    alpha_XX: float
    alpha_XY: float
    alpha_YX: float
    alpha_YY: float

    # Shifted-exponential jump sizes U = s + Exp(kappa)
    s_X: float
    kappa_X: float
    s_Y: float
    kappa_Y: float

def E_shifted_exp(eta: complex, s: float, kappa: float) -> complex:
    """Laplace transform E[e^{-eta (s + Exp(kappa))}] = exp(-eta s) * kappa/(kappa+eta)."""
    return np.exp(-eta * s) * (kappa / (kappa + eta))

def system_g(vec: np.ndarray, q: complex, p: Params) -> np.ndarray:
    """Returns (g_X, g_Y, g_0) for the mode (eta, uX, uY)."""
    eta, uX, uY = vec
    EX = E_shifted_exp(eta, p.s_X, p.kappa_X)
    EY = E_shifted_exp(eta, p.s_Y, p.kappa_Y)
    eX = np.exp(uX * p.alpha_XX + uY * p.alpha_YX) * EX
    eY = np.exp(uX * p.alpha_XY + uY * p.alpha_YY) * EY
    gX = -p.beta_X * uX + eX - 1.0
    gY = -p.beta_Y * uY + eY - 1.0
    g0 = -p.mu_h * eta + 0.5 * (p.sigma_h ** 2) * (eta ** 2) + p.beta_X * p.nu_X * uX + p.beta_Y * p.nu_Y * uY - q
    return np.array([gX, gY, g0], dtype=complex)

def jacobian_g(vec: np.ndarray, q: complex, p: Params) -> np.ndarray:
    """Jacobian of system_g w.r.t. (eta, uX, uY)."""
    eta, uX, uY = vec
    EX = E_shifted_exp(eta, p.s_X, p.kappa_X)
    EY = E_shifted_exp(eta, p.s_Y, p.kappa_Y)
    eX = np.exp(uX * p.alpha_XX + uY * p.alpha_YX) * EX
    eY = np.exp(uX * p.alpha_XY + uY * p.alpha_YY) * EY
    dEX_deta = EX * (-(p.s_X) - 1.0 / (p.kappa_X + eta))
    dEY_deta = EY * (-(p.s_Y) - 1.0 / (p.kappa_Y + eta))

    J = np.zeros((3, 3), dtype=complex)
    # gX
    J[0, 0] = dEX_deta * np.exp(uX * p.alpha_XX + uY * p.alpha_YX)  # d/deta
    J[0, 1] = -p.beta_X + eX * p.alpha_XX                           # d/duX
    J[0, 2] = eX * p.alpha_YX                                       # d/duY
    # gY
    J[1, 0] = dEY_deta * np.exp(uX * p.alpha_XY + uY * p.alpha_YY)  # d/deta
    J[1, 1] = eY * p.alpha_XY                                       # d/duX
    J[1, 2] = -p.beta_Y + eY * p.alpha_YY                           # d/duY
    # g0
    J[2, 0] = -p.mu_h + (p.sigma_h ** 2) * eta                       # d/deta
    J[2, 1] = p.beta_X * p.nu_X                                      # d/duX
    J[2, 2] = p.beta_Y * p.nu_Y                                      # d/duY
    return J

def newton_mode(vec0: np.ndarray, q: complex, p: Params, tol: float = 1e-12, maxit: int = 60) -> tuple[np.ndarray, bool]:
    """Damped Newton for complex root of system_g. Returns (vec, success)."""
    x = vec0.astype(complex)
    for _ in range(maxit):
        g = system_g(x, q, p)
        J = jacobian_g(x, q, p)
        try:
            dx = np.linalg.solve(J, -g)
        except np.linalg.LinAlgError:
            return x, False
        # backtracking
        alpha = 1.0
        base = norm(g)
        ok = False
        for _ in range(12):
            x_trial = x + alpha * dx
            if norm(system_g(x_trial, q, p)) < base:
                x = x_trial
                ok = True
                break
            alpha *= 0.5
        if not ok:
            x = x + dx  # last resort
        if norm(system_g(x, q, p)) < tol:
            return x, True
    return x, False

def dedup_modes(modes: List[np.ndarray], re_eta_min: float = 0.0, tol: float = 1e-6) -> List[np.ndarray]:
    """Remove near-duplicates and discard modes with Re(eta) <= threshold. Sort by Re(eta)."""
    kept: List[np.ndarray] = []
    for m in modes:
        if np.real(m[0]) <= re_eta_min:
            continue
        if all(norm(m - n) > tol for n in kept):
            kept.append(m)
    kept.sort(key=lambda v: (np.real(v[0]), np.imag(v[0])))
    return kept

def select_diverse_modes(modes: List[np.ndarray], K: int) -> List[np.ndarray]:
    """Greedy D-optimal-ish selection on features [1, Re(uX), Re(uY)] to improve span."""
    if len(modes) <= K:
        return modes
    U = np.stack([
        np.ones(len(modes)),
        np.real([m[1] for m in modes]),
        np.real([m[2] for m in modes])
    ], axis=1)
    chosen = []
    remaining = list(range(len(modes)))
    # start with smallest Re(eta)
    start = int(np.argmin([np.real(m[0]) for m in modes]))
    chosen.append(start)
    remaining.remove(start)
    # greedy add
    while len(chosen) < K and remaining:
        Xc = U[chosen]
        Gc = Xc.T @ Xc + 1e-12*np.eye(Xc.shape[1])
        Gc_inv = np.linalg.inv(Gc)
        best_idx, best_gain = None, -np.inf
        for i in remaining:
            v = U[i:i+1]
            gain = float(v @ Gc_inv @ v.T)
            if gain > best_gain:
                best_gain, best_idx = gain, i
        chosen.append(best_idx)
        remaining.remove(best_idx)
    return [modes[i] for i in chosen]

def initial_eta_guess_for_q(q: complex, p: Params) -> complex:
    """Quadratic root of -mu_h*eta + 0.5 sigma^2 eta^2 - q = 0 with positive Re(eta)."""
    a = 0.5 * p.sigma_h ** 2
    b = -p.mu_h
    c = -q
    disc = b*b - 4*a*c
    sq = np.sqrt(disc)
    eta1 = (-b + sq) / (2*a)
    eta2 = (-b - sq) / (2*a)
    return eta1 if np.real(eta1) > np.real(eta2) else eta2

def solve_modes_for_q(
    q: complex,
    p: Params,
    K: int = 4,
    prev_modes: Optional[List[np.ndarray]] = None,
    extra_eta_scales: tuple = (1.0, 2.0, 4.0, 8.0),
) -> List[np.ndarray]:
    """Find up to K modes (eta,uX,uY) for a given q, using continuation & seed diversity."""
    seeds: List[np.ndarray] = []
    if prev_modes:
        seeds.extend(prev_modes)
    eta0 = initial_eta_guess_for_q(q, p)
    for s in extra_eta_scales:
        seeds.append(np.array([s * eta0, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex))
    eps = 1e-6
    seeds.append(np.array([eta0 * (1 + 1j * eps), 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex))

    candidates: List[np.ndarray] = []
    for sd in seeds:
        sol, ok = newton_mode(sd, q, p)
        if ok:
            candidates.append(sol)

    modes = dedup_modes(candidates, re_eta_min=1e-12, tol=1e-6)
    if len(modes) < K:
        # try more jitters
        for k in range(12):
            jitter = (1 + 0.15j) ** (k+1)
            sd = np.array([eta0 * jitter, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
            sol, ok = newton_mode(sd, q, p)
            if ok:
                modes.append(sol)
                modes = dedup_modes(modes, re_eta_min=1e-12, tol=1e-6)
            if len(modes) >= K:
                break

    # select diverse K
    modes = select_diverse_modes(modes, K)
    return modes[:K]

def build_constraints_matrix(
    modes: List[np.ndarray],
    order: int = 2,
    collocation: Optional[List[Tuple[float, float]]] = None,
):
    """Build A, b so that A w ≈ b encodes boundary ≈ 1 at h=b."""
    K = len(modes)
    rows = []
    rhs = []

    uX = np.array([m[1] for m in modes])
    uY = np.array([m[2] for m in modes])

    # Constant
    rows.append(np.ones(K, dtype=complex)); rhs.append(1.0 + 0.0j)

    if order >= 1:
        rows.append(uX); rhs.append(0.0 + 0.0j)
        rows.append(uY); rhs.append(0.0 + 0.0j)

    if order >= 2:
        rows.append(uX*uX); rhs.append(0.0 + 0.0j)
        rows.append(uY*uY); rhs.append(0.0 + 0.0j)
        rows.append(uX*uY); rhs.append(0.0 + 0.0j)

    if collocation:
        for (lx, ly) in collocation:
            row = np.exp(uX * lx + uY * ly)
            rows.append(row)
            rhs.append(1.0 + 0.0j)

    A = np.vstack(rows)
    b = np.array(rhs, dtype=complex)
    return A, b

def fit_weights(
    modes: List[np.ndarray],
    order: int = 2,
    collocation: Optional[List[Tuple[float, float]]] = None,
    ridge: float = 1e-6,
    cond_warn: float = 1e8,
) -> np.ndarray:
    """Least-squares weights with optional Tikhonov regularization if ill-conditioned."""
    A, b = build_constraints_matrix(modes, order=order, collocation=collocation)
    # SVD condition number
    try:
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        cond = (s[0] / s[-1]) if s[-1] != 0 else np.inf
    except np.linalg.LinAlgError:
        cond = np.inf
    if cond > cond_warn:
        AtA = A.conj().T @ A + ridge * np.eye(A.shape[1], dtype=complex)
        Atb = A.conj().T @ b
        w = np.linalg.solve(AtA, Atb)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w

def boundary_error(modes: List[np.ndarray], w: np.ndarray, grid: List[Tuple[float, float]]) -> float:
    """Max abs error of Σ w e^{uX λX + uY λY} - 1 over a λ-grid."""
    uX = np.array([m[1] for m in modes])
    uY = np.array([m[2] for m in modes])
    errs = []
    for (lx, ly) in grid:
        val = np.sum(w * np.exp(uX * lx + uY * ly))
        errs.append(abs(val - 1.0))
    return float(np.max(errs))

def phi_q(
    q: complex,
    modes: List[np.ndarray],
    w: np.ndarray,
    h0: float,
    b: float,
    lX0: float,
    lY0: float,
) -> complex:
    """Compute φ(q) ≈ Σ w_k exp(-η_k(h0-b) + uX_k λX0 + uY_k λY0)."""
    eta = np.array([m[0] for m in modes])
    uX = np.array([m[1] for m in modes])
    uY = np.array([m[2] for m in modes])
    return np.sum(w * np.exp(-eta * (h0 - b) + uX * lX0 + uY * lY0))

def cf_tau_b(
    omegas: np.ndarray,
    p: Params,
    h0: float,
    b: float,
    lX0: float,
    lY0: float,
    K: int = 6,
    order: int = 2,
    collocation: Optional[List[Tuple[float, float]]] = None,
    lam_box_eval: Optional[List[Tuple[float, float]]] = None,
    ridge: float = 1e-6,
    cond_warn: float = 1e8,
):
    """
    Returns (psi, all_modes, all_weights)
      psi[j] = φ(q=-i ω_j)
    """
    psi = np.zeros_like(omegas, dtype=complex)
    prev_modes = None
    all_modes = []
    all_w = []
    for j, wfreq in enumerate(omegas):
        q = -1j * wfreq
        modes = solve_modes_for_q(q, p, K=K, prev_modes=prev_modes)
        wts = fit_weights(modes, order=order, collocation=collocation, ridge=ridge, cond_warn=cond_warn)
        psi[j] = phi_q(q, modes, wts, h0, b, lX0, lY0)
        prev_modes = modes
        all_modes.append(modes)
        all_w.append(wts)
    if lam_box_eval is not None and len(omegas) > 0:
        err = boundary_error(all_modes[-1], all_w[-1], lam_box_eval)
        print(f"[Boundary check @ last ω] max|boundary error| ≈ {err:.3e}")
    return psi, all_modes, all_w

def stationary_mean_intensities(p: Params) -> tuple[float, float]:
    """Compute the stationary mean (λ̄_X, λ̄_Y) = (I-Φ)^{-1} ν if ρ(Φ)<1, with Φ_ij = α_ij/β_i."""
    Phi = np.array([[p.alpha_XX/p.beta_X, p.alpha_XY/p.beta_X],
                    [p.alpha_YX/p.beta_Y, p.alpha_YY/p.beta_Y]], dtype=float)
    I = np.eye(2)
    nu = np.array([p.nu_X, p.nu_Y], dtype=float)
    lam_bar = np.linalg.solve(I - Phi, nu)
    return float(lam_bar[0]), float(lam_bar[1])
