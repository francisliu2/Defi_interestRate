
# fpt_cf_fixB_grid_robust.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Params:
    mu_h: float
    sigma_h: float
    nu_X: float
    nu_Y: float
    beta_X: float
    beta_Y: float
    alpha_XX: float
    alpha_XY: float
    alpha_YX: float
    alpha_YY: float
    s_X: float
    kappa_X: float
    s_Y: float
    kappa_Y: float

def E_shifted_exp(eta: complex, s: float, kappa: float) -> complex:
    return np.exp(-eta * s) * (kappa / (kappa + eta))

def stationary_mean_intensities(p: Params) -> Tuple[float, float]:
    Phi = np.array([[p.alpha_XX/p.beta_X, p.alpha_XY/p.beta_X],
                    [p.alpha_YX/p.beta_Y, p.alpha_YY/p.beta_Y]], dtype=float)
    I = np.eye(2)
    nu = np.array([p.nu_X, p.nu_Y], dtype=float)
    lam_bar = np.linalg.solve(I - Phi, nu)
    return float(lam_bar[0]), float(lam_bar[1])

# ---------------- Grid utilities ----------------

def build_lambda_grid_auto(
    p: Params,
    lX0: float,
    lY0: float,
    colloc_points: Optional[List[Tuple[float,float]]],
    safety: float = 3.5,
    Mx: int = 65,
    My: int = 65,
):
    """Auto-size grid to cover state, collocation, and one alpha shift outward."""
    lam_bar = stationary_mean_intensities(p)
    pts = [(0.0,0.0), (lX0, lY0), (lam_bar[0], lam_bar[1])]
    if colloc_points:
        pts += list(colloc_points)
    # include one-step shifts
    alpha_pts = [
        (x + p.alpha_XX, y + p.alpha_YX) for x,y in pts
    ] + [
        (x + p.alpha_XY, y + p.alpha_YY) for x,y in pts
    ]
    pts += alpha_pts
    Lx_max = max([x for x,_ in pts]) * 1.05 + 1e-6
    Ly_max = max([y for _,y in pts]) * 1.05 + 1e-6
    # ensure at least safety * mean intensity
    Lx_max = max(Lx_max, safety * lam_bar[0], p.alpha_XX + p.alpha_XY + 0.5)
    Ly_max = max(Ly_max, safety * lam_bar[1], p.alpha_YX + p.alpha_YY + 0.5)

    lx = np.linspace(0.0, Lx_max, Mx)
    ly = np.linspace(0.0, Ly_max, My)
    dx = lx[1] - lx[0]
    dy = ly[1] - ly[0]
    return lx, ly, dx, dy

def bilinear_interp(field: np.ndarray, lx: np.ndarray, ly: np.ndarray, x: float, y: float) -> complex:
    # clamp into the grid (grid is sized generously)
    x = max(lx[0], min(lx[-1], x))
    y = max(ly[0], min(ly[-1], y))
    i = np.searchsorted(lx, x) - 1
    j = np.searchsorted(ly, y) - 1
    i = max(0, min(i, len(lx)-2))
    j = max(0, min(j, len(ly)-2))
    x0, x1 = lx[i], lx[i+1]
    y0, y1 = ly[j], ly[j+1]
    tx = 0.0 if x1==x0 else (x - x0)/(x1 - x0)
    ty = 0.0 if y1==y0 else (y - y0)/(y1 - y0)
    f00 = field[i, j]
    f10 = field[i+1, j]
    f01 = field[i, j+1]
    f11 = field[i+1, j+1]
    return (1-tx)*(1-ty)*f00 + tx*(1-ty)*f10 + (1-tx)*ty*f01 + tx*ty*f11

# -------------- Operator assembly --------------

def assemble_operator_matrix(
    eta: complex,
    q: complex,
    p: Params,
    lx: np.ndarray,
    ly: np.ndarray,
    dx: float,
    dy: float,
    upwind_eps: float = 1e-12
):
    Mx, My = len(lx), len(ly)
    N = Mx * My
    A = np.zeros((N, N), dtype=complex)

    EX = E_shifted_exp(eta, p.s_X, p.kappa_X)
    EY = E_shifted_exp(eta, p.s_Y, p.kappa_Y)

    c0 = (p.mu_h * eta + 0.5 * (p.sigma_h**2) * (eta**2) - q)

    def idx(i, j): return i*My + j

    for i, lamx in enumerate(lx):
        for j, lamy in enumerate(ly):
            row = idx(i, j)
            A[row, row] += c0

            # drift in lambda_X
            vX = p.beta_X * (p.nu_X - lamx)
            if abs(vX) > upwind_eps:
                if vX > 0:
                    if i > 0:
                        A[row, row] += vX / dx
                        A[row, idx(i-1, j)] += -vX / dx
                else:
                    if i < Mx-1:
                        A[row, idx(i+1, j)] += -vX / dx
                        A[row, row] += vX / dx

            # drift in lambda_Y
            vY = p.beta_Y * (p.nu_Y - lamy)
            if abs(vY) > upwind_eps:
                if vY > 0:
                    if j > 0:
                        A[row, row] += vY / dy
                        A[row, idx(i, j-1)] += -vY / dy
                else:
                    if j < My-1:
                        A[row, idx(i, j+1)] += -vY / dy
                        A[row, row] += vY / dy

            # Hawkes shift terms
            A[row, row] += -lamx - lamy

            # add bilinear weights for shifts
            def add_shift(lx_shift, ly_shift, weight):
                # clamp in-grid
                x = min(max(lx[0], lx_shift), lx[-1])
                y = min(max(ly[0], ly_shift), ly[-1])
                ii = np.searchsorted(lx, x) - 1
                jj = np.searchsorted(ly, y) - 1
                ii = max(0, min(ii, Mx-2))
                jj = max(0, min(jj, My-2))
                x0, x1 = lx[ii], lx[ii+1]
                y0, y1 = ly[jj], ly[jj+1]
                tx = 0.0 if x1==x0 else (x - x0)/(x1 - x0)
                ty = 0.0 if y1==y0 else (y - y0)/(y1 - y0)

                A[row, idx(ii,   jj  )] += weight * (1-tx)*(1-ty)
                A[row, idx(ii+1, jj  )] += weight * (tx)*(1-ty)
                A[row, idx(ii,   jj+1)] += weight * (1-tx)*(ty)
                A[row, idx(ii+1, jj+1)] += weight * (tx)*(ty)

            add_shift(lamx + p.alpha_XX, lamy + p.alpha_YX, lamx * EX)
            add_shift(lamx + p.alpha_XY, lamy + p.alpha_YY, lamy * EY)

    return A

def solve_g_for_eta(
    eta: complex,
    q: complex,
    p: Params,
    lx: np.ndarray,
    ly: np.ndarray,
    dx: float,
    dy: float,
    ref_idx: Tuple[int, int],
) -> np.ndarray:
    """Solve A g = 0 with normalization g[ref_idx] = 1."""
    A = assemble_operator_matrix(eta, q, p, lx, ly, dx, dy)
    Mx, My = len(lx), len(ly)
    N = Mx * My
    def idx(i,j): return i*My + j

    b = np.zeros(N, dtype=complex)
    ri = idx(*ref_idx)
    A[ri, :] = 0.0
    A[ri, ri] = 1.0 + 0.0j
    b[ri] = 1.0 + 0.0j

    g = np.linalg.solve(A, b)
    return g.reshape((Mx, My))

# ---------------- Robust η selection & ref index ----------------

def diffusion_root_eta(q: complex, mu_h: float, sigma_h: float) -> complex:
    a = 0.5 * sigma_h**2
    b = -mu_h
    c = -q
    disc = b*b - 4*a*c
    sq = np.sqrt(disc)
    e1 = (-b + sq) / (2*a)
    e2 = (-b - sq) / (2*a)
    return e1 if np.real(e1) > np.real(e2) else e2

def choose_etas_robust(q: complex, p: Params, K: int) -> List[complex]:
    """
    Choose K etas: start at diffusion root, spread geometrically,
    add small imaginary jitters to avoid singularities.
    """
    base = diffusion_root_eta(q, p.mu_h, p.sigma_h)
    mags = np.geomspace(0.6, 2.2, K)
    etas = []
    for m in mags:
        et = base * m
        # add alternating small imag parts
        jit = 1e-3j if (len(etas) % 2 == 0) else -1e-3j
        etas.append(et + jit)
    return etas

def pick_ref_index(lx: np.ndarray, ly: np.ndarray, target: Tuple[float,float]) -> Tuple[int,int]:
    """Pick ref grid index nearest to target λ (typically (0,0) or state λ)."""
    tx, ty = target
    i = int(np.argmin(np.abs(lx - tx)))
    j = int(np.argmin(np.abs(ly - ty)))
    return (i, j)

# ---------------- φ(q) and Talbot inversion ----------------

def phi_q_fixB(
    q: complex,
    p: Params,
    h0: float,
    b: float,
    lX0: float,
    lY0: float,
    K: int = 6,
    colloc_points: Optional[List[Tuple[float,float]]] = None,
    grid_M: int = 65,
    safety: float = 3.5,
) -> complex:
    # pick collocation if not given
    if colloc_points is None:
        lam_bar = stationary_mean_intensities(p)
        colloc_points = [
            (0.0, 0.0),
            lam_bar,
            (0.5*lam_bar[0], 0.5*lam_bar[1]),
            (2.0*lam_bar[0], 2.0*lam_bar[1]),
            (lam_bar[0], 0.0),
            (0.0, lam_bar[1]),
        ]
    # auto grid sizing
    lx, ly, dx, dy = build_lambda_grid_auto(p, lX0, lY0, colloc_points, safety=safety, Mx=grid_M, My=grid_M)
    # pick normalization near (0,0) to stabilize weights
    ref_idx = pick_ref_index(lx, ly, (0.0, 0.0))

    # choose K etas
    etas = choose_etas_robust(q, p, K=K)

    # solve g for each eta
    Gmats = []
    for eta in etas:
        G = solve_g_for_eta(eta, q, p, lx, ly, dx, dy, ref_idx=ref_idx)
        Gmats.append(G)

    # Fit weights: at collocation points, sum_k w_k g_k(λ) = 1
    C = np.zeros((len(colloc_points), K), dtype=complex)
    d = np.ones(len(colloc_points), dtype=complex)
    for m, (lxm, lym) in enumerate(colloc_points):
        for k, G in enumerate(Gmats):
            C[m, k] = bilinear_interp(G, lx, ly, lxm, lym)
    # small ridge to stabilize
    ridge = 1e-6
    w = np.linalg.solve(C.conj().T @ C + ridge*np.eye(K, dtype=complex), C.conj().T @ d)

    # evaluate φ at state
    g_state = np.array([bilinear_interp(G, lx, ly, lX0, lY0) for G in Gmats])
    exp_h = np.exp(-np.array(etas) * (h0 - b))
    phi = np.sum(w * exp_h * g_state)
    return phi

def talbot_cdf(times: np.ndarray, laplace_phi, N: int = 48) -> np.ndarray:
    times = np.asarray(times, float)
    if np.any(times <= 0):
        raise ValueError("times must be > 0.")
    out = np.zeros_like(times, float)
    M = N
    h = 2*np.pi/M
    for it, t in enumerate(times):
        S = 0j
        for k in range(1, M+1):
            theta = (k - 0.5)*h - np.pi
            z = (M/(2.0*t))*(0.5*theta/np.tan(0.5*theta) - 1j*theta)
            dz = (M/(2.0*t))*((0.5*theta)/(np.sin(0.5*theta)**2) + 1j)*(-1j)*h
            phi = laplace_phi(z)
            S += np.exp(z*t) * (phi / z) * dz
        out[it] = (S/(2j*np.pi)).real
    out = np.maximum.accumulate(np.clip(out, -1e-6, 1.0+1e-6))
    return np.clip(out, 0.0, 1.0)
