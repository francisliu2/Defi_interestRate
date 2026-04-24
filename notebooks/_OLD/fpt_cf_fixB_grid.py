
# fpt_cf_fixB_grid.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class Params:
    # h dynamics
    mu_h: float
    sigma_h: float

    # Hawkes
    nu_X: float
    nu_Y: float
    beta_X: float
    beta_Y: float
    alpha_XX: float
    alpha_XY: float
    alpha_YX: float
    alpha_YY: float

    # Shifted exponential jumps U = s + Exp(kappa)
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

def build_lambda_grid(Lx_max: float, Ly_max: float, Mx: int, My: int):
    lx = np.linspace(0.0, Lx_max, Mx)
    ly = np.linspace(0.0, Ly_max, My)
    dx = lx[1] - lx[0]
    dy = ly[1] - ly[0]
    return lx, ly, dx, dy

def bilinear_interp(field: np.ndarray, lx: np.ndarray, ly: np.ndarray, x: float, y: float) -> complex:
    # clamp into the grid
    x = max(lx[0], min(lx[-1], x))
    y = max(ly[0], min(ly[-1], y))
    # find indices
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
    """
    Assemble the linear system A g = 0 for g on the (lx, ly) grid, where:
    L_eta[g] + (μ_h η + 0.5 σ^2 η^2 - q) g = 0
    L_eta includes λ-drifts and Hawkes shift terms.
    We discretize ∂_λ with upwind, and the shift g(λ+α) via bilinear interpolation.
    """
    Mx, My = len(lx), len(ly)
    N = Mx * My
    A = np.zeros((N, N), dtype=complex)

    EX = E_shifted_exp(eta, p.s_X, p.kappa_X)
    EY = E_shifted_exp(eta, p.s_Y, p.kappa_Y)

    c0 = (p.mu_h * eta + 0.5 * (p.sigma_h**2) * (eta**2) - q)  # constant term

    def idx(i, j): return i*My + j

    for i, lamx in enumerate(lx):
        for j, lamy in enumerate(ly):
            row = idx(i, j)

            # start with (μ_h η + 1/2 σ^2 η^2 - q)*g
            A[row, row] += c0

            # Drift terms: β_X ν_X ∂_λX g  - β_X λ_X ∂_λX g
            # Use upwind: velocity vX = β_X*(ν_X - λ_X)
            vX = p.beta_X * (p.nu_X - lamx)
            if abs(vX) > upwind_eps:
                if vX > 0:
                    # backward diff: (g(i,j)-g(i-1,j))/dx
                    if i > 0:
                        A[row, row] += vX / dx
                        A[row, idx(i-1, j)] += -vX / dx
                    else:
                        # Neumann at left boundary (zero gradient): g(-1) = g(0)
                        # => derivative ~ 0; skip contribution
                        pass
                else:
                    # forward diff: (g(i+1,j)-g(i,j))/dx
                    if i < Mx-1:
                        A[row, idx(i+1, j)] += -vX / dx
                        A[row, row] += vX / dx
                    else:
                        pass

            # Similarly for Y
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

            # Hawkes jump/shift terms:
            # λ_X ( EX * g(λ+α^{(X)}) - g(λ) ) + λ_Y ( EY * g(λ+α^{(Y)}) - g(λ) )
            # -g(λ) pieces add to diagonal
            A[row, row] += -lamx - lamy

            # For the shifted eval, we add the "interpolation row" explicitly
            # We'll accumulate into A via a stencil that represents bilinear interpolation weights.
            # Build weights for point (lamx+αXX, lamy+αYX) and (lamx+αXY, lamy+αYY)
            # We do manual bilinear weights:
            def add_shift(lx_shift, ly_shift, weight):
                # clamp
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

                # bilinear stencil
                A[row, idx(ii,   jj  )] += weight * (1-tx)*(1-ty)
                A[row, idx(ii+1, jj  )] += weight * (tx)*(1-ty)
                A[row, idx(ii,   jj+1)] += weight * (1-tx)*(ty)
                A[row, idx(ii+1, jj+1)] += weight * (tx)*(ty)

            # X-shift
            add_shift(lamx + p.alpha_XX, lamy + p.alpha_YX, lamx * EX)
            # Y-shift
            add_shift(lamx + p.alpha_XY, lamy + p.alpha_YY, lamy * EY)

    return A  # A g = 0 (homogeneous)

def solve_g_for_eta(
    eta: complex,
    q: complex,
    p: Params,
    lx: np.ndarray,
    ly: np.ndarray,
    dx: float,
    dy: float,
    ref_idx: Tuple[int, int] = (0,0),
) -> np.ndarray:
    """
    Solve A(eta,q) g = 0 with normalization g[ref_idx] = 1.
    Implement by replacing the row for ref_idx with an identity row.
    """
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

# ----------------- φ(q) evaluation via Fix B -----------------

def choose_etas(q: complex, p: Params, K: int) -> List[complex]:
    """
    Choose K eta's around the diffusion-only root to span the h-direction.
    You can tune this as needed.
    """
    # diffusion-only: 0.5 σ^2 η^2 - μ_h η - q = 0
    a = 0.5 * p.sigma_h**2
    b = -p.mu_h
    c = -q
    disc = b*b - 4*a*c
    eta_star = (-b + np.sqrt(disc)) / (2*a)
    # spread around eta_star on a geometric ladder
    scales = np.geomspace(0.5, 2.0, K)
    return [eta_star * s for s in scales]

def phi_q_fixB(
    q: complex,
    p: Params,
    h0: float,
    b: float,
    lX0: float,
    lY0: float,
    # grid
    Lx_max: float,
    Ly_max: float,
    Mx: int,
    My: int,
    K: int = 6,
    ref_idx: Tuple[int,int] = (0,0),
    colloc_points: Optional[List[Tuple[float,float]]] = None,
) -> complex:
    """
    Compute φ(q) by building K separated modes e^{-η(h-b)} g_η(λ),
    with each g_η solving the λ-system, and weights fitted to enforce
    Σ w_k g_ηk(λ) ≈ 1 over collocation points at the barrier.
    """
    lx, ly, dx, dy = build_lambda_grid(Lx_max, Ly_max, Mx, My)

    # pick K etas
    etas = choose_etas(q, p, K)

    # solve g for each eta
    Gmats = []
    for eta in etas:
        G = solve_g_for_eta(eta, q, p, lx, ly, dx, dy, ref_idx=ref_idx)
        Gmats.append(G)

    # Build boundary fit: for each collocation point λ^m, sum_k w_k g_k(λ^m) = 1
    if colloc_points is None:
        # default: a few points including origin and stationary mean multiples
        lam_bar = stationary_mean_intensities(p)
        colloc_points = [
            (0.0, 0.0),
            lam_bar,
            (0.5*lam_bar[0], 0.5*lam_bar[1]),
            (2.0*lam_bar[0], 2.0*lam_bar[1]),
            (lam_bar[0], 0.0),
            (0.0, lam_bar[1]),
        ]

    # Build matrix C w ≈ 1
    C = np.zeros((len(colloc_points), K), dtype=complex)
    d = np.ones(len(colloc_points), dtype=complex)

    # evaluate g_k at collocation points via bilinear interpolation
    for m, (lxm, lym) in enumerate(colloc_points):
        for k, G in enumerate(Gmats):
            C[m, k] = bilinear_interp(G, lx, ly, lxm, lym)

    # Least squares for weights
    w, *_ = np.linalg.lstsq(C, d, rcond=None)

    # Evaluate φ(q) at the current state
    # Interpolate each g_k at (lX0, lY0)
    g_at_state = np.array([bilinear_interp(G, lx, ly, lX0, lY0) for G in Gmats])
    exp_h = np.exp(-np.array(etas) * (h0 - b))
    phi = np.sum(w * exp_h * g_at_state)
    return phi

# ---------------- Talbot inversion for CDF (complex q) ----------------

def talbot_cdf(times: np.ndarray, laplace_phi, N: int = 32) -> np.ndarray:
    """
    Invert F(t) = L^{-1}[phi(q)/q](t) along Talbot contour.
    laplace_phi(q_complex) must accept complex q and return complex φ(q).
    """
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
            S += np.exp(z*t) * (phi / z) * dz   # divide by q for CDF
        out[it] = (S/(2j*np.pi)).real
    # enforce monotonicity and [0,1]
    out = np.maximum.accumulate(np.clip(out, -1e-6, 1.0+1e-6))
    return np.clip(out, 0.0, 1.0)
