# ==== Mean-field Hawkes first-passage moments (full, copiable) ====
# Requirements: numpy, mpmath
# pip install mpmath

import numpy as np
import mpmath as mp
from dataclasses import dataclass

mp.mp.dps = 50  # high precision helps with Laplace inversions and roots

# ---------------------------
# Parameters and mean-field rates
# ---------------------------

@dataclass
class HawkesMFParams:
    # log-health diffusion
    mu_h: float            # drift of h: mu_X - mu_Y - 0.5*sigma_h^2
    sigma_h: float         # vol of h (>=0)

    # jump-size laws (shifted exponential J = delta + Exp(eta), eta>0, delta>=0)
    eta_X: float
    eta_Y: float
    delta_X: float
    delta_Y: float

    # Hawkes intensity params (for mean-field rates)
    beta_X: float
    beta_Y: float
    alpha_XX: float
    alpha_XY: float
    alpha_YX: float
    alpha_YY: float
    mu_X_lambda: float
    mu_Y_lambda: float

    # initial intensities (if you want dynamic MF), else ignored for static
    lambda_X0: float = None
    lambda_Y0: float = None


def meanfield_rates_static(p: HawkesMFParams):
    """
    Return stationary mean-field intensities (if Hawkes is subcritical).
    Solve: E[lambda]' = A E[lambda] + b  =>  lambda_inf = -A^{-1} b
    """
    A = mp.matrix([
        [-p.beta_X + p.alpha_XX, p.alpha_XY],
        [p.alpha_YX,            -p.beta_Y + p.alpha_YY]
    ])
    b = mp.matrix([p.beta_X * p.mu_X_lambda, p.beta_Y * p.mu_Y_lambda])
    # Check subcriticality (spectral radius < 0 for A) roughly:
    try:
        evals = [complex(mp.eig(A)[0][i]) for i in range(2)]
    except Exception:
        evals = []
    if any(e.real >= 0 for e in evals):
        print("[warn] mean-field A not Hurwitz; stationary rates may not exist.")
    lam_inf = -A**-1 * b
    lX = float(lam_inf[0])
    lY = float(lam_inf[1])
    if lX <= 0 or lY <= 0:
        print("[warn] stationary MF rates non-positive, clipping at small positive.")
        lX = max(lX, 1e-10)
        lY = max(lY, 1e-10)
    return lX, lY

# ---------------------------
# Lévy exponent κ(θ) and Φ(s)
# ---------------------------

def kappa(theta, params_tuple):
    """
    Lévy exponent κ(θ) for h_t with mean-field Poisson jumps (downward):
      κ(θ) = μ_h θ + 1/2 σ_h^2 θ^2
             + λX*(E[e^{-θ J_X}] - 1) + λY*(E[e^{-θ J_Y}] - 1)
    where J = δ + Exp(η)  =>  E[e^{-θ J}] = e^{-θ δ} * η/(η+θ).
    """
    mu_h, sigma_h, lX, lY, etaX, etaY, dX, dY = params_tuple
    theta = mp.mpf(theta) if not isinstance(theta, (mp.mpf, mp.mpc)) else theta
    drift = mu_h * theta + 0.5 * (sigma_h**2) * (theta**2)
    EX = mp.e**(-theta * dX) * (etaX / (etaX + theta)) - 1.0
    EY = mp.e**(-theta * dY) * (etaY / (etaY + theta)) - 1.0
    return drift + lX * EX + lY * EY

def Phi_of_s(s, params_tuple, guess=None):
    """
    Right inverse Φ(s): solve κ(θ) = s for θ>0.
    We use mpmath findroot with a sane initial guess / bracket.
    """
    s = mp.mpf(s)
    # crude initial guess: small s -> small theta; large s -> scale by sigma
    if guess is None:
        mu_h, sigma_h, *_ = params_tuple
        g = (s / max(1e-8, abs(mu_h))) if abs(mu_h) > 1e-8 else mp.sqrt(2*s/max(1e-12, sigma_h**2))
        guess = max(1e-8, float(g))

    f = lambda th: kappa(th, params_tuple) - s
    # try two-seed findroot
    try:
        th = mp.findroot(f, (guess, guess*1.5))
        if th <= 0:
            raise ValueError
        return th
    except Exception:
        # fallback: bracket scan
        lo = mp.mpf('1e-10')
        hi = mp.mpf('1.0')
        # expand hi until f(lo)*f(hi) < 0 (simple bracket)
        flo = f(lo)
        fhi = f(hi)
        tries = 0
        while flo * fhi > 0 and tries < 40:
            hi *= 2
            fhi = f(hi)
            tries += 1
        if flo * fhi > 0:
            # last resort: return positive guess
            return mp.mpf(max(guess, 1e-6))
        # bisection polish then newton
        for _ in range(60):
            mid = 0.5*(lo+hi)
            fmid = f(mid)
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5*(lo+hi)

# ---------------------------
# Scale function W^{(s)} via Laplace inversion
# ---------------------------

def W_s(x, s, params_tuple, method="dehoog"):
    """
    Compute W^{(s)}(x) = L^{-1}_{θ->x} [ 1/(κ(θ) - s) ](x)
    using mpmath.invertlaplace (de Hoog). x >= 0.
    """
    if x < 0:
        return mp.mpf('0.0')
    if x == 0:
        # Right limit W(0+): for bounded var paths W(0)=0; with Brownian σ>0, W(0)=0.
        return mp.mpf('0.0')

    F = lambda th: 1.0 / (kappa(th, params_tuple) - s)  # Laplace image
    try:
        val = mp.invertlaplace(F, x, method=method)  # returns real by default
    except Exception:
        # fallback: crude Bromwich integral
        # not super-robust; better to fix parameters or de Hoog settings
        a = Phi_of_s(s, params_tuple)
        def integrand(u):
            th = a + 1j*u
            return mp.e**(th*x) / (kappa(th, params_tuple) - s)
        val = (1.0/(2*mp.pi)) * mp.quad(lambda u: integrand(u).real, [-mp.inf, mp.inf])
    return val

def Z_s(x, s, params_tuple):
    """ Z^{(s)}(x) = 1 + s * ∫_0^x W^{(s)}(y) dy """
    if x <= 0:
        return mp.mpf('1.0')
    f = lambda y: W_s(y, s, params_tuple)
    I = mp.quad(f, [0, x])
    return 1 + s*I

def psi_tau(s, h0, params_tuple):
    """
    ψ(s;h0) = E[e^{-s τ}] for first passage below 0 starting at h0>0:
      ψ = Z^{(s)}(h0) - (s/Φ(s)) W^{(s)}(h0)
    """
    if s == 0:
        return mp.mpf('1.0')
    Phi = Phi_of_s(s, params_tuple)
    W = W_s(h0, s, params_tuple)
    Z = Z_s(h0, s, params_tuple)
    return Z - (s/Phi)*W

# ---------------------------
# Moments from ψ via forward differences at 0+
# ---------------------------

def moments_from_psi(h0, params_tuple, s_step=1e-2):
    """
    Compute first four moments via 5-point one-sided forward differences at s=0+.
    Returns dict with mean, variance, skewness, kurtosis (Pearson).
    """
    s = mp.mpf(s_step)
    psi0 = mp.mpf('1.0')
    psi1 = psi_tau(s,   h0, params_tuple)
    psi2 = psi_tau(2*s, h0, params_tuple)
    psi3 = psi_tau(3*s, h0, params_tuple)
    psi4 = psi_tau(4*s, h0, params_tuple)

    # 5-point forward differences (O(s^4))
    d1 = (-25*psi0 + 48*psi1 - 36*psi2 + 16*psi3 - 3*psi4) / (12*s)
    d2 = ( 35*psi0 -104*psi1 + 114*psi2 - 56*psi3 + 11*psi4) / (12*s**2)
    d3 = (-50*psi0 +160*psi1 - 180*psi2 + 96*psi3 - 21*psi4) / (12*s**3)
    d4 = ( 35*psi0 -120*psi1 + 150*psi2 - 80*psi3 + 15*psi4) / (12*s**4)

    # Raw moments from derivatives of ψ at 0: m1 = -ψ'(0), m2 = ψ''(0), m3 = -ψ'''(0), m4 = ψ''''(0)
    m1 = -d1
    m2 =  d2
    m3 = -d3
    m4 =  d4

    var = m2 - m1**2
    mu3 = m3 - 3*m1*m2 + 2*m1**3
    mu4 = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4

    # guard tiny negative due to numerics
    var = float(var) if var >= -1e-12 else float(var)
    if var < 0 and abs(var) < 1e-10:
        var = 0.0
    skew = float(mu3 / (var**1.5)) if var > 0 else np.nan
    kurt = float(mu4 / (var**2))    if var > 0 else np.nan  # Pearson kurtosis

    return dict(
        mean=float(m1),
        variance=float(var),
        skewness=skew,
        kurtosis=kurt,
        raw_moments=(float(m1), float(m2), float(m3), float(m4))
    )

# ---------------------------
# Helper to run across LTVs
# ---------------------------

def run_across_LTVs(p: HawkesMFParams, LTV_values, s_step=5e-3, use_stationary_rates=True):
    """
    For each LTV, compute moments with mean-field STATIC rates.
    h0 = log(1/LTV).
    """
    if use_stationary_rates:
        lX, lY = meanfield_rates_static(p)
    else:
        # You can add time-averaged or dynamic MF here if you want later
        lX, lY = p.lambda_X0, p.lambda_Y0

    params_tuple = (
        p.mu_h,
        p.sigma_h,
        float(lX),
        float(lY),
        float(p.eta_X),
        float(p.eta_Y),
        float(p.delta_X),
        float(p.delta_Y),
    )

    print("LTV, mean, variance, skewness, kurtosis")
    for LTV in LTV_values:
        h0 = float(np.log(1.0 / LTV))
        out = moments_from_psi(h0, params_tuple, s_step=s_step)
        print(f"{LTV:.2f}, {out['mean']:.6g}, {out['variance']:.6g}, "
              f"{out['skewness']:.6g}, {out['kurtosis']:.6g}")

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Example parameter set (tweak to your case)
    # Important: both jumps LOWER h (harmful). mu_h = mu_X - mu_Y - 0.5*sigma_h^2
    p = HawkesMFParams(
        mu_h=0.03 - 0.01 - 0.5*(0.25**2),   # example: mu_X=3%, mu_Y=1%, sigma_h=25%
        sigma_h=0.25,
        eta_X=5.0,  eta_Y=5.0,
        delta_X=0.05, delta_Y=0.05,
        beta_X=2.0, beta_Y=2.0,
        alpha_XX=0.3, alpha_XY=0.1,
        alpha_YX=0.1, alpha_YY=0.3,
        mu_X_lambda=0.1, mu_Y_lambda=0.1,
        lambda_X0=0.1, lambda_Y0=0.1,
    )

    # LTV grid
    LTVs = np.linspace(0.1, 0.9, 12)

    # Print moments (s_step ~ 5e-3 is a good start; shrink if noisy)
    run_across_LTVs(p, LTVs, s_step=5e-3)
