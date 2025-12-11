from dataclasses import dataclass
from typing import Optional, Callable, Annotated
import numpy as np
from scipy.optimize import minimize

# Type aliases for clarity
PositiveFloat = Annotated[float, "Must be positive"]
NonNegativeFloat = Annotated[float, "Must be non-negative"]

@dataclass
class BaseParameters:
    """
    Parameters that apply system-wide
    """
    dt: PositiveFloat = 1/365  # time step for MC, daily data
    r: float = 0.0              # funding rate (if needed)

    def _post_init_(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive")


@dataclass
class JumpDiffusionParams:
    """
    Parameters of a spectrally negative jump-diffusion process
    """
    mu: float
    sigma: NonNegativeFloat
    lam: NonNegativeFloat              # jump intensity λ
    delta: NonNegativeFloat            # minimum jump magnitude
    eta: PositiveFloat                 # exponential rate for the jump tail
    rho: float = 0.0                   # correlation (only used for X)

    def _post_init_(self):
        """Validate all parameters are in valid ranges."""
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.lam < 0:
            raise ValueError("lambda (jump intensity) must be non-negative")
        if self.delta < 0:
            raise ValueError("delta must be non-negative")
        if self.eta <= 0:
            raise ValueError("eta must be strictly positive")
        if not (-1 <= self.rho <= 1):
            raise ValueError("rho (correlation) must be in [-1, 1]")

    # -----------------------------
    # MLE CALIBRATION
    # -----------------------------
    def fit_mle(self, returns: np.ndarray, dt: float = 1/365, verbose: bool = False):
        log_returns = np.array(returns)
        n = len(log_returns)

        def neg_log_likelihood(params):
            mu, sigma, lam, delta, eta = params
            if sigma <= 0 or lam < 0 or delta < 0 or eta <= 0:
                return np.inf

            # Approximate likelihood: diffusion + compound Poisson
            # For simplicity, assume at most one jump per dt (small dt)
            # PDF ≈ (1 - lam*dt) * Normal + lam*dt * shifted-exponential

            # Diffusion-only component
            mean_diff = mu * dt
            var_diff = sigma**2 * dt
            pdf_diff = (1 - lam*dt) * (1/np.sqrt(2*np.pi*var_diff)) * \
                       np.exp(-(log_returns - mean_diff)**2 / (2*var_diff))

            # Single-jump component (shift + exponential)
            pdf_jump = lam*dt * (1/eta) * np.exp(-(log_returns + delta)/eta)
            pdf_jump[log_returns > -delta] = 0  # jump support

            # Total likelihood
            likelihood = pdf_diff + pdf_jump
            # Avoid log(0)
            likelihood = np.maximum(likelihood, 1e-12)
            return -np.sum(np.log(likelihood))

        # Initial guess
        init_params = np.array([np.mean(log_returns)/dt, np.std(log_returns)/np.sqrt(dt),
                                0.01, 0.01, 1.0])

        bounds = [(-np.inf, np.inf), (1e-6, np.inf), (0, np.inf), (0, np.inf), (1e-6, np.inf)]
        result = minimize(neg_log_likelihood, init_params, bounds=bounds)

        if result.success:
            self.mu, self.sigma, self.lam, self.delta, self.eta = result.x
            if verbose:
                print("MLE success:", result.x)
        else:
            raise RuntimeError("MLE optimization failed: " + result.message)


@dataclass
class ModelParameters:
    X: JumpDiffusionParams
    Y: JumpDiffusionParams
    w_X: float
    w_Y: float
    b_X: PositiveFloat

    def _post_init_(self):
        if self.w_X - self.w_Y != 1:
            raise ValueError("weights must satisfy w_X - w_Y = 1")
        if self.b_X <= 0:
            raise ValueError("b_X must be positive")
        if not (-1 <= self.X.rho <= 1):
            raise ValueError("correlation must lie in [-1,1]")

        #check for conflicts before overwriting
        if hasattr(self.Y, 'rho') and self.Y.rho != self.X.rho and self.Y.rho != 0.0:
            raise ValueError(
                f"Y.rho ({self.Y.rho}) must equal X.rho ({self.X.rho}); "
                "only specify X.rho or ensure both are equal"
            )
        self.Y.rho = self.X.rho


@dataclass
class LogHealthParameters:
    pX: JumpDiffusionParams  # Parameters of X(t)
    pY: JumpDiffusionParams  # Parameters of Y(t)
    w_X: float
    w_Y: float
    b_X: PositiveFloat  # Collateral factor

    def _post_init_(self):
        """Validate constraints on log-health parameters."""
        if self.w_X - self.w_Y != 1:
            raise ValueError("weights must satisfy w_X - w_Y = 1")
        if self.b_X <= 0:
            raise ValueError("b_X must be positive")

    @property
    def drift_diffusion(self) -> float:
        pX, pY = self.pX, self.pY
        return (pX.mu - pY.mu) - 0.5 * (pX.sigma*2 + pY.sigma*2 - 2 * pX.rho * pX.sigma * pY.sigma)

    @property
    def variance_diffusion(self) -> float:
        pX, pY = self.pX, self.pY
        return (pX.sigma*2 + pY.sigma*2 - 2 * pX.rho * pX.sigma * pY.sigma)

    def mgf_jump_X(self, theta: float) -> float:
        p = self.pX
        if theta <= -p.eta:
            raise ValueError(f"theta={theta} must be > -{p.eta} for mgf_jump_X")
        log_exp_term = -theta * p.delta
        log_ratio = np.log(p.eta) - np.log(p.eta + theta)
        return np.exp(log_exp_term + log_ratio)

    def mgf_jump_Y(self, theta: float) -> float:
        p = self.pY
        if theta >= p.eta:
            raise ValueError(f"theta={theta} must be < {p.eta} for mgf_jump_Y")
        log_exp_term = theta * p.delta
        log_ratio = np.log(p.eta) - np.log(p.eta - theta)
        return np.exp(log_exp_term + log_ratio)

    def psi_h(self, theta: float) -> float:
        pX, pY = self.pX, self.pY
        lower_bound = -pX.eta
        upper_bound = pY.eta
        if not (lower_bound < theta < upper_bound):
            raise ValueError(f"theta={theta} outside valid domain ({lower_bound}, {upper_bound})")
        return (
            theta * self.drift_diffusion
            + 0.5 * theta**2 * self.variance_diffusion
            + pX.lam * (self.mgf_jump_X(theta) - 1.0)
            + pY.lam * (self.mgf_jump_Y(-theta) - 1.0)
        )