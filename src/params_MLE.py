"""
This module includes:

1. JumpDiffusionParams: parameters for a spectrally negative jump-diffusion process
2. ModelParameters: combined parameters for X and Y processes, weights, collateral factor(b_X)
3. LogHealthParameters 
4. MLE calibration 
"""
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
        """
        Fit jump-diffusion parameters using MLE with robust initialization and bounds.
        
        Parameters:
            returns: array of log-returns
            dt: time step (default 1/365 for daily data)
            verbose: print optimization details
        """
        log_returns = np.array(returns)
        n = len(log_returns)
        
        # Robust initial estimates from data
        mean_ret = np.mean(log_returns)
        std_ret = np.std(log_returns, ddof=1)
        
        # Annualized parameters for initialization
        mu_init = mean_ret / dt
        sigma_init = std_ret / np.sqrt(dt)
        
        # Detect jumps: large deviations from normal
        z_scores = np.abs((log_returns - mean_ret) / std_ret)
        jump_prob = np.mean(z_scores > 2.0)  # % of returns > 2 std devs
        lam_init = max(0.01, min(jump_prob * 10, 0.5))
        delta_init = 0.02
        eta_init = 2.0

        def neg_log_likelihood(params):
            mu, sigma, lam, delta, eta = params
            
            # Parameter validation
            if sigma <= 0 or eta <= 0:
                return 1e10
            if lam < 0 or delta < 0:
                return 1e10
            
            try:
                # Diffusion component: Normal(mu*dt, sigma^2*dt)
                mean_diff = mu * dt
                var_diff = np.maximum(sigma**2 * dt, 1e-12)
                
                # Normal PDF
                norm_pdf = (1.0 / np.sqrt(2 * np.pi * var_diff)) * \
                           np.exp(-0.5 * (log_returns - mean_diff)**2 / var_diff)
                
                # Jump component: exponential tail with shift
                # Support: y <= -delta (spectrally negative)
                jump_pdf = np.zeros_like(log_returns)
                jump_mask = log_returns <= -delta
                jump_pdf[jump_mask] = (1.0 / eta) * np.exp(-(log_returns[jump_mask] + delta) / eta)
                
                # Mixture: (1 - lam*dt) * diffusion + lam*dt * jump
                mixture = (1.0 - lam * dt) * norm_pdf + lam * dt * jump_pdf
                mixture = np.maximum(mixture, 1e-15)
                
                nll = -np.sum(np.log(mixture))
                return nll if np.isfinite(nll) else 1e10
            except:
                return 1e10

        # Bounds: mu (unbounded), sigma (positive), lam (0-1), delta (positive), eta (positive)
        bounds = [
            (-0.5, 0.5),           # mu (reasonable drift range)
            (1e-4, 2.0),           # sigma (volatility bounds)
            (0.0, 1.0),            # lam (jump intensity 0-100%)
            (0.0, 0.2),            # delta (jump magnitude)
            (0.5, 10.0)            # eta (tail decay)
        ]
        
        init_params = np.array([mu_init, sigma_init, lam_init, delta_init, eta_init])
        
        # Clip initial params to bounds
        for i in range(len(init_params)):
            init_params[i] = np.clip(init_params[i], bounds[i][0], bounds[i][1])
        
        if verbose:
            print(f"  Initial params: mu={init_params[0]:.6f}, sigma={init_params[1]:.6f}, " +
                  f"lam={init_params[2]:.6f}, delta={init_params[3]:.6f}, eta={init_params[4]:.6f}")
        
        try:
            result = minimize(
                neg_log_likelihood,
                init_params,
                bounds=bounds,
                method='L-BFGS-B',
                options={'ftol': 1e-8, 'maxiter': 500}
            )
            
            if result.success or result.nfev > 100:  # Accept if converged or made many function evals
                self.mu, self.sigma, self.lam, self.delta, self.eta = result.x
                if verbose:
                    print(f"  ✓ MLE converged (nfev={result.nfev}, nll={result.fun:.4f})")
                    print(f"    mu={self.mu:.6f}, sigma={self.sigma:.6f}, lam={self.lam:.6f}, " +
                          f"delta={self.delta:.6f}, eta={self.eta:.6f}")
            else:
                # Use initialized parameters if optimization fails
                self.mu, self.sigma, self.lam, self.delta, self.eta = init_params
                if verbose:
                    print(f"  ⚠ MLE did not converge fully (nfev={result.nfev}); using init params")
        except Exception as e:
            # Fallback: use initialized parameters if optimization crashes
            self.mu, self.sigma, self.lam, self.delta, self.eta = init_params
            if verbose:
                print(f"  ⚠ MLE optimization failed ({str(e)}); using init params")


@dataclass
class ModelParameters:
    X: JumpDiffusionParams
    Y: JumpDiffusionParams
    w_X: float
    w_Y: float
    b_X: PositiveFloat

    def _post_init_(self):
        if abs((self.w_X - self.w_Y) - 1.0) > 1e-12:
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