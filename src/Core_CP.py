"""
core.py
Reusable base classes
Includes:
- BaseParameters
- BaseStrategy
- CompoundPoissonParams
- CompoundPoissonProcess
"""

from dataclasses import dataclass
import numpy as np
from typing import Any, Tuple


# ============================================================
# 1. BaseParameters
# ============================================================

@dataclass
class BaseParameters:
    """
    Generic parameter container for strategies.
    """
    W0: float = 10000.0   # Initial wealth
    T: float = 1.0        # Time horizon in years
    b_X: float = 1.0      # Optional scaling parameter
    cp_params: Any = None # Compound Poisson parameters (set by strategy)

    def validate(self):
        if self.W0 <= 0:
            raise ValueError("Initial wealth W0 must be positive")
        if self.T <= 0:
            raise ValueError("Time horizon T must be positive")


# ============================================================
# 2. Compound Poisson Jump-Diffusion Model
# ============================================================

@dataclass
class CompoundPoissonParams:
    """
    Parameters for a two-leg compound Poisson jump-diffusion model.
    X = long leg, Y = short leg.
    """

    # Long leg
    mu_X: float
    sigma_X: float
    lambda_X: float
    delta_X: float
    eta_X: float

    # Short leg
    mu_Y: float
    sigma_Y: float
    lambda_Y: float
    delta_Y: float
    eta_Y: float


class CompoundPoissonProcess:
    """
    Compound Poisson jump-diffusion engine.
    Provides:
    - characteristic function
    - mean/variance computation
    """

    def __init__(self, params: CompoundPoissonParams):
        self.params = params

    # Characteristic exponent ψ(u)
    def _psi(self, u, mu, sigma, lam, delta):
        return (
            1j * mu * u
            - 0.5 * sigma**2 * u**2
            + lam * (np.exp(1j * u * delta) - 1)
        )

    def characteristic_function(self, u, leg: str, T: float):
        """
        φ(u) = exp(T * ψ(u))
        """
        p = self.params

        if leg == "X":
            psi = self._psi(u, p.mu_X, p.sigma_X, p.lambda_X, p.delta_X)
        elif leg == "Y":
            psi = self._psi(u, p.mu_Y, p.sigma_Y, p.lambda_Y, p.delta_Y)
        else:
            raise ValueError("leg must be 'X' or 'Y'")

        return np.exp(T * psi)

    def compute_moments(self, leg: str, T: float) -> Tuple[float, float]:
        """
        Returns (mean, variance) of the jump-diffusion return.
        """
        p = self.params

        if leg == "X":
            mu, sigma, lam, delta = p.mu_X, p.sigma_X, p.lambda_X, p.delta_X
        elif leg == "Y":
            mu, sigma, lam, delta = p.mu_Y, p.sigma_Y, p.lambda_Y, p.delta_Y
        else:
            raise ValueError("leg must be 'X' or 'Y'")

        mean = T * (mu + lam * delta)
        variance = T * (sigma**2 + lam * delta**2)

        return mean, variance


# ============================================================
# 3. BaseStrategy
# ============================================================

class BaseStrategy:
    """
    Strategy parent class.
    Provides:
    - access to parameters
    - compound Poisson process
    - utility function hooks
    - placeholder liquidation/health methods
    """

    def __init__(self, params: BaseParameters):
        params.validate()
        self.params = params

        if params.cp_params is None:
            raise ValueError("cp_params must be set before initializing strategy")

        self.cp_process = CompoundPoissonProcess(params.cp_params)

    # --------------------------------------------------------
    # Utility function (override in subclasses)
    # --------------------------------------------------------
    def utility_function(self, r: float, rho1: float, rho2: float) -> float:
        """
        Default utility: mean - rho1 * volatility - rho2 * |beta|
        Override in subclasses for custom behavior.
        """
        analysis = self.analyze_strategy(r)
        return (
            analysis["expected_return"]
            - rho1 * analysis["total_volatility"]
            - rho2 * abs(analysis["portfolio_beta"])
        )

    # --------------------------------------------------------
    # Placeholder methods (override in subclasses)
    # --------------------------------------------------------
    def analyze_strategy(self, r: float):
        raise NotImplementedError("Subclasses must implement analyze_strategy")

    def liquidation_probability(self, r: float) -> float:
        return 0.0

    def initial_health(self, r: float) -> float:
        return 1.0
