"""
Log-Health Process Module

This module provides the universal log-health process used across all
strategies. The log-health formula is strategy-agnostic;
only weights change.

We construct:

1. Drift of log-health
2. Variance of log-health
3. Lévy exponent 
4. Deterministic initial log-health h0
"""

from dataclasses import dataclass
from typing import Annotated
import numpy as np
import math


# Type aliases for clarity
PositiveFloat = Annotated[float, "Must be positive"]
NonNegativeFloat = Annotated[float, "Must be non-negative"]
CollateralFactor = Annotated[float, "Must be in (0, 1]"]


@dataclass
class HealthProcessParameters:
    """Parameters for the universal log-health process."""

    # Strategy weights
    w_X: float
    w_Y: PositiveFloat

    b_X: CollateralFactor = 1.0  # collateral factor: fraction of collateral considered (0 < b_X <= 1)

    # Diffusion parameters
    mu_X: float = 0.0
    mu_Y: float = 0.0
    sigma_X: NonNegativeFloat = 0.0
    sigma_Y: NonNegativeFloat = 0.0
    rho: float = 0.0

    # Jump intensities
    lambda_X: NonNegativeFloat = 0.0
    lambda_Y: NonNegativeFloat = 0.0

    # Jump size parameters (one-sided)
    delta_X: NonNegativeFloat = 0.0   # minimum downward jump for X (shift)
    delta_Y: NonNegativeFloat = 0.0   # minimum upward jump for Y (shift)
    eta_X: PositiveFloat = 1.0        # exp rate for X jump tail
    eta_Y: PositiveFloat = 1.0        # exp rate for Y jump tail

    # Current prices (only used for initial h0)
    X0: PositiveFloat = 1.0
    Y0: PositiveFloat = 1.0

    def __post_init__(self):
        """Validate all parameters are in valid ranges."""
        if self.w_Y <= 0:
            raise ValueError("w_Y must be positive (short-side exposure/funding).")
        # Allow for small floating-point rounding when checking the weights
        if not math.isclose(self.w_X - self.w_Y, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            diff = self.w_X - self.w_Y
            raise ValueError(f"weights must satisfy w_X - w_Y = 1 (got {diff})")
        if self.sigma_X < 0:
            raise ValueError("sigma_X must be non-negative")
        if self.sigma_Y < 0:
            raise ValueError("sigma_Y must be non-negative")
        if not (-1 <= self.rho <= 1):
            raise ValueError("rho (correlation) must be in [-1, 1]")
        if self.eta_X <= 0:
            raise ValueError("eta_X (jump exponential rate) must be strictly positive")
        if self.eta_Y <= 0:
            raise ValueError("eta_Y (jump exponential rate) must be strictly positive")
        if self.lambda_X < 0:
            raise ValueError("lambda_X (jump intensity) must be non-negative")
        if self.lambda_Y < 0:
            raise ValueError("lambda_Y (jump intensity) must be non-negative")
        if self.delta_X < 0:
            raise ValueError("delta_X must be non-negative")
        if self.delta_Y < 0:
            raise ValueError("delta_Y must be non-negative")
        if self.X0 <= 0:
            raise ValueError("Initial price X0 must be strictly positive")
        if self.Y0 <= 0:
            raise ValueError("Initial price Y0 must be strictly positive")
        if not (0.0 < self.b_X <= 1.0):
            raise ValueError("b_X must lie in (0, 1] for a standard collateral factor. "
                             "Set a different validation if your protocol uses other conventions.")


class LogHealthProcess:
    """
    Universal log-health process module for DeFi strategies.

    The log-health process h(t) = log(b_X * w_X * X(t) / (w_Y * Y(t)))
    is defined by its Lévy exponent and all relevant moments.
    """

    def __init__(self, params: HealthProcessParameters):
        self.p = params

    # --------------------------------------------------------------------------
    # INITIAL HEALTH
    # --------------------------------------------------------------------------
    def h0(self) -> float:
        """
        Compute deterministic initial log-health.

        Returns:
            Initial log-health: log(b_X * w_X * X0 / (w_Y * Y0))
        """
        p = self.p
        return np.log((p.b_X * p.w_X * p.X0) / (p.w_Y * p.Y0))

    # --------------------------------------------------------------------------
    # DIFFUSION CONTRIBUTIONS
    # --------------------------------------------------------------------------
    def drift_diffusion(self) -> float:
        """
        Drift of log(X/Y) from diffusion only (Itô adjustment included).

        Formula used: (μ_X - μ_Y) - ½(σ_X² + σ_Y² - 2ρσ_Xσ_Y)

        Returns:
            Drift coefficient of the diffusion component
        """
        p = self.p
        return (p.mu_X - p.mu_Y
                - 0.5 * (p.sigma_X**2 + p.sigma_Y**2 - 2 * p.rho * p.sigma_X * p.sigma_Y))

    def variance_diffusion(self) -> float:
        """
        Instantaneous variance of log(X/Y) from diffusion only.

        Formula: σ_X² + σ_Y² - 2ρσ_Xσ_Y

        Returns:
            Diffusion variance coefficient
        """
        p = self.p
        return (p.sigma_X**2
                + p.sigma_Y**2
                - 2 * p.rho * p.sigma_X * p.sigma_Y)

    # --------------------------------------------------------------------------
    # JUMP MGF CONTRIBUTIONS
    # --------------------------------------------------------------------------
    def mgf_jump_X(self, theta: float) -> float:
        """
        MGF of one-sided downward jump U_X = -(δ_X + Exp(η_X)).

        Formula: E[e^{θ U_X}] = exp(-θ δ_X) * η_X/(η_X + θ)

        Valid for θ > -η_X.

        Args:
            theta: Moment parameter

        Returns:
            MGF evaluated at theta

        Raises:
            ValueError: If theta is outside the valid domain
        """
        p = self.p

        if theta <= -p.eta_X:
            raise ValueError(
                f"theta={theta} must be > -{p.eta_X} for mgf_jump_X"
            )

        # Use log-space for numerical stability
        log_exp_term = -theta * p.delta_X
        log_ratio = np.log(p.eta_X) - np.log(p.eta_X + theta)
        return np.exp(log_exp_term + log_ratio)

    def mgf_jump_Y(self, theta: float) -> float:
        """
        MGF of one-sided upward jump U_Y = +(δ_Y + Exp(η_Y)).

        Formula: E[e^{θ U_Y}] = exp(+θ δ_Y) * η_Y/(η_Y - θ)

        Valid for θ < η_Y.

        Args:
            theta: Moment parameter

        Returns:
            MGF evaluated at theta

        Raises:
            ValueError: If theta is outside the valid domain
        """
        p = self.p

        if theta >= p.eta_Y:
            raise ValueError(
                f"theta={theta} must be < {p.eta_Y} for mgf_jump_Y"
            )

        # Use log-space for numerical stability
        log_exp_term = theta * p.delta_Y
        log_ratio = np.log(p.eta_Y) - np.log(p.eta_Y - theta)
        return np.exp(log_exp_term + log_ratio)

    # --------------------------------------------------------------------------
    # LÉVY EXPONENT
    # --------------------------------------------------------------------------
    def psi_h(self, theta: float) -> float:
        """
        Lévy exponent of log-health increments.

        Formula: ψ_h(θ) = θ·drift_diffusion
                        + ½θ²·variance_diffusion
                        + λ_X(MGF_X(θ) - 1)
                        + λ_Y(MGF_Y(-θ) - 1)

        Valid for -η_X < θ < η_Y. Note the sign flip (-θ) for the Y jump MGF
        due to the ratio structure in h(t) = log(X/Y).

        Args:
            theta: Moment parameter, must satisfy -η_X < θ < η_Y

        Returns:
            Lévy exponent value

        Raises:
            ValueError: If theta is outside the valid domain
        """
        p = self.p

        # Validate theta is in valid domain
        lower_bound = -p.eta_X
        upper_bound = p.eta_Y

        if not (lower_bound < theta < upper_bound):
            raise ValueError(
                f"theta={theta} outside valid domain "
                f"({lower_bound}, {upper_bound}). "
                f"Valid range: -η_X < θ < η_Y where "
                f"η_X={p.eta_X}, η_Y={p.eta_Y}"
            )

        diff = (theta * self.drift_diffusion()
                + 0.5 * (theta**2) * self.variance_diffusion())

        jumps_X = p.lambda_X * (self.mgf_jump_X(theta) - 1.0)
        jumps_Y = p.lambda_Y * (self.mgf_jump_Y(-theta) - 1.0)

        return diff + jumps_X + jumps_Y

    # --------------------------------------------------------------------------
    # DRIFT OF LOG-HEALTH (ψ'_h(0))
    # --------------------------------------------------------------------------
    def drift_health(self) -> float:
        """
        Expected drift of log-health: κ_h = ψ'_h(0).

        This is the instantaneous expected rate of change of log-health.

        Derivation:
            ψ'_h(θ) = drift_diffusion
                     + θ·variance_diffusion
                     + λ_X·η_X/(η_X+θ)² (MGF derivative)
                     + λ_Y·(-η_Y/(η_Y-θ)²) (MGF derivative, note sign)

        At θ=0:
            ψ'_h(0) = drift_diffusion
                    + λ_X/η_X
                    - λ_Y/η_Y

        Alternative formula using jump expectations:
            κ_h = drift_diffusion + λ_X·E[U_X] + λ_Y·E[U_Y]

        where E[U_X] = -(δ_X + 1/η_X) and E[U_Y] = +(δ_Y + 1/η_Y)

        Returns:
            Expected drift of log-health
        """
        p = self.p

        drift = self.drift_diffusion()

        # Jump contributions via Lévy-Khintchine formula
        # ψ'_h(0) = ∫ u ν(du) where ν is the jump measure
        jump_contribution_X = p.lambda_X / p.eta_X
        jump_contribution_Y = -p.lambda_Y / p.eta_Y

        return drift + jump_contribution_X + jump_contribution_Y