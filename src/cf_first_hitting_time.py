"""
Characteristic Function for First Hitting Time Analysis

This module implements the characteristic function approach for computing
first hitting time distributions of the log-health factor in DeFi liquidation
risk models with bivariate Hawkes jump-diffusion processes.

The implementation includes:
1. 12D complex-valued ODE system for CF Riccati equations
2. Integration with Gil-Pelaez inversion algorithm for distribution recovery
3. Integration with existing parameter structures and utilities

Author: [Author Name]
Date: 2025
"""

import warnings
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

# Import existing modules for reuse
from hawkes_process import HawkesParameters
from inversion_algorithms import GilPelaezInversion
from utils import create_summary_statistics, save_results


class CFFirstHittingTime:
    """
    Characteristic function computation for first hitting time of log-health factor.

    This class implements the 12D complex-valued ODE system for computing characteristic
    functions of first hitting times in the Hawkes jump-diffusion framework.
    """

    def __init__(self, parameters: HawkesParameters):
        """
        Initialize CF solver using existing HawkesParameters structure.

        Args:
            parameters: HawkesParameters object containing all model parameters
        """
        self.params = parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate model parameters for stability and convergence."""
        if self.params.sigma_h <= 0:
            raise ValueError("Volatility sigma_h must be positive")

        if self.params.eta_X <= 0 or self.params.eta_Y <= 0:
            raise ValueError("Jump intensity parameters eta_X, eta_Y must be positive")

        if self.params.beta_X <= 0 or self.params.beta_Y <= 0:
            raise ValueError(
                "Mean reversion parameters beta_X, beta_Y must be positive"
            )

        # Check stability using existing validation pattern
        spectral_radius = max(
            abs(self.params.alpha_XX / self.params.beta_X),
            abs(self.params.alpha_YY / self.params.beta_Y),
        )

        if spectral_radius >= 1.0:
            warnings.warn(
                f"Hawkes process may be unstable: spectral radius = {spectral_radius:.3f}"
            )

    def ode_system_real(self, h: float, y: np.ndarray, s: float) -> np.ndarray:
        """
        12D first-order ODE system for CF Riccati equations in real coordinates.

        Args:
            h: Current health factor value
            y: State vector [ReA, ReA', ReB, ReB', ReC, ReC', ImA, ImA', ImB, ImB', ImC, ImC']
            s: Frequency parameter

        Returns:
            Derivative vector
        """
        # Extract real and imaginary parts
        ReA, ReAp, ReB, ReBp, ReC, ReCp, ImA, ImAp, ImB, ImBp, ImC, ImCp = y

        # Reconstruct complex functions
        A = ReA + 1j * ImA
        A_p = ReAp + 1j * ImAp
        B = ReB + 1j * ImB
        B_p = ReBp + 1j * ImBp
        C = ReC + 1j * ImC
        C_p = ReCp + 1j * ImCp

        i_s = 1j * s

        # Compute drift of health factor from existing parameters
        mu_h = (
            self.params.mu_X
            - self.params.mu_Y
            - 0.5
            * (
                self.params.sigma_X**2
                + self.params.sigma_Y**2
                - 2 * self.params.rho * self.params.sigma_X * self.params.sigma_Y
            )
        )

        # Avoid division by zero
        denom_X = self.params.eta_X - i_s * A_p + 1e-10
        denom_Y = self.params.eta_Y + i_s * A_p + 1e-10

        # Characteristic functions for jump sizes
        f_X = (self.params.eta_X / denom_X) * np.exp(
            i_s * A_p * self.params.delta_X
            + self.params.alpha_XX * B
            + self.params.alpha_XY * C
        )
        f_Y = (self.params.eta_Y / denom_Y) * np.exp(
            -i_s * A_p * self.params.delta_Y
            + self.params.alpha_YY * C
            + self.params.alpha_YX * B
        )

        # Second derivatives from the Riccati system
        if abs(s) < 1e-12:
            # Use L'Hôpital's rule for s→0
            A_pp = (
                -2 * mu_h * A_p
                - 2
                * self.params.beta_X
                * self.params.mu_X_lambda
                * B
                / self.params.sigma_h**2
                - 2
                * self.params.beta_Y
                * self.params.mu_Y_lambda
                * C
                / self.params.sigma_h**2
                - 2j
            ) / self.params.sigma_h**2
        else:
            A_pp = (
                -2 * mu_h * i_s * A_p
                - self.params.sigma_h**2 * (i_s * A_p) ** 2
                - 2 * self.params.beta_X * self.params.mu_X_lambda * B
                - 2 * self.params.beta_Y * self.params.mu_Y_lambda * C
                - 2 * i_s
            ) / (self.params.sigma_h**2 * i_s)

        B_pp = (
            -2 * mu_h * B_p
            - self.params.sigma_h**2 * (B_p**2 + 2 * i_s * A_p * B_p)
            + 2 * self.params.beta_X * B
            - 2 * (f_X - 1)
        ) / self.params.sigma_h**2

        C_pp = (
            -2 * mu_h * C_p
            - self.params.sigma_h**2 * (C_p**2 + 2 * i_s * A_p * C_p)
            + 2 * self.params.beta_Y * C
            - 2 * (f_Y - 1)
        ) / self.params.sigma_h**2

        # Return derivative vector in real form
        dy = np.zeros(12)
        dy[0] = ReAp  # d(ReA)/dh
        dy[1] = A_pp.real  # d(ReA')/dh
        dy[2] = ReBp  # d(ReB)/dh
        dy[3] = B_pp.real  # d(ReB')/dh
        dy[4] = ReCp  # d(ReC)/dh
        dy[5] = C_pp.real  # d(ReC')/dh
        dy[6] = ImAp  # d(ImA)/dh
        dy[7] = A_pp.imag  # d(ImA')/dh
        dy[8] = ImBp  # d(ImB)/dh
        dy[9] = B_pp.imag  # d(ImB')/dh
        dy[10] = ImCp  # d(ImC)/dh
        dy[11] = C_pp.imag  # d(ImC')/dh

        return dy

    @lru_cache(maxsize=128)
    def characteristic_function(
        self,
        s: float,
        h0: float,
        lambda_X0: float,
        lambda_Y0: float,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> complex:
        """
        Compute the characteristic function φ(s) = E[exp(isτ)].

        Args:
            s: Frequency parameter
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            Complex characteristic function value
        """
        # Initial conditions: all derivatives set to small non-zero values
        y0 = np.zeros(12)
        y0[1] = 1e-10  # ReA'
        y0[3] = 0  # ReB'
        y0[5] = 0  # ReC'

        try:
            sol = solve_ivp(
                fun=lambda h, y: self.ode_system_real(h, y, s),
                t_span=[0, h0],
                y0=y0,
                method="BDF",
                rtol=rtol,
                atol=atol,
                max_step=min(h0 / 100, 0.1),
            )

            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")

            # Extract final values
            ReA, ReB, ReC = sol.y[0, -1], sol.y[2, -1], sol.y[4, -1]
            ImA, ImB, ImC = sol.y[6, -1], sol.y[8, -1], sol.y[10, -1]

            A = ReA + 1j * ImA
            B = ReB + 1j * ImB
            C = ReC + 1j * ImC

            # Compute characteristic function
            phi_value = np.exp(1j * s * A + B * lambda_X0 + C * lambda_Y0)
            return phi_value

        except Exception as e:
            warnings.warn(f"CF computation failed for s={s}: {str(e)}")
            return complex(np.nan)

    @lru_cache(maxsize=32)
    def batch_characteristic_function(
        self,
        s_tuple: tuple,
        h0: float,
        lambda_X0: float,
        lambda_Y0: float,
        rtol: float = 1e-8,
    ) -> tuple:
        """Batch computation of CF for multiple s values (cached version)."""
        s_array = np.array(s_tuple)
        phi_array = np.zeros(len(s_array), dtype=complex)

        for i, s in enumerate(s_array):
            try:
                phi_array[i] = self.characteristic_function(
                    s, h0, lambda_X0, lambda_Y0, rtol
                )
            except:
                phi_array[i] = complex(np.nan)

        return tuple(phi_array)

    def clear_cache(self) -> None:
        """Clear all LRU caches to free memory."""
        self.characteristic_function.cache_clear()
        self.batch_characteristic_function.cache_clear()

    def cache_info(self) -> Dict:
        """Get cache statistics for all cached methods."""
        return {
            "characteristic_function": self.characteristic_function.cache_info(),
            "batch_characteristic_function": self.batch_characteristic_function.cache_info(),
        }

    def validate_solution(
        self, s: float, h0: float, lambda_X0: float, lambda_Y0: float
    ) -> Dict:
        """
        Validate the CF solution by checking basic properties.

        Args:
            s: Frequency parameter to test
            h0: Health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity

        Returns:
            Dictionary with validation results
        """
        try:
            # Test if characteristic function can be computed
            phi = self.characteristic_function(s, h0, lambda_X0, lambda_Y0)

            # Basic sanity checks
            if np.isnan(phi) or np.isinf(phi):
                return {
                    "valid": False,
                    "error": "Characteristic function is NaN or Inf",
                }

            # Check if |φ(0)| = 1 (normalization)
            if abs(s) < 1e-6:
                phi_0 = self.characteristic_function(0, h0, lambda_X0, lambda_Y0)
                if abs(abs(phi_0) - 1.0) > 1e-6:
                    return {
                        "valid": False,
                        "error": f"φ(0) normalization failed: |φ(0)| = {abs(phi_0):.8f}",
                    }

            return {"valid": True, "phi_value": phi, "phi_magnitude": abs(phi)}

        except Exception as e:
            return {"valid": False, "error": str(e)}


class FirstHittingTimeDistribution:
    """
    Complete first hitting time distribution computation using CF and Gil-Pelaez inversion.

    This class provides the interface between the CF solver and the Gil-Pelaez inversion
    algorithm to compute CDFs and PDFs of first hitting times.
    """

    def __init__(self, cf_solver: CFFirstHittingTime, adaptive_params: bool = True):
        """
        Initialize distribution solver.

        Args:
            cf_solver: CF solver instance
            adaptive_params: Whether to use adaptive parameters for Gil-Pelaez
        """
        self.cf_solver = cf_solver
        self.gil_pelaez = GilPelaezInversion(adaptive_params=adaptive_params)

    @lru_cache(maxsize=256)
    def gil_pelaez_cdf(
        self,
        T: float,
        h0: float,
        lambda_X0: float,
        lambda_Y0: float,
        s_max: Optional[float] = None,
        n_points: Optional[int] = None,
    ) -> float:
        """
        Compute first hitting time CDF P(τ ≤ T) using CF and Gil-Pelaez inversion.

        Args:
            T: Evaluation time
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            s_max: Integration upper limit (None for adaptive)
            n_points: Number of discretization points (None for adaptive)

        Returns:
            CDF value at time T
        """

        def char_function(s):
            """Characteristic function φ(s)"""
            try:
                phi = self.cf_solver.characteristic_function(
                    s, h0, lambda_X0, lambda_Y0
                )
                return phi if np.isfinite(phi) else complex(0.0)
            except:
                return complex(0.0)

        # Use Gil-Pelaez inversion
        cdf_value = self.gil_pelaez.invert(char_function, T, h0, s_max, n_points)

        # Ensure CDF is in [0,1]
        return np.clip(cdf_value, 0.0, 1.0)

    def first_passage_pdf(
        self, T: float, h0: float, lambda_X0: float, lambda_Y0: float, dT: float = 1e-4
    ) -> float:
        """
        Compute first hitting time PDF using numerical differentiation.

        Args:
            T: Evaluation time
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            dT: Step size for numerical differentiation

        Returns:
            PDF value at time T
        """
        if T <= dT:
            # Use forward difference at T=0
            cdf_T = self.gil_pelaez_cdf(T + dT, h0, lambda_X0, lambda_Y0)
            cdf_0 = 0.0
            return (cdf_T - cdf_0) / dT
        else:
            # Use central difference
            cdf_plus = self.gil_pelaez_cdf(T + dT / 2, h0, lambda_X0, lambda_Y0)
            cdf_minus = self.gil_pelaez_cdf(T - dT / 2, h0, lambda_X0, lambda_Y0)
            return (cdf_plus - cdf_minus) / dT

    @lru_cache(maxsize=256)
    def survival_function(
        self, T: float, h0: float, lambda_X0: float, lambda_Y0: float
    ) -> float:
        """
        Compute survival function P(τ > T) = 1 - F(T).

        Args:
            T: Evaluation time
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity

        Returns:
            Survival probability at time T
        """
        return 1.0 - self.gil_pelaez_cdf(T, h0, lambda_X0, lambda_Y0)

    def clear_cache(self) -> None:
        """Clear all LRU caches to free memory."""
        self.gil_pelaez_cdf.cache_clear()
        self.survival_function.cache_clear()
        self.cf_solver.clear_cache()

    def cache_info(self) -> Dict:
        """Get cache statistics for all cached methods."""
        return {
            "gil_pelaez_cdf": self.gil_pelaez_cdf.cache_info(),
            "survival_function": self.survival_function.cache_info(),
            "cf_solver": self.cf_solver.cache_info(),
        }


def create_example_parameters() -> HawkesParameters:
    """Create example parameter set for testing using existing HawkesParameters structure."""
    return HawkesParameters(
        # Health factor and diffusion
        h0=1.25,
        sigma_h=0.25,
        # Drift parameters
        mu_X=0.04,  # collateral drift
        mu_Y=0.02,  # borrowed asset drift
        # Jump size parameters
        eta_X=5.0,  # jump rate
        eta_Y=5.0,
        delta_X=0.05,  # minimum jump size
        delta_Y=0.05,
        # Hawkes intensity parameters
        mu_X_lambda=0.1,  # baseline intensity
        mu_Y_lambda=0.1,
        beta_X=2.0,  # mean reversion
        beta_Y=2.0,
        alpha_XX=0.3,  # self-excitation
        alpha_XY=0.1,  # cross-excitation
        alpha_YX=0.1,
        alpha_YY=0.3,
        # Initial intensities
        lambda_X0=0.1,
        lambda_Y0=0.1,
    )


def validate_implementation():
    """Validation function for the implementation"""
    print("Validating CF First Hitting Time Implementation...")
    print("=" * 60)

    # Create example parameters using existing structure
    parameters = create_example_parameters()

    # Initialize solver
    cf_solver = CFFirstHittingTime(parameters)

    # Test CF computation
    h0 = 0.5
    lambda_X0 = parameters.lambda_X0
    lambda_Y0 = parameters.lambda_Y0

    print(f"Initial conditions: h0={h0}, λ_X0={lambda_X0}, λ_Y0={lambda_Y0}")

    # Test different s values
    s_values = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for s in s_values:
        phi_val = cf_solver.characteristic_function(s, h0, lambda_X0, lambda_Y0)
        if np.isfinite(phi_val):
            print(f"φ({s}) = {phi_val:.6f}")
            results[f"phi_{s}"] = {
                "real": float(np.real(phi_val)),
                "imag": float(np.imag(phi_val)),
            }
        else:
            print(f"φ({s}) = Failed")
            results[f"phi_{s}"] = None

    # Test distribution computation
    print("\nTesting distribution computation...")
    dist_solver = FirstHittingTimeDistribution(cf_solver)

    T_values = [0.5, 1.0, 2.0, 5.0]
    distribution_results = {}

    for T in T_values:
        cdf_val = dist_solver.gil_pelaez_cdf(T, h0, lambda_X0, lambda_Y0)
        pdf_val = dist_solver.first_passage_pdf(T, h0, lambda_X0, lambda_Y0)
        print(f"T={T}: F(T)={cdf_val:.6f}, f(T)={pdf_val:.6f}")
        distribution_results[f"T_{T}"] = {"cdf": cdf_val, "pdf": pdf_val}

    # Create summary statistics
    cdf_values = [
        distribution_results[key]["cdf"] for key in distribution_results.keys()
    ]
    cdf_stats = create_summary_statistics(np.array(cdf_values), "CDF_values")

    # Save results using existing utility
    all_results = {
        "parameters": parameters.__dict__,
        "cf_values": results,
        "distribution_values": distribution_results,
        "summary_statistics": cdf_stats,
        "cache_info": {
            "cf_solver": cf_solver.cache_info(),
            "dist_solver": dist_solver.cache_info(),
        },
    }

    try:
        save_results(all_results, "results/cf_validation_results")
        print(
            "Validation completed. Results saved to results/cf_validation_results.yaml"
        )
    except Exception as e:
        print(f"Validation completed. Could not save results: {e}")

    return all_results


if __name__ == "__main__":
    validate_implementation()
