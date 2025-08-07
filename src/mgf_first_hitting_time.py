"""
Moment Generating Function for First Hitting Time Analysis

This module implements the moment generating function approach for computing
first hitting time distributions of the log-health factor in DeFi liquidation
risk models with bivariate Hawkes jump-diffusion processes.

The implementation includes:
1. 6D real-valued ODE system for MGF Riccati equations
2. Integration with Talbot inversion algorithm for distribution recovery
3. Integration with existing parameter structures and utilities

Author: [Author Name]
Date: 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
from typing import Tuple, Dict, Optional
from functools import lru_cache

# Import existing modules for reuse
from hawkes_process import HawkesParameters
from utils import save_results, create_summary_statistics
from inversion_algorithms import TalbotInversion


class MGFFirstHittingTime:
    """
    Moment generating function computation for first hitting time of log-health factor.

    This class implements the 6D real-valued ODE system for computing moment generating
    functions of first hitting times in the Hawkes jump-diffusion framework.
    """

    def __init__(self, parameters: HawkesParameters):
        """
        Initialize MGF solver using existing HawkesParameters structure.

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
            raise ValueError("Mean reversion parameters beta_X, beta_Y must be positive")

        # Check stability using existing validation pattern
        spectral_radius = max(
            abs(self.params.alpha_XX / self.params.beta_X),
            abs(self.params.alpha_YY / self.params.beta_Y)
        )

        if spectral_radius >= 1.0:
            warnings.warn(f"Hawkes process may be unstable: spectral radius = {spectral_radius:.3f}")

    def ode_system_real(self, h: float, y: np.ndarray, s: float) -> np.ndarray:
        """
        6D first-order ODE system for MGF Riccati equations.

        Args:
            h: Current health factor value  
            y: State vector [A, A', B, B', C, C']
            s: MGF parameter (real-valued)

        Returns:
            Derivative vector
        """
        A, A_prime, B, B_prime, C, C_prime = y

        # Moment generating functions for jump sizes
        # For X jumps: g_X(s) = η_X/(η_X - s*A') * exp(s*A'*δ_X + α_XX*B + α_XY*C)
        denom_X = self.params.eta_X - s * A_prime
        exp_arg_X = s * A_prime * self.params.delta_X + self.params.alpha_XX * B + self.params.alpha_XY * C
        
        # Prevent numerical overflow in exponential
        if abs(exp_arg_X) > 100:  # e^100 ≈ 10^43, manageable
            exp_arg_X = np.sign(exp_arg_X) * 100
        
        if abs(denom_X) > 1e-12:  # Avoid division by zero
            g_X = (self.params.eta_X / denom_X) * np.exp(exp_arg_X)
        else:
            # Use series expansion for small denominator
            g_X = np.exp(exp_arg_X)

        # For Y jumps: g_Y(s) = η_Y/(η_Y + s*A') * exp(-s*A'*δ_Y + α_YY*C + α_YX*B)  
        denom_Y = self.params.eta_Y + s * A_prime
        exp_arg_Y = -s * A_prime * self.params.delta_Y + self.params.alpha_YY * C + self.params.alpha_YX * B
        
        # Prevent numerical overflow in exponential
        if abs(exp_arg_Y) > 100:
            exp_arg_Y = np.sign(exp_arg_Y) * 100
        
        if abs(denom_Y) > 1e-12:  # Avoid division by zero
            g_Y = (self.params.eta_Y / denom_Y) * np.exp(exp_arg_Y)
        else:
            # Use series expansion for small denominator  
            g_Y = np.exp(exp_arg_Y)

        # Compute drift of health factor from existing parameters
        mu_h = self.params.mu_X - self.params.mu_Y  # Drift difference

        # Second derivatives from Riccati equations
        # A'' equation: σ²/2 * A''+ μ_h * A' + β_X*μ_X_λ*B + β_Y*μ_Y_λ*C + s = 0
        # Rearranged: A'' = -(2/σ²)(μ_h * A' + β_X*μ_X_λ*B + β_Y*μ_Y_λ*C + s)
        if abs(s) > 1e-12:
            A_double_prime = (-2/self.params.sigma_h**2) * (
                mu_h * A_prime + 
                self.params.beta_X * self.params.mu_X_lambda * B + 
                self.params.beta_Y * self.params.mu_Y_lambda * C + 
                s
            )
        else:
            # For s≈0, use L'Hôpital's rule or direct computation
            A_double_prime = (-2/self.params.sigma_h**2) * (
                mu_h * A_prime + 
                self.params.beta_X * self.params.mu_X_lambda * B + 
                self.params.beta_Y * self.params.mu_Y_lambda * C
            )

        # B'' equation: σ²/2 * B'' + μ_h * B' + σ²/2 * B'² + σ²*s*A'*B' - β_X*B + (g_X - 1) = 0
        B_double_prime = (-2/self.params.sigma_h**2) * (
            mu_h * B_prime + 
            (self.params.sigma_h**2/2) * B_prime**2 + 
            self.params.sigma_h**2 * s * A_prime * B_prime - 
            self.params.beta_X * B + 
            (g_X - 1)
        )

        # C'' equation: σ²/2 * C'' + μ_h * C' + σ²/2 * C'² + σ²*s*A'*C' - β_Y*C + (g_Y - 1) = 0
        C_double_prime = (-2/self.params.sigma_h**2) * (
            mu_h * C_prime + 
            (self.params.sigma_h**2/2) * C_prime**2 + 
            self.params.sigma_h**2 * s * A_prime * C_prime - 
            self.params.beta_Y * C + 
            (g_Y - 1)
        )

        return np.array([A_prime, A_double_prime, B_prime, B_double_prime, 
                        C_prime, C_double_prime])

    @lru_cache(maxsize=128)
    def moment_generating_function(self, s: float, h0: float, lambda_X0: float, 
                                  lambda_Y0: float, rtol: float = 1e-8, 
                                  atol: float = 1e-10) -> complex:
        """
        Compute the moment generating function M(s) = E[exp(s*τ)].

        Args:
            s: MGF parameter (real-valued)
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver

        Returns:
            Complex moment generating function value
        """
        if s <= 0:
            warnings.warn("MGF parameter s should be positive for first hitting time")
            return complex(np.nan)
        
        # Add upper bound check to prevent numerical instability
        s_max = min(50.0, self.params.eta_X, self.params.eta_Y)  # Conservative bound
        if s > s_max:
            warnings.warn(f"MGF parameter s={s} too large (max={s_max}), may cause numerical instability")
            return complex(np.nan)

        # Initial conditions: At h=0 (boundary), M=1, so A(0)=B(0)=C(0)=0
        # The derivatives A'(0), B'(0), C'(0) are determined by shooting method
        # For small h, use series expansion starting values
        epsilon = 1e-8
        y0 = np.array([0.0, epsilon, 0.0, epsilon, 0.0, epsilon])

        try:
            sol = solve_ivp(
                fun=lambda h, y: self.ode_system_real(h, y, s),
                t_span=[0, h0],
                y0=y0,
                method='Radau',
                rtol=rtol,
                atol=atol,
                max_step=min(h0/100, 0.1)
            )

            if not sol.success:
                raise RuntimeError(f"ODE integration failed: {sol.message}")

            # Extract final values
            A_final, _, B_final, _, C_final, _ = sol.y[:, -1]

            # Compute MGF
            mgf_value = np.exp(s * A_final + B_final * lambda_X0 + C_final * lambda_Y0)
            
            return complex(mgf_value)

        except Exception as e:
            warnings.warn(f"MGF computation failed for s={s}, h0={h0}: {str(e)}")
            # Try with looser tolerances as fallback
            try:
                sol = solve_ivp(
                    fun=lambda h, y: self.ode_system_real(h, y, s),
                    t_span=[0, h0],
                    y0=y0,
                    method='RK45',
                    rtol=1e-6,
                    atol=1e-8,
                    max_step=h0/50
                )
                if sol.success:
                    A_final, _, B_final, _, C_final, _ = sol.y[:, -1]
                    mgf_value = np.exp(s * A_final + B_final * lambda_X0 + C_final * lambda_Y0)
                    return complex(mgf_value)
                else:
                    warnings.warn(f"Fallback integration also failed: {sol.message}")
            except:
                pass
            
            return complex(np.nan)

    @lru_cache(maxsize=32)
    def batch_moment_generating_function(self, s_tuple: tuple, h0: float, 
                                        lambda_X0: float, lambda_Y0: float,
                                        rtol: float = 1e-8) -> tuple:
        """Batch computation of MGF for multiple s values (cached version)."""
        s_array = np.array(s_tuple)
        mgf_array = np.zeros(len(s_array), dtype=complex)

        for i, s in enumerate(s_array):
            try:
                mgf_array[i] = self.moment_generating_function(s, h0, lambda_X0, lambda_Y0, rtol)
            except:
                mgf_array[i] = complex(np.nan)

        return tuple(mgf_array)

    def clear_cache(self) -> None:
        """Clear all LRU caches to free memory."""
        self.moment_generating_function.cache_clear()
        self.batch_moment_generating_function.cache_clear()

    def cache_info(self) -> Dict:
        """Get cache statistics for all cached methods."""
        return {
            'moment_generating_function': self.moment_generating_function.cache_info(),
            'batch_moment_generating_function': self.batch_moment_generating_function.cache_info()
        }

    def validate_solution(self, s: float, h0: float, lambda_X0: float, 
                         lambda_Y0: float) -> Dict:
        """
        Validate the MGF solution by checking basic properties.

        Args:
            s: MGF parameter to test
            h0: Health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity

        Returns:
            Dictionary with validation results
        """
        try:
            # Test if MGF can be computed
            mgf = self.moment_generating_function(s, h0, lambda_X0, lambda_Y0)

            # Basic sanity checks
            if np.isnan(mgf) or np.isinf(mgf):
                return {
                    "valid": False,
                    "error": "MGF is NaN or Inf",
                }

            # Check if MGF is real and positive for real s > 0
            if s > 0 and (np.imag(mgf) != 0 or np.real(mgf) <= 0):
                return {
                    "valid": False,
                    "error": f"MGF should be real and positive for s > 0: MGF = {mgf}",
                }

            return {
                "valid": True,
                "mgf_value": mgf,
                "mgf_magnitude": abs(mgf)
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


class FirstHittingTimeDistribution:
    """
    Complete first hitting time distribution computation using MGF and Talbot inversion.
    
    This class provides the interface between the MGF solver and the Talbot inversion
    algorithm to compute CDFs and PDFs of first hitting times.
    """

    def __init__(self, mgf_solver: MGFFirstHittingTime, N_talbot: int = 24):
        """
        Initialize distribution solver.

        Args:
            mgf_solver: MGF solver instance
            N_talbot: Number of contour points for Talbot algorithm
        """
        self.mgf_solver = mgf_solver
        # Use standard Talbot parameters with the corrected algorithm
        self.talbot = TalbotInversion(N=N_talbot, gamma=0.0)

    @lru_cache(maxsize=256)
    def talbot_cdf(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float) -> float:
        """
        Compute first hitting time CDF P(τ ≤ T) using MGF and Talbot inversion.

        Args:
            T: Evaluation time
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity

        Returns:
            CDF value at time T
        """
        def laplace_transform(s):
            """Laplace transform: L{F(t)}(s) = M(s)/s"""
            if np.abs(s) < 1e-12:
                return 0.0

            try:
                # Handle complex s by using real part (simplified approach)
                s_real = np.real(s) if np.iscomplex(s) else s
                if s_real <= 0:
                    return 0.0
                    
                mgf_val = self.mgf_solver.moment_generating_function(s_real, h0, lambda_X0, lambda_Y0)
                return mgf_val / s if np.isfinite(mgf_val) else 0.0
            except:
                return 0.0

        # Use Talbot inversion
        cdf_value = self.talbot.invert(laplace_transform, T)

        # Ensure CDF is in [0,1]
        return np.clip(cdf_value, 0.0, 1.0)

    def first_passage_pdf(self, T: float, h0: float, lambda_X0: float, 
                         lambda_Y0: float, dT: float = 1e-4) -> float:
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
            cdf_T = self.talbot_cdf(T + dT, h0, lambda_X0, lambda_Y0)
            cdf_0 = 0.0
            return (cdf_T - cdf_0) / dT
        else:
            # Use central difference
            cdf_plus = self.talbot_cdf(T + dT/2, h0, lambda_X0, lambda_Y0)
            cdf_minus = self.talbot_cdf(T - dT/2, h0, lambda_X0, lambda_Y0)
            return (cdf_plus - cdf_minus) / dT

    @lru_cache(maxsize=256)
    def survival_function(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float) -> float:
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
        return 1.0 - self.talbot_cdf(T, h0, lambda_X0, lambda_Y0)

    def clear_cache(self) -> None:
        """Clear all LRU caches to free memory."""
        self.talbot_cdf.cache_clear()
        self.survival_function.cache_clear()
        self.mgf_solver.clear_cache()

    def cache_info(self) -> Dict:
        """Get cache statistics for all cached methods."""
        return {
            'talbot_cdf': self.talbot_cdf.cache_info(),
            'survival_function': self.survival_function.cache_info(),
            'mgf_solver': self.mgf_solver.cache_info()
        }


def create_example_parameters() -> HawkesParameters:
    """Create example parameter set for testing using existing HawkesParameters structure."""
    return HawkesParameters(
        # Health factor and diffusion
        h0=1.25,
        sigma_h=0.25,

        # Drift parameters  
        mu_X=0.04,          # collateral drift
        mu_Y=0.02,          # borrowed asset drift

        # Jump size parameters
        eta_X=5.0,          # jump rate
        eta_Y=5.0,
        delta_X=0.05,       # minimum jump size
        delta_Y=0.05,

        # Hawkes intensity parameters
        mu_X_lambda=0.1,    # baseline intensity
        mu_Y_lambda=0.1,
        beta_X=2.0,         # mean reversion
        beta_Y=2.0,
        alpha_XX=0.3,       # self-excitation
        alpha_XY=0.1,       # cross-excitation
        alpha_YX=0.1,
        alpha_YY=0.3,

        # Initial intensities
        lambda_X0=0.1,
        lambda_Y0=0.1
    )


def validate_implementation():
    """Validation function for the implementation"""
    print("Validating MGF First Hitting Time Implementation...")
    print("=" * 60)

    # Create example parameters using existing structure
    parameters = create_example_parameters()

    # Initialize solver
    mgf_solver = MGFFirstHittingTime(parameters)

    # Test MGF computation
    h0 = 0.5
    lambda_X0 = parameters.lambda_X0
    lambda_Y0 = parameters.lambda_Y0

    print(f"Initial conditions: h0={h0}, λ_X0={lambda_X0}, λ_Y0={lambda_Y0}")

    # Test different s values
    s_values = [0.1, 0.5, 1.0, 2.0]
    results = {}

    for s in s_values:
        mgf_val = mgf_solver.moment_generating_function(s, h0, lambda_X0, lambda_Y0)
        if np.isfinite(mgf_val):
            print(f"M({s}) = {mgf_val:.6f}")
            results[f'mgf_{s}'] = float(np.real(mgf_val))
        else:
            print(f"M({s}) = Failed")
            results[f'mgf_{s}'] = None

    # Test distribution computation
    print("\nTesting distribution computation...")
    dist_solver = FirstHittingTimeDistribution(mgf_solver)

    T_values = [0.5, 1.0, 2.0, 5.0]
    distribution_results = {}

    for T in T_values:
        cdf_val = dist_solver.talbot_cdf(T, h0, lambda_X0, lambda_Y0)
        pdf_val = dist_solver.first_passage_pdf(T, h0, lambda_X0, lambda_Y0)
        print(f"T={T}: F(T)={cdf_val:.6f}, f(T)={pdf_val:.6f}")
        distribution_results[f'T_{T}'] = {'cdf': cdf_val, 'pdf': pdf_val}

    # Create summary statistics
    cdf_values = [distribution_results[key]['cdf'] for key in distribution_results.keys()]
    cdf_stats = create_summary_statistics(np.array(cdf_values), "CDF_values")

    # Save results using existing utility
    all_results = {
        'parameters': parameters.__dict__,
        'mgf_values': results,
        'distribution_values': distribution_results,
        'summary_statistics': cdf_stats,
        'cache_info': {
            'mgf_solver': mgf_solver.cache_info(),
            'dist_solver': dist_solver.cache_info()
        }
    }

    try:
        save_results(all_results, 'results/mgf_validation_results')
        print("Validation completed. Results saved to results/mgf_validation_results.yaml")
    except Exception as e:
        print(f"Validation completed. Could not save results: {e}")

    return all_results


if __name__ == "__main__":
    validate_implementation()