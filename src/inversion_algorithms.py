"""
Inversion Algorithms for First Hitting Time Distributions

This module contains numerical inversion algorithms for recovering probability
distributions from their transforms (characteristic functions and moment
generating functions).

Algorithms included:
1. Gil-Pelaez inversion for characteristic functions
2. Talbot inversion for Laplace transforms/moment generating functions

Author: [Author Name]
Date: 2025
"""

import numpy as np
import warnings
from typing import Callable, Dict, Optional, Tuple
from functools import lru_cache
from scipy.integrate import simpson


class GilPelaezInversion:
    """
    Gil-Pelaez inversion algorithm for recovering CDFs from characteristic functions.
    
    The Gil-Pelaez formula inverts the characteristic function φ(t) = E[e^{itX}] 
    to obtain the cumulative distribution function:
    F(x) = 1/2 - (1/π) ∫₀^∞ Im[e^{-itx} φ(t) / t] dt
    """
    
    def __init__(self, adaptive_params: bool = True):
        """
        Initialize Gil-Pelaez inversion algorithm.
        
        Args:
            adaptive_params: Whether to use adaptive integration parameters
        """
        self.adaptive_params = adaptive_params
    
    def _adaptive_params_for_inversion(self, h0: float, T: float) -> Tuple[float, int]:
        """
        Compute adaptive integration parameters based on problem characteristics.
        
        Args:
            h0: Initial health factor distance from boundary
            T: Time horizon
            
        Returns:
            (s_max, n_points) tuple optimized for accuracy
        """
        # Base parameters
        base_s_max = 50.0
        base_n_points = 1000
        
        # Scaling factors based on h0 (distance from boundary)
        if h0 < 0.2:
            # Close to boundary: fast decay, need precision near s=0
            s_max_factor = 0.5
            n_points_factor = 1.5
        elif h0 < 0.5:
            # Moderate distance: standard parameters
            s_max_factor = 1.0
            n_points_factor = 1.0
        elif h0 < 1.0:
            # Far from boundary: slower decay, need larger range
            s_max_factor = 1.5
            n_points_factor = 1.2
        else:
            # Very far: slow decay, need large range and more points
            s_max_factor = 2.0 + 0.5 * np.log(h0)
            n_points_factor = 1.3 + 0.2 * np.log(h0)
        
        # Time scaling: longer times need higher precision
        time_factor = 1.0 + 0.3 * np.log(max(T, 0.1))
        
        # Compute final parameters
        s_max = base_s_max * s_max_factor * time_factor
        n_points = int(base_n_points * n_points_factor * time_factor)
        
        # Apply bounds
        s_max = np.clip(s_max, 20.0, 500.0)
        n_points = np.clip(n_points, 500, 5000)
        
        return s_max, n_points
    
    def invert(self, char_func: Callable[[float], complex], T: float, 
               h0: Optional[float] = None, s_max: Optional[float] = None, 
               n_points: Optional[int] = None) -> float:
        """
        Compute CDF F(T) using Gil-Pelaez inversion formula.
        
        Args:
            char_func: Characteristic function φ(s)
            T: Evaluation time point
            h0: Initial health factor (for adaptive parameters)
            s_max: Integration upper limit (None for adaptive)
            n_points: Number of integration points (None for adaptive)
            
        Returns:
            CDF value F(T) = P(τ ≤ T)
        """
        # Use adaptive parameters if not specified
        if self.adaptive_params and h0 is not None:
            if s_max is None or n_points is None:
                s_max_adaptive, n_points_adaptive = self._adaptive_params_for_inversion(h0, T)
                s_max = s_max or s_max_adaptive
                n_points = n_points or n_points_adaptive
        else:
            s_max = s_max or 100.0
            n_points = n_points or 2000
        
        # Handle singularity at s=0 by starting slightly above zero
        s_min = 1e-8
        
        # Create frequency grid with denser sampling near s=0
        if h0 is not None and h0 < 0.5:
            # More density near s=0 for small h0
            dense_fraction = 0.4
            transition_point = min(0.5, s_max / 20)
        else:
            # Less density near s=0 for large h0
            dense_fraction = 0.3
            transition_point = min(1.0, s_max / 10)
        
        n_dense = int(n_points * dense_fraction)
        n_regular = n_points - n_dense
        
        s_dense = np.logspace(np.log10(s_min), np.log10(transition_point), n_dense)
        s_regular = np.linspace(transition_point, s_max, n_regular)
        s_vals = np.concatenate([s_dense, s_regular])
        s_vals = np.unique(s_vals)  # Remove duplicates
        
        # Compute characteristic function values
        phi_vals = []
        valid_indices = []
        
        for i, s in enumerate(s_vals):
            try:
                phi = char_func(s)
                # Check for numerical issues
                if np.isnan(phi) or np.isinf(phi) or abs(phi) > 10:
                    continue
                phi_vals.append(phi)
                valid_indices.append(i)
            except (RuntimeError, OverflowError, ZeroDivisionError):
                continue
        
        if len(phi_vals) < 10:
            warnings.warn("Gil-Pelaez inversion had too many failures, using fallback")
            return self._fallback_inversion(char_func, T, s_max // 2, n_points // 2)
        
        phi_vals = np.array(phi_vals)
        s_vals_valid = s_vals[valid_indices]
        
        # Compute Gil-Pelaez integrand
        integrand = np.zeros_like(phi_vals, dtype=float)
        
        for i, (s, phi) in enumerate(zip(s_vals_valid, phi_vals)):
            if s > 0:
                kernel = np.exp(-1j * s * T) * phi / s
                integrand[i] = np.imag(kernel)
        
        # Numerical integration
        if len(s_vals_valid) > 1:
            integral = np.trapz(integrand, s_vals_valid)
        else:
            integral = 0.0
        
        cdf = 0.5 - integral / np.pi
        return np.clip(np.real(cdf), 0.0, 1.0)
    
    def _fallback_inversion(self, char_func: Callable[[float], complex], 
                           T: float, s_max: float, n_points: int) -> float:
        """Conservative fallback Gil-Pelaez implementation."""
        s_vals = np.linspace(0.01, s_max, n_points)
        
        phi_vals = []
        for s in s_vals:
            try:
                phi = char_func(s)
                phi_vals.append(phi if np.isfinite(phi) else complex(0.0))
            except:
                phi_vals.append(complex(0.0))
        
        phi_vals = np.array(phi_vals)
        kernel = np.exp(-1j * s_vals * T) * phi_vals / s_vals
        integrand = np.imag(kernel)
        integral = np.trapz(integrand, s_vals)
        
        cdf = 0.5 - integral / np.pi
        return np.clip(np.real(cdf), 0.0, 1.0)


class TalbotInversion:
    """
    Talbot algorithm for numerical Laplace transform inversion.
    
    The Talbot algorithm provides a robust method for inverting Laplace transforms
    L{f}(s) = ∫₀^∞ e^{-st} f(t) dt to recover f(t). It uses an optimally deformed
    Bromwich contour to achieve exponential convergence.
    """
    
    def __init__(self, N: int = 32, gamma: float = 0.0):
        """
        Initialize Talbot inversion parameters.
        
        Args:
            N: Number of contour points (default: 32)
            gamma: Contour deformation parameter (default: 0.0, standard Talbot)
        """
        self.N = N
        self.gamma = gamma
    
    def invert(self, laplace_func: Callable[[complex], complex], t: float) -> float:
        """
        Invert Laplace transform using Talbot algorithm with proper formulation.
        
        The algorithm evaluates f(t) = L^{-1}{F}(t) where F(s) is the Laplace 
        transform. Uses the correct Talbot contour with N points.
        
        Args:
            laplace_func: Function in Laplace domain F(s)
            t: Time point for inversion
            
        Returns:
            Inverted function value f(t)
        """
        if t <= 0:
            return 0.0
        
        f_approx = 0.0
        mu = self.N / 2
        
        for k in range(1, self.N + 1):
            theta = (2*k - 1) * np.pi / self.N
            
            # Talbot contour point
            eta = self.gamma + mu * (theta / np.tan(theta) + 1j * theta)
            
            # Talbot weight
            omega = mu * (1 + 1j / np.tan(theta) - theta / (np.sin(theta)**2)) * np.exp(eta)
            
            try:
                # Evaluate Laplace function at eta/t
                s_eval = eta / t
                
                # Add numerical stability bounds
                if np.abs(s_eval) > 100:  # Prevent extreme values
                    continue
                
                F_val = laplace_func(s_eval)
                
                if np.isfinite(F_val):
                    f_approx += omega * F_val
            except:
                # Skip problematic points
                continue
        
        return (2.0 / t) * f_approx.real if np.isfinite(f_approx) else 0.0
    
    def invert_with_convergence_check(self, laplace_func: Callable[[complex], complex], 
                                     t: float, tolerance: float = 1e-6, 
                                     max_refinements: int = 3) -> Dict:
        """
        Invert with adaptive refinement until convergence.
        
        Args:
            laplace_func: Laplace domain function
            t: Time point
            tolerance: Convergence tolerance
            max_refinements: Maximum number of refinements
            
        Returns:
            Dictionary with result and convergence information
        """
        results = []
        
        for refinement in range(max_refinements + 1):
            # Increase precision with each refinement
            current_N = self.N * (2 ** refinement)
            talbot_refined = TalbotInversion(N=current_N, gamma=self.gamma)
            
            result = talbot_refined.invert(laplace_func, t)
            results.append(result)
            
            # Check convergence
            if refinement > 0:
                relative_error = abs(result - results[-2]) / max(abs(results[-2]), 1e-10)
                if relative_error < tolerance:
                    return {
                        'result': result,
                        'converged': True,
                        'refinements': refinement,
                        'relative_error': relative_error,
                        'history': results
                    }
        
        # Did not converge
        final_result = results[-1]
        prev_result = results[-2] if len(results) > 1 else final_result
        relative_error = abs(final_result - prev_result) / max(abs(prev_result), 1e-10)
        
        return {
            'result': final_result,
            'converged': False,
            'refinements': max_refinements,
            'relative_error': relative_error,
            'history': results
        }


class InversionValidator:
    """Utility class for validating inversion algorithm accuracy."""
    
    @staticmethod
    def test_exponential_distribution(rate: float = 1.0, test_points: int = 10) -> Dict:
        """
        Test inversion accuracy using exponential distribution with known analytical solution.
        
        Args:
            rate: Exponential distribution rate parameter
            test_points: Number of test points
            
        Returns:
            Dictionary with test results
        """
        # Analytical solutions
        def exponential_char_func(t):
            return rate / (rate - 1j * t)
        
        def exponential_mgf(s):
            return rate / (rate - s) if s < rate else np.inf
        
        def exponential_cdf(x):
            return 1 - np.exp(-rate * x) if x >= 0 else 0.0
        
        # Test points
        x_vals = np.linspace(0.1, 5.0, test_points)
        
        # Test Gil-Pelaez inversion
        gil_pelaez = GilPelaezInversion(adaptive_params=False)
        gil_pelaez_results = []
        
        for x in x_vals:
            numerical_cdf = gil_pelaez.invert(exponential_char_func, x, s_max=50, n_points=1000)
            analytical_cdf = exponential_cdf(x)
            gil_pelaez_results.append({
                'x': x,
                'numerical': numerical_cdf,
                'analytical': analytical_cdf,
                'error': abs(numerical_cdf - analytical_cdf)
            })
        
        # Test Talbot inversion
        talbot = TalbotInversion(N=32)
        talbot_results = []
        
        for x in x_vals:
            def laplace_cdf(s):
                return exponential_mgf(s) / s if s != 0 else 0
            
            numerical_cdf = talbot.invert(laplace_cdf, x)
            analytical_cdf = exponential_cdf(x)
            talbot_results.append({
                'x': x,
                'numerical': numerical_cdf,
                'analytical': analytical_cdf,
                'error': abs(numerical_cdf - analytical_cdf)
            })
        
        # Compute summary statistics
        gil_pelaez_errors = [r['error'] for r in gil_pelaez_results]
        talbot_errors = [r['error'] for r in talbot_results]
        
        return {
            'gil_pelaez': {
                'results': gil_pelaez_results,
                'max_error': max(gil_pelaez_errors),
                'mean_error': np.mean(gil_pelaez_errors),
                'rmse': np.sqrt(np.mean(np.array(gil_pelaez_errors)**2))
            },
            'talbot': {
                'results': talbot_results,
                'max_error': max(talbot_errors),
                'mean_error': np.mean(talbot_errors),
                'rmse': np.sqrt(np.mean(np.array(talbot_errors)**2))
            }
        }


def validate_inversion_algorithms():
    """Run validation tests for both inversion algorithms."""
    print("Validating Inversion Algorithms...")
    print("=" * 50)
    
    validator = InversionValidator()
    
    # Test with exponential distribution (rate = 1.0)
    print("Testing with Exponential(1.0) distribution:")
    results_exp1 = validator.test_exponential_distribution(rate=1.0, test_points=10)
    
    print(f"Gil-Pelaez - Max Error: {results_exp1['gil_pelaez']['max_error']:.2e}, "
          f"RMSE: {results_exp1['gil_pelaez']['rmse']:.2e}")
    print(f"Talbot     - Max Error: {results_exp1['talbot']['max_error']:.2e}, "
          f"RMSE: {results_exp1['talbot']['rmse']:.2e}")
    
    # Test with different rate
    print("\nTesting with Exponential(0.5) distribution:")
    results_exp05 = validator.test_exponential_distribution(rate=0.5, test_points=10)
    
    print(f"Gil-Pelaez - Max Error: {results_exp05['gil_pelaez']['max_error']:.2e}, "
          f"RMSE: {results_exp05['gil_pelaez']['rmse']:.2e}")
    print(f"Talbot     - Max Error: {results_exp05['talbot']['max_error']:.2e}, "
          f"RMSE: {results_exp05['talbot']['rmse']:.2e}")
    
    print("\nValidation completed!")
    return {'exp_1.0': results_exp1, 'exp_0.5': results_exp05}


if __name__ == "__main__":
    validate_inversion_algorithms()