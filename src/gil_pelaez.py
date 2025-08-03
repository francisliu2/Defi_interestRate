"""
Gil-Pelaez CDF Inversion Formula Implementation

This module implements the Gil-Pelaez inversion formula for computing cumulative
distribution functions from characteristic functions. This is essential for
computing first passage time probabilities in the Hawkes jump-diffusion model.

The Gil-Pelaez formula states:
F(x) = 1/2 - (1/π) * ∫[0,∞] Im[φ(t) * exp(-itx)] / t dt

Where φ(t) is the characteristic function.
"""

import numpy as np
from scipy.integrate import quad
from typing import Optional, Dict, Callable, Tuple
import warnings
from dataclasses import dataclass
from .riccati_solver import RiccatiSolver


@dataclass
class GilPelaezParameters:
    """Parameters for Gil-Pelaez CDF computation."""
    
    integration_limit: float = 50.0
    n_integration_points: int = 1000
    damping_factor: float = 0.75
    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    max_subdivisions: int = 100


class GilPelaezCDF:
    """
    Gil-Pelaez CDF inversion for Hawkes jump-diffusion first passage times.
    
    This class computes P(h_T ≤ x) using the Gil-Pelaez inversion formula
    applied to the characteristic function obtained from the Riccati solver.
    """
    
    def __init__(self, riccati_solver: RiccatiSolver, 
                 parameters: Optional[GilPelaezParameters] = None):
        """
        Initialize Gil-Pelaez CDF computer.
        
        Args:
            riccati_solver: RiccatiSolver instance for characteristic functions
            parameters: GilPelaezParameters for integration settings
        """
        self.riccati_solver = riccati_solver
        self.params = parameters or GilPelaezParameters()
        
    def _integrand(self, t: float, x: float, tau: float, lambda_X: float, 
                  lambda_Y: float, damping: bool = True) -> float:
        """
        Compute the integrand for Gil-Pelaez formula.
        
        Args:
            t: Integration variable (frequency)
            x: Point at which to evaluate CDF
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            damping: Whether to apply exponential damping
            
        Returns:
            Integrand value
        """
        if t == 0:
            return 0.0
        
        try:
            # Compute characteristic function
            u = complex(0, t)  # Pure imaginary frequency
            phi = self.riccati_solver.characteristic_function(u, tau, lambda_X, lambda_Y)
            
            # Apply exponential damping to improve convergence
            if damping:
                phi *= np.exp(-self.params.damping_factor * t)
            
            # Gil-Pelaez integrand: Im[φ(it) * exp(-itx)] / t
            integrand = np.imag(phi * np.exp(-1j * t * x)) / t
            
            return integrand
            
        except (RuntimeError, ValueError, ZeroDivisionError):
            # Handle numerical issues gracefully
            return 0.0
    
    def cdf_single_point(self, x: float, tau: float, lambda_X: float, 
                        lambda_Y: float, use_damping: bool = True) -> Dict:
        """
        Compute CDF at a single point using Gil-Pelaez formula.
        
        Args:
            x: Point at which to evaluate CDF
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            use_damping: Whether to use exponential damping
            
        Returns:
            Dictionary with CDF value and integration diagnostics
        """
        try:
            # Define integrand function
            def integrand(t):
                return self._integrand(t, x, tau, lambda_X, lambda_Y, use_damping)
            
            # Perform numerical integration
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                integral, error = quad(
                    integrand,
                    0,
                    self.params.integration_limit,
                    epsabs=self.params.absolute_tolerance,
                    epsrel=self.params.relative_tolerance,
                    limit=self.params.max_subdivisions
                )
            
            # Apply Gil-Pelaez formula
            cdf_value = 0.5 - integral / np.pi
            
            # Ensure CDF is in [0, 1]
            cdf_value = np.clip(cdf_value, 0.0, 1.0)
            
            return {
                'cdf': cdf_value,
                'integration_error': error,
                'success': True,
                'n_evaluations': None  # scipy.quad doesn't provide this
            }
            
        except Exception as e:
            return {
                'cdf': np.nan,
                'integration_error': np.inf,
                'success': False,
                'error': str(e)
            }
    
    def cdf_array(self, x_array: np.ndarray, tau: float, lambda_X: float,
                  lambda_Y: float, use_damping: bool = True) -> Dict:
        """
        Compute CDF at multiple points.
        
        Args:
            x_array: Array of points at which to evaluate CDF
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            use_damping: Whether to use exponential damping
            
        Returns:
            Dictionary with CDF array and diagnostics
        """
        n_points = len(x_array)
        cdf_values = np.zeros(n_points)
        integration_errors = np.zeros(n_points)
        success_flags = np.zeros(n_points, dtype=bool)
        
        for i, x in enumerate(x_array):
            result = self.cdf_single_point(x, tau, lambda_X, lambda_Y, use_damping)
            
            cdf_values[i] = result['cdf']
            integration_errors[i] = result['integration_error']
            success_flags[i] = result['success']
        
        return {
            'x': x_array,
            'cdf': cdf_values,
            'integration_errors': integration_errors,
            'success_flags': success_flags,
            'success_rate': np.mean(success_flags)
        }
    
    def survival_function(self, x: float, tau: float, lambda_X: float,
                         lambda_Y: float, use_damping: bool = True) -> Dict:
        """
        Compute survival function S(x) = 1 - F(x).
        
        Args:
            x: Point at which to evaluate survival function
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            use_damping: Whether to use exponential damping
            
        Returns:
            Dictionary with survival function value and diagnostics
        """
        cdf_result = self.cdf_single_point(x, tau, lambda_X, lambda_Y, use_damping)
        
        if cdf_result['success']:
            survival = 1.0 - cdf_result['cdf']
            return {
                'survival': survival,
                'cdf': cdf_result['cdf'],
                'integration_error': cdf_result['integration_error'],
                'success': True
            }
        else:
            return {
                'survival': np.nan,
                'cdf': np.nan,
                'integration_error': np.inf,
                'success': False,
                'error': cdf_result.get('error')
            }
    
    def liquidation_probability(self, tau: float, lambda_X: float, lambda_Y: float,
                              threshold: float = 1.0, use_damping: bool = True) -> Dict:
        """
        Compute liquidation probability P(h_T ≤ threshold).
        
        Args:
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            threshold: Liquidation threshold (default: 1.0)
            use_damping: Whether to use exponential damping
            
        Returns:
            Dictionary with liquidation probability and diagnostics
        """
        return self.cdf_single_point(threshold, tau, lambda_X, lambda_Y, use_damping)
    
    def quantile_function(self, p: float, tau: float, lambda_X: float, lambda_Y: float,
                         x_bounds: Tuple[float, float] = (0.1, 3.0),
                         use_damping: bool = True, tolerance: float = 1e-6,
                         max_iterations: int = 100) -> Dict:
        """
        Compute quantile function (inverse CDF) using bisection method.
        
        Args:
            p: Probability level (0 < p < 1)
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            x_bounds: Search bounds for quantile
            use_damping: Whether to use exponential damping
            tolerance: Convergence tolerance
            max_iterations: Maximum bisection iterations
            
        Returns:
            Dictionary with quantile value and convergence info
        """
        if not (0 < p < 1):
            raise ValueError("Probability p must be in (0, 1)")
        
        x_low, x_high = x_bounds
        
        # Check bounds
        cdf_low = self.cdf_single_point(x_low, tau, lambda_X, lambda_Y, use_damping)['cdf']
        cdf_high = self.cdf_single_point(x_high, tau, lambda_X, lambda_Y, use_damping)['cdf']
        
        if cdf_low > p:
            return {
                'quantile': x_low,
                'success': False,
                'error': f"Lower bound too high: F({x_low}) = {cdf_low} > {p}"
            }
        
        if cdf_high < p:
            return {
                'quantile': x_high,
                'success': False,
                'error': f"Upper bound too low: F({x_high}) = {cdf_high} < {p}"
            }
        
        # Bisection method
        for iteration in range(max_iterations):
            x_mid = (x_low + x_high) / 2
            cdf_mid = self.cdf_single_point(x_mid, tau, lambda_X, lambda_Y, use_damping)['cdf']
            
            if abs(cdf_mid - p) < tolerance:
                return {
                    'quantile': x_mid,
                    'cdf_value': cdf_mid,
                    'target_probability': p,
                    'iterations': iteration + 1,
                    'success': True
                }
            
            if cdf_mid < p:
                x_low = x_mid
            else:
                x_high = x_mid
        
        # Convergence failed
        x_final = (x_low + x_high) / 2
        cdf_final = self.cdf_single_point(x_final, tau, lambda_X, lambda_Y, use_damping)['cdf']
        
        return {
            'quantile': x_final,
            'cdf_value': cdf_final,
            'target_probability': p,
            'iterations': max_iterations,
            'success': False,
            'error': f"Bisection failed to converge after {max_iterations} iterations"
        }
    
    def pdf_single_point(self, x: float, tau: float, lambda_X: float, lambda_Y: float,
                        dx: float = 1e-6, use_damping: bool = True) -> Dict:
        """
        Compute probability density function using numerical differentiation.
        
        Args:
            x: Point at which to evaluate PDF
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            dx: Step size for numerical differentiation
            use_damping: Whether to use exponential damping
            
        Returns:
            Dictionary with PDF value and diagnostics
        """
        # Compute CDF at x + dx and x - dx
        cdf_plus = self.cdf_single_point(x + dx, tau, lambda_X, lambda_Y, use_damping)
        cdf_minus = self.cdf_single_point(x - dx, tau, lambda_X, lambda_Y, use_damping)
        
        if cdf_plus['success'] and cdf_minus['success']:
            # Central difference approximation
            pdf_value = (cdf_plus['cdf'] - cdf_minus['cdf']) / (2 * dx)
            
            return {
                'pdf': max(pdf_value, 0.0),  # Ensure non-negative
                'cdf_plus': cdf_plus['cdf'],
                'cdf_minus': cdf_minus['cdf'],
                'dx': dx,
                'success': True
            }
        else:
            return {
                'pdf': np.nan,
                'success': False,
                'error': "CDF evaluation failed"
            }
    
    def validate_cdf_properties(self, tau: float, lambda_X: float, lambda_Y: float,
                              x_test: np.ndarray = None) -> Dict:
        """
        Validate that computed CDF satisfies basic properties.
        
        Args:
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            x_test: Test points (if None, use default range)
            
        Returns:
            Dictionary with validation results
        """
        if x_test is None:
            x_test = np.linspace(0.1, 3.0, 20)
        
        cdf_result = self.cdf_array(x_test, tau, lambda_X, lambda_Y)
        
        if cdf_result['success_rate'] < 0.9:
            return {
                'valid': False,
                'error': f"Low success rate: {cdf_result['success_rate']:.2%}"
            }
        
        cdf_values = cdf_result['cdf']
        
        # Check monotonicity
        is_monotonic = np.all(np.diff(cdf_values) >= -1e-6)  # Allow small numerical errors
        
        # Check bounds
        in_bounds = np.all((cdf_values >= -1e-6) & (cdf_values <= 1 + 1e-6))
        
        # Check limits
        cdf_low = self.cdf_single_point(x_test[0], tau, lambda_X, lambda_Y)['cdf']
        cdf_high = self.cdf_single_point(x_test[-1], tau, lambda_X, lambda_Y)['cdf']
        
        return {
            'valid': is_monotonic and in_bounds,
            'monotonic': is_monotonic,
            'in_bounds': in_bounds,
            'cdf_range': (cdf_low, cdf_high),
            'mean_integration_error': np.mean(cdf_result['integration_errors']),
            'max_integration_error': np.max(cdf_result['integration_errors'])
        }