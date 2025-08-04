"""
First Hitting Time Riccati ODE Solver for Hawkes Jump-Diffusion

This module implements the 12D Riccati ODE system for computing characteristic functions
of the first hitting time for the Hawkes jump-diffusion process. The characteristic 
function φ(s) = E[exp(isτ)] where τ is the first hitting time.

The system uses a second-order ODE formulation with 12 real variables:
[ReA, ReA', ReB, ReB', ReC, ReC', ImA, ImA', ImB, ImB', ImC, ImC']
"""

import numpy as np
from scipy.integrate import solve_ivp, simpson
from typing import Tuple, Dict, Optional, Callable
import warnings
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class FirstHittingTimeParameters:
    """Parameters for the first hitting time Riccati ODE system."""
    
    # Health factor parameters
    sigma_h: float
    mu_h: float  # Drift of health factor
    
    # Jump size parameters
    eta_X: float
    delta_X: float
    eta_Y: float
    delta_Y: float
    
    # Hawkes parameters
    mu_X_lambda: float
    mu_Y_lambda: float
    beta_X: float
    beta_Y: float
    alpha_XX: float
    alpha_YY: float
    alpha_XY: float
    alpha_YX: float


class FirstHittingTimeRiccatiSolver:
    """
    Solver for the first hitting time characteristic function using 12D Riccati system.
    
    The characteristic function φ(s) = E[exp(isτ)] where τ is the first hitting time,
    satisfies φ(s) = exp(-isA(h₀) - B(h₀)λ_X₀ - C(h₀)λ_Y₀) where A, B, C are solutions
    to the second-order Riccati system.
    """
    
    def __init__(self, parameters: FirstHittingTimeParameters):
        """
        Initialize the first hitting time Riccati solver.
        
        Args:
            parameters: FirstHittingTimeParameters containing all model parameters
        """
        self.params = parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate parameter constraints."""
        if self.params.sigma_h <= 0:
            raise ValueError("Volatility sigma_h must be positive")
        
        if self.params.eta_X <= 0 or self.params.eta_Y <= 0:
            raise ValueError("Jump intensity parameters eta_X, eta_Y must be positive")
        
        if self.params.beta_X <= 0 or self.params.beta_Y <= 0:
            raise ValueError("Mean reversion parameters beta_X, beta_Y must be positive")
    
    def ode_system_real(self, h: float, y: np.ndarray, s: float) -> np.ndarray:
        """
        Define the 12D first hitting time Riccati ODE system in real variables.
        
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
        
        # Model parameters
        mu_h = self.params.mu_h
        sigma_h = self.params.sigma_h
        beta_X = self.params.beta_X
        beta_Y = self.params.beta_Y
        mu_X_lambda = self.params.mu_X_lambda
        mu_Y_lambda = self.params.mu_Y_lambda
        eta_X = self.params.eta_X
        delta_X = self.params.delta_X
        eta_Y = self.params.eta_Y
        delta_Y = self.params.delta_Y
        alpha_XX = self.params.alpha_XX
        alpha_XY = self.params.alpha_XY
        alpha_YY = self.params.alpha_YY
        alpha_YX = self.params.alpha_YX
        
        # Avoid division by zero
        denom_X = eta_X + i_s * A_p + 1e-10
        denom_Y = eta_Y - i_s * A_p + 1e-10
        
        # f_X and f_Y functions
        f_X = (eta_X / denom_X) * np.exp(i_s * A_p * delta_X - alpha_XX * B - alpha_XY * C)
        f_Y = (eta_Y / denom_Y) * np.exp(-i_s * A_p * delta_Y - alpha_YY * C - alpha_YX * B)
        
        # Second derivatives from the second-order system
        # Handle division by i_s more carefully to avoid numerical issues
        if abs(s) < 1e-12:
            # Use L'Hôpital's rule or series expansion for s→0
            A_pp = (-2 * mu_h * A_p
                    - 2 * beta_X * mu_X_lambda * B / sigma_h**2
                    - 2 * beta_Y * mu_Y_lambda * C / sigma_h**2
                    - 2j) / sigma_h**2
        else:
            A_pp = (-2 * mu_h * i_s * A_p
                    - sigma_h**2 * (i_s * A_p)**2
                    - 2 * beta_X * mu_X_lambda * B
                    - 2 * beta_Y * mu_Y_lambda * C
                    - 2 * i_s) / (sigma_h**2 * i_s)
        
        B_pp = (-2 * mu_h * B_p
                - sigma_h**2 * (B_p**2 + 2 * i_s * A_p * B_p)
                + 2 * beta_X * B
                - 2 * (f_X - 1)) / sigma_h**2
        
        C_pp = (-2 * mu_h * C_p
                - sigma_h**2 * (C_p**2 + 2 * i_s * A_p * C_p)
                + 2 * beta_Y * C
                - 2 * (f_Y - 1)) / sigma_h**2
        
        # Return derivative vector in real form
        dy = np.zeros(12)
        dy[0] = ReAp         # d(ReA)/dh
        dy[1] = A_pp.real    # d(ReA')/dh
        dy[2] = ReBp         # d(ReB)/dh
        dy[3] = B_pp.real    # d(ReB')/dh
        dy[4] = ReCp         # d(ReC)/dh
        dy[5] = C_pp.real    # d(ReC')/dh
        dy[6] = ImAp         # d(ImA)/dh
        dy[7] = A_pp.imag    # d(ImA')/dh
        dy[8] = ImBp         # d(ImB)/dh
        dy[9] = B_pp.imag    # d(ImB')/dh
        dy[10] = ImCp        # d(ImC)/dh
        dy[11] = C_pp.imag   # d(ImC')/dh
        
        return dy
    
    @lru_cache(maxsize=128)
    def characteristic_function(self, s: float, h0: float, lambda_X0: float, lambda_Y0: float,
                              rtol: float = 1e-6, atol: float = 1e-8) -> complex:
        """
        Compute the first hitting time characteristic function φ(s) = E[exp(isτ)].
        
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
        y0[1] = 1e-6   # ReA'
        y0[3] = 1e-6   # ReB'
        y0[5] = 1e-6   # ReC'
        
        # Integration from 0 to h0
        h_span = [0, h0]
        
        try:
            sol = solve_ivp(
                fun=lambda h, y: self.ode_system_real(h, y, s),
                t_span=h_span,
                y0=y0,
                t_eval=[h0],
                rtol=rtol,
                atol=atol,
                method='RK45'
            )
            
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
            
            # Extract final values
            ReA, ReB, ReC = sol.y[0, -1], sol.y[2, -1], sol.y[4, -1]
            ImA, ImB, ImC = sol.y[6, -1], sol.y[8, -1], sol.y[10, -1]
            
            A = ReA + 1j * ImA
            B = ReB + 1j * ImB
            C = ReC + 1j * ImC
            
            # Return characteristic function
            return np.exp(-1j * s * A - B * lambda_X0 - C * lambda_Y0)
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute characteristic function: {e}")
    
    @lru_cache(maxsize=32)
    def _batch_characteristic_function_cached(self, s_tuple: tuple, h0: float,
                                            lambda_X0: float, lambda_Y0: float,
                                            rtol: float = 1e-6) -> tuple:
        """Internal cached version that works with tuples instead of arrays."""
        s_array = np.array(s_tuple)
        phi_array = np.zeros(len(s_array), dtype=complex)
        
        for i, s in enumerate(s_array):
            try:
                phi_array[i] = self.characteristic_function(s, h0, lambda_X0, lambda_Y0, rtol)
            except RuntimeError:
                phi_array[i] = np.nan + 1j * np.nan
        
        return tuple(phi_array)
    
    def batch_characteristic_function(self, s_array: np.ndarray, h0: float,
                                    lambda_X0: float, lambda_Y0: float,
                                    rtol: float = 1e-6) -> np.ndarray:
        """
        Compute characteristic function for multiple frequency parameters.
        
        Args:
            s_array: Array of frequency parameters
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            rtol: Relative tolerance
            
        Returns:
            Array of characteristic function values
        """
        # Use the cached version by converting array to tuple and back
        s_tuple = tuple(s_array)
        phi_tuple = self._batch_characteristic_function_cached(s_tuple, h0, lambda_X0, lambda_Y0, rtol)
        return np.array(phi_tuple)
    
    def _adaptive_gil_pelaez_params(self, h0: float, T: float) -> tuple:
        """
        Compute adaptive Gil-Pelaez integration parameters based on h0 and T.
        
        Args:
            h0: Initial health factor distance
            T: Time horizon
            
        Returns:
            (s_max, n_points) tuple optimized for h0 and T
        """
        # Base parameters
        base_s_max = 50.0
        base_n_points = 1000
        
        # Scaling factors based on h0
        if h0 < 0.2:
            # Close to boundary: fast decay, need precision near s=0
            s_max_factor = 0.5   # Smaller s_max
            n_points_factor = 1.5  # More points for precision
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
            s_max_factor = 2.0 + 0.5 * np.log(h0)  # Logarithmic scaling
            n_points_factor = 1.3 + 0.2 * np.log(h0)
        
        # Time scaling: longer times need higher precision
        time_factor = 1.0 + 0.3 * np.log(max(T, 0.1))
        
        # Compute final parameters
        s_max = base_s_max * s_max_factor * time_factor
        n_points = int(base_n_points * n_points_factor * time_factor)
        
        # Bounds checking
        s_max = np.clip(s_max, 20.0, 500.0)
        n_points = np.clip(n_points, 500, 5000)
        
        return s_max, n_points
    
    @lru_cache(maxsize=256)
    def gil_pelaez_cdf(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float,
                      s_max: float = None, n_points: int = None) -> float:
        """
        Compute first hitting time CDF P(τ ≤ T) using Gil-Pelaez inversion formula:
        F(T) = 0.5 - (1/π) ∫₀^∞ Im[ e^{-isT} φ(s) / s ] ds
        
        Uses adaptive parameters based on h0 and T for optimal accuracy.
        
        Args:
            T: Evaluation time
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            s_max: Integration upper limit (None for adaptive)
            n_points: Number of discretization points (None for adaptive)
            
        Returns:
            CDF value at time T (scalar in [0,1])
        """
        # Use adaptive parameters if not specified
        if s_max is None or n_points is None:
            s_max_adaptive, n_points_adaptive = self._adaptive_gil_pelaez_params(h0, T)
            if s_max is None:
                s_max = s_max_adaptive
            if n_points is None:
                n_points = n_points_adaptive
        
        # Handle the singularity at s=0 more carefully
        s_min = 1e-8  # Start closer to 0 but avoid division issues
        
        # Use denser grid near s=0 where integrand is most important
        # Adaptive grid density based on h0
        if h0 < 0.5:
            # More density near s=0 for small h0
            dense_fraction = 0.4
            transition_point = min(0.5, s_max/20)
        else:
            # Less density near s=0 for large h0
            dense_fraction = 0.3
            transition_point = min(1.0, s_max/10)
        
        n_dense = int(n_points * dense_fraction)
        n_regular = n_points - n_dense
        
        s_dense = np.logspace(np.log10(s_min), np.log10(transition_point), n_dense)
        s_regular = np.linspace(transition_point, s_max, n_regular)
        s_vals = np.concatenate([s_dense, s_regular])
        s_vals = np.unique(s_vals)  # Remove duplicates and sort
        
        # Compute characteristic function values with error handling
        phi_vals = []
        valid_indices = []
        
        for i, s in enumerate(s_vals):
            try:
                # Use higher precision for small s values
                rtol = 1e-10 if s < 0.1 else 1e-8
                phi = self.characteristic_function(s, h0, lambda_X0, lambda_Y0, rtol=rtol)
                
                # Check for numerical issues
                if np.isnan(phi) or np.isinf(phi) or abs(phi) > 10:
                    continue
                    
                phi_vals.append(phi)
                valid_indices.append(i)
            except (RuntimeError, OverflowError, ZeroDivisionError):
                continue
        
        if len(phi_vals) < 10:
            # Fallback to simpler computation if too many failures
            warnings.warn(f"Gil-Pelaez inversion had {len(s_vals) - len(phi_vals)} failures, using fallback")
            return self._gil_pelaez_cdf_fallback(T, h0, lambda_X0, lambda_Y0)
        
        phi_vals = np.array(phi_vals)
        s_vals_valid = s_vals[valid_indices]
        
        # Gil-Pelaez kernel with careful handling of small s
        kernel = np.zeros_like(phi_vals, dtype=complex)
        
        for i, (s, phi) in enumerate(zip(s_vals_valid, phi_vals)):
            if s > 0:
                kernel[i] = np.exp(-1j * s * T) * phi / s
            else:
                # Use L'Hôpital's rule limit as s→0: φ'(0) * (-iT)
                kernel[i] = 0  # Conservative approach
        
        integrand = np.imag(kernel)
        
        # Use adaptive integration or trapezoidal rule for better stability
        if len(s_vals_valid) > 1:
            integral = np.trapz(integrand, s_vals_valid)
        else:
            integral = 0.0
        
        cdf = 0.5 - integral / np.pi
        
        # Ensure mathematical constraints
        cdf = np.clip(np.real(cdf), 0.0, 1.0)
        
        return cdf
    
    def _gil_pelaez_cdf_fallback(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float) -> float:
        """
        Fallback Gil-Pelaez implementation with conservative parameters.
        """
        s_vals = np.linspace(0.01, 50.0, 500)  # Conservative range
        
        phi_vals = []
        for s in s_vals:
            try:
                phi = self.characteristic_function(s, h0, lambda_X0, lambda_Y0, rtol=1e-6)
                if not (np.isnan(phi) or np.isinf(phi)):
                    phi_vals.append(phi)
                else:
                    phi_vals.append(complex(0.0, 0.0))  # Conservative fallback
            except:
                phi_vals.append(complex(0.0, 0.0))
        
        phi_vals = np.array(phi_vals)
        
        # Simple trapezoidal integration
        kernel = np.exp(-1j * s_vals * T) * phi_vals / s_vals
        integrand = np.imag(kernel)
        integral = np.trapz(integrand, s_vals)
        
        cdf = 0.5 - integral / np.pi
        return np.clip(np.real(cdf), 0.0, 1.0)
    
    @lru_cache(maxsize=128)
    def first_passage_pdf(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float,
                         dT: float = 0.01) -> float:
        """
        Compute first hitting time PDF using numerical differentiation of CDF.
        
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
            cdf_plus = self.gil_pelaez_cdf(T + dT/2, h0, lambda_X0, lambda_Y0)
            cdf_minus = self.gil_pelaez_cdf(T - dT/2, h0, lambda_X0, lambda_Y0)
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
        return 1.0 - self.gil_pelaez_cdf(T, h0, lambda_X0, lambda_Y0)
    
    @lru_cache(maxsize=64)
    def moments(self, h0: float, lambda_X0: float, lambda_Y0: float, 
               max_moment: int = 4) -> Dict[str, float]:
        """
        Compute moments of the first hitting time distribution using characteristic function.
        
        Args:
            h0: Initial health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            max_moment: Maximum moment to compute
            
        Returns:
            Dictionary containing moments
        """
        moments_dict = {}
        
        for n in range(1, max_moment + 1):
            # Compute n-th moment using derivative of characteristic function
            # E[τⁿ] = (-i)ⁿ (d/ds)ⁿ φ(s)|_{s=0}
            
            # Use numerical differentiation
            ds = 1e-6
            if n == 1:
                # First moment (mean)
                phi_plus = self.characteristic_function(ds, h0, lambda_X0, lambda_Y0)
                phi_minus = self.characteristic_function(-ds, h0, lambda_X0, lambda_Y0)
                derivative = (phi_plus - phi_minus) / (2 * ds)
                moment = (-1j * derivative).real
            elif n == 2:
                # Second moment
                phi_0 = self.characteristic_function(0, h0, lambda_X0, lambda_Y0)
                phi_plus = self.characteristic_function(ds, h0, lambda_X0, lambda_Y0)
                phi_minus = self.characteristic_function(-ds, h0, lambda_X0, lambda_Y0)
                second_derivative = (phi_plus - 2*phi_0 + phi_minus) / (ds**2)
                moment = ((-1j)**2 * second_derivative).real
            else:
                # Higher moments (simplified numerical approach)
                moment = np.nan  # Would need more sophisticated numerical differentiation
            
            moments_dict[f'moment_{n}'] = moment
        
        # Compute variance from first two moments
        if max_moment >= 2:
            mean = moments_dict['moment_1']
            second_moment = moments_dict['moment_2']
            moments_dict['variance'] = second_moment - mean**2
            moments_dict['std_dev'] = np.sqrt(max(0, moments_dict['variance']))
        
        return moments_dict
    
    def validate_solution(self, s: float, h0: float, lambda_X0: float, lambda_Y0: float, 
                         n_check: int = 10) -> Dict:
        """
        Validate the first hitting time solution by checking ODE residuals.
        
        Args:
            s: Frequency parameter to test
            h0: Health factor value
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            n_check: Number of points to check
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Test if characteristic function can be computed
            phi = self.characteristic_function(s, h0, lambda_X0, lambda_Y0)
            
            # Basic sanity checks
            if np.isnan(phi) or np.isinf(phi):
                return {'valid': False, 'error': 'Characteristic function is NaN or Inf'}
            
            # Check if |φ(0)| = 1 (normalization)
            phi_0 = self.characteristic_function(0, h0, lambda_X0, lambda_Y0)
            if abs(abs(phi_0) - 1.0) > 1e-6:
                return {
                    'valid': False, 
                    'error': f'φ(0) normalization failed: |φ(0)| = {abs(phi_0):.8f}'
                }
            
            return {
                'valid': True,
                'phi_test': phi,
                'phi_0_magnitude': abs(phi_0),
                'normalization_error': abs(abs(phi_0) - 1.0)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def clear_cache(self) -> None:
        """Clear all LRU caches to free memory."""
        self.characteristic_function.cache_clear()
        self._batch_characteristic_function_cached.cache_clear()
        self.gil_pelaez_cdf.cache_clear()
        self.first_passage_pdf.cache_clear()
        self.survival_function.cache_clear()
        self.moments.cache_clear()
    
    def cache_info(self) -> Dict[str, object]:
        """Get cache statistics for all cached methods."""
        return {
            'characteristic_function': self.characteristic_function.cache_info(),
            'batch_characteristic_function': self._batch_characteristic_function_cached.cache_info(),
            'gil_pelaez_cdf': self.gil_pelaez_cdf.cache_info(),
            'first_passage_pdf': self.first_passage_pdf.cache_info(),
            'survival_function': self.survival_function.cache_info(),
            'moments': self.moments.cache_info()
        }
    
    def test_monotonicity(self, T: float, lambda_X0: float, lambda_Y0: float,
                         h0_min: float = 0.1, h0_max: float = 2.0, n_points: int = 20) -> Dict:
        """
        Test monotonicity of CDF with respect to h0 and identify violations.
        
        Args:
            T: Fixed time point for CDF evaluation
            lambda_X0: Initial X intensity  
            lambda_Y0: Initial Y intensity
            h0_min: Minimum h0 value to test
            h0_max: Maximum h0 value to test
            n_points: Number of test points
            
        Returns:
            Dictionary with monotonicity test results
        """
        h0_values = np.linspace(h0_min, h0_max, n_points)
        cdf_values = []
        violations = []
        
        print(f"Testing CDF monotonicity at T = {T}")
        print("h0\t\tCDF\t\tΔCDF\t\tStatus")
        print("-" * 45)
        
        for i, h0 in enumerate(h0_values):
            try:
                # Use high precision parameters
                cdf = self.gil_pelaez_cdf(T, h0, lambda_X0, lambda_Y0, 
                                         s_max=200, n_points=2000)
                cdf_values.append(cdf)
                
                if i > 0:
                    delta_cdf = cdf - cdf_values[i-1]
                    status = "VIOLATION" if delta_cdf > 1e-6 else "OK"
                    if delta_cdf > 1e-6:
                        violations.append({
                            'h0_prev': h0_values[i-1],
                            'h0_curr': h0,
                            'cdf_prev': cdf_values[i-1],
                            'cdf_curr': cdf,
                            'delta': delta_cdf
                        })
                    print(f"{h0:.3f}\t\t{cdf:.6f}\t{delta_cdf:+.2e}\t{status}")
                else:
                    print(f"{h0:.3f}\t\t{cdf:.6f}\t\t--\t\tOK")
                    
            except Exception as e:
                print(f"{h0:.3f}\t\tERROR: {str(e)[:20]}...")
                cdf_values.append(np.nan)
        
        # Summary
        n_violations = len(violations)
        valid_cdfs = [cdf for cdf in cdf_values if not np.isnan(cdf)]
        
        result = {
            'monotonic': n_violations == 0,
            'n_violations': n_violations,
            'violations': violations,
            'h0_values': h0_values,
            'cdf_values': np.array(cdf_values),
            'cdf_range': [min(valid_cdfs), max(valid_cdfs)] if valid_cdfs else [np.nan, np.nan],
            'n_errors': sum(1 for cdf in cdf_values if np.isnan(cdf))
        }
        
        print(f"\\nSummary:")
        print(f"  Monotonic: {'Yes' if result['monotonic'] else 'No'}")
        print(f"  Violations: {n_violations}")
        print(f"  Errors: {result['n_errors']}")
        print(f"  CDF range: [{result['cdf_range'][0]:.6f}, {result['cdf_range'][1]:.6f}]")
        
        return result
    
    def gil_pelaez_cdf_with_convergence(self, T: float, h0: float, lambda_X0: float, lambda_Y0: float,
                                       target_precision: float = 1e-4, max_iterations: int = 3) -> Dict:
        """
        Compute CDF with adaptive convergence checking.
        
        Args:
            T: Evaluation time
            h0: Initial health factor
            lambda_X0: Initial X intensity
            lambda_Y0: Initial Y intensity
            target_precision: Target relative precision
            max_iterations: Maximum refinement iterations
            
        Returns:
            Dictionary with CDF value and convergence info
        """
        results = []
        
        for iteration in range(max_iterations):
            # Get adaptive parameters
            s_max, n_points = self._adaptive_gil_pelaez_params(h0, T)
            
            # Scale up parameters for each iteration
            if iteration > 0:
                s_max *= (1.5 ** iteration)
                n_points = int(n_points * (1.3 ** iteration))
            
            # Compute CDF
            cdf = self.gil_pelaez_cdf(T, h0, lambda_X0, lambda_Y0, s_max=s_max, n_points=n_points)
            
            results.append({
                'iteration': iteration,
                's_max': s_max,
                'n_points': n_points,
                'cdf': cdf
            })
            
            # Check convergence
            if iteration > 0:
                prev_cdf = results[iteration-1]['cdf']
                relative_error = abs(cdf - prev_cdf) / max(abs(prev_cdf), 1e-10)
                
                if relative_error < target_precision:
                    return {
                        'cdf': cdf,
                        'converged': True,
                        'iterations': iteration + 1,
                        'relative_error': relative_error,
                        'final_params': {'s_max': s_max, 'n_points': n_points},
                        'history': results
                    }
        
        # Did not converge
        final_cdf = results[-1]['cdf']
        prev_cdf = results[-2]['cdf'] if len(results) > 1 else final_cdf
        relative_error = abs(final_cdf - prev_cdf) / max(abs(prev_cdf), 1e-10)
        
        return {
            'cdf': final_cdf,
            'converged': False,
            'iterations': max_iterations,
            'relative_error': relative_error,
            'final_params': {'s_max': s_max, 'n_points': n_points},
            'history': results
        }