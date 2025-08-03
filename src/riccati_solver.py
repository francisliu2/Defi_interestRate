"""
Riccati ODE Solver for Hawkes Jump-Diffusion Characteristic Functions

This module implements the Riccati ODE solver for computing characteristic functions
of the Hawkes jump-diffusion process. The characteristic function has the form:

φ(u, t, T) = exp(A(u, T-t) + B(u, T-t) * λ_X(t) + C(u, T-t) * λ_Y(t))

Where A, B, C satisfy a system of Riccati ODEs.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Optional, Callable
import warnings
from dataclasses import dataclass


@dataclass
class RiccatiParameters:
    """Parameters for the Riccati ODE system."""
    
    # Health factor parameters
    sigma_h: float
    mu_X: float
    mu_Y: float
    
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


class RiccatiSolver:
    """
    Solver for the Riccati ODE system arising from Hawkes jump-diffusion processes.
    
    The characteristic function φ(u, t, T) = E[exp(iu * h_T) | F_t] satisfies:
    φ(u, t, T) = exp(A(u, τ) + B(u, τ) * λ_X(t) + C(u, τ) * λ_Y(t))
    
    where τ = T - t and A, B, C satisfy Riccati ODEs.
    """
    
    def __init__(self, parameters: RiccatiParameters):
        """
        Initialize the Riccati solver.
        
        Args:
            parameters: RiccatiParameters containing all model parameters
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
    
    def _jump_transform_X(self, u: complex) -> complex:
        """Compute jump transform for X process: E[exp(iu * J_X)]."""
        # For exponential jump sizes: J_X = -(delta_X + Z) where Z ~ Exp(eta_X)
        # E[exp(iu * (-(delta_X + Z)))] = exp(-iu * delta_X) * eta_X / (eta_X + iu)
        return np.exp(-1j * u * self.params.delta_X) * self.params.eta_X / (self.params.eta_X + 1j * u)
    
    def _jump_transform_Y(self, u: complex) -> complex:
        """Compute jump transform for Y process: E[exp(iu * J_Y)]."""
        # For exponential jump sizes: J_Y = delta_Y + Z where Z ~ Exp(eta_Y)
        # E[exp(iu * (delta_Y + Z))] = exp(iu * delta_Y) * eta_Y / (eta_Y - iu)
        return np.exp(1j * u * self.params.delta_Y) * self.params.eta_Y / (self.params.eta_Y - 1j * u)
    
    def riccati_system(self, tau: float, y: np.ndarray, u: complex) -> np.ndarray:
        """
        Define the Riccati ODE system.
        
        The system is:
        dA/dτ = -0.5 * σ_h² * u² - μ_X_λ * (ψ_X(u) - 1) - μ_Y_λ * (ψ_Y(u) - 1)
        dB/dτ = -β_X * B + α_XX * (ψ_X(u) - 1) + α_YX * (ψ_Y(u) - 1)
        dC/dτ = -β_Y * C + α_YY * (ψ_Y(u) - 1) + α_XY * (ψ_X(u) - 1)
        
        Args:
            tau: Time to maturity (T - t)
            y: State vector [A, B_real, B_imag, C_real, C_imag]
            u: Frequency parameter
            
        Returns:
            Derivative vector
        """
        A = y[0]
        B = complex(y[1], y[2])
        C = complex(y[3], y[4])
        
        # Jump transforms
        psi_X = self._jump_transform_X(u)
        psi_Y = self._jump_transform_Y(u)
        
        # Drift term
        drift_term = 1j * u * (self.params.mu_X - self.params.mu_Y)
        
        # Riccati equations
        dA_dtau = (-0.5 * self.params.sigma_h**2 * u**2 
                   - self.params.mu_X_lambda * (psi_X - 1)
                   - self.params.mu_Y_lambda * (psi_Y - 1)
                   + drift_term)
        
        dB_dtau = (-self.params.beta_X * B 
                   + self.params.alpha_XX * (psi_X - 1)
                   + self.params.alpha_YX * (psi_Y - 1))
        
        dC_dtau = (-self.params.beta_Y * C
                   + self.params.alpha_YY * (psi_Y - 1)
                   + self.params.alpha_XY * (psi_X - 1))
        
        return np.array([
            dA_dtau.real if np.isreal(dA_dtau) else dA_dtau,
            dB_dtau.real,
            dB_dtau.imag,
            dC_dtau.real,
            dC_dtau.imag
        ]).astype(complex)
    
    def solve_riccati(self, u: complex, tau_max: float, 
                     rtol: float = 1e-8, atol: float = 1e-10,
                     max_step: float = 0.1) -> Dict:
        """
        Solve the Riccati ODE system for given frequency parameter.
        
        Args:
            u: Frequency parameter (complex)
            tau_max: Maximum time to maturity
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            max_step: Maximum step size for ODE solver
            
        Returns:
            Dictionary containing solution arrays and metadata
        """
        # Initial conditions: A(0) = B(0) = C(0) = 0
        y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
        
        # Time span
        tau_span = (0, tau_max)
        
        try:
            # Solve ODE system
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                sol = solve_ivp(
                    fun=lambda t, y: self.riccati_system(t, y, u),
                    t_span=tau_span,
                    y0=y0,
                    method='RK45',
                    rtol=rtol,
                    atol=atol,
                    max_step=max_step,
                    dense_output=True
                )
            
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
            
            # Extract solutions
            tau = sol.t
            A = sol.y[0]
            B = sol.y[1] + 1j * sol.y[2]
            C = sol.y[3] + 1j * sol.y[4]
            
            return {
                'tau': tau,
                'A': A,
                'B': B,
                'C': C,
                'success': True,
                'message': sol.message,
                'nfev': sol.nfev,
                'sol': sol  # Dense output for interpolation
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tau': None,
                'A': None,
                'B': None,
                'C': None
            }
    
    def characteristic_function(self, u: complex, tau: float, lambda_X: float, 
                              lambda_Y: float, rtol: float = 1e-8) -> complex:
        """
        Compute the characteristic function at given parameters.
        
        Args:
            u: Frequency parameter
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            rtol: Relative tolerance for ODE solver
            
        Returns:
            Complex characteristic function value
        """
        # Solve Riccati system
        result = self.solve_riccati(u, tau, rtol=rtol)
        
        if not result['success']:
            raise RuntimeError(f"Riccati solver failed: {result.get('error', 'Unknown error')}")
        
        # Interpolate at desired tau
        A_tau = result['sol'].sol(tau)[0]
        B_tau = complex(result['sol'].sol(tau)[1], result['sol'].sol(tau)[2])
        C_tau = complex(result['sol'].sol(tau)[3], result['sol'].sol(tau)[4])
        
        # Compute characteristic function
        phi = np.exp(A_tau + B_tau * lambda_X + C_tau * lambda_Y)
        
        return phi
    
    def batch_characteristic_function(self, u_array: np.ndarray, tau: float,
                                    lambda_X: float, lambda_Y: float,
                                    rtol: float = 1e-8) -> np.ndarray:
        """
        Compute characteristic function for multiple frequency parameters.
        
        Args:
            u_array: Array of frequency parameters
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            rtol: Relative tolerance
            
        Returns:
            Array of characteristic function values
        """
        phi_array = np.zeros(len(u_array), dtype=complex)
        
        for i, u in enumerate(u_array):
            try:
                phi_array[i] = self.characteristic_function(u, tau, lambda_X, lambda_Y, rtol)
            except RuntimeError:
                # Handle solver failures gracefully
                phi_array[i] = np.nan + 1j * np.nan
        
        return phi_array
    
    def moment_generating_function(self, s: float, tau: float, lambda_X: float,
                                 lambda_Y: float, rtol: float = 1e-8) -> float:
        """
        Compute moment generating function M(s) = E[exp(s * h_T)].
        
        This is related to the characteristic function by φ(u) = M(-iu).
        
        Args:
            s: Real parameter
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            rtol: Relative tolerance
            
        Returns:
            Moment generating function value (real)
        """
        u = -1j * s
        phi = self.characteristic_function(u, tau, lambda_X, lambda_Y, rtol)
        return phi.real  # Should be real for real s
    
    def cumulant_generating_function(self, s: float, tau: float, lambda_X: float,
                                   lambda_Y: float, rtol: float = 1e-8) -> float:
        """
        Compute cumulant generating function K(s) = log(M(s)).
        
        Args:
            s: Real parameter
            tau: Time to maturity
            lambda_X: Current X intensity
            lambda_Y: Current Y intensity
            rtol: Relative tolerance
            
        Returns:
            Cumulant generating function value
        """
        mgf = self.moment_generating_function(s, tau, lambda_X, lambda_Y, rtol)
        return np.log(max(mgf, 1e-100))  # Avoid log(0)
    
    def validate_solution(self, u: complex, tau_max: float, n_check: int = 10) -> Dict:
        """
        Validate the Riccati solution by checking ODE residuals.
        
        Args:
            u: Frequency parameter to test
            tau_max: Maximum time to maturity
            n_check: Number of points to check
            
        Returns:
            Dictionary with validation results
        """
        result = self.solve_riccati(u, tau_max)
        
        if not result['success']:
            return {'valid': False, 'error': result.get('error')}
        
        # Check at random points
        tau_check = np.linspace(0.1, tau_max, n_check)
        max_residual = 0.0
        
        for tau in tau_check:
            # Evaluate solution and derivative
            y = result['sol'].sol(tau)
            dydt = result['sol'].sol(tau, der=1)
            
            # Compute expected derivative from Riccati system
            expected_dydt = self.riccati_system(tau, y, u)
            
            # Compute residual
            residual = np.linalg.norm(dydt - expected_dydt)
            max_residual = max(max_residual, residual)
        
        return {
            'valid': max_residual < 1e-6,
            'max_residual': max_residual,
            'tolerance': 1e-6
        }