"""
Hawkes Jump-Diffusion Process Implementation

This module implements the core Hawkes jump-diffusion process for modeling
health factor dynamics in DeFi lending platforms with cross-excitation effects.

The health factor follows:
dh_t = (μ_X w_X/S_X_t - μ_Y w_Y/S_Y_t) dt + σ_h dW_t 
       - δ_X dN_X_t + δ_Y dN_Y_t

Where N_X_t and N_Y_t are Hawkes processes with cross-excitation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class HawkesParameters:
    """Parameters for the Hawkes jump-diffusion process."""
    
    # Health factor parameters
    h0: float = 1.25
    sigma_h: float = 0.8
    
    # Drift parameters
    mu_X: float = 0.02
    mu_Y: float = 0.05
    
    # Jump size parameters
    eta_X: float = 2.0
    delta_X: float = 0.08
    eta_Y: float = 1.8
    delta_Y: float = 0.08
    
    # Hawkes intensity parameters
    mu_X_lambda: float = 2.0
    mu_Y_lambda: float = 2.0
    beta_X: float = 1.5
    beta_Y: float = 1.5
    alpha_XX: float = 0.5
    alpha_YY: float = 0.6
    alpha_XY: float = 0.3
    alpha_YX: float = 0.3
    
    # Initial intensities
    lambda_X0: float = 2.0
    lambda_Y0: float = 2.0
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'HawkesParameters':
        """Load parameters from YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            h0=config['health_factor']['initial_value'],
            sigma_h=config['diffusion']['volatility'],
            mu_X=config['drift']['collateral_rate'],
            mu_Y=config['drift']['borrowed_rate'],
            eta_X=config['jump_sizes']['eta_X'],
            delta_X=config['jump_sizes']['delta_X'],
            eta_Y=config['jump_sizes']['eta_Y'],
            delta_Y=config['jump_sizes']['delta_Y'],
            mu_X_lambda=config['hawkes_intensities']['mu_X_lambda'],
            mu_Y_lambda=config['hawkes_intensities']['mu_Y_lambda'],
            beta_X=config['hawkes_intensities']['beta_X'],
            beta_Y=config['hawkes_intensities']['beta_Y'],
            alpha_XX=config['hawkes_intensities']['alpha_XX'],
            alpha_YY=config['hawkes_intensities']['alpha_YY'],
            alpha_XY=config['hawkes_intensities']['alpha_XY'],
            alpha_YX=config['hawkes_intensities']['alpha_YX'],
            lambda_X0=config['initial_intensities']['lambda_X0'],
            lambda_Y0=config['initial_intensities']['lambda_Y0']
        )


class HawkesJumpDiffusion:
    """
    Hawkes jump-diffusion process for health factor dynamics.
    
    This class implements the bivariate Hawkes process with cross-excitation
    for modeling correlated jump events in collateral and borrowed asset prices.
    """
    
    def __init__(self, parameters: HawkesParameters):
        """
        Initialize the Hawkes jump-diffusion process.
        
        Args:
            parameters: HawkesParameters object containing all model parameters
        """
        self.params = parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate parameter constraints for model stability."""
        # Check stability conditions for Hawkes process
        spectral_radius = max(
            abs(self.params.alpha_XX / self.params.beta_X),
            abs(self.params.alpha_YY / self.params.beta_Y)
        )
        
        if spectral_radius >= 1.0:
            raise ValueError(
                f"Hawkes process is not stationary: spectral radius = {spectral_radius:.3f} >= 1.0"
            )
        
        # Check positive parameters
        positive_params = [
            'h0', 'sigma_h', 'eta_X', 'eta_Y', 'delta_X', 'delta_Y',
            'mu_X_lambda', 'mu_Y_lambda', 'beta_X', 'beta_Y',
            'alpha_XX', 'alpha_YY', 'alpha_XY', 'alpha_YX',
            'lambda_X0', 'lambda_Y0'
        ]
        
        for param_name in positive_params:
            value = getattr(self.params, param_name)
            if value <= 0:
                raise ValueError(f"Parameter {param_name} must be positive, got {value}")
    
    def intensity_process(self, t: np.ndarray, jump_times_X: List[float], 
                         jump_times_Y: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute intensity processes λ_X(t) and λ_Y(t) given jump histories.
        
        Args:
            t: Time points at which to evaluate intensities
            jump_times_X: List of jump times for process X
            jump_times_Y: List of jump times for process Y
            
        Returns:
            Tuple of (lambda_X, lambda_Y) intensity arrays
        """
        lambda_X = np.zeros_like(t)
        lambda_Y = np.zeros_like(t)
        
        for i, time in enumerate(t):
            # Baseline intensities
            lambda_X[i] = self.params.mu_X_lambda
            lambda_Y[i] = self.params.mu_Y_lambda
            
            # Add contributions from past X jumps
            for jump_time in jump_times_X:
                if jump_time < time:
                    decay_factor = np.exp(-self.params.beta_X * (time - jump_time))
                    lambda_X[i] += self.params.alpha_XX * decay_factor
                    lambda_Y[i] += self.params.alpha_XY * decay_factor
            
            # Add contributions from past Y jumps
            for jump_time in jump_times_Y:
                if jump_time < time:
                    decay_factor = np.exp(-self.params.beta_Y * (time - jump_time))
                    lambda_Y[i] += self.params.alpha_YY * decay_factor
                    lambda_X[i] += self.params.alpha_YX * decay_factor
        
        return lambda_X, lambda_Y
    
    def simulate_hawkes_jumps(self, T: float, dt: float = 0.01) -> Tuple[List[float], List[float]]:
        """
        Simulate jump times for the bivariate Hawkes process using thinning algorithm.
        
        Args:
            T: Time horizon
            dt: Time step for intensity updates
            
        Returns:
            Tuple of (jump_times_X, jump_times_Y)
        """
        jump_times_X = []
        jump_times_Y = []
        
        # Current intensities
        lambda_X = self.params.lambda_X0
        lambda_Y = self.params.lambda_Y0
        
        t = 0.0
        
        while t < T:
            # Upper bound for intensities (for thinning)
            lambda_max = max(lambda_X, lambda_Y) * 1.2
            
            # Generate candidate inter-arrival time
            u1 = np.random.exponential(1.0 / lambda_max)
            t_candidate = t + u1
            
            if t_candidate >= T:
                break
            
            # Update intensities at candidate time
            lambda_X_new = self.params.mu_X_lambda
            lambda_Y_new = self.params.mu_Y_lambda
            
            # Add contributions from past jumps
            for jump_time in jump_times_X:
                if jump_time < t_candidate:
                    decay_factor = np.exp(-self.params.beta_X * (t_candidate - jump_time))
                    lambda_X_new += self.params.alpha_XX * decay_factor
                    lambda_Y_new += self.params.alpha_XY * decay_factor
            
            for jump_time in jump_times_Y:
                if jump_time < t_candidate:
                    decay_factor = np.exp(-self.params.beta_Y * (t_candidate - jump_time))
                    lambda_Y_new += self.params.alpha_YY * decay_factor
                    lambda_X_new += self.params.alpha_YX * decay_factor
            
            # Thinning step
            u2 = np.random.uniform()
            total_intensity = lambda_X_new + lambda_Y_new
            
            if u2 <= total_intensity / lambda_max:
                # Accept the jump, determine which process
                u3 = np.random.uniform()
                if u3 <= lambda_X_new / total_intensity:
                    jump_times_X.append(t_candidate)
                else:
                    jump_times_Y.append(t_candidate)
                
                # Update current intensities after jump
                lambda_X = lambda_X_new
                lambda_Y = lambda_Y_new
            
            t = t_candidate
        
        return jump_times_X, jump_times_Y
    
    def generate_jump_sizes(self, n_jumps_X: int, n_jumps_Y: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate jump sizes from exponential distributions with shifting.
        
        Args:
            n_jumps_X: Number of X jumps
            n_jumps_Y: Number of Y jumps
            
        Returns:
            Tuple of (jump_sizes_X, jump_sizes_Y)
        """
        # X jumps are downward (negative)
        jump_sizes_X = -(self.params.delta_X + 
                        np.random.exponential(1.0/self.params.eta_X, n_jumps_X))
        
        # Y jumps are upward (positive)
        jump_sizes_Y = (self.params.delta_Y + 
                       np.random.exponential(1.0/self.params.eta_Y, n_jumps_Y))
        
        return jump_sizes_X, jump_sizes_Y
    
    def health_factor_path(self, T: float, dt: float = 0.01, 
                          w_X: float = 1.0, w_Y: float = 0.5) -> Dict:
        """
        Simulate a complete health factor path with jumps and diffusion.
        
        Args:
            T: Time horizon
            dt: Time step
            w_X: Collateral weight
            w_Y: Borrowed weight
            
        Returns:
            Dictionary containing time grid, health factor path, jump times, etc.
        """
        # Simulate jump times
        jump_times_X, jump_times_Y = self.simulate_hawkes_jumps(T, dt)
        
        # Generate jump sizes
        jump_sizes_X, jump_sizes_Y = self.generate_jump_sizes(
            len(jump_times_X), len(jump_times_Y)
        )
        
        # Create time grid
        t = np.arange(0, T + dt, dt)
        n_steps = len(t)
        
        # Initialize health factor path
        h = np.zeros(n_steps)
        h[0] = self.params.h0
        
        # Drift coefficient
        drift = self.params.mu_X * w_X - self.params.mu_Y * w_Y
        
        # Simulate diffusion part
        dW = np.random.normal(0, np.sqrt(dt), n_steps - 1)
        
        # Add jumps to appropriate time points
        jump_contributions = np.zeros(n_steps)
        
        for i, jump_time in enumerate(jump_times_X):
            jump_idx = int(jump_time / dt)
            if jump_idx < n_steps:
                jump_contributions[jump_idx] += jump_sizes_X[i]
        
        for i, jump_time in enumerate(jump_times_Y):
            jump_idx = int(jump_time / dt)
            if jump_idx < n_steps:
                jump_contributions[jump_idx] += jump_sizes_Y[i]
        
        # Build complete path
        for i in range(1, n_steps):
            h[i] = (h[i-1] + drift * dt + 
                   self.params.sigma_h * dW[i-1] + 
                   jump_contributions[i])
        
        return {
            'time': t,
            'health_factor': h,
            'jump_times_X': jump_times_X,
            'jump_times_Y': jump_times_Y,
            'jump_sizes_X': jump_sizes_X,
            'jump_sizes_Y': jump_sizes_Y,
            'first_passage_time': self._find_first_passage_time(t, h),
            'liquidated': np.any(h <= 1.0)
        }
    
    def _find_first_passage_time(self, t: np.ndarray, h: np.ndarray) -> Optional[float]:
        """Find first time when health factor hits liquidation threshold."""
        liquidation_indices = np.where(h <= 1.0)[0]
        if len(liquidation_indices) > 0:
            return t[liquidation_indices[0]]
        return None
    
    def monte_carlo_analysis(self, T: float, n_paths: int = 10000, 
                           dt: float = 0.01, w_X: float = 10.0, 
                           w_Y: float = 15000.0) -> Dict:
        """
        Run Monte Carlo analysis for first passage time statistics.
        
        Args:
            T: Time horizon
            n_paths: Number of simulation paths
            dt: Time step
            w_X: Collateral weight
            w_Y: Borrowed weight
            
        Returns:
            Dictionary with liquidation statistics
        """
        first_passage_times = []
        liquidation_count = 0
        
        for _ in range(n_paths):
            path_result = self.health_factor_path(T, dt, w_X, w_Y)
            
            if path_result['liquidated']:
                liquidation_count += 1
                first_passage_times.append(path_result['first_passage_time'])
        
        liquidation_probability = liquidation_count / n_paths
        
        results = {
            'liquidation_probability': liquidation_probability,
            'n_liquidations': liquidation_count,
            'n_paths': n_paths,
            'time_horizon': T
        }
        
        if first_passage_times:
            first_passage_times = np.array(first_passage_times)
            results.update({
                'mean_first_passage_time': np.mean(first_passage_times),
                'std_first_passage_time': np.std(first_passage_times),
                'median_first_passage_time': np.median(first_passage_times),
                'min_first_passage_time': np.min(first_passage_times),
                'max_first_passage_time': np.max(first_passage_times)
            })
        
        return results