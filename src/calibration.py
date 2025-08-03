"""
Parameter Calibration for Hawkes Jump-Diffusion Models

This module implements calibration methods for Hawkes jump-diffusion parameters
using Peak-Over-Threshold (POT) method for jump detection and Maximum Likelihood
Estimation (MLE) for intensity parameters.

The calibration process involves:
1. Jump detection using POT method on price returns
2. Jump size distribution estimation
3. MLE estimation of Hawkes intensity parameters
4. Goodness-of-fit testing and validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass, asdict
import yaml
from pathlib import Path


@dataclass
class CalibrationData:
    """Container for market data used in calibration."""
    
    timestamps: np.ndarray
    collateral_returns: np.ndarray
    borrowed_returns: np.ndarray
    collateral_prices: np.ndarray
    borrowed_prices: np.ndarray
    
    @classmethod
    def from_csv(cls, filepath: str, 
                 timestamp_col: str = 'timestamp',
                 collateral_return_col: str = 'collateral_return',
                 borrowed_return_col: str = 'borrowed_return',
                 collateral_price_col: str = 'collateral_price',
                 borrowed_price_col: str = 'borrowed_price') -> 'CalibrationData':
        """Load calibration data from CSV file."""
        
        df = pd.read_csv(filepath)
        
        # Parse timestamps if they're strings
        if df[timestamp_col].dtype == 'object':
            timestamps = pd.to_datetime(df[timestamp_col]).values
        else:
            timestamps = df[timestamp_col].values
        
        return cls(
            timestamps=timestamps,
            collateral_returns=df[collateral_return_col].values,
            borrowed_returns=df[borrowed_return_col].values,
            collateral_prices=df[collateral_price_col].values,
            borrowed_prices=df[borrowed_price_col].values
        )


@dataclass
class POTParameters:
    """Parameters for Peak-Over-Threshold jump detection."""
    
    threshold_quantile: float = 0.95
    min_jump_size: float = 0.01
    min_cluster_separation: int = 1  # Minimum time steps between jumps
    

class POTCalibrator:
    """
    Peak-Over-Threshold calibrator for jump detection and size estimation.
    
    This class identifies significant price movements as jumps and estimates
    the parameters of their size distributions.
    """
    
    def __init__(self, parameters: POTParameters = None):
        """
        Initialize POT calibrator.
        
        Args:
            parameters: POT parameters (uses defaults if None)
        """
        self.params = parameters or POTParameters()
        
    def detect_jumps(self, returns: np.ndarray, direction: str = 'both') -> Dict:
        """
        Detect jumps using Peak-Over-Threshold method.
        
        Args:
            returns: Array of price returns
            direction: 'up', 'down', or 'both' for jump direction
            
        Returns:
            Dictionary with jump information
        """
        if direction == 'down':
            # Look for large negative returns (collateral drops)
            threshold = np.quantile(returns, 1 - self.params.threshold_quantile)
            candidates = returns <= threshold
            jump_returns = returns[candidates]
            jump_sizes = -jump_returns  # Convert to positive values
            
        elif direction == 'up':
            # Look for large positive returns (borrowed asset increases)
            threshold = np.quantile(returns, self.params.threshold_quantile)
            candidates = returns >= threshold
            jump_returns = returns[candidates]
            jump_sizes = jump_returns
            
        else:  # both directions
            # Use absolute returns
            abs_returns = np.abs(returns)
            threshold = np.quantile(abs_returns, self.params.threshold_quantile)
            candidates = abs_returns >= threshold
            jump_returns = returns[candidates]
            jump_sizes = abs_returns[candidates]
        
        # Filter by minimum jump size
        large_enough = jump_sizes >= self.params.min_jump_size
        jump_indices = np.where(candidates)[0][large_enough]
        jump_sizes = jump_sizes[large_enough]
        jump_returns = jump_returns[large_enough]
        
        # Remove clustered jumps (keep only the largest in each cluster)
        if len(jump_indices) > 1:
            filtered_indices = []
            filtered_sizes = []
            filtered_returns = []
            
            i = 0
            while i < len(jump_indices):
                # Find cluster of nearby jumps
                cluster_end = i
                while (cluster_end + 1 < len(jump_indices) and 
                       jump_indices[cluster_end + 1] - jump_indices[i] <= self.params.min_cluster_separation):
                    cluster_end += 1
                
                # Keep the largest jump in the cluster
                cluster_idx = np.arange(i, cluster_end + 1)
                max_idx = cluster_idx[np.argmax(jump_sizes[cluster_idx])]
                
                filtered_indices.append(jump_indices[max_idx])
                filtered_sizes.append(jump_sizes[max_idx])
                filtered_returns.append(jump_returns[max_idx])
                
                i = cluster_end + 1
            
            jump_indices = np.array(filtered_indices)
            jump_sizes = np.array(filtered_sizes)
            jump_returns = np.array(filtered_returns)
        
        return {
            'jump_indices': jump_indices,
            'jump_sizes': jump_sizes,
            'jump_returns': jump_returns,
            'threshold': threshold,
            'n_jumps': len(jump_indices),
            'jump_rate': len(jump_indices) / len(returns)
        }
    
    def fit_jump_distribution(self, jump_sizes: np.ndarray, 
                             distribution: str = 'exponential') -> Dict:
        """
        Fit distribution to detected jump sizes.
        
        Args:
            jump_sizes: Array of jump sizes (positive values)
            distribution: 'exponential' or 'gamma'
            
        Returns:
            Dictionary with fitted parameters and goodness-of-fit
        """
        if len(jump_sizes) == 0:
            return {
                'success': False,
                'error': 'No jumps detected'
            }
        
        try:
            if distribution == 'exponential':
                # Fit shifted exponential: f(x) = eta * exp(-eta * (x - delta))
                # Use method of moments for initial guess
                mean_excess = np.mean(jump_sizes)
                eta_init = 1.0 / mean_excess
                delta_init = np.min(jump_sizes) * 0.9
                
                # MLE estimation
                def neg_log_likelihood(params):
                    eta, delta = params
                    if eta <= 0 or delta < 0 or np.any(jump_sizes <= delta):
                        return np.inf
                    
                    excess = jump_sizes - delta
                    log_lik = np.sum(np.log(eta) - eta * excess)
                    return -log_lik
                
                result = minimize(
                    neg_log_likelihood,
                    [eta_init, delta_init],
                    method='L-BFGS-B',
                    bounds=[(1e-6, None), (0, np.min(jump_sizes) * 0.99)]
                )
                
                if result.success:
                    eta_hat, delta_hat = result.x
                    
                    # Goodness-of-fit test
                    excess = jump_sizes - delta_hat
                    ks_stat, ks_pvalue = stats.kstest(
                        excess, 
                        lambda x: stats.expon.cdf(x, scale=1/eta_hat)
                    )
                    
                    return {
                        'success': True,
                        'distribution': 'exponential',
                        'eta': eta_hat,
                        'delta': delta_hat,
                        'log_likelihood': -result.fun,
                        'aic': 2 * 2 - 2 * (-result.fun),  # 2 parameters
                        'bic': 2 * np.log(len(jump_sizes)) - 2 * (-result.fun),
                        'ks_statistic': ks_stat,
                        'ks_pvalue': ks_pvalue,
                        'n_jumps': len(jump_sizes)
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Optimization failed: {result.message}'
                    }
                    
            else:
                raise NotImplementedError(f"Distribution '{distribution}' not implemented")
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def calibrate_jumps(self, data: CalibrationData) -> Dict:
        """
        Full jump calibration for both collateral and borrowed assets.
        
        Args:
            data: CalibrationData object
            
        Returns:
            Dictionary with calibration results for both assets
        """
        # Detect jumps in collateral (downward)
        collateral_jumps = self.detect_jumps(data.collateral_returns, direction='down')
        
        # Detect jumps in borrowed asset (upward)  
        borrowed_jumps = self.detect_jumps(data.borrowed_returns, direction='up')
        
        # Fit distributions
        collateral_dist = self.fit_jump_distribution(collateral_jumps['jump_sizes'])
        borrowed_dist = self.fit_jump_distribution(borrowed_jumps['jump_sizes'])
        
        return {
            'collateral_jumps': collateral_jumps,
            'borrowed_jumps': borrowed_jumps,
            'collateral_distribution': collateral_dist,
            'borrowed_distribution': borrowed_dist,
            'summary': {
                'n_collateral_jumps': collateral_jumps['n_jumps'],
                'n_borrowed_jumps': borrowed_jumps['n_jumps'],
                'collateral_jump_rate': collateral_jumps['jump_rate'],
                'borrowed_jump_rate': borrowed_jumps['jump_rate']
            }
        }


class MLECalibrator:
    """
    Maximum Likelihood Estimator for Hawkes intensity parameters.
    
    This class estimates the parameters of the bivariate Hawkes process
    given detected jump times and sizes.
    """
    
    def __init__(self, max_iter: int = 1000, tolerance: float = 1e-6):
        """
        Initialize MLE calibrator.
        
        Args:
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def _hawkes_log_likelihood(self, params: np.ndarray, jump_times_X: np.ndarray,
                              jump_times_Y: np.ndarray, T: float) -> float:
        """
        Compute log-likelihood for bivariate Hawkes process.
        
        Args:
            params: [mu_X, mu_Y, alpha_XX, alpha_XY, alpha_YX, alpha_YY, beta_X, beta_Y]
            jump_times_X: Jump times for process X
            jump_times_Y: Jump times for process Y
            T: Observation period
            
        Returns:
            Negative log-likelihood (for minimization)
        """
        mu_X, mu_Y, alpha_XX, alpha_XY, alpha_YX, alpha_YY, beta_X, beta_Y = params
        
        # Check parameter constraints
        if any(p <= 0 for p in params):
            return np.inf
        
        # Stability check
        if alpha_XX / beta_X >= 1 or alpha_YY / beta_Y >= 1:
            return np.inf
        
        try:
            # Combine and sort all jump times
            all_jumps = []
            for t in jump_times_X:
                all_jumps.append((t, 'X'))
            for t in jump_times_Y:
                all_jumps.append((t, 'Y'))
            
            all_jumps.sort()
            
            log_lik = 0.0
            lambda_X = mu_X
            lambda_Y = mu_Y
            
            prev_time = 0.0
            
            for jump_time, jump_type in all_jumps:
                dt = jump_time - prev_time
                
                # Update intensities (decay)
                lambda_X = mu_X + (lambda_X - mu_X) * np.exp(-beta_X * dt)
                lambda_Y = mu_Y + (lambda_Y - mu_Y) * np.exp(-beta_Y * dt)
                
                # Add log-intensity at jump time
                if jump_type == 'X':
                    log_lik += np.log(max(lambda_X, 1e-10))
                    # Update intensities after jump
                    lambda_X += alpha_XX
                    lambda_Y += alpha_XY
                else:  # jump_type == 'Y'
                    log_lik += np.log(max(lambda_Y, 1e-10))
                    # Update intensities after jump
                    lambda_Y += alpha_YY
                    lambda_X += alpha_YX
                
                prev_time = jump_time
            
            # Integral term (compensator)
            dt_final = T - prev_time
            if dt_final > 0:
                lambda_X_final = mu_X + (lambda_X - mu_X) * np.exp(-beta_X * dt_final)
                lambda_Y_final = mu_Y + (lambda_Y - mu_Y) * np.exp(-beta_Y * dt_final)
            else:
                lambda_X_final = lambda_X
                lambda_Y_final = lambda_Y
            
            # Compute compensator integrals
            integral_X = self._compute_compensator_integral(
                jump_times_X, jump_times_Y, T, mu_X, alpha_XX, alpha_YX, beta_X
            )
            integral_Y = self._compute_compensator_integral(
                jump_times_Y, jump_times_X, T, mu_Y, alpha_YY, alpha_XY, beta_Y
            )
            
            log_lik -= (integral_X + integral_Y)
            
            return -log_lik  # Return negative for minimization
            
        except Exception:
            return np.inf
    
    def _compute_compensator_integral(self, own_jumps: np.ndarray, other_jumps: np.ndarray,
                                    T: float, mu: float, alpha_own: float, 
                                    alpha_cross: float, beta: float) -> float:
        """Compute compensator integral for one component of Hawkes process."""
        
        # Baseline term
        integral = mu * T
        
        # Self-excitation terms
        for t_j in own_jumps:
            integral += (alpha_own / beta) * (1 - np.exp(-beta * (T - t_j)))
        
        # Cross-excitation terms
        for t_j in other_jumps:
            integral += (alpha_cross / beta) * (1 - np.exp(-beta * (T - t_j)))
        
        return integral
    
    def fit_hawkes_parameters(self, jump_times_X: np.ndarray, jump_times_Y: np.ndarray,
                             T: float, initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        Fit Hawkes parameters using MLE.
        
        Args:
            jump_times_X: Jump times for process X
            jump_times_Y: Jump times for process Y
            T: Total observation time
            initial_guess: Initial parameter guess
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        if len(jump_times_X) == 0 and len(jump_times_Y) == 0:
            return {
                'success': False,
                'error': 'No jumps detected in either process'
            }
        
        # Initial guess
        if initial_guess is None:
            rate_X = len(jump_times_X) / T
            rate_Y = len(jump_times_Y) / T
            
            initial_guess = np.array([
                max(rate_X, 0.1),  # mu_X
                max(rate_Y, 0.1),  # mu_Y  
                0.3,               # alpha_XX
                0.2,               # alpha_XY
                0.2,               # alpha_YX
                0.3,               # alpha_YY
                1.0,               # beta_X
                1.0                # beta_Y
            ])
        
        # Parameter bounds
        bounds = [
            (1e-6, None),  # mu_X > 0
            (1e-6, None),  # mu_Y > 0
            (1e-6, 0.99),  # 0 < alpha_XX < 1
            (1e-6, 0.99),  # 0 < alpha_XY < 1
            (1e-6, 0.99),  # 0 < alpha_YX < 1
            (1e-6, 0.99),  # 0 < alpha_YY < 1
            (1e-6, None),  # beta_X > 0
            (1e-6, None)   # beta_Y > 0
        ]
        
        try:
            # Use differential evolution for global optimization
            result = differential_evolution(
                lambda params: self._hawkes_log_likelihood(params, jump_times_X, jump_times_Y, T),
                bounds,
                seed=42,
                maxiter=self.max_iter,
                tol=self.tolerance,
                workers=1
            )
            
            if result.success:
                params = result.x
                mu_X, mu_Y, alpha_XX, alpha_XY, alpha_YX, alpha_YY, beta_X, beta_Y = params
                
                # Compute AIC and BIC
                log_lik = -result.fun
                n_params = len(params)
                n_obs = len(jump_times_X) + len(jump_times_Y)
                
                aic = 2 * n_params - 2 * log_lik
                bic = n_params * np.log(n_obs) - 2 * log_lik
                
                return {
                    'success': True,
                    'parameters': {
                        'mu_X_lambda': mu_X,
                        'mu_Y_lambda': mu_Y,
                        'alpha_XX': alpha_XX,
                        'alpha_XY': alpha_XY,
                        'alpha_YX': alpha_YX,
                        'alpha_YY': alpha_YY,
                        'beta_X': beta_X,
                        'beta_Y': beta_Y
                    },
                    'log_likelihood': log_lik,
                    'aic': aic,
                    'bic': bic,
                    'n_observations': n_obs,
                    'optimization_result': result
                }
            else:
                return {
                    'success': False,
                    'error': f"Optimization failed: {result.message}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def full_calibration_pipeline(data: CalibrationData, 
                            pot_params: Optional[POTParameters] = None,
                            output_path: Optional[str] = None) -> Dict:
    """
    Run complete calibration pipeline.
    
    Args:
        data: Market data for calibration
        pot_params: POT parameters
        output_path: Path to save calibration results
        
    Returns:
        Dictionary with complete calibration results
    """
    # Initialize calibrators
    pot_calibrator = POTCalibrator(pot_params)
    mle_calibrator = MLECalibrator()
    
    # Step 1: Jump detection and size estimation
    jump_calibration = pot_calibrator.calibrate_jumps(data)
    
    if not (jump_calibration['collateral_distribution']['success'] and 
            jump_calibration['borrowed_distribution']['success']):
        return {
            'success': False,
            'error': 'Jump size estimation failed',
            'jump_calibration': jump_calibration
        }
    
    # Step 2: Extract jump times
    collateral_indices = jump_calibration['collateral_jumps']['jump_indices']
    borrowed_indices = jump_calibration['borrowed_jumps']['jump_indices']
    
    # Convert indices to time (assuming daily data)
    jump_times_X = collateral_indices.astype(float)
    jump_times_Y = borrowed_indices.astype(float)
    T = float(len(data.timestamps))
    
    # Step 3: Hawkes parameter estimation
    hawkes_calibration = mle_calibrator.fit_hawkes_parameters(jump_times_X, jump_times_Y, T)
    
    # Combine results
    results = {
        'success': hawkes_calibration['success'],
        'calibration_metadata': {
            'data_length': len(data.timestamps),
            'calibration_method': 'POT + MLE',
            'n_collateral_jumps': jump_calibration['summary']['n_collateral_jumps'],
            'n_borrowed_jumps': jump_calibration['summary']['n_borrowed_jumps']
        },
        'jump_parameters': {
            'eta_X': jump_calibration['collateral_distribution']['eta'],
            'delta_X': jump_calibration['collateral_distribution']['delta'],
            'eta_Y': jump_calibration['borrowed_distribution']['eta'],
            'delta_Y': jump_calibration['borrowed_distribution']['delta']
        },
        'hawkes_parameters': hawkes_calibration.get('parameters', {}),
        'goodness_of_fit': {
            'log_likelihood': hawkes_calibration.get('log_likelihood'),
            'aic': hawkes_calibration.get('aic'),
            'bic': hawkes_calibration.get('bic'),
            'collateral_ks_pvalue': jump_calibration['collateral_distribution'].get('ks_pvalue'),
            'borrowed_ks_pvalue': jump_calibration['borrowed_distribution'].get('ks_pvalue')
        }
    }
    
    # Save results if path provided
    if output_path and results['success']:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    
    return results