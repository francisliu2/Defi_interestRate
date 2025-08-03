"""
Monte Carlo Simulation and Analysis Tools

This module provides high-level simulation tools for the Hawkes jump-diffusion
process, including batch simulations, sensitivity analysis, and statistical
analysis of first passage times.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import warnings
from pathlib import Path
import yaml

from .hawkes_process import HawkesJumpDiffusion, HawkesParameters
from .riccati_solver import RiccatiSolver, RiccatiParameters
from .gil_pelaez import GilPelaezCDF, GilPelaezParameters


@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulations."""
    
    # Simulation settings
    n_paths: int = 10000
    time_horizon: float = 30.0
    dt: float = 0.01
    
    # Portfolio settings
    w_X: float = 10.0  # Collateral weight
    w_Y: float = 15000.0  # Borrowed weight
    
    # Parallel processing
    n_cores: Optional[int] = None
    batch_size: int = 1000
    
    # Random seed
    random_seed: Optional[int] = None


class HawkesSimulator:
    """
    High-level simulator for Hawkes jump-diffusion processes.
    
    This class provides batch simulation capabilities, sensitivity analysis,
    and integration with analytical methods for validation.
    """
    
    def __init__(self, hawkes_params: HawkesParameters, 
                 sim_params: Optional[SimulationParameters] = None):
        """
        Initialize the simulator.
        
        Args:
            hawkes_params: Hawkes process parameters
            sim_params: Simulation parameters (uses defaults if None)
        """
        self.hawkes_params = hawkes_params
        self.sim_params = sim_params or SimulationParameters()
        self.hawkes_process = HawkesJumpDiffusion(hawkes_params)
        
        # Set random seed if provided
        if self.sim_params.random_seed is not None:
            np.random.seed(self.sim_params.random_seed)
    
    def _simulate_single_batch(self, batch_idx: int, batch_size: int) -> Dict:
        """Simulate a single batch of paths."""
        # Set different seed for each batch
        if self.sim_params.random_seed is not None:
            np.random.seed(self.sim_params.random_seed + batch_idx)
        
        first_passage_times = []
        liquidation_count = 0
        jump_statistics = {'n_X_jumps': [], 'n_Y_jumps': [], 'total_jumps': []}
        
        for _ in range(batch_size):
            path_result = self.hawkes_process.health_factor_path(
                T=self.sim_params.time_horizon,
                dt=self.sim_params.dt,
                w_X=self.sim_params.w_X,
                w_Y=self.sim_params.w_Y
            )
            
            if path_result['liquidated']:
                liquidation_count += 1
                first_passage_times.append(path_result['first_passage_time'])
            
            # Collect jump statistics
            jump_statistics['n_X_jumps'].append(len(path_result['jump_times_X']))
            jump_statistics['n_Y_jumps'].append(len(path_result['jump_times_Y']))
            jump_statistics['total_jumps'].append(
                len(path_result['jump_times_X']) + len(path_result['jump_times_Y'])
            )
        
        return {
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'first_passage_times': first_passage_times,
            'liquidation_count': liquidation_count,
            'jump_statistics': jump_statistics
        }
    
    def run_monte_carlo(self, save_paths: bool = False, 
                       output_dir: Optional[str] = None) -> Dict:
        """
        Run Monte Carlo simulation with optional parallel processing.
        
        Args:
            save_paths: Whether to save individual path results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with simulation results and statistics
        """
        # Determine number of cores
        n_cores = self.sim_params.n_cores
        if n_cores is None:
            n_cores = max(1, mp.cpu_count() - 1)
        
        # Calculate batch configuration
        n_batches = max(1, self.sim_params.n_paths // self.sim_params.batch_size)
        remainder = self.sim_params.n_paths % self.sim_params.batch_size
        
        batch_sizes = [self.sim_params.batch_size] * n_batches
        if remainder > 0:
            batch_sizes.append(remainder)
            n_batches += 1
        
        print(f"Running {self.sim_params.n_paths} simulations in {n_batches} batches using {n_cores} cores")
        
        # Run simulations
        if n_cores == 1:
            # Sequential processing
            batch_results = []
            for i, batch_size in enumerate(batch_sizes):
                result = self._simulate_single_batch(i, batch_size)
                batch_results.append(result)
                
                if (i + 1) % max(1, n_batches // 10) == 0:
                    print(f"Completed {i + 1}/{n_batches} batches")
        else:
            # Parallel processing
            with mp.Pool(processes=n_cores) as pool:
                batch_results = pool.starmap(
                    self._simulate_single_batch,
                    [(i, batch_size) for i, batch_size in enumerate(batch_sizes)]
                )
        
        # Aggregate results
        all_first_passage_times = []
        total_liquidations = 0
        all_jump_stats = {'n_X_jumps': [], 'n_Y_jumps': [], 'total_jumps': []}
        
        for result in batch_results:
            all_first_passage_times.extend(result['first_passage_times'])
            total_liquidations += result['liquidation_count']
            
            for key in all_jump_stats:
                all_jump_stats[key].extend(result['jump_statistics'][key])
        
        # Calculate statistics
        liquidation_probability = total_liquidations / self.sim_params.n_paths
        
        results = {
            'simulation_parameters': {
                'n_paths': self.sim_params.n_paths,
                'time_horizon': self.sim_params.time_horizon,
                'w_X': self.sim_params.w_X,
                'w_Y': self.sim_params.w_Y,
                'dt': self.sim_params.dt
            },
            'liquidation_statistics': {
                'liquidation_probability': liquidation_probability,
                'n_liquidations': total_liquidations,
                'survival_probability': 1 - liquidation_probability
            },
            'jump_statistics': {
                'mean_X_jumps': np.mean(all_jump_stats['n_X_jumps']),
                'mean_Y_jumps': np.mean(all_jump_stats['n_Y_jumps']),
                'mean_total_jumps': np.mean(all_jump_stats['total_jumps']),
                'std_X_jumps': np.std(all_jump_stats['n_X_jumps']),
                'std_Y_jumps': np.std(all_jump_stats['n_Y_jumps']),
                'std_total_jumps': np.std(all_jump_stats['total_jumps'])
            }
        }
        
        # First passage time statistics
        if all_first_passage_times:
            fpt_array = np.array(all_first_passage_times)
            results['first_passage_statistics'] = {
                'mean': np.mean(fpt_array),
                'std': np.std(fpt_array),
                'median': np.median(fpt_array),
                'q25': np.quantile(fpt_array, 0.25),
                'q75': np.quantile(fpt_array, 0.75),
                'min': np.min(fpt_array),
                'max': np.max(fpt_array)
            }
        else:
            results['first_passage_statistics'] = None
        
        # Save results if requested
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save summary statistics
            with open(output_path / 'simulation_results.yaml', 'w') as f:
                yaml.dump(results, f, default_flow_style=False)
            
            # Save first passage times
            if all_first_passage_times:
                pd.DataFrame({
                    'first_passage_time': all_first_passage_times
                }).to_csv(output_path / 'first_passage_times.csv', index=False)
        
        return results
    
    def sensitivity_analysis(self, parameter_ranges: Dict[str, Tuple[float, float]],
                           n_points: int = 10, n_paths_per_point: int = 1000) -> Dict:
        """
        Perform sensitivity analysis by varying model parameters.
        
        Args:
            parameter_ranges: Dict mapping parameter names to (min, max) ranges
            n_points: Number of points to sample in each parameter range
            n_paths_per_point: Number of Monte Carlo paths per parameter point
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        original_params = self.hawkes_params
        sensitivity_results = {}
        
        for param_name, (param_min, param_max) in parameter_ranges.items():
            if not hasattr(original_params, param_name):
                warnings.warn(f"Parameter '{param_name}' not found in HawkesParameters")
                continue
            
            param_values = np.linspace(param_min, param_max, n_points)
            liquidation_probs = []
            mean_fpt_times = []
            
            original_value = getattr(original_params, param_name)
            
            for param_value in param_values:
                # Create modified parameters
                modified_params = HawkesParameters(**{
                    field.name: getattr(original_params, field.name) 
                    for field in original_params.__dataclass_fields__.values()
                })
                setattr(modified_params, param_name, param_value)
                
                # Create temporary simulator
                temp_sim_params = SimulationParameters(
                    n_paths=n_paths_per_point,
                    time_horizon=self.sim_params.time_horizon,
                    dt=self.sim_params.dt,
                    w_X=self.sim_params.w_X,
                    w_Y=self.sim_params.w_Y,
                    n_cores=1  # Use single core for sensitivity analysis
                )
                
                temp_simulator = HawkesSimulator(modified_params, temp_sim_params)
                result = temp_simulator.run_monte_carlo()
                
                liquidation_probs.append(result['liquidation_statistics']['liquidation_probability'])
                
                if result['first_passage_statistics']:
                    mean_fpt_times.append(result['first_passage_statistics']['mean'])
                else:
                    mean_fpt_times.append(np.nan)
            
            # Calculate sensitivity metrics
            base_idx = np.argmin(np.abs(param_values - original_value))
            base_prob = liquidation_probs[base_idx]
            
            sensitivity_results[param_name] = {
                'parameter_values': param_values.tolist(),
                'liquidation_probabilities': liquidation_probs,
                'mean_first_passage_times': mean_fpt_times,
                'base_value': original_value,
                'sensitivity_index': np.std(liquidation_probs) / (base_prob + 1e-6),
                'max_change': np.max(liquidation_probs) - np.min(liquidation_probs)
            }
        
        return sensitivity_results
    
    def validate_with_analytical(self, riccati_solver: RiccatiSolver,
                               gil_pelaez: GilPelaezCDF, tau: float = 1.0,
                               threshold: float = 1.0, n_validation_paths: int = 10000) -> Dict:
        """
        Validate Monte Carlo results against analytical Gil-Pelaez computation.
        
        Args:
            riccati_solver: RiccatiSolver instance
            gil_pelaez: GilPelaezCDF instance  
            tau: Time horizon for comparison
            threshold: Health factor threshold
            n_validation_paths: Number of paths for validation
            
        Returns:
            Dictionary comparing MC and analytical results
        """
        # Run focused Monte Carlo simulation
        validation_sim_params = SimulationParameters(
            n_paths=n_validation_paths,
            time_horizon=tau,
            dt=self.sim_params.dt,
            w_X=self.sim_params.w_X,
            w_Y=self.sim_params.w_Y,
            n_cores=1
        )
        
        validation_simulator = HawkesSimulator(self.hawkes_params, validation_sim_params)
        mc_result = validation_simulator.run_monte_carlo()
        
        # Compute analytical result
        try:
            analytical_result = gil_pelaez.liquidation_probability(
                tau, 
                self.hawkes_params.lambda_X0,
                self.hawkes_params.lambda_Y0,
                threshold
            )
            
            if analytical_result['success']:
                analytical_prob = analytical_result['cdf']
                mc_prob = mc_result['liquidation_statistics']['liquidation_probability']
                
                # Calculate comparison metrics
                absolute_error = abs(analytical_prob - mc_prob)
                relative_error = absolute_error / (analytical_prob + 1e-10)
                
                # Confidence interval for MC estimate
                mc_std_error = np.sqrt(mc_prob * (1 - mc_prob) / n_validation_paths)
                confidence_interval = (
                    mc_prob - 1.96 * mc_std_error,
                    mc_prob + 1.96 * mc_std_error
                )
                
                return {
                    'validation_successful': True,
                    'monte_carlo': {
                        'probability': mc_prob,
                        'std_error': mc_std_error,
                        'confidence_interval': confidence_interval,
                        'n_paths': n_validation_paths
                    },
                    'analytical': {
                        'probability': analytical_prob,
                        'integration_error': analytical_result.get('integration_error', 0)
                    },
                    'comparison': {
                        'absolute_error': absolute_error,
                        'relative_error': relative_error,
                        'within_confidence_interval': (
                            confidence_interval[0] <= analytical_prob <= confidence_interval[1]
                        )
                    },
                    'parameters': {
                        'tau': tau,
                        'threshold': threshold,
                        'lambda_X0': self.hawkes_params.lambda_X0,
                        'lambda_Y0': self.hawkes_params.lambda_Y0
                    }
                }
            else:
                return {
                    'validation_successful': False,
                    'error': f"Analytical computation failed: {analytical_result.get('error')}"
                }
                
        except Exception as e:
            return {
                'validation_successful': False,
                'error': f"Validation failed: {str(e)}"
            }
    
    def convergence_analysis(self, path_counts: List[int], 
                           reference_paths: int = 100000) -> Dict:
        """
        Analyze Monte Carlo convergence by varying the number of simulation paths.
        
        Args:
            path_counts: List of path counts to test
            reference_paths: Number of paths for reference "true" value
            
        Returns:
            Dictionary with convergence analysis results
        """
        # Compute reference value with many paths
        ref_sim_params = SimulationParameters(
            n_paths=reference_paths,
            time_horizon=self.sim_params.time_horizon,
            dt=self.sim_params.dt,
            w_X=self.sim_params.w_X,
            w_Y=self.sim_params.w_Y,
            random_seed=42  # Fixed seed for reproducibility
        )
        
        ref_simulator = HawkesSimulator(self.hawkes_params, ref_sim_params)
        ref_result = ref_simulator.run_monte_carlo()
        ref_prob = ref_result['liquidation_statistics']['liquidation_probability']
        
        # Test convergence for different path counts
        convergence_results = {
            'path_counts': [],
            'liquidation_probabilities': [],
            'absolute_errors': [],
            'relative_errors': [],
            'std_errors': []
        }
        
        for n_paths in sorted(path_counts):
            if n_paths >= reference_paths:
                continue
                
            test_sim_params = SimulationParameters(
                n_paths=n_paths,
                time_horizon=self.sim_params.time_horizon,
                dt=self.sim_params.dt,
                w_X=self.sim_params.w_X,
                w_Y=self.sim_params.w_Y,
                random_seed=42  # Same seed for fair comparison
            )
            
            test_simulator = HawkesSimulator(self.hawkes_params, test_sim_params)
            test_result = test_simulator.run_monte_carlo()
            test_prob = test_result['liquidation_statistics']['liquidation_probability']
            
            # Calculate errors
            abs_error = abs(test_prob - ref_prob)
            rel_error = abs_error / (ref_prob + 1e-10)
            std_error = np.sqrt(test_prob * (1 - test_prob) / n_paths)
            
            convergence_results['path_counts'].append(n_paths)
            convergence_results['liquidation_probabilities'].append(test_prob)
            convergence_results['absolute_errors'].append(abs_error)
            convergence_results['relative_errors'].append(rel_error)
            convergence_results['std_errors'].append(std_error)
        
        return {
            'reference_probability': ref_prob,
            'reference_paths': reference_paths,
            'convergence_results': convergence_results,
            'theoretical_std_error': lambda n: np.sqrt(ref_prob * (1 - ref_prob) / n)
        }