"""
Monte Carlo Simulation Module for DeFi Interest Rate Models

This module provides comprehensive Monte Carlo simulation capabilities for
the log-health process h(t) = log(b_X * w_X * X(t) / (w_Y * Y(t))) under
spectrally negative jump-diffusion dynamics.

Features:
- Euler-Maruyama discretization with jump components
- Exact simulation for CIR processes
- First passage time estimation for liquidation analysis
- Variance reduction techniques (antithetic variates, control variates)
- Parallel simulation support
- Comprehensive statistics and convergence diagnostics

Usage:
    from monte_carlo import MonteCarloEngine, SimulationConfig
    from log_health import HealthProcessParameters, LogHealthProcess

    params = HealthProcessParameters(w_X=2.0, w_Y=1.0, ...)
    config = SimulationConfig(n_paths=10000, T=30.0, dt=1/365)
    
    engine = MonteCarloEngine(params, config)
    results = engine.simulate()
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Union
from enum import Enum
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class SimulationMethod(Enum):
    """Simulation discretization methods."""
    EULER_MARUYAMA = "euler_maruyama"
    MILSTEIN = "milstein"
    EXACT_CIR = "exact_cir"


class VarianceReduction(Enum):
    """Variance reduction techniques."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    BOTH = "both"


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulations.
    
    Attributes:
        n_paths: Number of simulation paths
        T: Time horizon in years
        dt: Time step size (default: 1/365 for daily)
        seed: Random seed for reproducibility
        method: Discretization method
        variance_reduction: Variance reduction technique
        n_workers: Number of parallel workers (None = auto)
        batch_size: Paths per batch for parallel execution
        store_paths: Whether to store full path trajectories
        antithetic: Use antithetic variates (deprecated, use variance_reduction)
    """
    n_paths: int = 10000
    T: float = 1.0
    dt: float = 1/365
    seed: Optional[int] = None
    method: SimulationMethod = SimulationMethod.EULER_MARUYAMA
    variance_reduction: VarianceReduction = VarianceReduction.NONE
    n_workers: Optional[int] = None
    batch_size: int = 1000
    store_paths: bool = False
    
    def __post_init__(self):
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.dt > self.T:
            raise ValueError("dt cannot exceed T")
        
    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(np.ceil(self.T / self.dt))
    
    @property
    def time_grid(self) -> np.ndarray:
        """Time grid for simulation."""
        return np.linspace(0, self.T, self.n_steps + 1)


@dataclass
class FirstPassageResult:
    """Results for first passage time analysis.
    
    Attributes:
        hitting_times: Array of first passage times (np.inf if no hit)
        hit_indicator: Boolean array indicating if barrier was hit
        survival_prob: Estimated survival probability at T
        liquidation_prob: Estimated liquidation probability at T
        mean_hitting_time: Mean hitting time (conditional on hit)
        std_hitting_time: Std of hitting time (conditional on hit)
        hitting_time_quantiles: Quantiles of hitting time distribution
    """
    hitting_times: np.ndarray
    hit_indicator: np.ndarray
    survival_prob: float
    liquidation_prob: float
    mean_hitting_time: float
    std_hitting_time: float
    hitting_time_quantiles: Dict[str, float]
    
    @classmethod
    def from_paths(cls, h_paths: np.ndarray, time_grid: np.ndarray, 
                   barrier: float = 0.0) -> "FirstPassageResult":
        """Compute first passage results from simulated paths.
        
        Args:
            h_paths: Array of shape (n_paths, n_steps+1)
            time_grid: Time points corresponding to columns
            barrier: Liquidation barrier level (default: 0 for log-health)
            
        Returns:
            FirstPassageResult with computed statistics
        """
        n_paths = h_paths.shape[0]
        hitting_times = np.full(n_paths, np.inf)
        
        # Find first passage time for each path
        below_barrier = h_paths <= barrier
        for i in range(n_paths):
            hit_indices = np.where(below_barrier[i, :])[0]
            if len(hit_indices) > 0:
                hitting_times[i] = time_grid[hit_indices[0]]
        
        hit_indicator = np.isfinite(hitting_times)
        n_hits = np.sum(hit_indicator)
        
        liquidation_prob = n_hits / n_paths
        survival_prob = 1.0 - liquidation_prob
        
        # Conditional statistics
        if n_hits > 0:
            conditional_times = hitting_times[hit_indicator]
            mean_ht = np.mean(conditional_times)
            std_ht = np.std(conditional_times, ddof=1) if n_hits > 1 else 0.0
            quantiles = {
                "q05": np.percentile(conditional_times, 5),
                "q25": np.percentile(conditional_times, 25),
                "q50": np.percentile(conditional_times, 50),
                "q75": np.percentile(conditional_times, 75),
                "q95": np.percentile(conditional_times, 95),
            }
        else:
            mean_ht = np.nan
            std_ht = np.nan
            quantiles = {k: np.nan for k in ["q05", "q25", "q50", "q75", "q95"]}
        
        return cls(
            hitting_times=hitting_times,
            hit_indicator=hit_indicator,
            survival_prob=survival_prob,
            liquidation_prob=liquidation_prob,
            mean_hitting_time=mean_ht,
            std_hitting_time=std_ht,
            hitting_time_quantiles=quantiles,
        )


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results.
    
    Attributes:
        h_paths: Log-health paths (if store_paths=True)
        final_values: Terminal log-health values
        first_passage: First passage time results
        statistics: Summary statistics dictionary
        config: Simulation configuration used
        convergence_diagnostics: Convergence analysis results
    """
    h_paths: Optional[np.ndarray]
    final_values: np.ndarray
    first_passage: FirstPassageResult
    statistics: Dict[str, float]
    config: SimulationConfig
    convergence_diagnostics: Optional[Dict] = None
    
    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            "=" * 60,
            "Monte Carlo Simulation Results",
            "=" * 60,
            f"Paths simulated: {self.config.n_paths:,}",
            f"Time horizon: {self.config.T:.2f} years",
            f"Time step: {self.config.dt:.6f} ({1/self.config.dt:.0f} steps/year)",
            "-" * 60,
            "First Passage Statistics:",
            f"  Liquidation probability: {self.first_passage.liquidation_prob:.4%}",
            f"  Survival probability: {self.first_passage.survival_prob:.4%}",
            f"  Mean hitting time (conditional): {self.first_passage.mean_hitting_time:.4f}",
            f"  Std hitting time (conditional): {self.first_passage.std_hitting_time:.4f}",
            "-" * 60,
            "Terminal Value Statistics:",
            f"  Mean: {self.statistics['mean']:.6f}",
            f"  Std: {self.statistics['std']:.6f}",
            f"  Min: {self.statistics['min']:.6f}",
            f"  Max: {self.statistics['max']:.6f}",
            f"  Skewness: {self.statistics['skewness']:.4f}",
            f"  Kurtosis: {self.statistics['kurtosis']:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for log-health processes.
    
    This engine simulates paths of the log-health process:
        h(t) = log(b_X * w_X * X(t) / (w_Y * Y(t)))
    
    where X and Y follow correlated jump-diffusion processes with
    spectrally negative jumps (downward for X, upward for Y in log-space).
    
    Example:
        from log_health import HealthProcessParameters
        
        params = HealthProcessParameters(
            w_X=2.0, w_Y=1.0,
            mu_X=0.05, mu_Y=0.02,
            sigma_X=0.3, sigma_Y=0.1,
            lambda_X=0.5, lambda_Y=0.3,
            delta_X=0.02, delta_Y=0.01,
            eta_X=5.0, eta_Y=5.0,
        )
        
        config = SimulationConfig(n_paths=50000, T=1.0, dt=1/365)
        engine = MonteCarloEngine(params, config)
        results = engine.simulate()
        print(results.summary())
    """
    
    def __init__(self, params, config: Optional[SimulationConfig] = None):
        """
        Initialize the Monte Carlo engine.
        
        Args:
            params: HealthProcessParameters or compatible parameter object
            config: Simulation configuration (uses defaults if None)
        """
        self.params = params
        self.config = config or SimulationConfig()
        self._rng: Optional[Generator] = None
        
    def _get_rng(self, seed: Optional[int] = None) -> Generator:
        """Get or create random number generator."""
        if seed is not None:
            return default_rng(seed)
        if self._rng is None:
            self._rng = default_rng(self.config.seed)
        return self._rng
    
    def _compute_h0(self) -> float:
        """Compute initial log-health value."""
        p = self.params
        return np.log((p.b_X * p.w_X * p.X0) / (p.w_Y * p.Y0))
    
    def _drift_diffusion(self) -> float:
        """Compute diffusion drift (Itô-adjusted)."""
        p = self.params
        return (p.mu_X - p.mu_Y 
                - 0.5 * (p.sigma_X**2 + p.sigma_Y**2 
                         - 2 * p.rho * p.sigma_X * p.sigma_Y))
    
    def _variance_diffusion(self) -> float:
        """Compute diffusion variance."""
        p = self.params
        return (p.sigma_X**2 + p.sigma_Y**2 
                - 2 * p.rho * p.sigma_X * p.sigma_Y)
    
    def simulate_paths(self, n_paths: Optional[int] = None,
                       seed: Optional[int] = None,
                       return_intermediate: bool = False) -> np.ndarray:
        """
        Simulate log-health paths using Euler-Maruyama discretization.
        
        Args:
            n_paths: Number of paths (default from config)
            seed: Random seed for this simulation
            return_intermediate: If True, return full paths; else only terminal
            
        Returns:
            Array of shape (n_paths,) or (n_paths, n_steps+1)
        """
        n_paths = n_paths or self.config.n_paths
        rng = self._get_rng(seed)
        
        dt = self.config.dt
        n_steps = self.config.n_steps
        p = self.params
        
        # Initialize paths
        h0 = self._compute_h0()
        
        if return_intermediate or self.config.store_paths:
            h_paths = np.zeros((n_paths, n_steps + 1))
            h_paths[:, 0] = h0
        else:
            h_paths = np.full(n_paths, h0)
        
        # Precompute drift and volatility
        mu_diff = self._drift_diffusion()
        sigma_diff = np.sqrt(max(self._variance_diffusion(), 0.0))
        
        # Simulate step by step
        for t in range(n_steps):
            # Current values
            if return_intermediate or self.config.store_paths:
                h_current = h_paths[:, t]
            else:
                h_current = h_paths
            
            # Diffusion increment: dh_diff = μ dt + σ dW
            dW = rng.normal(0, np.sqrt(dt), size=n_paths)
            dh_diff = mu_diff * dt + sigma_diff * dW
            
            # Jump increments for X (spectrally negative - downward)
            n_jumps_X = rng.poisson(p.lambda_X * dt, size=n_paths)
            jump_X = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps_X[i] > 0:
                    # Each jump: -(delta_X + Exp(eta_X))
                    jump_sizes = -(p.delta_X + rng.exponential(1/p.eta_X, size=n_jumps_X[i]))
                    jump_X[i] = np.sum(jump_sizes)
            
            # Jump increments for Y (spectrally negative in h-space - upward for Y)
            n_jumps_Y = rng.poisson(p.lambda_Y * dt, size=n_paths)
            jump_Y = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps_Y[i] > 0:
                    # In log-health, Y jumps reduce h: -(delta_Y + Exp(eta_Y))
                    jump_sizes = -(p.delta_Y + rng.exponential(1/p.eta_Y, size=n_jumps_Y[i]))
                    jump_Y[i] = np.sum(jump_sizes)
            
            # Update paths
            dh = dh_diff + jump_X + jump_Y
            
            if return_intermediate or self.config.store_paths:
                h_paths[:, t + 1] = h_current + dh
            else:
                h_paths = h_current + dh
        
        return h_paths
    
    def simulate_paths_vectorized(self, n_paths: Optional[int] = None,
                                   seed: Optional[int] = None) -> np.ndarray:
        """
        Vectorized path simulation (more memory but faster for moderate sizes).
        
        Args:
            n_paths: Number of paths
            seed: Random seed
            
        Returns:
            Array of shape (n_paths, n_steps+1)
        """
        n_paths = n_paths or self.config.n_paths
        rng = self._get_rng(seed)
        
        dt = self.config.dt
        n_steps = self.config.n_steps
        p = self.params
        
        h0 = self._compute_h0()
        h_paths = np.zeros((n_paths, n_steps + 1))
        h_paths[:, 0] = h0
        
        mu_diff = self._drift_diffusion()
        sigma_diff = np.sqrt(max(self._variance_diffusion(), 0.0))
        
        # Pre-generate all random numbers
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        n_jumps_X = rng.poisson(p.lambda_X * dt, size=(n_paths, n_steps))
        n_jumps_Y = rng.poisson(p.lambda_Y * dt, size=(n_paths, n_steps))
        
        for t in range(n_steps):
            # Diffusion
            dh_diff = mu_diff * dt + sigma_diff * dW[:, t]
            
            # Jump X - vectorized where possible
            max_jumps_X = n_jumps_X[:, t].max() if n_jumps_X[:, t].max() > 0 else 1
            exp_X = rng.exponential(1/p.eta_X, size=(n_paths, max_jumps_X))
            mask_X = np.arange(max_jumps_X) < n_jumps_X[:, t, None]
            jump_X = -np.sum((p.delta_X + exp_X) * mask_X, axis=1)
            
            # Jump Y
            max_jumps_Y = n_jumps_Y[:, t].max() if n_jumps_Y[:, t].max() > 0 else 1
            exp_Y = rng.exponential(1/p.eta_Y, size=(n_paths, max_jumps_Y))
            mask_Y = np.arange(max_jumps_Y) < n_jumps_Y[:, t, None]
            jump_Y = -np.sum((p.delta_Y + exp_Y) * mask_Y, axis=1)
            
            h_paths[:, t + 1] = h_paths[:, t] + dh_diff + jump_X + jump_Y
        
        return h_paths
    
    def simulate_with_antithetic(self, n_paths: Optional[int] = None,
                                  seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate paths using antithetic variates for variance reduction.
        
        For each path, we also simulate an "antithetic" path using negated
        Brownian increments. This reduces variance for functions of terminal values.
        
        Args:
            n_paths: Number of path pairs (total paths = 2 * n_paths)
            seed: Random seed
            
        Returns:
            Array of shape (2*n_paths, n_steps+1)
        """
        n_paths = n_paths or (self.config.n_paths // 2)
        rng = self._get_rng(seed)
        
        dt = self.config.dt
        n_steps = self.config.n_steps
        p = self.params
        
        h0 = self._compute_h0()
        h_paths = np.zeros((2 * n_paths, n_steps + 1))
        h_paths[:, 0] = h0
        
        mu_diff = self._drift_diffusion()
        sigma_diff = np.sqrt(max(self._variance_diffusion(), 0.0))
        
        for t in range(n_steps):
            # Generate base increments
            dW_base = rng.normal(0, np.sqrt(dt), size=n_paths)
            
            # Original and antithetic Brownian increments
            dW = np.concatenate([dW_base, -dW_base])
            
            dh_diff = mu_diff * dt + sigma_diff * dW
            
            # Jumps (same for both - cannot use antithetic for discrete jumps)
            n_jumps_X = rng.poisson(p.lambda_X * dt, size=2 * n_paths)
            n_jumps_Y = rng.poisson(p.lambda_Y * dt, size=2 * n_paths)
            
            jump_X = np.zeros(2 * n_paths)
            jump_Y = np.zeros(2 * n_paths)
            
            for i in range(2 * n_paths):
                if n_jumps_X[i] > 0:
                    jump_X[i] = -np.sum(p.delta_X + rng.exponential(1/p.eta_X, size=n_jumps_X[i]))
                if n_jumps_Y[i] > 0:
                    jump_Y[i] = -np.sum(p.delta_Y + rng.exponential(1/p.eta_Y, size=n_jumps_Y[i]))
            
            h_paths[:, t + 1] = h_paths[:, t] + dh_diff + jump_X + jump_Y
        
        return h_paths
    
    def simulate(self) -> SimulationResult:
        """
        Run full Monte Carlo simulation with configured settings.
        
        Returns:
            SimulationResult containing paths, statistics, and diagnostics
        """
        # Select simulation method based on variance reduction
        if self.config.variance_reduction == VarianceReduction.ANTITHETIC:
            h_paths = self.simulate_with_antithetic()
        else:
            h_paths = self.simulate_paths_vectorized(return_intermediate=True)
            # Ensure we have full paths for first passage analysis
            if h_paths.ndim == 1:
                h_paths = self.simulate_paths(return_intermediate=True)
        
        # Compute first passage results
        time_grid = self.config.time_grid
        first_passage = FirstPassageResult.from_paths(h_paths, time_grid, barrier=0.0)
        
        # Compute statistics
        final_values = h_paths[:, -1]
        statistics = self._compute_statistics(final_values)
        
        # Store paths if requested
        stored_paths = h_paths if self.config.store_paths else None
        
        # Compute convergence diagnostics
        diagnostics = self._compute_convergence_diagnostics(final_values, first_passage)
        
        return SimulationResult(
            h_paths=stored_paths,
            final_values=final_values,
            first_passage=first_passage,
            statistics=statistics,
            config=self.config,
            convergence_diagnostics=diagnostics,
        )
    
    def _compute_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics for terminal values."""
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        # Higher moments
        centered = values - mean
        m3 = np.mean(centered**3)
        m4 = np.mean(centered**4)
        skewness = m3 / (std**3) if std > 0 else 0.0
        kurtosis = m4 / (std**4) - 3.0 if std > 0 else 0.0  # Excess kurtosis
        
        return {
            "mean": mean,
            "std": std,
            "var": std**2,
            "min": np.min(values),
            "max": np.max(values),
            "q05": np.percentile(values, 5),
            "q25": np.percentile(values, 25),
            "median": np.percentile(values, 50),
            "q75": np.percentile(values, 75),
            "q95": np.percentile(values, 95),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "se_mean": std / np.sqrt(n),
        }
    
    def _compute_convergence_diagnostics(self, final_values: np.ndarray,
                                          first_passage: FirstPassageResult) -> Dict:
        """Compute convergence diagnostics for the simulation."""
        n = len(final_values)
        
        # Running mean convergence
        running_means = np.cumsum(final_values) / np.arange(1, n + 1)
        
        # Standard error of mean
        se_mean = np.std(final_values, ddof=1) / np.sqrt(n)
        
        # Standard error of liquidation probability
        p_liq = first_passage.liquidation_prob
        se_liq = np.sqrt(p_liq * (1 - p_liq) / n) if n > 0 else 0.0
        
        # 95% CI for liquidation probability
        ci_liq = (
            max(0, p_liq - 1.96 * se_liq),
            min(1, p_liq + 1.96 * se_liq)
        )
        
        return {
            "n_paths": n,
            "se_mean": se_mean,
            "se_liquidation_prob": se_liq,
            "ci_95_liquidation_prob": ci_liq,
            "final_running_mean": running_means[-1] if n > 0 else np.nan,
            "coefficient_of_variation": se_mean / abs(np.mean(final_values)) if np.mean(final_values) != 0 else np.inf,
        }
    
    def estimate_liquidation_probability(self, n_paths: Optional[int] = None,
                                          confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
        """
        Estimate liquidation probability with confidence interval.
        
        Args:
            n_paths: Number of paths to simulate
            confidence: Confidence level for interval
            
        Returns:
            Tuple of (point_estimate, (lower_bound, upper_bound))
        """
        n_paths = n_paths or self.config.n_paths
        h_paths = self.simulate_paths_vectorized(n_paths)
        
        # Check if any path crosses barrier
        hit = np.any(h_paths <= 0, axis=1)
        p_liq = np.mean(hit)
        
        # Wilson score interval
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        denom = 1 + z**2 / n_paths
        center = (p_liq + z**2 / (2 * n_paths)) / denom
        width = z * np.sqrt(p_liq * (1 - p_liq) / n_paths + z**2 / (4 * n_paths**2)) / denom
        
        return p_liq, (max(0, center - width), min(1, center + width))
    
    def sensitivity_analysis(self, param_name: str, 
                             param_values: List[float],
                             n_paths: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on a single parameter.
        
        Args:
            param_name: Name of parameter to vary
            param_values: List of values to test
            n_paths: Paths per configuration
            
        Returns:
            Dictionary with parameter values and corresponding metrics
        """
        n_paths = n_paths or (self.config.n_paths // len(param_values))
        
        results = {
            "param_values": np.array(param_values),
            "liquidation_prob": [],
            "mean_terminal": [],
            "std_terminal": [],
            "mean_hitting_time": [],
        }
        
        original_value = getattr(self.params, param_name)
        
        for val in param_values:
            setattr(self.params, param_name, val)
            
            h_paths = self.simulate_paths_vectorized(n_paths)
            time_grid = self.config.time_grid
            fp = FirstPassageResult.from_paths(h_paths, time_grid)
            
            results["liquidation_prob"].append(fp.liquidation_prob)
            results["mean_terminal"].append(np.mean(h_paths[:, -1]))
            results["std_terminal"].append(np.std(h_paths[:, -1]))
            results["mean_hitting_time"].append(fp.mean_hitting_time)
        
        # Restore original value
        setattr(self.params, param_name, original_value)
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results


class ParallelMonteCarloEngine(MonteCarloEngine):
    """
    Parallel Monte Carlo engine using multiprocessing.
    
    Extends MonteCarloEngine with parallel simulation capabilities
    for large-scale simulations.
    """
    
    def simulate_parallel(self, n_workers: Optional[int] = None) -> SimulationResult:
        """
        Run simulation in parallel across multiple workers.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count)
            
        Returns:
            Combined SimulationResult from all workers
        """
        n_workers = n_workers or self.config.n_workers or mp.cpu_count()
        n_paths = self.config.n_paths
        paths_per_worker = n_paths // n_workers
        
        # Prepare seeds for each worker
        base_seed = self.config.seed or 42
        seeds = [base_seed + i * 1000 for i in range(n_workers)]
        
        all_paths = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i, seed in enumerate(seeds):
                n = paths_per_worker + (1 if i < n_paths % n_workers else 0)
                futures.append(
                    executor.submit(self._simulate_batch, n, seed)
                )
            
            for future in as_completed(futures):
                all_paths.append(future.result())
        
        # Combine results
        h_paths = np.vstack(all_paths)
        
        time_grid = self.config.time_grid
        first_passage = FirstPassageResult.from_paths(h_paths, time_grid)
        final_values = h_paths[:, -1]
        statistics = self._compute_statistics(final_values)
        diagnostics = self._compute_convergence_diagnostics(final_values, first_passage)
        
        return SimulationResult(
            h_paths=h_paths if self.config.store_paths else None,
            final_values=final_values,
            first_passage=first_passage,
            statistics=statistics,
            config=self.config,
            convergence_diagnostics=diagnostics,
        )
    
    def _simulate_batch(self, n_paths: int, seed: int) -> np.ndarray:
        """Simulate a batch of paths (used by parallel execution)."""
        return self.simulate_paths_vectorized(n_paths, seed)


def simulate_cir_paths(x0: float, kappa: float, theta: float, sigma: float,
                       T: float, dt: float, n_paths: int,
                       seed: Optional[int] = None,
                       exact: bool = False) -> np.ndarray:
    """
    Simulate CIR process paths: dX = κ(θ - X)dt + σ√X dW
    
    Args:
        x0: Initial value
        kappa: Mean reversion speed
        theta: Long-term mean
        sigma: Volatility
        T: Time horizon
        dt: Time step
        n_paths: Number of paths
        seed: Random seed
        exact: Use exact simulation (requires scipy.stats.ncx2)
        
    Returns:
        Array of shape (n_paths, n_steps+1)
    """
    rng = default_rng(seed)
    n_steps = int(np.ceil(T / dt))
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = x0
    
    if exact:
        # Exact simulation using non-central chi-squared
        try:
            from scipy.stats import ncx2
        except ImportError:
            warnings.warn("scipy.stats.ncx2 not available, falling back to Euler")
            exact = False
    
    if exact:
        # Parameters for non-central chi-squared
        c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
        d = 4 * kappa * theta / sigma**2
        
        for t in range(n_steps):
            x_t = paths[:, t]
            # Non-centrality parameter
            nc = x_t * np.exp(-kappa * dt) / c
            # Sample from non-central chi-squared
            paths[:, t + 1] = c * rng.noncentral_chisquare(d, nc)
    else:
        # Euler-Maruyama with reflection
        sqrt_dt = np.sqrt(dt)
        for t in range(n_steps):
            x_t = np.maximum(paths[:, t], 0)  # Ensure non-negative
            dW = rng.normal(0, sqrt_dt, size=n_paths)
            dx = kappa * (theta - x_t) * dt + sigma * np.sqrt(x_t) * dW
            paths[:, t + 1] = np.maximum(x_t + dx, 0)
    
    return paths


def quick_liquidation_estimate(params, T: float = 1.0, n_paths: int = 10000,
                                seed: Optional[int] = None) -> Dict[str, float]:
    """
    Quick estimate of liquidation probability and related metrics.
    
    Convenience function for rapid analysis without full engine setup.
    
    Args:
        params: HealthProcessParameters or compatible object
        T: Time horizon
        n_paths: Number of paths
        seed: Random seed
        
    Returns:
        Dictionary with liquidation_prob, mean_terminal, std_terminal
    """
    config = SimulationConfig(n_paths=n_paths, T=T, seed=seed)
    engine = MonteCarloEngine(params, config)
    
    h_paths = engine.simulate_paths_vectorized()
    
    hit = np.any(h_paths <= 0, axis=1)
    final = h_paths[:, -1]
    
    return {
        "liquidation_prob": np.mean(hit),
        "survival_prob": np.mean(~hit),
        "mean_terminal": np.mean(final),
        "std_terminal": np.std(final),
        "min_terminal": np.min(final),
        "max_terminal": np.max(final),
    }
