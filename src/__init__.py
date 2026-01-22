"""
Hawkes First Passage Time Analysis for DeFi Lending

This package implements Hawkes jump-diffusion models for analyzing first passage times
in DeFi lending platforms, focusing on optimal long-short positioning and liquidation
risk management.

Modules:
    hawkes_process: Core Hawkes process implementation
    riccati_solver: Riccati ODE solver for characteristic functions
    gil_pelaez: Gil-Pelaez CDF inversion formula
    calibration: Parameter calibration using POT and MLE methods
    simulation: Monte Carlo simulation tools
    monte_carlo: Advanced Monte Carlo simulation engine
    utils: Utility functions and helpers
"""

__version__ = "0.1.0"
__author__ = "Francis Liu"
__email__ = "francis.liu@example.com"

# Import main classes for easy access
from .hawkes_process import HawkesJumpDiffusion
from .riccati_solver import FirstHittingTimeRiccatiSolver
from .gil_pelaez import GilPelaezCDF
from .calibration import POTCalibrator, MLECalibrator
from .simulation import HawkesSimulator
from .monte_carlo import (
    MonteCarloEngine,
    ParallelMonteCarloEngine,
    SimulationConfig,
    SimulationResult,
    FirstPassageResult,
    simulate_cir_paths,
    quick_liquidation_estimate,
)

__all__ = [
    "HawkesJumpDiffusion",
    "FirstHittingTimeRiccatiSolver", 
    "GilPelaezCDF",
    "POTCalibrator",
    "MLECalibrator",
    "HawkesSimulator",
    # Monte Carlo
    "MonteCarloEngine",
    "ParallelMonteCarloEngine",
    "SimulationConfig",
    "SimulationResult",
    "FirstPassageResult",
    "simulate_cir_paths",
    "quick_liquidation_estimate",
]