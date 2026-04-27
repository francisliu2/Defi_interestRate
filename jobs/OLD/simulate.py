#!/usr/bin/env python3
"""
Monte Carlo Simulation Script for Hawkes Jump-Diffusion Process

This script provides a command-line interface for running Monte Carlo simulations
of the Hawkes jump-diffusion process to analyze first passage time statistics
and liquidation probabilities.

Usage:
    python -m scripts.simulate --config config/model_parameters.yaml --n-paths 10000

Example:
    hawkes-simulate --config model_params.yaml --n-paths 50000 --time-horizon 30 --output simulation_results/
"""

import argparse
import sys
from pathlib import Path
import logging
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hawkes_process import HawkesParameters
from simulation import HawkesSimulator, SimulationParameters
from utils import load_config, save_results, format_results_table, PlottingUtils


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Monte Carlo simulation for Hawkes jump-diffusion process',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file containing model parameters'
    )
    
    # Simulation parameters
    parser.add_argument(
        '--n-paths',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulation paths'
    )
    
    parser.add_argument(
        '--time-horizon',
        type=float,
        default=30.0,
        help='Time horizon for simulation (days)'
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step for simulation'
    )
    
    # Portfolio parameters
    parser.add_argument(
        '--w-X',
        type=float,
        default=10.0,
        help='Collateral weight'
    )
    
    parser.add_argument(
        '--w-Y',
        type=float,
        default=15000.0,
        help='Borrowed weight'
    )
    
    # Processing parameters
    parser.add_argument(
        '--n-cores',
        type=int,
        default=None,
        help='Number of CPU cores to use (None for auto-detect)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for parallel processing'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    # Output parameters
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='simulation_results/',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-paths',
        action='store_true',
        help='Save individual simulation paths (warning: large files)'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create visualization plots'
    )
    
    # Analysis options
    parser.add_argument(
        '--sensitivity-analysis',
        action='store_true',
        help='Run sensitivity analysis'
    )
    
    parser.add_argument(
        '--convergence-analysis',
        action='store_true',
        help='Run convergence analysis'
    )
    
    # Display options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--print-table',
        action='store_true',
        help='Print formatted results table to console'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Validate simulation parameters
    if args.n_paths <= 0:
        raise ValueError("Number of paths must be positive")
    
    if args.time_horizon <= 0:
        raise ValueError("Time horizon must be positive")
    
    if args.dt <= 0 or args.dt > args.time_horizon:
        raise ValueError("Time step must be positive and less than time horizon")
    
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")


def run_sensitivity_analysis(simulator: HawkesSimulator, logger: logging.Logger) -> dict:
    """Run sensitivity analysis for key parameters."""
    logger.info("Running sensitivity analysis...")
    
    # Define parameter ranges for sensitivity analysis
    parameter_ranges = {
        'sigma_h': (0.5, 1.2),
        'alpha_XX': (0.3, 0.8),
        'alpha_YY': (0.3, 0.8),
        'eta_X': (1.0, 3.0),
        'eta_Y': (1.0, 3.0)
    }
    
    return simulator.sensitivity_analysis(
        parameter_ranges=parameter_ranges,
        n_points=8,
        n_paths_per_point=2000
    )


def run_convergence_analysis(simulator: HawkesSimulator, logger: logging.Logger) -> dict:
    """Run convergence analysis for Monte Carlo estimates."""
    logger.info("Running convergence analysis...")
    
    path_counts = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    return simulator.convergence_analysis(
        path_counts=path_counts,
        reference_paths=100000
    )


def main():
    """Main simulation function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Hawkes jump-diffusion simulation")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Output directory: {args.output}")
        
        # Validate arguments
        validate_arguments(args)
        
        # Load configuration
        logger.info("Loading model parameters...")
        try:
            config = load_config(args.config)
            hawkes_params = HawkesParameters.from_yaml(args.config)
            logger.info("Model parameters loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
        
        # Setup simulation parameters
        sim_params = SimulationParameters(
            n_paths=args.n_paths,
            time_horizon=args.time_horizon,
            dt=args.dt,
            w_X=args.w_X,
            w_Y=args.w_Y,
            n_cores=args.n_cores,
            batch_size=args.batch_size,
            random_seed=args.random_seed
        )
        
        logger.info(f"Simulation parameters: {args.n_paths} paths, "
                   f"T={args.time_horizon}, dt={args.dt}")
        logger.info(f"Portfolio weights: w_X={args.w_X}, w_Y={args.w_Y}")
        
        # Create simulator
        simulator = HawkesSimulator(hawkes_params, sim_params)
        
        # Run main simulation
        logger.info("Starting Monte Carlo simulation...")
        start_time = time.time()
        
        results = simulator.run_monte_carlo(
            save_paths=args.save_paths,
            output_dir=args.output
        )
        
        simulation_time = time.time() - start_time
        logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
        
        # Add timing information to results
        results['computation_time'] = {
            'simulation_time_seconds': simulation_time,
            'paths_per_second': args.n_paths / simulation_time
        }
        
        # Run additional analyses if requested
        if args.sensitivity_analysis:
            sensitivity_results = run_sensitivity_analysis(simulator, logger)
            results['sensitivity_analysis'] = sensitivity_results
        
        if args.convergence_analysis:
            convergence_results = run_convergence_analysis(simulator, logger)
            results['convergence_analysis'] = convergence_results
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        save_results(results, output_dir / 'simulation_results.yaml', format='yaml')
        save_results(results, output_dir / 'simulation_results.json', format='json')
        
        # Create plots if requested
        if args.create_plots:
            logger.info("Creating visualization plots...")
            plotter = PlottingUtils()
            
            # First passage time distribution plot
            if results['first_passage_statistics']:
                # Note: This would require loading the individual first passage times
                # which are saved separately in first_passage_times.csv
                logger.info("First passage time plots would be created here")
            
            # Sensitivity analysis plots
            if args.sensitivity_analysis:
                fig = plotter.plot_sensitivity_analysis(sensitivity_results)
                fig.savefig(output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # Convergence analysis plots
            if args.convergence_analysis:
                fig = plotter.plot_convergence_analysis(convergence_results)
                fig.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Print summary
        logger.info("Simulation completed successfully!")
        logger.info(f"Liquidation probability: {results['liquidation_statistics']['liquidation_probability']:.4f}")
        logger.info(f"Number of liquidations: {results['liquidation_statistics']['n_liquidations']}")
        
        if results['first_passage_statistics']:
            logger.info(f"Mean first passage time: {results['first_passage_statistics']['mean']:.2f}")
            logger.info(f"Median first passage time: {results['first_passage_statistics']['median']:.2f}")
        
        # Print formatted table if requested
        if args.print_table:
            print("\n" + format_results_table(results))
        
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())