#!/usr/bin/env python3
"""
Calibration Script for Hawkes Jump-Diffusion Parameters

This script provides a command-line interface for calibrating Hawkes jump-diffusion
model parameters from market data using the Peak-Over-Threshold (POT) method
and Maximum Likelihood Estimation (MLE).

Usage:
    python -m scripts.calibrate --data data/raw/market_data.csv --output results/calibration.yaml

Example:
    hawkes-calibrate --data data/raw/eth_usdc_data.csv --threshold-quantile 0.95 --output calibration_results.yaml
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calibration import CalibrationData, POTParameters, full_calibration_pipeline
from utils import load_config, save_results, format_results_table


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
        description='Calibrate Hawkes jump-diffusion parameters from market data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to CSV file containing market data'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='calibration_results.yaml',
        help='Output file path for calibration results'
    )
    
    parser.add_argument(
        '--threshold-quantile',
        type=float,
        default=0.95,
        help='Quantile threshold for POT jump detection (0.90-0.99)'
    )
    
    parser.add_argument(
        '--min-jump-size',
        type=float,
        default=0.01,
        help='Minimum jump size to consider significant'
    )
    
    parser.add_argument(
        '--min-cluster-separation',
        type=int,
        default=1,
        help='Minimum time steps between jumps in same cluster'
    )
    
    parser.add_argument(
        '--timestamp-col',
        type=str,
        default='timestamp',
        help='Name of timestamp column in CSV'
    )
    
    parser.add_argument(
        '--collateral-return-col',
        type=str,
        default='collateral_return',
        help='Name of collateral return column in CSV'
    )
    
    parser.add_argument(
        '--borrowed-return-col',
        type=str,
        default='borrowed_return',
        help='Name of borrowed return column in CSV'
    )
    
    parser.add_argument(
        '--collateral-price-col',
        type=str,
        default='collateral_price',
        help='Name of collateral price column in CSV'
    )
    
    parser.add_argument(
        '--borrowed-price-col',
        type=str,
        default='borrowed_price',
        help='Name of borrowed price column in CSV'
    )
    
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
    # Check data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Validate threshold quantile
    if not (0.90 <= args.threshold_quantile <= 0.99):
        raise ValueError("Threshold quantile must be between 0.90 and 0.99")
    
    # Validate minimum jump size
    if args.min_jump_size <= 0:
        raise ValueError("Minimum jump size must be positive")


def main():
    """Main calibration function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Hawkes parameter calibration")
        logger.info(f"Data file: {args.data}")
        logger.info(f"Output file: {args.output}")
        
        # Validate arguments
        validate_arguments(args)
        
        # Load data
        logger.info("Loading market data...")
        try:
            data = CalibrationData.from_csv(
                args.data,
                timestamp_col=args.timestamp_col,
                collateral_return_col=args.collateral_return_col,
                borrowed_return_col=args.borrowed_return_col,
                collateral_price_col=args.collateral_price_col,
                borrowed_price_col=args.borrowed_price_col
            )
            logger.info(f"Loaded {len(data.timestamps)} data points")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            sys.exit(1)
        
        # Setup POT parameters
        pot_params = POTParameters(
            threshold_quantile=args.threshold_quantile,
            min_jump_size=args.min_jump_size,
            min_cluster_separation=args.min_cluster_separation
        )
        
        logger.info(f"POT parameters: threshold_quantile={pot_params.threshold_quantile}, "
                   f"min_jump_size={pot_params.min_jump_size}")
        
        # Run calibration pipeline
        logger.info("Running calibration pipeline...")
        results = full_calibration_pipeline(
            data=data,
            pot_params=pot_params,
            output_path=args.output
        )
        
        # Check if calibration was successful
        if not results['success']:
            logger.error(f"Calibration failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Save results
        logger.info(f"Saving results to {args.output}")
        save_results(results, args.output, format='yaml')
        
        # Print summary
        logger.info("Calibration completed successfully!")
        logger.info(f"Collateral jumps detected: {results['calibration_metadata']['n_collateral_jumps']}")
        logger.info(f"Borrowed jumps detected: {results['calibration_metadata']['n_borrowed_jumps']}")
        
        if results['goodness_of_fit']['log_likelihood']:
            logger.info(f"Log-likelihood: {results['goodness_of_fit']['log_likelihood']:.2f}")
            logger.info(f"AIC: {results['goodness_of_fit']['aic']:.2f}")
            logger.info(f"BIC: {results['goodness_of_fit']['bic']:.2f}")
        
        # Print formatted table if requested
        if args.print_table:
            print("\n" + format_results_table(results))
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())