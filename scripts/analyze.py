#!/usr/bin/env python3
"""
Analysis Script for Hawkes Jump-Diffusion Results

This script provides a command-line interface for analyzing simulation results,
creating visualizations, and computing additional statistics from saved results.

Usage:
    python -m scripts.analyze --results simulation_results/ --create-plots

Example:
    hawkes-analyze --results results/baseline/ --compare results/sensitivity/ --output analysis_report.html
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import PlottingUtils, create_summary_statistics, format_results_table


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
        description='Analyze Hawkes jump-diffusion simulation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--results', '-r',
        type=str,
        required=True,
        help='Path to directory containing simulation results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--compare',
        type=str,
        default=None,
        help='Path to second results directory for comparison'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='analysis_results/',
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--create-plots',
        action='store_true',
        help='Create visualization plots'
    )
    
    parser.add_argument(
        '--create-report',
        action='store_true',
        help='Create HTML analysis report'
    )
    
    parser.add_argument(
        '--format',
        choices=['yaml', 'json', 'csv'],
        default='yaml',
        help='Output format for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print summary statistics to console'
    )
    
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    """Load simulation results from directory."""
    results_files = {
        'yaml': results_dir / 'simulation_results.yaml',
        'json': results_dir / 'simulation_results.json'
    }
    
    # Try to load YAML first, then JSON
    for format_type, file_path in results_files.items():
        if file_path.exists():
            if format_type == 'yaml':
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            else:  # json
                with open(file_path, 'r') as f:
                    return json.load(f)
    
    raise FileNotFoundError(f"No results files found in {results_dir}")


def load_first_passage_times(results_dir: Path) -> np.ndarray:
    """Load first passage times from CSV file if available."""
    fpt_file = results_dir / 'first_passage_times.csv'
    
    if fpt_file.exists():
        df = pd.read_csv(fpt_file)
        return df['first_passage_time'].values
    else:
        return np.array([])


def analyze_liquidation_statistics(results: dict) -> dict:
    """Analyze liquidation statistics and compute additional metrics."""
    liq_stats = results.get('liquidation_statistics', {})
    
    analysis = {
        'liquidation_probability': liq_stats.get('liquidation_probability', 0),
        'survival_probability': liq_stats.get('survival_probability', 1),
        'n_liquidations': liq_stats.get('n_liquidations', 0),
        'n_paths': results.get('simulation_parameters', {}).get('n_paths', 0)
    }
    
    # Calculate confidence intervals
    if analysis['n_paths'] > 0:
        p = analysis['liquidation_probability']
        n = analysis['n_paths']
        
        # Wilson score interval
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
        
        analysis['confidence_interval'] = {
            'method': 'Wilson score',
            'confidence_level': 0.95,
            'lower': max(0, center - margin),
            'upper': min(1, center + margin)
        }
    
    return analysis


def analyze_first_passage_times(fpt_times: np.ndarray) -> dict:
    """Analyze first passage time distribution."""
    if len(fpt_times) == 0:
        return {'error': 'No first passage times available'}
    
    analysis = create_summary_statistics(fpt_times, 'first_passage_time')
    
    # Add additional statistics
    analysis['percentiles'] = {
        'p01': np.percentile(fpt_times, 1),
        'p05': np.percentile(fpt_times, 5),
        'p10': np.percentile(fpt_times, 10),
        'p90': np.percentile(fpt_times, 90),
        'p95': np.percentile(fpt_times, 95),
        'p99': np.percentile(fpt_times, 99)
    }
    
    # Fit exponential distribution (common for first passage times)
    try:
        from scipy import stats
        
        # Fit exponential distribution
        exp_params = stats.expon.fit(fpt_times)
        exp_rate = 1.0 / exp_params[1]  # Rate parameter
        
        # Goodness of fit test
        ks_stat, ks_pvalue = stats.kstest(fpt_times, lambda x: stats.expon.cdf(x, *exp_params))
        
        analysis['exponential_fit'] = {
            'rate_parameter': exp_rate,
            'mean_theoretical': exp_params[1],
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'good_fit': ks_pvalue > 0.05
        }
        
    except ImportError:
        analysis['exponential_fit'] = {'error': 'scipy not available for distribution fitting'}
    
    return analysis


def compare_results(results1: dict, results2: dict, label1: str = "Results 1", label2: str = "Results 2") -> dict:
    """Compare two sets of simulation results."""
    comparison = {
        'labels': [label1, label2],
        'liquidation_probability': {
            label1: results1.get('liquidation_statistics', {}).get('liquidation_probability', 0),
            label2: results2.get('liquidation_statistics', {}).get('liquidation_probability', 0)
        },
        'n_paths': {
            label1: results1.get('simulation_parameters', {}).get('n_paths', 0),
            label2: results2.get('simulation_parameters', {}).get('n_paths', 0)
        }
    }
    
    # Calculate difference
    p1 = comparison['liquidation_probability'][label1]
    p2 = comparison['liquidation_probability'][label2]
    
    comparison['difference'] = {
        'absolute': p2 - p1,
        'relative': (p2 - p1) / (p1 + 1e-10),
        'relative_percent': ((p2 - p1) / (p1 + 1e-10)) * 100
    }
    
    # Statistical significance test (if both have enough samples)
    n1 = comparison['n_paths'][label1]
    n2 = comparison['n_paths'][label2]
    
    if n1 > 0 and n2 > 0:
        # Two-proportion z-test
        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
        
        if se > 0:
            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            comparison['significance_test'] = {
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant_at_5pct': p_value < 0.05,
                'significant_at_1pct': p_value < 0.01
            }
    
    return comparison


def create_analysis_plots(results: dict, fpt_times: np.ndarray, output_dir: Path) -> None:
    """Create analysis plots."""
    plotter = PlottingUtils()
    
    # First passage time distribution plot
    if len(fpt_times) > 0:
        fig = plotter.plot_first_passage_distribution(fpt_times)
        fig.savefig(output_dir / 'first_passage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Sensitivity analysis plot (if available)
    if 'sensitivity_analysis' in results:
        fig = plotter.plot_sensitivity_analysis(results['sensitivity_analysis'])
        fig.savefig(output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Convergence analysis plot (if available)
    if 'convergence_analysis' in results:
        fig = plotter.plot_convergence_analysis(results['convergence_analysis'])
        fig.savefig(output_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def create_html_report(analysis_results: dict, output_path: Path) -> None:
    """Create HTML analysis report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hawkes Jump-Diffusion Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2 { color: #2c3e50; }
            .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
            .metric { margin: 10px 0; }
            .metric-label { font-weight: bold; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot { text-align: center; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Hawkes Jump-Diffusion Analysis Report</h1>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
    """
    
    # Add liquidation statistics
    if 'liquidation_analysis' in analysis_results:
        liq_stats = analysis_results['liquidation_analysis']
        html_content += f"""
            <div class="metric">
                <span class="metric-label">Liquidation Probability:</span> 
                {liq_stats.get('liquidation_probability', 0):.4f}
            </div>
            <div class="metric">
                <span class="metric-label">Number of Liquidations:</span> 
                {liq_stats.get('n_liquidations', 0)}
            </div>
            <div class="metric">
                <span class="metric-label">Total Paths:</span> 
                {liq_stats.get('n_paths', 0)}
            </div>
        """
    
    # Add first passage time statistics
    if 'first_passage_analysis' in analysis_results:
        fpt_stats = analysis_results['first_passage_analysis']
        if 'error' not in fpt_stats:
            html_content += f"""
                <div class="metric">
                    <span class="metric-label">Mean First Passage Time:</span> 
                    {fpt_stats.get('first_passage_time_mean', 0):.2f}
                </div>
                <div class="metric">
                    <span class="metric-label">Median First Passage Time:</span> 
                    {fpt_stats.get('first_passage_time_median', 0):.2f}
                </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    """Main analysis function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Hawkes jump-diffusion results analysis")
        logger.info(f"Results directory: {args.results}")
        
        # Validate results directory
        results_dir = Path(args.results)
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {args.results}")
        
        # Load main results
        logger.info("Loading simulation results...")
        results = load_results(results_dir)
        
        # Load first passage times
        fpt_times = load_first_passage_times(results_dir)
        logger.info(f"Loaded {len(fpt_times)} first passage times")
        
        # Analyze results
        logger.info("Analyzing results...")
        analysis_results = {
            'liquidation_analysis': analyze_liquidation_statistics(results),
            'first_passage_analysis': analyze_first_passage_times(fpt_times)
        }
        
        # Add original results for reference
        analysis_results['original_results'] = results
        
        # Compare with second results if provided
        if args.compare:
            logger.info(f"Loading comparison results from {args.compare}")
            compare_dir = Path(args.compare)
            compare_results = load_results(compare_dir)
            
            comparison = compare_results(
                results, compare_results,
                label1="Primary", label2="Comparison"
            )
            analysis_results['comparison'] = comparison
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        if args.format == 'yaml':
            with open(output_dir / 'analysis_results.yaml', 'w') as f:
                yaml.dump(analysis_results, f, default_flow_style=False)
        elif args.format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy(analysis_results)
            with open(output_dir / 'analysis_results.json', 'w') as f:
                json.dump(json_results, f, indent=2)
        
        # Create plots if requested
        if args.create_plots:
            logger.info("Creating analysis plots...")
            create_analysis_plots(results, fpt_times, output_dir)
        
        # Create HTML report if requested
        if args.create_report:
            logger.info("Creating HTML report...")
            create_html_report(analysis_results, output_dir / 'analysis_report.html')
        
        # Print summary if requested
        if args.print_summary:
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            
            liq_analysis = analysis_results['liquidation_analysis']
            print(f"Liquidation Probability: {liq_analysis['liquidation_probability']:.4f}")
            print(f"95% Confidence Interval: [{liq_analysis.get('confidence_interval', {}).get('lower', 0):.4f}, "
                  f"{liq_analysis.get('confidence_interval', {}).get('upper', 1):.4f}]")
            
            fpt_analysis = analysis_results['first_passage_analysis']
            if 'error' not in fpt_analysis:
                print(f"Mean First Passage Time: {fpt_analysis.get('first_passage_time_mean', 0):.2f}")
                print(f"Median First Passage Time: {fpt_analysis.get('first_passage_time_median', 0):.2f}")
            
            if 'comparison' in analysis_results:
                comp = analysis_results['comparison']
                print(f"\nComparison:")
                print(f"Difference: {comp['difference']['absolute']:.4f} "
                      f"({comp['difference']['relative_percent']:.1f}%)")
                if 'significance_test' in comp:
                    sig_test = comp['significance_test']
                    print(f"Statistical significance: p-value = {sig_test['p_value']:.4f}")
            
            print("="*60)
        
        logger.info(f"Analysis completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())