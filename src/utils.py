"""
Utility Functions and Helper Classes

This module provides utility functions for data processing, visualization,
statistical analysis, and configuration management for the Hawkes jump-diffusion
first passage time analysis package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Basic validation
        required_sections = ['health_factor', 'diffusion', 'jump_sizes', 'hawkes_intensities']
        for section in required_sections:
            if section not in config:
                warnings.warn(f"Missing configuration section: {section}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def save_results(results: Dict[str, Any], output_path: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    Save results to file in specified format.
    
    Args:
        results: Dictionary with results to save
        output_path: Output file path
        format: Output format ('yaml', 'json', or 'csv' for DataFrame-like results)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to results
    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    if format.lower() == 'yaml':
        with open(output_path.with_suffix('.yaml'), 'w') as f:
            yaml.dump(results_with_metadata, f, default_flow_style=False)
    
    elif format.lower() == 'json':
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
        
        json_results = convert_numpy(results_with_metadata)
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_summary_statistics(data: np.ndarray, name: str = "Data") -> Dict[str, float]:
    """
    Create comprehensive summary statistics for a data array.
    
    Args:
        data: Input data array
        name: Name for the dataset
        
    Returns:
        Dictionary with summary statistics
    """
    if len(data) == 0:
        return {'error': 'Empty dataset'}
    
    return {
        f'{name}_count': len(data),
        f'{name}_mean': np.mean(data),
        f'{name}_std': np.std(data),
        f'{name}_min': np.min(data),
        f'{name}_max': np.max(data),
        f'{name}_median': np.median(data),
        f'{name}_q25': np.quantile(data, 0.25),
        f'{name}_q75': np.quantile(data, 0.75),
        f'{name}_skewness': _calculate_skewness(data),
        f'{name}_kurtosis': _calculate_kurtosis(data)
    }


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate sample skewness."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    
    n = len(data)
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate sample kurtosis (excess kurtosis)."""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if std == 0:
        return 0.0
    
    n = len(data)
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3.0
    return kurt


def bootstrap_confidence_interval(data: np.ndarray, statistic: callable,
                                confidence_level: float = 0.95,
                                n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic: Function that computes the statistic (e.g., np.mean)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return (
        np.percentile(bootstrap_stats, lower_percentile),
        np.percentile(bootstrap_stats, upper_percentile)
    )


class PlottingUtils:
    """Utility class for creating standard plots and visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize plotting utilities.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
            warnings.warn(f"Style '{style}' not available, using default")
    
    def plot_health_factor_path(self, time: np.ndarray, health_factor: np.ndarray,
                               jump_times_X: List[float] = None,
                               jump_times_Y: List[float] = None,
                               title: str = "Health Factor Path") -> plt.Figure:
        """
        Plot health factor path with jump markers.
        
        Args:
            time: Time array
            health_factor: Health factor values
            jump_times_X: X jump times (optional)
            jump_times_Y: Y jump times (optional)
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot health factor path
        ax.plot(time, health_factor, 'b-', linewidth=1.5, alpha=0.8, label='Health Factor')
        
        # Add liquidation threshold
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        # Mark jumps
        if jump_times_X:
            for jump_time in jump_times_X:
                if jump_time <= time[-1]:
                    idx = np.argmin(np.abs(time - jump_time))
                    ax.scatter(jump_time, health_factor[idx], color='red', 
                             marker='v', s=50, alpha=0.8, zorder=5)
        
        if jump_times_Y:
            for jump_time in jump_times_Y:
                if jump_time <= time[-1]:
                    idx = np.argmin(np.abs(time - jump_time))
                    ax.scatter(jump_time, health_factor[idx], color='orange', 
                             marker='^', s=50, alpha=0.8, zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Health Factor')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_first_passage_distribution(self, first_passage_times: np.ndarray,
                                       bins: int = 50, 
                                       title: str = "First Passage Time Distribution") -> plt.Figure:
        """
        Plot histogram and density of first passage times.
        
        Args:
            first_passage_times: Array of first passage times
            bins: Number of histogram bins
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(first_passage_times, bins=bins, alpha=0.7, density=True, color='skyblue')
        ax1.set_xlabel('First Passage Time')
        ax1.set_ylabel('Density')
        ax1.set_title('Histogram')
        ax1.grid(True, alpha=0.3)
        
        # Empirical CDF
        sorted_times = np.sort(first_passage_times)
        y_values = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, y_values, 'b-', linewidth=2)
        ax2.set_xlabel('First Passage Time')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Empirical CDF')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, Dict],
                                 title: str = "Sensitivity Analysis") -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Args:
            sensitivity_results: Results from sensitivity analysis
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        param_names = list(sensitivity_results.keys())[:4]  # Show up to 4 parameters
        
        for i, param_name in enumerate(param_names):
            if i >= 4:
                break
                
            result = sensitivity_results[param_name]
            param_values = result['parameter_values']
            liquidation_probs = result['liquidation_probabilities']
            base_value = result['base_value']
            
            ax = axes[i]
            ax.plot(param_values, liquidation_probs, 'bo-', linewidth=2, markersize=6)
            ax.axvline(x=base_value, color='r', linestyle='--', alpha=0.7, label='Base Value')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Liquidation Probability')
            ax.set_title(f'Sensitivity to {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(param_names), 4):
            axes[i].set_visible(False)
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_convergence_analysis(self, convergence_results: Dict,
                                 title: str = "Monte Carlo Convergence") -> plt.Figure:
        """
        Plot Monte Carlo convergence analysis.
        
        Args:
            convergence_results: Results from convergence analysis
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        conv_data = convergence_results['convergence_results']
        path_counts = conv_data['path_counts']
        liquidation_probs = conv_data['liquidation_probabilities']
        abs_errors = conv_data['absolute_errors']
        std_errors = conv_data['std_errors']
        ref_prob = convergence_results['reference_probability']
        
        # Convergence plot
        ax1.semilogx(path_counts, liquidation_probs, 'bo-', label='MC Estimate')
        ax1.axhline(y=ref_prob, color='r', linestyle='--', label='Reference Value')
        ax1.fill_between(path_counts, 
                        np.array(liquidation_probs) - 1.96 * np.array(std_errors),
                        np.array(liquidation_probs) + 1.96 * np.array(std_errors),
                        alpha=0.3, label='95% CI')
        
        ax1.set_xlabel('Number of Paths')
        ax1.set_ylabel('Liquidation Probability')
        ax1.set_title('Convergence to Reference Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error plot
        ax2.loglog(path_counts, abs_errors, 'ro-', label='Absolute Error')
        ax2.loglog(path_counts, std_errors, 'go-', label='Standard Error')
        
        # Theoretical 1/sqrt(n) line
        theoretical_line = std_errors[0] * np.sqrt(path_counts[0] / np.array(path_counts))
        ax2.loglog(path_counts, theoretical_line, 'k--', alpha=0.7, label=r'$1/\sqrt{n}$')
        
        ax2.set_xlabel('Number of Paths')
        ax2.set_ylabel('Error')
        ax2.set_title('Error vs. Sample Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        return fig


class DataProcessor:
    """Utility class for processing market data and preparing it for calibration."""
    
    @staticmethod
    def compute_returns(prices: np.ndarray, method: str = 'log') -> np.ndarray:
        """
        Compute returns from price series.
        
        Args:
            prices: Array of prices
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            Array of returns
        """
        if method == 'log':
            return np.diff(np.log(prices))
        elif method == 'simple':
            return np.diff(prices) / prices[:-1]
        else:
            raise ValueError("Method must be 'log' or 'simple'")
    
    @staticmethod
    def clean_data(df: pd.DataFrame, columns: List[str], 
                  remove_outliers: bool = True, outlier_std: float = 5.0) -> pd.DataFrame:
        """
        Clean market data by removing NaN values and outliers.
        
        Args:
            df: Input DataFrame
            columns: List of columns to clean
            remove_outliers: Whether to remove outliers
            outlier_std: Number of standard deviations for outlier detection
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove NaN values
        df_clean = df_clean.dropna(subset=columns)
        
        # Remove outliers if requested
        if remove_outliers:
            for col in columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                outlier_mask = np.abs(df_clean[col] - mean) > outlier_std * std
                df_clean = df_clean[~outlier_mask]
        
        return df_clean
    
    @staticmethod
    def resample_data(df: pd.DataFrame, timestamp_col: str = 'timestamp',
                     frequency: str = '1H') -> pd.DataFrame:
        """
        Resample time series data to different frequency.
        
        Args:
            df: Input DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            frequency: Target frequency (e.g., '1H', '15min', '1D')
            
        Returns:
            Resampled DataFrame
        """
        df_resampled = df.copy()
        df_resampled[timestamp_col] = pd.to_datetime(df_resampled[timestamp_col])
        df_resampled = df_resampled.set_index(timestamp_col)
        
        # Resample using last observation carried forward for prices
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        return_cols = [col for col in df.columns if 'return' in col.lower()]
        
        resampled_data = {}
        
        # Use last value for prices
        for col in price_cols:
            if col in df_resampled.columns:
                resampled_data[col] = df_resampled[col].resample(frequency).last()
        
        # Use sum for returns (to get period returns)
        for col in return_cols:
            if col in df_resampled.columns:
                resampled_data[col] = df_resampled[col].resample(frequency).sum()
        
        # Use mean for other numeric columns
        for col in df_resampled.columns:
            if col not in price_cols + return_cols and df_resampled[col].dtype in ['float64', 'int64']:
                resampled_data[col] = df_resampled[col].resample(frequency).mean()
        
        result_df = pd.DataFrame(resampled_data)
        result_df = result_df.dropna()
        result_df.reset_index(inplace=True)
        
        return result_df


def parameter_validation(params: Dict[str, Any], param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    """
    Validate parameters against specified bounds.
    
    Args:
        params: Dictionary of parameters to validate
        param_bounds: Dictionary mapping parameter names to (min, max) bounds
        
    Returns:
        Dictionary mapping parameter names to validation results
    """
    validation_results = {}
    
    for param_name, bounds in param_bounds.items():
        if param_name in params:
            value = params[param_name]
            min_val, max_val = bounds
            
            is_valid = min_val <= value <= max_val
            validation_results[param_name] = is_valid
            
            if not is_valid:
                warnings.warn(f"Parameter {param_name} = {value} is outside bounds [{min_val}, {max_val}]")
        else:
            validation_results[param_name] = False
            warnings.warn(f"Parameter {param_name} not found in parameter dictionary")
    
    return validation_results


def format_results_table(results: Dict[str, Any], precision: int = 4) -> str:
    """
    Format results dictionary as a readable table string.
    
    Args:
        results: Dictionary of results
        precision: Number of decimal places for float formatting
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION RESULTS")
    lines.append("=" * 60)
    
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        elif isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, dict):
            return "Dict[...]"
        elif isinstance(value, list):
            return f"List[{len(value)} items]"
        else:
            return str(value)
    
    def add_section(data, section_name="", indent=0):
        if section_name:
            lines.append(" " * indent + f"{section_name}:")
            lines.append(" " * indent + "-" * len(section_name))
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(" " * (indent + 2) + f"{key}:")
                add_section(value, "", indent + 4)
            else:
                formatted_value = format_value(value)
                lines.append(" " * (indent + 2) + f"{key}: {formatted_value}")
        
        if section_name:
            lines.append("")
    
    add_section(results)
    lines.append("=" * 60)
    
    return "\n".join(lines)