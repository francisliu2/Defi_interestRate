from Core_CP import BaseStrategy, BaseParameters
from Core_CP import CompoundPoissonParams
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Tuple, Optional


@dataclass
class BetaNeutralLongShortParams(BaseParameters):
    """Parameters for Beta-Neutral Long-Short Strategy"""

    # Beta and alpha parameters
    beta_L: float = 1.2      # Long leg beta
    beta_S: float = 1.2      # Short leg beta (should be similar for beta neutrality)
    alpha_L: float = 0.08    # Long leg alpha (idiosyncratic return)
    alpha_S: float = 0.04    # Short leg alpha

    # Market parameters
    market_volatility: float = 0.60  # Overall crypto market volatility
    beta_correlation: float = 0.8    # Correlation for beta adjustment

    # Long leg characteristics
    mu_L: float = 0.08    # Total return = alpha_L + beta_L * market_return
    sigma_L: float = 0.75 # Volatility
    lambda_L: float = 0.08 # Jump intensity
    delta_L: float = 0.20 # Jump sizes

    # Short leg characteristics
    mu_S: float = 0.04    # Total return = alpha_S + beta_S * market_return
    sigma_S: float = 0.70 # Volatility
    lambda_S: float = 0.12 # Jump intensity
    delta_S: float = 0.22 # Jump sizes

    # Beta neutrality tolerance
    beta_tolerance: float = 0.1  # Maximum allowed portfolio beta

    def __post_init__(self):
        # Validate betas
        if self.beta_L <= 0:
            raise ValueError("beta_L must be positive")
        if self.beta_S <= 0:
            raise ValueError("beta_S must be positive")
        # Validate alphas are numeric
        if not isinstance(self.alpha_L, (int, float)):
            raise TypeError("alpha_L must be numeric")
        if not isinstance(self.alpha_S, (int, float)):
            raise TypeError("alpha_S must be numeric")
        # Validate market volatility
        if self.market_volatility <= 0:
            raise ValueError("market_volatility must be positive")
        # Validate beta correlation in [-1,1]
        if not (-1.0 <= self.beta_correlation <= 1.0):
            raise ValueError("beta_correlation must be between -1 and 1")
        # Validate volatilities, jump intensities, jump sizes non-negative
        for val, name in [(self.sigma_L, "sigma_L"), (self.sigma_S, "sigma_S"),
                          (self.lambda_L, "lambda_L"), (self.lambda_S, "lambda_S"),
                          (self.delta_L, "delta_L"), (self.delta_S, "delta_S")]:
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        # Validate beta_tolerance non-negative
        if self.beta_tolerance < 0:
            raise ValueError("beta_tolerance must be non-negative")


class BetaNeutralLongShortStrategy(BaseStrategy):
    """
    Beta-Neutral Long-Short Strategy implementation
    Long high-alpha assets, short low-alpha assets with similar beta
    """

    def __init__(self, params: BetaNeutralLongShortParams):
        # Initialize compound Poisson parameters for both legs
        cp_params = CompoundPoissonParams(
            # Long leg (high alpha)
            mu_X=params.mu_L, sigma_X=params.sigma_L,
            lambda_X=params.lambda_L, delta_X=params.delta_L, eta_X=5.0,
            # Short leg (low alpha)
            mu_Y=params.mu_S, sigma_Y=params.sigma_S,
            lambda_Y=params.lambda_S, delta_Y=params.delta_S, eta_Y=5.0
        )

        # Update params with CP parameters
        params.cp_params = cp_params
        super().__init__(params)
        self.specific_params = params

        # Track selected assets
        self.long_asset: Optional[str] = None
        self.short_asset: Optional[str] = None
        self.L0: Optional[float] = None
        self.S0: Optional[float] = None

    def calculate_beta_neutral_weights(self, r: float) -> Tuple[float, float]:
        """
        Calculate weights that achieve beta neutrality
        w_L * beta_L - w_S * beta_S ≈ 0
        """
        beta_L = self.specific_params.beta_L
        beta_S = self.specific_params.beta_S

        beta_neutral_ratio = beta_S / beta_L

        if abs(r - beta_neutral_ratio) > self.specific_params.beta_tolerance:
            w_S = 1.0
            w_L = beta_neutral_ratio
        else:
            w_S = 1.0
            w_L = r

        return w_L, w_S

    def portfolio_beta(self, r: float) -> float:
        """Calculate portfolio beta for given allocation ratio"""
        w_L, w_S = self.calculate_beta_neutral_weights(r)
        return w_L * self.specific_params.beta_L - w_S * self.specific_params.beta_S

    def alpha_spread(self, r: float) -> float:
        """Calculate alpha spread between long and short legs"""
        w_L, w_S = self.calculate_beta_neutral_weights(r)
        return (w_L * self.specific_params.alpha_L - w_S * self.specific_params.alpha_S) * self.params.T

    def expected_wealth(self, r: float) -> float:
        """
        E[W_T] = W_0 + w_L*(e^{Tψ_L(1)}-1) - w_S*(e^{Tψ_S(1)}-1)
        """
        w_L, w_S = self.calculate_beta_neutral_weights(r)

        cf_L = self.cp_process.characteristic_function(-1j, 'X', self.params.T)
        cf_S = self.cp_process.characteristic_function(-1j, 'Y', self.params.T)

        E_return_L = cf_L.real - 1
        E_return_S = cf_S.real - 1

        wealth_component = w_L * E_return_L - w_S * E_return_S
        expected_wealth = self.params.W0 * (1 + wealth_component)

        return expected_wealth

    def wealth_variance(self, r: float) -> float:
        """
        Var(W_T) = w_L² Var[R_L(T)] + w_S² Var[R_S(T)] - 2 w_L w_S Cov[R_L(T), R_S(T)]
        """
        w_L, w_S = self.calculate_beta_neutral_weights(r)

        _, var_L = self.cp_process.compute_moments('X', self.params.T)
        _, var_S = self.cp_process.compute_moments('Y', self.params.T)

        correlation = self.specific_params.beta_correlation
        cov_LS = correlation * np.sqrt(var_L * var_S)

        total_variance = (w_L**2 * var_L + w_S**2 * var_S - 2 * w_L * w_S * cov_LS) * self.params.W0**2

        return total_variance

    def systematic_risk_component(self, r: float) -> float:
        """Calculate systematic risk component from market exposure"""
        portfolio_beta = self.portfolio_beta(r)
        market_variance = (self.specific_params.market_volatility ** 2) * self.params.T
        return (portfolio_beta ** 2) * market_variance * (self.params.W0 ** 2)

    def idiosyncratic_risk_component(self, r: float) -> float:
        """Calculate idiosyncratic risk component"""
        total_variance = self.wealth_variance(r)
        systematic_risk = self.systematic_risk_component(r)
        return max(0, total_variance - systematic_risk)

    def analyze_strategy(self, r: float) -> Dict[str, Any]:
        """Comprehensive strategy analysis for a given allocation"""
        analysis = {}

        analysis['allocation_ratio'] = r
        w_L, w_S = self.calculate_beta_neutral_weights(r)
        analysis['w_L'] = w_L
        analysis['w_S'] = w_S

        analysis['portfolio_beta'] = self.portfolio_beta(r)
        analysis['is_beta_neutral'] = abs(analysis['portfolio_beta']) <= self.specific_params.beta_tolerance
        analysis['alpha_spread'] = self.alpha_spread(r)

        analysis['expected_wealth'] = self.expected_wealth(r)
        analysis['expected_return'] = analysis['expected_wealth'] - self.params.W0
        analysis['idiosyncratic_return'] = analysis['alpha_spread'] * self.params.W0

        analysis['total_variance'] = self.wealth_variance(r)
        analysis['total_volatility'] = np.sqrt(analysis['total_variance'])
        analysis['systematic_risk'] = self.systematic_risk_component(r)
        analysis['idiosyncratic_risk'] = self.idiosyncratic_risk_component(r)
        analysis['systematic_volatility'] = np.sqrt(analysis['systematic_risk'])
        analysis['idiosyncratic_volatility'] = np.sqrt(analysis['idiosyncratic_risk'])

        if analysis['total_volatility'] > 0:
            analysis['sharpe_ratio'] = analysis['expected_return'] / analysis['total_volatility']
            analysis['information_ratio'] = analysis['idiosyncratic_return'] / analysis['idiosyncratic_volatility']
        else:
            analysis['sharpe_ratio'] = 0
            analysis['information_ratio'] = 0

        analysis['initial_health'] = self.initial_health(r)
        analysis['liquidation_prob'] = self.liquidation_probability(r)

        analysis['risk_reduction_ratio'] = (
            analysis['idiosyncratic_volatility'] / analysis['total_volatility']
            if analysis['total_volatility'] > 0 else 0
        )

        return analysis

    def sensitivity_analysis(self, base_r: float) -> Dict[str, Any]:
        """Analyze sensitivity to parameter changes"""
        sensitivities = {}

        base_analysis = self.analyze_strategy(base_r)

        original_beta_S = self.specific_params.beta_S
        self.specific_params.beta_S += 0.2
        high_beta_mismatch_analysis = self.analyze_strategy(base_r)
        self.specific_params.beta_S = original_beta_S

        sensitivities['beta_mismatch_sensitivity'] = (
            high_beta_mismatch_analysis['systematic_risk'] - base_analysis['systematic_risk']
        )

        original_alpha_L = self.specific_params.alpha_L
        self.specific_params.alpha_L += 0.03
        high_alpha_analysis = self.analyze_strategy(base_r)
        self.specific_params.alpha_L = original_alpha_L

        sensitivities['alpha_spread_sensitivity'] = (
            high_alpha_analysis['expected_return'] - base_analysis['expected_return']
        )

        original_market_vol = self.specific_params.market_volatility
        self.specific_params.market_volatility += 0.2
        high_market_vol_analysis = self.analyze_strategy(base_r)
        self.specific_params.market_volatility = original_market_vol

        sensitivities['market_vol_sensitivity'] = (
            high_market_vol_analysis['systematic_risk'] - base_analysis['systematic_risk']
        )

        return sensitivities

    def find_beta_neutral_allocation(self, r_bounds: Tuple[float, float] = (0.5, 2.0)) -> float:
        """Find allocation that minimizes portfolio beta"""
        from scipy.optimize import minimize_scalar

        def objective(r):
            return abs(self.portfolio_beta(r))

        result = minimize_scalar(objective, bounds=r_bounds, method='bounded')

        if result.success:
            return result.x
        else:
            return self.specific_params.beta_S / self.specific_params.beta_L

    def optimize_position_ratio(self, rho1: float = 0.1, rho2: float = 10.0,
                                r_bounds: Tuple[float, float] = (0.5, 2.0)) -> Dict[str, Any]:
        """
        Optimize long-short ratio using mean-variance utility with beta neutrality constraint
        """
        from scipy.optimize import minimize_scalar

        def objective(r):
            beta_penalty = abs(self.portfolio_beta(r)) * 1000
            return -self.utility_function(r, rho1, rho2) + beta_penalty

        result = minimize_scalar(objective, bounds=r_bounds, method='bounded')

        if result.success:
            r_opt = result.x
            optimal_utility = self.utility_function(r_opt, rho1, rho2)
            optimal_analysis = self.analyze_strategy(r_opt)

            return {
                'optimal_ratio': r_opt,
                'optimal_utility': optimal_utility,
                'optimal_weights': self.calculate_beta_neutral_weights(r_opt),
                'portfolio_beta': self.portfolio_beta(r_opt),
                'analysis': optimal_analysis
            }
        else:
            raise ValueError("Optimization failed")

    def plot_strategy_analysis(self, rho1: float = 0.1, rho2: float = 10.0):
        """Comprehensive visualization of beta-neutral strategy trade-offs"""
        import matplotlib.pyplot as plt

        r_values = np.linspace(0.5, 2.0, 50)
        analyses = [self.analyze_strategy(r) for r in r_values]
        utilities = [self.utility_function(r, rho1, rho2) for r in r_values]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        returns = [a['expected_return'] for a in analyses]
        total_volatilities = [a['total_volatility'] for a in analyses]
        idiosyncratic_volatilities = [a['idiosyncratic_volatility'] for a in analyses]
        portfolio_betas = [a['portfolio_beta'] for a in analyses]

        scatter = ax1.scatter(total_volatilities, returns, c=portfolio_betas,
                              cmap='RdYlBu_r', s=50, alpha=0.7)
        ax1.set_xlabel('Total Portfolio Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Beta-Neutral Strategy: Risk-Return Trade-off\n(Color: Portfolio Beta)')
        plt.colorbar(scatter, ax=ax1)
        ax1.grid(True, alpha=0.3)

        beta_neutral_returns = [returns[i] for i, a in enumerate(analyses) if a['is_beta_neutral']]
        beta_neutral_vols = [total_volatilities[i] for i, a in enumerate(analyses) if a['is_beta_neutral']]
        if beta_neutral_returns:
            ax1.scatter(beta_neutral_vols, beta_neutral_returns, color='green', s=100,
                        marker='*', label='Beta Neutral', edgecolors='black')
            ax1.legend()

        ax2.plot(r_values, total_volatilities, 'k-', linewidth=2, label='Total Risk')
        ax2.plot(r_values, [np.sqrt(a['systematic_risk']) for a in analyses], 'r--', linewidth=2, label='Systematic Risk')
        ax2.plot(r_values, idiosyncratic_volatilities, 'g--', linewidth=2, label='Idiosyncratic Risk')
        ax2.set_xlabel('Long-Short Ratio (r)')
        ax2.set_ylabel('Volatility Components')
        ax2.set_title('Risk Decomposition: Systematic vs Idiosyncratic')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        alpha_spreads = [a['alpha_spread'] for a in analyses]
        information_ratios = [a['information_ratio'] for a in analyses]

        ax3.plot(r_values, alpha_spreads, 'purple', linewidth=2, label='Alpha Spread')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(r_values, information_ratios, 'orange', linewidth=2, label='Information Ratio')
        ax3.set_xlabel('Long-Short Ratio (r)')
        ax3.set_ylabel('Alpha Spread', color='purple')
        ax3_twin.set_ylabel('Information Ratio', color='orange')
        ax3.set_title('Alpha Capture Efficiency')
        ax3.grid(True, alpha=0.3)

        ax4.plot(r_values, utilities, 'b-', linewidth=2, label='Utility')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(r_values, portfolio_betas, 'red', linewidth=2, label='Portfolio Beta')
        ax4.set_xlabel('Long-Short Ratio (r)')
        ax4.set_ylabel('Utility', color='b')
        ax4_twin.set_ylabel('Portfolio Beta', color='red')
        ax4.set_title('Utility vs Beta Neutrality')
        ax4.grid(True, alpha=0.3)

        ax4_twin.axhline(y=0, color='green', linestyle=':', alpha=0.7, label='Beta Neutral')
        ax4_twin.legend()

        plt.tight_layout()
        plt.show()

        return analyses


# Utility factory function
def create_beta_neutral_strategy(
    W0: float = 10000,
    days: int = 30,
    beta_L: float = 1.2,
    beta_S: float = 1.2,
    alpha_L: float = 0.08,
    alpha_S: float = 0.04,
    mu_L: float = 0.08,
    mu_S: float = 0.04,
    sigma_L: float = 0.75,
    sigma_S: float = 0.70,
    lambda_L: float = 0.08,
    lambda_S: float = 0.12,
    delta_L: float = 0.20,
    delta_S: float = 0.22,
    market_volatility: float = 0.60,
    b_L: float = 1.5
) -> BetaNeutralLongShortStrategy:
    """Create a beta-neutral strategy instance with defaults or provided params."""

    params = BetaNeutralLongShortParams(
        W0=W0,
        T=days / 365,
        b_X=b_L,
        beta_L=beta_L,
        beta_S=beta_S,
        alpha_L=alpha_L,
        alpha_S=alpha_S,
        mu_L=mu_L,
        mu_S=mu_S,
        sigma_L=sigma_L,
        sigma_S=sigma_S,
        lambda_L=lambda_L,
        lambda_S=lambda_S,
        delta_L=delta_L,
        delta_S=delta_S,
        market_volatility=market_volatility
    )

    return BetaNeutralLongShortStrategy(params)