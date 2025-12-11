from Core_CP import BaseStrategy, BaseParameters
from Core_CP import CompoundPoissonParams
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class MomentumLongShortParams(BaseParameters):
    """Parameters for Long-Short Momentum Strategy"""

    # Momentum formation period (days) - not final, will examine lit/industry recs
    formation_period: int = 30

    # Asset selection parameters
    momentum_percentile: float = 0.2  # Top/bottom % for winner/loser selection

    # Long leg (momentum winner) characteristics - placeholder values
    mu_L: float = 0.10    # Higher drift for winners
    sigma_L: float = 0.80 # Higher volatility
    lambda_L: float = 0.10 # Moderate jump intensity
    delta_L: float = 0.20 # Moderate jump sizes
    eta_L: float = 5.0    # Tail risk parameter

    # Short leg (momentum loser) characteristics - placeholder values
    mu_S: float = 0.05    # Lower/negative drift for losers
    sigma_S: float = 0.70 # Slightly lower volatility
    lambda_S: float = 0.20 # Higher jump intensity (more downside risk)
    delta_S: float = 0.25 # Larger negative jumps
    eta_S: float = 4.0    # Heavier tails

    # Correlation between assets
    correlation: float = 0.3  # placeholder

    def __post_init__(self):
        """Validate strategy parameters."""
        # Call parent validation if it exists
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        # Validate momentum parameters
        if not (0.0 < self.formation_period <= 252):
            raise ValueError("formation_period must be in (0, 252] days")
        if not (0.0 < self.momentum_percentile <= 0.5):
            raise ValueError("momentum_percentile must be in (0, 0.5]")

        # Validate long leg parameters
        if self.sigma_L < 0:
            raise ValueError("sigma_L must be non-negative")
        if self.lambda_L < 0:
            raise ValueError("lambda_L (jump intensity) must be non-negative")
        if self.delta_L < 0:
            raise ValueError("delta_L must be non-negative")
        if self.eta_L <= 0:
            raise ValueError("eta_L (exponential rate) must be strictly positive")

        # Validate short leg parameters
        if self.sigma_S < 0:
            raise ValueError("sigma_S must be non-negative")
        if self.lambda_S < 0:
            raise ValueError("lambda_S (jump intensity) must be non-negative")
        if self.delta_S < 0:
            raise ValueError("delta_S must be non-negative")
        if self.eta_S <= 0:
            raise ValueError("eta_S (exponential rate) must be strictly positive")

        # Validate correlation
        if not (-1 <= self.correlation <= 1):
            raise ValueError("correlation must be in [-1, 1]")


class MomentumLongShortStrategy(BaseStrategy):
    """
    Long-Short Momentum Strategy implementation.

    Strategy: Long momentum winners, short momentum losers.
    Uses compound Poisson jump-diffusion processes for both legs.
    """

    def __init__(self, params: MomentumLongShortParams):
        # Initialize compound Poisson parameters for both legs
        cp_params = CompoundPoissonParams(
            # Long leg (winner)
            mu_X=params.mu_L, sigma_X=params.sigma_L,
            lambda_X=params.lambda_L, delta_X=params.delta_L, eta_X=params.eta_L,
            # Short leg (loser)
            mu_Y=params.mu_S, sigma_Y=params.sigma_S,
            lambda_Y=params.lambda_S, delta_Y=params.delta_S, eta_Y=params.eta_S
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

    # -------------------------------------------------------------------------
    # ASSET SELECTION
    # -------------------------------------------------------------------------
    def select_momentum_assets(self, price_data: Dict[str, np.ndarray]) -> Tuple[str, str]:
        """
        Select momentum winner (long) and loser (short) based on formation period returns.

        Args:
            price_data: Dictionary mapping asset names to price arrays

        Returns:
            Tuple of (long_asset_name, short_asset_name)

        Raises:
            ValueError: If insufficient assets or price history
        """
        formation_period = self.specific_params.formation_period

        # Calculate formation period returns for all assets
        momentum_returns = {}
        for asset, prices in price_data.items():
            if len(prices) > formation_period:
                return_ = (prices[-1] / prices[-formation_period]) - 1
                momentum_returns[asset] = return_

        if len(momentum_returns) < 2:
            raise ValueError("Need at least 2 assets with sufficient price history")

        # Sort by momentum returns (descending)
        sorted_assets = sorted(momentum_returns.items(), key=lambda x: x[1], reverse=True)

        # Select winner (highest) and loser (lowest)
        long_asset, long_return = sorted_assets[0]
        short_asset, short_return = sorted_assets[-1]

        # Store initial prices for position sizing
        self.long_asset = long_asset
        self.short_asset = short_asset
        self.L0 = price_data[long_asset][-1]
        self.S0 = price_data[short_asset][-1]

        print(f"Momentum Winner (Long): {long_asset} (Return: {long_return:.2%})")
        print(f"Momentum Loser (Short): {short_asset} (Return: {short_return:.2%})")

        return long_asset, short_asset

    # -------------------------------------------------------------------------
    # WEALTH AND RETURN METRICS
    # -------------------------------------------------------------------------
    def expected_wealth(self, r: float) -> float:
        """
        Compute expected terminal wealth.

        Formula: E[W_T] = W_0 * (1 + w_L * E[R_L] - w_S * E[R_S])

        where E[R] = exp(T * ψ(1)) - 1 from the Lévy exponent.

        Args:
            r: Long-short allocation ratio

        Returns:
            Expected terminal wealth

        Raises:
            ValueError: If characteristic function evaluation fails
        """
        w_L, w_S = r, 1.0  # Normalize: w_S = 1, w_L = r

        # Expected returns from Lévy processes
        # ψ(1) gives the log MGF for gross return e^R
        try:
            cf_L = self.cp_process.characteristic_function(-1j, 'X', self.params.T)
            cf_S = self.cp_process.characteristic_function(-1j, 'Y', self.params.T)
        except Exception as e:
            raise ValueError(f"Characteristic function evaluation failed: {e}")

        E_return_L = cf_L.real - 1  # Expected return of long leg
        E_return_S = cf_S.real - 1  # Expected return of short leg

        # Total expected wealth
        wealth_component = w_L * E_return_L - w_S * E_return_S
        expected_wealth = self.params.W0 * (1 + wealth_component)

        return expected_wealth

    def wealth_variance(self, r: float, cov_XY: Optional[float] = None) -> float:
        """
        Compute variance of terminal wealth.

        Formula: Var(W_T) = W_0² * [w_L² Var(R_L) + w_S² Var(R_S) - 2w_L w_S Cov(R_L, R_S)]

        Args:
            r: Long-short allocation ratio
            cov_XY: External covariance between long and short returns.
                   If None, estimated from correlation parameter.

        Returns:
            Variance of terminal wealth (non-negative)
        """
        w_L, w_S = r, 1.0

        # Get variances using characteristic functions
        try:
            _, var_L = self.cp_process.compute_moments('X', self.params.T)
            _, var_S = self.cp_process.compute_moments('Y', self.params.T)
        except Exception as e:
            raise ValueError(f"Moment computation failed: {e}")

        # Estimate or use provided covariance
        if cov_XY is None:
            correlation = self.specific_params.correlation
            cov_LS = correlation * np.sqrt(var_L * var_S)
        else:
            cov_LS = cov_XY

        # Total variance (scaled by initial wealth)
        total_variance = (
            w_L**2 * var_L + w_S**2 * var_S - 2 * w_L * w_S * cov_LS
        ) * (self.params.W0 ** 2)

        return max(0.0, total_variance)

    # -------------------------------------------------------------------------
    # STRATEGY DECOMPOSITION
    # -------------------------------------------------------------------------
    def momentum_component(self, r: float) -> float:
        """
        Compute the momentum spread component of expected returns.

        This isolates the drift differential between long and short legs.

        Formula: w_L * E[R_L] - w_S * E[R_S]

        Args:
            r: Long-short allocation ratio

        Returns:
            Expected momentum spread (in wealth units)
        """
        w_L, w_S = r, 1.0

        try:
            cf_L = self.cp_process.characteristic_function(-1j, 'X', self.params.T)
            cf_S = self.cp_process.characteristic_function(-1j, 'Y', self.params.T)
        except Exception as e:
            raise ValueError(f"Characteristic function evaluation failed: {e}")

        E_return_L = cf_L.real - 1
        E_return_S = cf_S.real - 1

        momentum_spread = (E_return_L - E_return_S) * self.params.W0
        return momentum_spread

    def jump_risk_component(self, r: float) -> Dict[str, float]:
        """
        Analyze jump risk contributions from both legs.

        Decomposes net jump exposure into intensity and size components.

        Args:
            r: Long-short allocation ratio

        Returns:
            Dictionary with jump risk metrics:
                - jump_intensity_contribution: λ_L - λ_S (weighted)
                - jump_size_risk: δ_L - δ_S (weighted)
                - net_jump_exposure: product of above
        """
        w_L, w_S = r, 1.0

        # Jump intensity contribution
        jump_intensity_contrib = (
            w_L * self.specific_params.lambda_L -
            w_S * self.specific_params.lambda_S
        )

        # Jump size risk
        jump_size_risk = (
            w_L * self.specific_params.delta_L -
            w_S * self.specific_params.delta_S
        )

        return {
            'jump_intensity_contribution': jump_intensity_contrib,
            'jump_size_risk': jump_size_risk,
            'net_jump_exposure': jump_intensity_contrib * jump_size_risk
        }

    # -------------------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------------------
    def calculate_position_weights(self, r: float) -> Tuple[float, float]:
        """
        Calculate normalized position weights from allocation ratio.

        Ensures weights sum to 1 for interpretation as portfolio composition.

        Formula: w_S = 1/(1+r), w_L = r*w_S

        Args:
            r: Long-short allocation ratio

        Returns:
            Tuple of (weight_long, weight_short)
        """
        if r <= 0:
            raise ValueError("r must be positive")
        w_S = 1.0 / (1 + r)
        w_L = r * w_S
        return w_L, w_S

    # -------------------------------------------------------------------------
    # COMPREHENSIVE ANALYSIS
    # -------------------------------------------------------------------------
    def analyze_strategy(self, r: float) -> Dict[str, Any]:
        """
        Comprehensive strategy analysis for a given allocation ratio.

        Computes expected wealth, variance, momentum component, jump risks,
        and strategy characteristics.

        Args:
            r: Long-short allocation ratio

        Returns:
            Dictionary containing:
                - allocation_ratio: input r
                - w_L, w_S: normalized weights
                - expected_wealth: E[W_T]
                - expected_return: E[W_T] - W_0
                - momentum_component: drift differential component
                - variance: Var(W_T)
                - volatility: sqrt(Var(W_T))
                - sharpe_ratio: expected_return / volatility
                - jump risk metrics
                - drift_differential, volatility_ratio, jump_risk_ratio
                - initial_health, liquidation_prob (from base strategy)
        """
        analysis = {}

        # Basic metrics
        analysis['allocation_ratio'] = r
        w_L, w_S = self.calculate_position_weights(r)
        analysis['w_L'] = w_L
        analysis['w_S'] = w_S

        # Wealth components
        analysis['expected_wealth'] = self.expected_wealth(r)
        analysis['expected_return'] = analysis['expected_wealth'] - self.params.W0
        analysis['momentum_component'] = self.momentum_component(r)

        # Risk metrics
        analysis['variance'] = self.wealth_variance(r)
        analysis['volatility'] = np.sqrt(analysis['variance'])
        analysis['sharpe_ratio'] = (
            analysis['expected_return'] / analysis['volatility']
            if analysis['volatility'] > 0 else 0
        )

        # Jump risk analysis
        jump_analysis = self.jump_risk_component(r)
        analysis.update(jump_analysis)

        # Strategy characteristics
        analysis['drift_differential'] = (
            (self.specific_params.mu_L - self.specific_params.mu_S) * self.params.T
        )
        analysis['volatility_ratio'] = (
            self.specific_params.sigma_L / self.specific_params.sigma_S
            if self.specific_params.sigma_S > 0 else float('inf')
        )
        analysis['jump_risk_ratio'] = (
            (self.specific_params.lambda_L * self.specific_params.delta_L) /
            (self.specific_params.lambda_S * self.specific_params.delta_S)
            if (self.specific_params.lambda_S * self.specific_params.delta_S) > 0 else float('inf')
        )

        # Health and constraints
        try:
            analysis['initial_health'] = self.initial_health(r)
            analysis['liquidation_prob'] = self.liquidation_probability(r)
        except Exception as e:
            analysis['initial_health'] = None
            analysis['liquidation_prob'] = None

        return analysis

    # -------------------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # -------------------------------------------------------------------------
    def sensitivity_analysis(self, base_r: float) -> Dict[str, float]:
        """
        Analyze sensitivity to parameter changes.

        Computes marginal impact on expected return and volatility from:
        - 5% increase in long drift
        - 0.2 increase in correlation
        - 10% increase in short leg jump intensity

        Args:
            base_r: Base allocation ratio for analysis

        Returns:
            Dictionary with sensitivity metrics:
                - drift_sensitivity: change in expected return
                - correlation_sensitivity: change in volatility
                - jump_risk_sensitivity: change in volatility
        """
        sensitivities = {}

        # Base case
        base_analysis = self.analyze_strategy(base_r)

        # Sensitivity to drift differential
        original_mu_L = self.specific_params.mu_L
        self.specific_params.mu_L += 0.05  # +5% drift
        high_drift_analysis = self.analyze_strategy(base_r)
        self.specific_params.mu_L = original_mu_L

        sensitivities['drift_sensitivity'] = (
            high_drift_analysis['expected_return'] - base_analysis['expected_return']
        )

        # Sensitivity to correlation
        original_corr = self.specific_params.correlation
        self.specific_params.correlation = min(1.0, original_corr + 0.2)  # +0.2 correlation
        high_corr_analysis = self.analyze_strategy(base_r)
        self.specific_params.correlation = original_corr

        sensitivities['correlation_sensitivity'] = (
            high_corr_analysis['volatility'] - base_analysis['volatility']
        )

        # Sensitivity to jump risk
        original_lambda_S = self.specific_params.lambda_S
        self.specific_params.lambda_S += 0.1  # +10% jump intensity
        high_jump_analysis = self.analyze_strategy(base_r)
        self.specific_params.lambda_S = original_lambda_S

        sensitivities['jump_risk_sensitivity'] = (
            high_jump_analysis['volatility'] - base_analysis['volatility']
        )

        return sensitivities

    # -------------------------------------------------------------------------
    # OPTIMIZATION
    # -------------------------------------------------------------------------
    def optimize_position_ratio(
        self,
        rho1: float = 0.1,
        rho2: float = 10.0,
        r_bounds: Tuple[float, float] = (0.5, 3.0)
    ) -> Dict[str, Any]:
        """
        Optimize long-short ratio using mean-variance utility.

        Formula: max_r { E[W_T] - ρ₁ * Var(W_T) - ρ₂ * P(liq ≤ T) }

        Args:
            rho1: Risk-aversion coefficient for variance
            rho2: Risk-aversion coefficient for liquidation probability
            r_bounds: Bounds for optimization (lower, upper)

        Returns:
            Dictionary with optimal allocation:
                - optimal_ratio: optimal r
                - optimal_utility: utility at optimum
                - optimal_weights: (w_L, w_S) at optimum
                - analysis: full strategy analysis at optimum

        Raises:
            ValueError: If optimization fails
        """
        from scipy.optimize import minimize_scalar

        def objective(r):
            try:
                return -self.utility_function(r, rho1, rho2)
            except:
                return 1e10  # Penalize invalid allocations

        result = minimize_scalar(objective, bounds=r_bounds, method='bounded')

        if result.success:
            r_opt = result.x
            optimal_utility = -result.fun
            optimal_analysis = self.analyze_strategy(r_opt)

            return {
                'optimal_ratio': r_opt,
                'optimal_utility': optimal_utility,
                'optimal_weights': self.calculate_position_weights(r_opt),
                'analysis': optimal_analysis
            }
        else:
            raise ValueError("Optimization failed to converge")

    # -------------------------------------------------------------------------
    # VISUALIZATION - not final, just ideas for now
    # -------------------------------------------------------------------------
    def plot_strategy_analysis(self, rho1: float = 0.1, rho2: float = 10.0):
        """
        Comprehensive visualization of momentum strategy trade-offs.

        Creates 2x2 subplot grid showing:
        1. Risk-Return frontier (Sharpe ratio color scale)
        2. Wealth decomposition and jump risk
        3. Utility optimization vs liquidation risk
        4. Strategy characteristics (drift, volatility ratio)

        Args:
            rho1: Risk-aversion coefficient for variance
            rho2: Risk-aversion coefficient for liquidation probability
        """
        import matplotlib.pyplot as plt

        r_values = np.linspace(0.5, 3.0, 50)
        analyses = [self.analyze_strategy(r) for r in r_values]
        utilities = [self.utility_function(r, rho1, rho2) for r in r_values]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Risk-Return Trade-off
        returns = [a['expected_return'] for a in analyses]
        volatilities = [a['volatility'] for a in analyses]
        sharpe_ratios = [a['sharpe_ratio'] for a in analyses]

        scatter = ax1.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', s=50)
        ax1.set_xlabel('Portfolio Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Momentum Strategy: Risk-Return Trade-off\n(Color: Sharpe Ratio)')
        plt.colorbar(scatter, ax=ax1)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Momentum Component vs Allocation
        momentum_components = [a['momentum_component'] for a in analyses]
        jump_risks = [a['net_jump_exposure'] for a in analyses]

        ax2.plot(r_values, returns, 'k-', linewidth=2, label='Total Return')
        ax2.plot(r_values, momentum_components, 'g--', linewidth=2, label='Momentum Spread')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(r_values, jump_risks, 'r:', linewidth=2, label='Jump Risk')
        ax2.set_xlabel('Long-Short Ratio (r)')
        ax2.set_ylabel('Return Components')
        ax2_twin.set_ylabel('Jump Risk', color='r')
        ax2.set_title('Wealth Decomposition and Jump Risk')
        ax2.legend()
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Utility Optimization
        ax3.plot(r_values, utilities, 'b-', linewidth=2, label='Utility')
        ax3_twin = ax3.twinx()
        liq_probs = [a['liquidation_prob'] if a['liquidation_prob'] is not None else 0 for a in analyses]
        ax3_twin.plot(r_values, liq_probs, 'orange', linewidth=2, label='Liquidation Prob')
        ax3.set_xlabel('Long-Short Ratio (r)')
        ax3.set_ylabel('Utility', color='b')
        ax3_twin.set_ylabel('Liquidation Probability', color='orange')
        ax3.set_title('Utility Optimization vs Risk')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Strategy Characteristics
        drift_differentials = [a['drift_differential'] for a in analyses]
        vol_ratios = [a['volatility_ratio'] for a in analyses]

        ax4.plot(r_values, drift_differentials, 'purple', linewidth=2, label='Drift Differential')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(r_values, vol_ratios, 'brown', linewidth=2, label='Vol Ratio (L/S)')
        ax4.set_xlabel('Long-Short Ratio (r)')
        ax4.set_ylabel('Drift Differential', color='purple')
        ax4_twin.set_ylabel('Volatility Ratio', color='brown')
        ax4.set_title('Strategy Characteristics')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return analyses


# ============================================================================
# FACTORY FUNCTION
# ============================================================================
def create_momentum_strategy(
    W0: float = 10000,
    days: int = 30,
    formation_period: int = 30,
    mu_L: float = 0.10,
    mu_S: float = 0.05,
    sigma_L: float = 0.80,
    sigma_S: float = 0.70,
    lambda_L: float = 0.10,
    lambda_S: float = 0.20,
    delta_L: float = 0.20,
    delta_S: float = 0.25,
    eta_L: float = 5.0,
    eta_S: float = 4.0,
    correlation: float = 0.3,
    b_L: float = 1.5
) -> MomentumLongShortStrategy:
    """
    Convenience factory to create momentum strategy with common parameters.

    Args:
        W0: Initial wealth (currency units)
        days: Time horizon in days
        formation_period: Lookback period for momentum calculation (days)
        mu_L, mu_S: Drift rates for long and short legs
        sigma_L, sigma_S: Volatilities for long and short legs
        lambda_L, lambda_S: Jump intensities for long and short legs
        delta_L, delta_S: Jump sizes for long and short legs
        eta_L, eta_S: Exponential rates for jump tail for long and short legs
        correlation: Correlation between long and short asset returns
        b_L: Collateral factor for long leg

    Returns:
        Initialized MomentumLongShortStrategy
    """
    params = MomentumLongShortParams(
        W0=W0,
        T=days / 365,  # Convert days to years
        b_X=b_L,  # Using b_X as collateral factor for long leg
        formation_period=formation_period,
        mu_L=mu_L,
        mu_S=mu_S,
        sigma_L=sigma_L,
        sigma_S=sigma_S,
        lambda_L=lambda_L,
        lambda_S=lambda_S,
        delta_L=delta_L,
        delta_S=delta_S,
        eta_L=eta_L,
        eta_S=eta_S,
        correlation=correlation
    )

    return MomentumLongShortStrategy(params)