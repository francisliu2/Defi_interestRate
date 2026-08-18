# Benchmark Analysis: DeFi Trading Strategy Comparison

## Objective
Benchmark the current DeFi trading strategy against four alternative approaches to evaluate relative performance, risk-adjusted returns, and robustness across market conditions.

## Strategies to Compare

### 1. Current DeFi Trading Strategy
- **Source:** Active strategy currently deployed
- **Parameters:** Extract from existing configuration/logic
- **Focus:** Document all entry/exit rules, position sizing, and protocol interactions

### 2. Taleb-Inspired "Antifragile" Strategy
- **Core Principle:** Asymmetric payoff profiles that benefit from volatility and disorder
- **Implementation Guidelines:**
  - Allocate 85-90% to ultra-safe assets (stablecoin yield farming, high-quality lending pools)
  - Allocate 10-15% to highly convex, long-tail opportunities (out-of-the-money options, early-stage liquidity provision in volatile pairs, black swan hedging positions)
  - No leverage on speculative portion
  - Rebalance only when convex positions experience extreme moves (>3 sigma)
  - Avoid mean-reverting strategies; favor trend-following with downside protection
- **Key Metrics:** Maximum drawdown, skewness, kurtosis, tail-risk ratio

### 3. Pure Supply & Demand Strategy
- **Core Principle:** Price action driven entirely by observable on-chain supply/demand imbalances
- **Implementation Guidelines:**
  - Entry signals: Large wallet accumulation/distribution (>$100k moves), exchange net flows, liquidity pool depth changes >20%
  - Exit signals: Supply/demand equilibrium restoration, inventory rebalancing
  - No technical indicators, no sentiment analysis
  - Position size proportional to imbalance magnitude
  - Timeframe: 4h-24h holding periods
- **Data Sources:** Nansen, Arkham, Dune Analytics, on-chain order book depth

### 4. Black-Litterman Portfolio Optimization Strategy
- **Core Principle:** Bayesian portfolio construction combining market equilibrium with investor views
- **Implementation Guidelines:**
  - Market-implied equilibrium returns from TVL-weighted DeFi assets
  - Incorporate 2-3 active views with confidence levels (e.g., "ETH outperforms BTC by 5% this month, 60% confidence")
  - Use DeFi correlation matrix (30-day rolling, weekly rebalance)
  - Position limits: Max 25% single asset, max 15% single protocol
  - Rebalance threshold: 2% drift from target weights
- **Risk Constraints:** Target volatility 20%, max drawdown target <25%

### 5. Momentum & Factor Rotation Strategy
- **Core Principle:** Systematic factor timing across DeFi yield sources and tokens
- **Factors to Track:**
  - Yield momentum (30d change in protocol APY)
  - TVL momentum (14d change in total value locked)
  - Volume momentum (7d DEX volume trend)
  - Volatility regime (GARCH-estimated vs implied)
  - Funding rate momentum (perp market sentiment)
- **Rotation Rules:**
  - Long top 3 factors, short bottom 2 (if shorting available)
  - Weekly signal generation, bi-weekly rebalancing
  - Market regime filter: Reduce exposure 50% when VIX proxy >30

## Analysis Framework

### Performance Metrics (Calculate All)
| Metric | Description |
|--------|-------------|
| Cumulative Return | Total % return over backtest period |
| Annualized Return | Geometric mean annualized |
| Annualized Volatility | Standard deviation of returns |
| Sharpe Ratio | (Return - Risk-Free) / Volatility |
| Sortino Ratio | Downside deviation only |
| Maximum Drawdown | Peak-to-trough decline |
| Calmar Ratio | Annualized Return / Max Drawdown |
| Win Rate | % of profitable trades/periods |
| Profit Factor | Gross Profit / Gross Loss |
| Tail Risk (VaR 95%, CVaR 95%) | Value at Risk, Conditional VaR |
| Skewness | Return distribution asymmetry |
| Kurtosis | Fat-tail measurement |
| Omega Ratio | Probability-weighted gains/losses |

### Market Regime Analysis
Run all strategies through these market conditions:
- **Bull market** (sustained uptrend, low volatility)
- **Bear market** (sustained downtrend, elevated volatility)
- **Sideways/choppy** (range-bound, mean-reverting)
- **Crisis event** (flash crash, protocol exploit, depeg event)
- **High volatility** (VIX proxy >40, frequent liquidations)

### Time Periods
- 30-day rolling windows (12 months of data)
- 90-day extended backtest
- Include at least one major stress event (e.g., SVB collapse March 2023, FTX November 2022)

## Data Requirements

### On-Chain Data
- Wallet tracking (whale accumulation/distribution)
- Exchange net flows (CEX and DEX)
- Liquidity pool depths and composition
- Protocol TVL changes
- Liquidations data
- Gas prices (network congestion proxy)

### Market Data
- OHLCV for all traded pairs (1h and 1d granularity)
- Perpetual futures funding rates
- Options implied volatility (if available via Deribit, Dopex, etc.)
- Stablecoin depeg events
- DEX slippage estimates

### Protocol-Specific Data
- Lending pool utilization rates
- Yield farming APY history
- Governance token emissions schedules
- Protocol revenue and fee generation

## Execution Instructions

### Step 1: Strategy Specification
```python
# For each strategy, produce:
{
    "strategy_name": "...",
    "entry_conditions": [...],
    "exit_conditions": [...],
    "position_sizing_rules": {...},
    "risk_management": {...},
    "rebalancing_frequency": "...",
    "assets_universe": [...],
    "protocols_used": [...]
}
