# Illustration Notebooks

These notebooks demonstrate focused workflows from the paper and the
`optimal_long_short` package. Each notebook shows how to call particular
package APIs, inspect returned objects, convert results into tables, and build
visualisations:

1. `01_health_factor_and_leverage.ipynb`
   - Run `MarketParams` and `UnitExposureLongShortStrategy`
   - Health-factor inputs
   - Unit-equity holdings
   - Buffer and leverage tradeoff

2. `02_survival_and_integer_moments.ipynb`
   - Run `ConditionalMoments`, `h0_liquidation_moment_report`, and `MonteCarlo`
   - Survival and liquidation probabilities
   - Integer-order moment admissibility
   - Conditional variance, skewness, and kurtosis

3. `03_sizing_frontier_and_drift_views.ipynb`
   - Build reusable sizing-grid and objective helpers from package results
   - Mean-variance-liquidation frontier
   - Penalised sizing objective
   - Price-growth drift scenarios

4. `04_kou_calibration_diagnostics.ipynb`
   - Run `simulate_kou_returns`, `StandardizedCalibrationGrid`, and `calibrate_ecf`
   - Simulated bivariate Kou returns
   - Standardized ECF frequency grid
   - Spread-direction ECF and tail diagnostics

The notebooks load the saved WBTC/WETH calibration from
`results/params_WBTC_WETH.json`. They are stored with executed outputs.

To rerun all notebooks from the repository root:

```bash
for notebook in notebooks/*.ipynb; do
  .venv/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=300 \
    "$notebook"
done
```
