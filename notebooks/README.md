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

5. `05_integrated_framework.ipynb` -- **complete project overview**
   - Component map across all ten layers, with the source file behind each result
   - Aave v3 data layer: fetcher, published rate-curve parameters, the kink; a check that
     `jobs/build_merged_4h.py` rebuilds the headline 4h panel exactly
   - Utilization by curve inversion; Jacobi diffusion replacing the failed CIR/JCIR fits
   - Pre-registered screen and the 37-pair cross-section; the carry-to-barrier diagnostic
   - Liquidation recovery `R_tau` and the first-passage transform, continuous-crossing versus
     exact jump overshoot
   - Regime-switching carry and the 6x6 resolvent, with its limit tests; an illustrative
     stress spec next to the calibrated (inert) result
   - Barbell legs: supplier long the kink, keeper long the barrier
   - Sizing on the map (paper §7.4–7.5, §9): the overshoot law and bad-debt probability,
     expected log wealth, the closed-form break-even drift, sample drifts, posteriors and the
     Bayes–Kelly rule, the realised 37-pair backtest, the model-versus-realised co-jump check,
     and the out-of-sample liquidation backtest
   - Full source index: modules, jobs, data, results, paper sections

   Regenerate from source with `python notebooks/build_05.py`; the notebook is
   built by a script rather than hand-edited so it stays reproducible.

6. `06_tutorial_from_scratch.ipynb` -- **first-principles tutorial on the additions**
   - Written for a reader with first-year calculus and probability only
   - Parts I--III build the position, the Kou process and the first-passage
     machinery from zero, deriving the compensator identity and the Laplace method
   - Parts IV--XIII then take each addition in turn, with a full derivation and a
     numerical check for every claim: the evaluation map, the Aave data layer,
     carry as a drift shift, the 37-pair cross-section and the carry-to-barrier
     ratio, the depeg overlay, the rate curve / Jacobi utilization model, the
     recovery term `R_tau`, its Esscher-tilt first-passage transform and the exact
     jump-overshoot version, the regime-switching 6x6 resolvent with its limit tests
     (illustrative and calibrated), the barbell, and sizing on the map: the
     first-unit-of-debt condition, a step-by-step derivation of the closed-form Kelly
     break-even, why drifts cannot be estimated, the Bayes–Kelly rule, the realised
     37-pair backtest, and what it reveals about the model's independent jumps
   - Part XIV: summary of all ten additions, source map, and a glossary of every
     symbol used

   Regenerate from source with `python notebooks/build_06.py`; like notebook 05
   it is built by a script rather than hand-edited so it stays reproducible.

The carry notebook `optimal_long_short/carry_optimal_long_short.ipynb` (illustrative
carry books, depeg overlay, walk-forward benchmark) lives next to the package, and the
trading-strategy notebooks live in `strategies/`.

Notebooks 05--06 follow the merged paper (`latex/optimal_long_short_merged.tex`) and use
its benchmark, `results/params_WBTC_WETH_benchmark.json` (calibrated shape parameters with
sample drifts); notebooks 01--04 load the calibration `results/params_WBTC_WETH.json`.
Notebooks 05--06 also read `results/merged/regime_calibrated.json` and the saved outputs of the
paper's jobs (`results/bootstrap_se.json`, `breakeven_drift.json`,
`realised_backtest.json`, `realised_backtest_pairs.csv`, `horizon_analysis.json`,
`rolling_backtest.json`, `rolling_backtest_bias.json`, `regime_calibrated.json`,
`model_comparison.json`); the realised and co-jump cells also read `data/AAVE_full/`. They are stored with executed
outputs. To rerun them from the repository root:

```bash
python notebooks/build_05.py && python notebooks/build_06.py
for notebook in notebooks/*.ipynb optimal_long_short/carry_optimal_long_short.ipynb; do
  python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 "$notebook"
done
```
