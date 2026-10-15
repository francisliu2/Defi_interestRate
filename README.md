# DeFi Long-Short Liquidation Risk

This repository studies leveraged long-short positions on Aave using a
bivariate Kou jump-diffusion model.  It calibrates WBTC/WETH dynamics from
Aave v3 Ethereum data, then uses a Laplace-resolvent method to compute
liquidation probabilities, killed payoff moments, liquidation recovery, and
carry effects over admissible initial health factors.  The accompanying paper is
`latex/optimal_long_short.tex`.

## Merged paper (current)

`latex/optimal_long_short_merged.tex` (slides: `latex/optimal_long_short_merged_slides.tex`)
merges the coauthor's version (Sections 2-5 and Appendix A) with this repository's
recovery, sizing and empirical work. It supersedes `latex/optimal_long_short.tex`, which
is kept for reference together with its job table below.

Its benchmark keeps the drift outside the ECF fit: `jobs/build_benchmark_params.py` writes
`results/params_WBTC_WETH_benchmark.json` (ECF shape parameters, sample drifts including
Aave rates). Sections 6-7 and the regime table are reproduced at that benchmark by

```bash
bash jobs/run_merged_benchmark.sh      # ~50 min; outputs in results/merged/
python jobs/decision_rules.py          # Table 6 (decision rules)
```

Sections 8.2-8.4 and the appendices use the same jobs as before
(`breakeven_drift.py`, `realised_backtest.py`, `fig_realised_backtest.py`,
`horizon_analysis.py`, `rolling_backtest.py`, `backtest_bias.py`, `bootstrap_se.py`).
Notebooks 05-06 follow the merged paper and its benchmark.

## Repository Layout

```text
.
|-- optimal_long_short/        # Python package (installable)
|   |-- calibration/           # Penalised ECF calibration, grid, initializer
|   |-- job_runners/           # Implementations behind the original jobs/*.py entry points
|   |-- kou_model.py           # Bivariate Kou exponent, tilted dynamics of Z = X1 - X2
|   |-- laplace_resolvent.py   # Particular/homogeneous solutions, 3x3 barrier system
|   |-- moments.py             # Survival probability and killed/conditional moments
|   |-- inversion.py           # Talbot Laplace inversion
|   |-- recovery.py            # Continuous-crossing liquidation recovery
|   |-- overshoot_recovery.py  # Exact jump-overshoot recovery (bivariate tilt)
|   |-- carry.py, rate_curve.py, regime_switching.py, screen.py   # Carry section
|   |-- drift_bayes.py         # Bayesian sizing distribution
|   |-- kelly.py               # Semi-analytic E[log W_T] (Kelly), overshoot law
|   |-- full_history.py        # Stale-aware pair panels from the full hourly history
|   `-- aave_hourly.py         # Aave v3 hourly reserve fetcher (The Graph)
|-- jobs/                      # Scripts that reproduce the paper's tables/figures
|-- tests/                     # pytest suite
|-- notebooks/                 # Package walkthroughs (see notebooks/README.md)
|-- data/AAVE/                 # Hourly reserve snapshots + the 4h WBTC/WETH panel
|-- results/                   # Calibrated parameters and job outputs (JSON/CSV)
|-- latex/                     # Paper, slides, figures
|-- strategies/                # Separate trading-strategy benchmarks (not used by the paper)
|-- docs/reviews/              # Referee-style review and revision reports
`-- aave-ts/                   # Older TypeScript on-chain fetcher (RPC-based)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,dev]"      # or: pip install -r requirements.txt
```

Run jobs from the repository root.

## Tests

```bash
pytest                    # full suite
pytest -m "not slow"      # skip the sizing-grid regression (a few minutes)
```

The core tests pin the resolvent against independent quantities: the tilted
Levy exponent, the no-shorting limit, the closed-form Brownian first-passage
probability in the vanishing-jump limit, and the k=0 first-passage transform.

## Reproducing the paper

### 1. Data

Hourly reserve snapshots (rates, TVL, balances, utilization) are fetched from
The Graph by `optimal_long_short/aave_hourly.py` (API key in `.env` as
`THE_GRAPH_API_KEY`); `jobs/fetch_universe.py` fetches the full universe and the reserve
configurations into `data/AAVE/`.

The headline WBTC/WETH panel is built from those files:

```bash
python jobs/build_merged_4h.py      # -> data/AAVE/merged_wbtc_weth_4h.parquet
```

Prices are oracle-implied, `TVL_USD / (token_balance / 10**decimals)`, resampled
to four hours by taking the last hourly value.  With the default window
(2026-03-17 16:00 to 2026-08-04 16:00 UTC) this reproduces the stored panel
exactly.

> Hourly files written before the `price_usd` fix hold `TVL / raw balance`
> in that column, off by `10**decimals`; log returns are unaffected, and
> `build_merged_4h.py` recomputes prices from TVL and balance directly.

### 2. Calibration

```bash
python jobs/calibrate_btc_eth.py --merged   # -> results/params_WBTC_WETH.json, latex/fig_ecf_empirical.pdf
```

This reproduces the stored headline parameters bit for bit.  Without
`--merged` the job reads the older `aave-ts/data/AAVE/manifest.csv` pipeline.

### 3. Tables and figures

| Script | Paper output |
|---|---|
| `jobs/numerical_comparison.py` | Table 3 (Laplace vs Monte Carlo) |
| `jobs/mc_convergence.py` | Table 4 (bridge-corrected MC) |
| `jobs/sensitivity_analysis.py` | Figure 1 |
| `jobs/overshoot_table.py` | Table 5 (jump-overshoot recovery) |
| `jobs/horizon_analysis.py` (writes `results/pair_cross_section_daily.csv`) | Tables 6, 12 (pair cross-section, daily σ̄_Z), Table 10 (full-history calibrations) |
| `jobs/regime_calibration.py` | Table 7 (regime-switching carry) |
| `jobs/breakeven_drift.py` | Table 8 (break-even drift views), Figure 2 (optimal leverage vs drift view), §9 posteriors |
| `jobs/realised_backtest.py`, `jobs/fig_realised_backtest.py` | §9.3 realised 37-pair backtest (Table 9, Figure 3); writes results/realised_backtest.json, results/realised_backtest_pairs.csv |
| `jobs/drift_bayes.py` | Legacy capped-grid thresholds (`results/task4_numbers.json`, used by notebooks 05/06; not the paper's break-evens) |
| `jobs/kelly_sizing.py` | §7.5 Monte Carlo validation of E[log W_T], `latex/fig_kelly.pdf` |
| `jobs/bootstrap_se.py` | Table 9 (block bootstrap, headline bounds) |
| `jobs/model_comparison.py` | GBM / Merton / Kou comparison (App. A.2) |
| `jobs/recalibrate_extended.py --recal` | Extended-sample robustness (Appendix A.2) |
| `jobs/fetch_universe.py --start 2023-01-27 --out-dir data/AAVE_full` | Full Aave v3 history (input to the two jobs below) |
| `jobs/horizon_analysis.py` | Full-history calibrations (Table 10), sample drift and SE (§9.3), tail risk (§7.4) |
| `jobs/rolling_backtest.py` | Table 11 (out-of-sample one-month liquidation backtest, 2024-01 onward) |

`bootstrap_se.py` takes `[B] [n_starts] [workers]` and runs in parallel.

### 4. Paper

```bash
cd latex && latexmk -pdf -interaction=nonstopmode -halt-on-error optimal_long_short.tex
```

## Drift convention

`KouParams.mu1`/`mu2` are annualised price-growth rates,
`E[exp(X_i(t))] = exp(mu_i t)`.  All numerics use the derived log-process drift
`muX_i = mu_i - 0.5 sigma_i^2 - lambda_i E[exp(J_i) - 1]`.  Apply user views to
`mu_i` (see `optimal_long_short.drift`).  High-frequency calibration identifies
the drifts weakly (bootstrap SE of the spread drift is larger than the estimate),
so the fitted `mu_i` are normalizers, not forecasts.

## Notes

- `data/OLD_AAVE/` is still read by `strategies/ccaave_v3_carry_trade_analysis.ipynb`.
- `optimal_long_short/carry_optimal_long_short.ipynb` writes
  `walk_forward_oos_results.csv` next to itself via an absolute path.

## License

MIT License.  See [LICENSE](LICENSE).
