# DeFi Long-Short Liquidation Risk

This repository studies leveraged long-short positions on AAVE using a
bivariate Kou jump-diffusion model.  The current empirical workflow calibrates
WETH/WBTC dynamics from AAVE v3 Ethereum on-chain data, with WETH as the
long collateral leg and WBTC as the short borrowed leg, then uses a
Laplace-resolvent method to compute liquidation probabilities, killed payoff
moments, and conditional surviving-path moments over admissible initial health
factors.

The primary contribution is the evaluation engine: survival probabilities and
killed payoff moments under a ratio barrier. Initial health-buffer sizing is a
stylized, objective-specific application of those outputs, not a universal
optimal-leverage rule or a protocol-level trading strategy.

The project has two main parts:

- `aave-ts/`: TypeScript data fetcher for AAVE v3 Ethereum market/history data.
- `optimal_long_short/`: Python model, calibration, moments, simulation, and job runners.

The `jobs/` directory contains thin entry-point scripts only.  Implementation
lives under `optimal_long_short/job_runners/` and `optimal_long_short/`.

## Repository Layout

```text
.
|-- aave-ts/                       # AAVE v3 Ethereum data fetcher
|   |-- src/run.ts                 # TypeScript CLI entry point
|   `-- data/AAVE/                 # Parquet history files and manifest.csv
|-- jobs/                          # Python entry points only
|-- optimal_long_short/            # Core Python package
|   |-- calibration/               # ECF calibration, initializers, diagnostics
|   |-- job_runners/               # Implementations behind jobs/*.py
|   |-- model_params.py            # Kou parameter convention
|   |-- moments.py                 # Survival, killed, and survivor-conditional moments
|   |-- laplace_resolvent.py       # Laplace-resolvent machinery
|   |-- risk_report.py             # H0/h0 liquidation reports
|   |-- sizing.py                  # Explicit objective-specific selection rules
|   |-- calibration_uncertainty.py # Paired block bootstrap and propagation
|   |-- numerical_diagnostics.py   # Inversion/root/conditioning/MC diagnostics
|   `-- drift.py                   # Drift-view helpers and diagnostics
|-- latex/                         # Paper and generated figures
|-- results/                       # Calibrated parameter JSON and reports
`-- requirements.txt               # Python dependencies
```

## Setup

Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most Python jobs are run from the repository root:

```bash
.venv/bin/python jobs/<job_name>.py
```

AAVE TypeScript fetcher:

```bash
cd aave-ts
cp .env.example .env   # set RPC_URL
npm install
```

See [aave-ts/README.md](aave-ts/README.md) for the full fetcher CLI, persistence
schema, supported frequencies, and RPC notes.

## Data Workflow

Fetch aligned AAVE history first.  For free-tier RPC endpoints, keep
concurrency low and use moderate history frequency:

```bash
cd aave-ts
npm run run -- --mode=history --assets=WETH,WBTC --days=90 --frequency=6h
```

History fetches are persisted by default to:

```text
aave-ts/data/AAVE/hist_<id>.parquet
aave-ts/data/AAVE/manifest.csv
```

The manifest records fetch parameters, realized block/date ranges, frequency,
asset, chain, RPC host hash, and the parquet ID.  When multiple assets are
passed in a single history fetch, the TypeScript fetcher uses the same target
block schedule across assets, so WETH/WBTC rows can be aligned by block.

## Empirical Calibration

The main empirical calibration is WETH collateral versus WBTC borrow:

```bash
.venv/bin/python jobs/calibrate_eth_btc.py
```

This reads the latest WBTC and WETH rows from `aave-ts/data/AAVE/manifest.csv`,
merges their parquet data by block, computes log returns, subtracts an
exponentially weighted mean, and calibrates bivariate Kou parameters with a
normalized ECF criterion plus disclosed soft moment/POT anchors.

Outputs:

```text
results/params_WETH_WBTC.json
latex/fig_ecf_empirical.pdf
```

The saved JSON includes:

- rate-adjusted calibrated parameters for downstream analysis
- raw ECF parameters before AAVE rate adjustment
- terminal-block WETH collateral constraints (`b`, maximum LTV, `h0_min`, `H0_min`)
- last aligned WETH/WBTC prices used as initial prices
- drift summaries and first-moment diagnostics

## Drift Convention

`KouParams.mu1` and `KouParams.mu2` are annualized price-growth normalizers:

```text
E[exp(X_i(t))] = exp(mu_i * t)
```

The characteristic function, moment calculations, Laplace-resolvent machinery,
and Monte Carlo simulation use the derived log-process drift:

```text
muX_i = mu_i - 0.5 * sigma_i^2 - lambda_i * E[exp(J_i) - 1]
```

User views should usually be applied to `mu_i` as price-growth views.  Use
`optimal_long_short.drift.with_muX_drift_view` only when the view is explicitly
stated as a log-process drift.

Important empirical caveat: high-frequency ECF calibration can weakly identify
the first moment.  The fitted `mu_i` values should be treated as model
normalizers, not out-of-sample return forecasts.  The calibration JSON reports
empirical first-moment diagnostics to make this visible.

## Analysis Jobs

Run from the repository root.

```bash
# Calibrate long-WETH/short-WBTC from AAVE history
.venv/bin/python jobs/calibrate_eth_btc.py

# Numerical comparison table: Laplace-resolvent vs Monte Carlo
.venv/bin/python jobs/numerical_comparison.py

# Parameter sensitivity figure
.venv/bin/python jobs/sensitivity_analysis.py

# Health-buffer evaluation-map figure
.venv/bin/python jobs/health_buffer_evaluation_map.py

# CSV report over h0 / H0 grid
.venv/bin/python jobs/h0_liquidation_report.py

# Compare explicitly parameterized initial-buffer decision rules
.venv/bin/python jobs/objective_comparison.py

# Talbot/root/conditioning and discrete-monitoring diagnostics
.venv/bin/python jobs/numerical_diagnostics.py

# Paired moving-block calibration and sizing uncertainty
.venv/bin/python jobs/calibration_uncertainty.py
```

The objective-comparison defaults reproduce
`results/sizing_objective_comparison.csv` on a 200-point health-factor grid.
The bootstrap job writes replicate-level estimates, percentile summaries, and
run metadata to:

```text
results/calibration_uncertainty_bootstrap.csv
results/calibration_uncertainty_summary.csv
results/calibration_uncertainty_metadata.json
```

These intervals describe sampling variability conditional on the Kou model,
ECF criterion, return preprocessing, block length, protocol terms, and sizing
grid. They are not model-selection or out-of-sample forecast intervals.

`health_buffer_evaluation_map.py` and `sensitivity_analysis.py` load
`results/params_WETH_WBTC.json` by default, including the same calibrated
parameters, AAVE constraints, initial prices, and one-month horizon used in the
empirical paper section. Optional drift views can be passed to the evaluation-map job
with `--mu1`, `--mu2`, `--delta-mu1`, and `--delta-mu2`.

## Illustration Notebooks

The executed notebooks in `notebooks/` provide focused package walkthroughs:

- health factor, holdings, and leverage
- survival probability and integer-order moments
- sizing maps, objectives, and drift views
- Kou calibration and spread-direction ECF diagnostics

See `notebooks/README.md` for the notebook index and rerun command.

## Paper

The main paper is:

```text
latex/optimal_long_short.tex
```

Its title is *Optimal Sizing of Leveraged Crypto Long--Short Positions under
Kou Jump-Diffusion via Killed Moments*; DeFi health-buffer sizing is the
application section.

Build it with:

```bash
cd latex
latexmk -pdf -interaction=nonstopmode -halt-on-error optimal_long_short.tex
```

Generated figures used by the paper include:

```text
latex/fig_ecf_empirical.pdf
latex/fig_ecf_fit.pdf
latex/fig_sensitivity.pdf
latex/fig_health_buffer_evaluation_map.pdf
```

## Notes

- `jobs/OLD/`, old notebooks, and some packaging metadata still reflect earlier
  Hawkes-oriented experiments.  The current empirical pipeline is the
  AAVE/Kou/Laplace-resolvent workflow described above.
- The current AAVE fetcher defaults to RPC concurrency `1`, suitable for
  free-tier endpoints.
- Historical `eth_call` requires an archive-capable or sufficiently capable RPC
  endpoint.  If a free endpoint fails for older blocks, reduce `--days`, lower
  sample count, or use an archive RPC.

## License

MIT License.  See [LICENSE](LICENSE).
