# DeFi Long-Short Liquidation Risk

This repository studies leveraged long-short positions on AAVE using a
bivariate Kou jump-diffusion model.  The current empirical workflow calibrates
WETH/WBTC dynamics from AAVE v3 Ethereum on-chain data, evaluates both possible
role assignments with the applicable AAVE carry, and uses the assignment with
the larger positive empirical-mu spread.  It then uses a
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

The `jobs/` directory contains thin entry-point scripts only. Implementation
lives in the responsibility-specific subpackages under `optimal_long_short/`.

## Repository Layout

```text
.
|-- aave-ts/                       # AAVE v3 Ethereum data fetcher
|   |-- src/run.ts                 # TypeScript CLI entry point
|   `-- data/AAVE/                 # Parquet history files and manifest.csv
|-- jobs/                          # Current paper result-generation scripts
|   `-- legacy_jobs/               # Jobs unused by the current paper and note
|-- optimal_long_short/            # Core Python package
|   |-- calibration/               # Reusable ECF calibration and bootstrap services
|   |-- laplace/                   # Inversion, roots, resolvents, and diagnostics
|   |-- model/                     # Parameters, dynamics, drift service, strategy, sizing, moments, and reports
|   |-- monte_carlo/               # Simulation engine
|   |-- strategy/                  # Colleague research assets
|   `-- utils/helpers.py           # Shared serialization and analysis helpers
|-- latex/                         # Paper and generated figures
|-- results/                       # Calibrated parameter JSON and reports
|-- pyproject.toml                 # Package metadata and dependency declarations
`-- uv.lock                        # Reproducible Python dependency lock
```

## Setup

Install [uv](https://docs.astral.sh/uv/) and create the locked development
environment from the repository root:

```bash
uv sync --locked
```

uv creates and manages `.venv` automatically. Run Python, tests, and jobs
through `uv run`; no manual activation is required:

```bash
uv run python -m jobs.<job_name>
uv run pytest
```

When dependencies change, use `uv add <package>` for runtime dependencies or
`uv add --dev <package>` for development tools, then commit both
`pyproject.toml` and `uv.lock`. CI and reproducible local runs should use
`uv sync --locked`; a runtime-only environment can use
`uv sync --locked --no-dev`.

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

The main empirical calibration selects the long/short orientation between WETH
and WBTC from the empirical price-growth exponents:

```bash
uv run ols-calibrate
```

This reads the latest WBTC and WETH rows from `aave-ts/data/AAVE/manifest.csv`,
merges their parquet data by block, and computes log returns. It estimates a
causal exponentially weighted mean (EWM) with a one-month half-life and
subtracts the lagged EWM mean once from each observed log-return increment. The
resulting residual increments are passed directly to a shape-only empirical
characteristic function (ECF) fit for the bivariate Kou diffusion, jump, and
dependence parameters; no additional sample demeaning is applied. The residual
expected log-return drifts are fixed at zero as a population model restriction.

For each asset, a carry-free empirical price-growth exponent combines the
endpoint EWM log-return trend with the shape-implied residual price-growth
correction.  Because financing depends on the assigned role, the workflow then
evaluates both ordered role assignments after adding the applicable AAVE supply
rate to the candidate long leg and borrowing rate to the candidate short leg.
It selects the assignment with the larger positive long-minus-short empirical
mu spread.

Outputs:

```text
results/params_empirical_showcase.json
results/params_<LONG>_<SHORT>.json
latex/fig_ecf_empirical.pdf
```

The saved JSON includes:

- the selected long/short orientation and its selection inputs
- final empirical parameters in long-then-short order for downstream analysis
- shape-only residual parameters before EWM trends and AAVE carry are added
- the selected collateral asset's terminal-block constraints (`b`, maximum LTV,
  `h0_min`, `H0_min`)
- last aligned prices in the selected asset order
- EWM preprocessing and drift-component diagnostics

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
`optimal_long_short.model.drift_service.with_muX_drift_view` only when the view is explicitly
stated as a log-process drift.

The showcase does not estimate `mu_i` freely in the high-frequency ECF fit.
Instead, it fixes each residual expected log-return drift at zero and constructs
the final `mu_i` from the endpoint EWM trend, the residual diffusion/jump
correction required by the convention above, and role-specific AAVE carry.  The
EWM trend is a horizon-matched empirical scenario input, not a structural
out-of-sample expected-return estimate.

For the empirical section, define
`g_i = endpoint EWM log mean + role-specific annual carry`.  The downstream
log-price coefficient is `muX_i = g_i - lambda_i E[J_i]`; the diffusion term is
not subtracted from `g_i`.  It has already been included when the stored
price-growth `mu_i` is converted to `muX_i`.

## Paper Result-Generation Jobs

Run from the repository root.

```bash
# Calibrate the data-selected WETH/WBTC orientation from AAVE history
uv run ols-calibrate

# Synthetic shape-only ECF diagnostic used in the calibration note
uv run ols-calibrate-kou

# Semi-analytical / seeded Monte Carlo comparison table
uv run ols-compare-methods

# Health-buffer evaluation-map figure
uv run ols-health-map

# Section 6 log-return-mean spread sensitivity CSV and figure
uv run ols-mean-spread

# Compare explicitly parameterized initial-buffer decision rules
uv run ols-objective-comparison
```

The objective-comparison defaults reproduce
`results/sizing_objective_comparison.csv` on a 200-point health-factor grid.
`health_buffer_evaluation_map.py` loads
`results/params_empirical_showcase.json` by default, including the selected
asset order, calibrated parameters, AAVE constraints, initial prices, and
one-month horizon used in the empirical paper section. Optional drift views can
be passed to the evaluation-map job with `--mu1`, `--mu2`, `--delta-mu1`, and
`--delta-mu2`; asset 1 is the selected long leg and asset 2 the selected short
leg.

Jobs not used to produce a table or figure in the current main paper or
calibration note are retained under `jobs/legacy_jobs/`. The colleague-provided
research assets under `optimal_long_short/strategy/` are intentionally outside
this classification and remain untouched.

Both TeX documents define `\showresultscriptpathsfalse` near the top. Change it
to `\showresultscriptpathstrue` to print the generating script beneath each
numerical table or figure; switch it back to hide those paths for distribution.

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
