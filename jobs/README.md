# Paper result-generation jobs

Run `uv sync --locked` once, then invoke these jobs from the repository root
through their installed commands (for example, `uv run ols-calibrate`). They
are the complete set used by the current main paper and empirical calibration
note:

| Script | Main output used by TeX |
|---|---|
| `calibrate_eth_btc.py` | `results/params_empirical_showcase.json` |
| `calibrate_kou.py` | `latex/fig_ecf_fit.pdf` |
| `empirical_method_comparison.py` | `results/empirical_method_comparison.csv` |
| `health_buffer_evaluation_map.py` | `latex/fig_health_buffer_evaluation_map.pdf` |
| `mu_spread_sensitivity.py` | spread-sensitivity CSVs and three PDF figures |
| `objective_comparison.py` | `results/sizing_objective_comparison.csv` |

The empirical calibration must be run before the five downstream empirical
jobs. `calibrate_kou.py` is independent and generates the synthetic ECF
diagnostic in the companion note.

Scripts not used by either current TeX document are retained in
`legacy_jobs/`. The colleague-provided material under
`optimal_long_short/strategy/` is intentionally not part of this job archive.

Both TeX files default to hiding script paths. Change
`\showresultscriptpathsfalse` to `\showresultscriptpathstrue` in the relevant
preamble to print each generating script beneath its table or figure.
