#!/bin/bash
# Reproduce Sections 6-7 and Table D.2 of latex/optimal_long_short_merged.tex.
#
# Those results use the benchmark parameters (ECF shape parameters with sample drifts),
# results/params_WBTC_WETH_benchmark.json, built by jobs/build_benchmark_params.py.
# The jobs below read results/params_WBTC_WETH.json and write into results/ and latex/,
# so they are run in a temporary copy of the repository where the benchmark file takes
# the calibration file's place; nothing in the working tree is overwritten. Outputs are
# collected into results/merged/ (and the sensitivity figure into
# latex/fig_sensitivity_merged.pdf).
#
# Usage: bash jobs/run_merged_benchmark.sh        (about 50 minutes)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/jobs/build_benchmark_params.py"

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
cp -R "$ROOT/optimal_long_short" "$ROOT/jobs" "$ROOT/results" "$W/"
mkdir -p "$W/latex" "$W/logs"
ln -s "$ROOT/data" "$W/data"
cp "$ROOT/results/params_WBTC_WETH_benchmark.json" "$W/results/params_WBTC_WETH.json"

cd "$W"
for job in numerical_comparison mc_convergence sensitivity_analysis overshoot_table \
           regime_calibration kelly_sizing; do
  echo "running $job"
  PYTHONPATH="$W" python3 -u "jobs/$job.py" > "logs/$job.log" 2>&1
done

OUT="$ROOT/results/merged"
mkdir -p "$OUT"
cp logs/*.log "$OUT/"
cp results/a1_results_row.json results/mc_convergence.json results/revision_moderate.json \
   results/regime_calibrated.json results/kelly.json "$OUT/"
cp latex/fig_sensitivity.pdf "$ROOT/latex/fig_sensitivity_merged.pdf"
echo "outputs in $OUT"
