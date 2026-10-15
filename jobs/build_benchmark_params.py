"""Benchmark parameters for the merged paper: calibrated shape, sample drift.

The ECF calibration in results/params_WBTC_WETH.json is fitted to EWM-de-meaned
returns, so its mu_i are normalizers, not drift estimates.  Following the
drift-outside-the-ECF construction, the benchmark keeps every diffusion, jump and
correlation parameter of that fit and replaces the two drifts with the headline
window's sample drifts in the price-growth convention, carry included:

    mu_i = log mean(exp(r_i)) / dt + r_i^role,

the same quantities as ``sample_drift_headline`` in jobs/breakeven_drift.py.

Writes results/params_WBTC_WETH_benchmark.json (same layout as the calibration
file, so any job can read it in place of params_WBTC_WETH.json).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
from breakeven_drift import sample_drift_headline  # noqa: E402
from optimal_long_short.model_params import KouParams  # noqa: E402

KEYS = ["mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
        "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho"]


def main():
    blob = json.load(open(ROOT / "results/params_WBTC_WETH.json"))
    d = sample_drift_headline()
    P = dict(blob["params"])
    P["mu1"], P["mu2"] = d["mu1"], d["mu2"]
    kp = KouParams(**{k: P[k] for k in KEYS})
    P["muX1"], P["muX2"] = kp.muX1, kp.muX2
    blob["params"] = P
    blob["_note"] = ("Benchmark for the merged paper: ECF shape parameters of params_WBTC_WETH.json "
                     "with the headline window's sample drifts (price growth plus carry) in place of "
                     "the ECF drift normalizers. Built by jobs/build_benchmark_params.py.")
    blob["benchmark_drift"] = d
    blob.pop("drift_summary", None)
    out = ROOT / "results/params_WBTC_WETH_benchmark.json"
    json.dump(blob, open(out, "w"), indent=2, default=float)
    print(f"mu1={d['mu1']:+.4f} mu2={d['mu2']:+.4f} nu={d['mu1'] - d['mu2']:+.4f}  -> {out}")


if __name__ == "__main__":
    main()
