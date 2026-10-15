"""Significance of the out-of-sample over-prediction in results/rolling_backtest.json.

For each predictor and buffer: mean predicted minus realised crossing frequency,
its z-score under independent monthly Bernoulli outcomes, and a moving-block
bootstrap (blocks of 3 months) that respects serial dependence across months.

Writes results/rolling_backtest_bias.json.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BT = json.load(open(ROOT / "results/rolling_backtest.json"))
BLOCK, B = 3, 20000


def main():
    rng = np.random.default_rng(0)
    out = {}
    for key in BT["summary"]:
        scheme, var = key.split("/")
        rows = sorted((r for r in BT["rows"] if r["scheme"] == scheme and "error" not in r),
                      key=lambda r: r["month"])
        res = {}
        for h in map(str, BT["h0_grid"]):
            p = np.array([r[var][h] for r in rows])
            y = np.array([BT["outcomes"][r["month"]][h] for r in rows], float)
            d = p - y
            z = d.sum() / math.sqrt((p * (1 - p)).sum())
            n = len(d)
            starts = rng.integers(0, n - BLOCK + 1, size=(B, math.ceil(n / BLOCK)))
            idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]
            boot = d[idx].mean(axis=1)
            res[h] = {"mean_pred": float(p.mean()), "realised": float(y.mean()),
                      "bias": float(d.mean()), "z_independent": float(z),
                      "block_bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                      "block_bootstrap_p_bias_le_0": float((boot <= 0).mean())}
        out[key] = res
        print(key + "  " + "  ".join(
            f"h0={h}: z={v['z_independent']:.2f}, CI=[{v['block_bootstrap_ci95'][0]:+.3f},{v['block_bootstrap_ci95'][1]:+.3f}]"
            for h, v in res.items() if h in ("0.05", "0.1", "0.15")))
    json.dump(out, open(ROOT / "results/rolling_backtest_bias.json", "w"), indent=2)
    print("SAVED results/rolling_backtest_bias.json")


if __name__ == "__main__":
    main()
