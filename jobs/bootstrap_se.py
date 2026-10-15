"""T11/A3: block-bootstrap standard errors for the 13 ECF-calibrated parameters,
using the SAME optimiser bounds as the headline fit (``CALIB_BOUNDS`` from
``calibrate_btc_eth``: lambda_max=5000, p in [0.001, 0.999], eta minima 1e-5).
Circular block bootstrap (L=12) on the 840 four-hour demeaned log-returns.

  * each resample is recalibrated with a multi-start cloud (n_starts=8; the
    headline fit uses 30);
  * the fraction of resamples at the jump-intensity bound is reported;
  * the bootstrap SE of the spread drift mu1 - mu2 is reported directly,
    together with the correlation between the two drift estimates.

An earlier run used the library-default bounds (lambda_max=500), which is tighter
than the headline fit; its output is kept in results/bootstrap_se_lam500.json.

Usage: python jobs/bootstrap_se.py [B=200] [n_starts=8] [workers=7]
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd
from optimal_long_short.calibration import calibrate_ecf
from optimal_long_short.job_runners.calibrate_btc_eth import CALIB_BOUNDS
from optimal_long_short.model_params import KouParams

DT = 4/(24*365)
df = pd.read_parquet("data/AAVE/merged_wbtc_weth_4h.parquet")
lb = np.diff(np.log(df["close_btc"].to_numpy()))
le = np.diff(np.log(df["close_eth"].to_numpy()))
# EWM demean, span 30, to match the stored calibration pipeline
lb = lb - pd.Series(lb).ewm(span=30).mean().to_numpy()
le = le - pd.Series(le).ewm(span=30).mean().to_numpy()
n = len(lb)

blob = json.load(open("results/params_WBTC_WETH.json")); pd_ = blob["params"]
NAMES = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
base = KouParams(**{k: pd_[k] for k in NAMES})

# A resample is counted as "railed" if either intensity lands within LAM_TOL of
# the (headline) optimiser bound.
LAM_BOUND = float(CALIB_BOUNDS.lambda_max)
LAM_TOL = 1.0

B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NSTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
L = 12
nb = int(np.ceil(n/L))
rng = np.random.default_rng(0)
INDICES = [np.concatenate([(np.arange(s, s+L) % n) for s in rng.integers(0, n, size=nb)])[:n]
           for _ in range(B)]


def _one(b):
    idx = INDICES[b]
    try:
        p = calibrate_ecf(lb[idx], le[idx], DT, n_starts=NSTARTS, seed=b, bounds=CALIB_BOUNDS).params
        return b, {k: getattr(p, k) for k in NAMES}
    except Exception as e:  # noqa: BLE001
        return b, str(e)


if __name__ == "__main__":
    print("n returns", n, "B", B, "n_starts", NSTARTS, "lam_bound", LAM_BOUND, flush=True)
    draws = {k: [] for k in NAMES}
    draws["spread"] = []
    n_railed = 0
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (b, res) in enumerate(pool.imap_unordered(_one, range(B)), 1):
            if isinstance(res, str):
                print("  resample %d failed: %s" % (b, res), flush=True)
            else:
                for k in NAMES:
                    draws[k].append(res[k])
                draws["spread"].append(res["mu1"] - res["mu2"])
                if max(res["lam1"], res["lam2"]) >= LAM_BOUND - LAM_TOL:
                    n_railed += 1
            if i % 10 == 0:
                print("  bootstrap %d/%d  elapsed %.0fs  kept %d  railed %d"
                      % (i, B, time.time()-t0, len(draws["sigma1"]), n_railed), flush=True)

    kept = len(draws["sigma1"])
    mu1a = np.array(draws["mu1"]); mu2a = np.array(draws["mu2"]); spa = np.array(draws["spread"])
    out = {
        "n_boot_kept": kept, "block_len": L, "n_starts": NSTARTS,
        "lam_bound": LAM_BOUND,
        "frac_railed": n_railed / kept if kept else None,
        "point": {k: getattr(base, k) for k in NAMES},
        "point_spread": float(base.mu1 - base.mu2),
        "se": {}, "ci": {},
        "spread": {
            "point": float(base.mu1 - base.mu2),
            "se": float(np.std(spa, ddof=1)) if kept > 1 else None,
            "ci": [float(np.quantile(spa, 0.025)), float(np.quantile(spa, 0.975))] if kept > 1 else None,
        },
        "drift_corr": float(np.corrcoef(mu1a, mu2a)[0, 1]) if kept > 1 else None,
    }
    for k in NAMES:
        a = np.array(draws[k])
        out["se"][k] = float(np.std(a, ddof=1)) if len(a) > 1 else None
        out["ci"][k] = [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))] if len(a) > 1 else None
    json.dump(out, open("results/bootstrap_se.json", "w"), indent=2)
    print("SAVED results/bootstrap_se.json", flush=True)
    print("frac_railed=%.3f  drift_corr=%.3f  spread SE=%.3f  spread point=%.3f"
          % (out["frac_railed"], out["drift_corr"], out["spread"]["se"], out["spread"]["point"]), flush=True)
    for k in NAMES:
        print("  %-10s point=%+.5f  se=%.5f  95%%CI=[%+.4f,%+.4f]"
              % (k, out["point"][k], out["se"][k], out["ci"][k][0], out["ci"][k][1]), flush=True)
