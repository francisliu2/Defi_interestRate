"""Kelly (log-utility) sizing on the calibrated WBTC/WETH book.

1. Validates the semi-analytic E[log W_T] of optimal_long_short.kelly against a
   Brownian-bridge-corrected Monte Carlo that records exact leg values at liquidation.
2. Traces the Kelly objective over the paper's buffer grid (H0 in [H0_min, 2]).
3. Finds the Kelly interior threshold nu*_Kelly by bisection on the spread drift,
   the Kelly buffer at nu = +0.2 (the paper's mean-variance anchor), and the
   half-Kelly buffer (half the Kelly debt).
4. Bayes-Kelly: maximises the posterior expectation of E[log W_T] over the
   carry-anchored drift posterior of results/task4_numbers.json (Gauss-Hermite).

Writes results/kelly.json and latex/fig_kelly.pdf.

Usage: python jobs/kelly_sizing.py [--mc-paths N] [--mc-steps N] [--no-mc]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimal_long_short.utils.helpers import load_calibrated_params  # noqa: E402
from optimal_long_short.kelly import (  # noqa: E402
    KellyGrid, expected_log_wealth, half_kelly_buffer, kelly_buffer, kelly_threshold)

HEADLINE_PARAMS = ROOT / "results" / "params_WBTC_WETH_headline.json"

T = 1.0 / 12.0


# ---------------------------------------------------------------- Monte Carlo
def _kou_jump(rng, lam, prob, ep, en, n, dt):
    nj = rng.poisson(lam * dt, size=n)
    nu = rng.binomial(nj, prob)
    nd = nj - nu
    up = np.where(nu > 0, rng.gamma(np.maximum(nu, 1), ep), 0.0)
    dn = np.where(nd > 0, rng.gamma(np.maximum(nd, 1), en), 0.0)
    return up - dn


def mc_log_wealth(p, h0, b, liq_bonus, n_paths, n_steps, seed, w_floor=1e-4):
    """E[log W_T] by simulation, per unit equity.

    Diffusive crossings (grid or Brownian-bridge) are continuous, so the health
    factor is exactly one there: overshoot 0, debt leg taken at the step midpoint.
    Jump crossings use the actual post-jump leg values, i.e. the exact overshoot.
    """
    rng = np.random.default_rng(seed)
    H0 = math.exp(h0)
    c = b * (1.0 + liq_bonus)
    scale = H0 - b
    m = -h0
    dt = T / n_steps
    sdt = math.sqrt(dt)
    sq = math.sqrt(1.0 - p.rho**2)
    sigZ2 = p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2
    X1 = np.zeros(n_paths)
    X2 = np.zeros(n_paths)
    alive = np.ones(n_paths, bool)
    logw = np.zeros(n_paths)
    for _ in range(n_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        n = idx.size
        x1, x2 = X1[idx], X2[idx]
        w1 = rng.standard_normal(n)
        w2 = rng.standard_normal(n)
        d1 = p.muX1 * dt + p.sigma1 * sdt * w1
        d2 = p.muX2 * dt + p.sigma2 * sdt * (p.rho * w1 + sq * w2)
        z0, zm = x1 - x2, (x1 + d1) - (x2 + d2)
        diff_cross = zm <= m
        safe = ~diff_cross
        pb = np.zeros(n)
        pb[safe] = np.exp(-2.0 * (z0[safe] - m) * (zm[safe] - m) / (sigZ2 * dt))
        diff_cross |= rng.random(n) < pb
        J1 = _kou_jump(rng, p.lam1, p.p1, p.eta1_pos, p.eta1_neg, n, dt)
        J2 = _kou_jump(rng, p.lam2, p.p2, p.eta2_pos, p.eta2_neg, n, dt)
        x1n, x2n = x1 + d1 + J1, x2 + d2 + J2
        jump_cross = (~diff_cross) & ((x1n - x2n) <= m)
        # continuous crossing: h_tau = 0
        i_d = idx[diff_cross]
        logw[i_d] = (x2[diff_cross] + 0.5 * d2[diff_cross]) - math.log(scale) + math.log(1.0 - c)
        # jump crossing: exact leg values after the jump
        i_j = idx[jump_cross]
        h_tau = h0 + (x1n - x2n)[jump_cross]
        r = np.exp(x2n[jump_cross]) * (np.exp(h_tau) - c) / scale
        logw[i_j] = np.log(np.maximum(r, w_floor))
        dead = diff_cross | jump_cross
        alive[idx[dead]] = False
        keep = ~dead
        X1[idx[keep]] = x1n[keep]
        X2[idx[keep]] = x2n[keep]
    s = np.flatnonzero(alive)
    logw[s] = X2[s] - math.log(scale) + np.log(np.exp(h0 + X1[s] - X2[s]) - b)
    return float(logw.mean()), float(logw.std(ddof=1) / math.sqrt(n_paths)), float(alive.mean())


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-paths", type=int, default=200_000)
    ap.add_argument("--mc-steps", type=int, default=2000)
    ap.add_argument("--no-mc", action="store_true")
    args = ap.parse_args()

    params, market = load_calibrated_params(HEADLINE_PARAMS)
    b, lam = market["b"], market["liq_bonus"]
    nu_cal = params.mu1 - params.mu2
    out: dict = {"b": b, "liq_bonus": lam, "T": T, "nu_calibrated": nu_cal,
                 "note": ("thresholds/anchors/bayes_kelly here use the capped grid H0 <= 2 and the ECF "
                          "drift normalizer; the paper's break-even views and posteriors come from "
                          "results/breakeven_drift.json. The paper uses this file for the Monte Carlo "
                          "validation of E[log W_T] only.")}

    # 1. validation ----------------------------------------------------------
    if not args.no_mc:
        rows = []
        for H in (1.07, 1.16, 1.30):
            h = math.log(H)
            sa = expected_log_wealth(params, h, b=b, liq_bonus=lam, T=T)
            t0 = time.time()
            mc, se, ps = mc_log_wealth(params, h, b, lam, args.mc_paths, args.mc_steps, seed=7)
            rows.append({"H0": H, "semi_analytic": sa["E_log_W"], "mc": mc, "mc_se": se,
                         "z": (mc - sa["E_log_W"]) / se, "p_surv_sa": sa["p_surv"],
                         "p_surv_mc": ps, "mc_secs": time.time() - t0})
            print(f"H0={H:.2f}  E[log W] semi-analytic {sa['E_log_W']:+.5f}   "
                  f"MC {mc:+.5f} +- {se:.5f}  (z={rows[-1]['z']:+.2f})", flush=True)
        out["validation"] = {"mc_paths": args.mc_paths, "mc_steps": args.mc_steps, "rows": rows}

    # 2. Kelly curve at the calibrated drift ----------------------------------
    grid = KellyGrid(params=params, b=b, liq_bonus=lam, T=T)
    details = [expected_log_wealth(params, float(h), b=b, liq_bonus=lam, T=T) for h in grid.h_grid]
    out["benchmark_curve"] = [
        {k: d[k] for k in ("H0", "initial_leverage", "p_liq", "E_log_W", "p_bad_debt")}
        for d in details
    ]
    H_cal, int_cal = kelly_buffer(grid, nu_cal)
    print(f"calibrated nu={nu_cal:+.3f}: Kelly H0*={H_cal:.3f} interior={int_cal}", flush=True)

    # 3. thresholds --------------------------------------------------------------
    lo, hi = 0.0, 0.2
    while kelly_buffer(grid, lo)[1]:
        lo -= 0.1
    while not kelly_buffer(grid, hi)[1]:
        hi += 0.1
    nu_k = kelly_threshold(grid, lo, hi)
    def half_interior(nu: float) -> bool:
        H, interior = kelly_buffer(grid, nu)
        return interior and half_kelly_buffer(H, b) < grid.H0_hi

    lo2, hi2 = nu_k, max(hi, nu_k + 0.1)
    while not half_interior(hi2):
        hi2 += 0.1
    while hi2 - lo2 > 0.005:
        mid = 0.5 * (lo2 + hi2)
        hi2, lo2 = (mid, lo2) if half_interior(mid) else (hi2, mid)
    nu_half = hi2
    anchor = {}
    for nu in (0.1, 0.2, 0.4):
        H, interior = kelly_buffer(grid, nu)
        Hh = half_kelly_buffer(H, b) if interior else grid.H0_hi
        anchor[str(nu)] = {"kelly_H0": H, "interior": interior,
                           "half_kelly_H0": min(Hh, grid.H0_hi),
                           "kelly_leverage": H / (H - b)}
        print(f"nu={nu:+.2f}: Kelly H0*={H:.3f} (L0={H/(H-b):.2f}), half-Kelly H0={min(Hh, 2.0):.3f}",
              flush=True)
    out["thresholds"] = {"kelly": nu_k, "half_kelly": nu_half}
    out["anchors"] = anchor
    print(f"Kelly threshold nu* = {nu_k:+.3f}; half-Kelly threshold = {nu_half:+.3f}", flush=True)

    # 4. Bayes-Kelly over the drift posterior -------------------------------------
    tb = json.load(open(ROOT / "results/task4_numbers.json"))
    x, w = np.polynomial.hermite.hermgauss(11)
    bayes = {}
    for tag in ("carry_s.20__ecf", "carry_s.20__emp", "carry_s.10__ecf", "carry_s.40__ecf"):
        post = tb["posteriors"][tag]
        m, s = post["post_mean"], post["post_sd"]
        nus = m + math.sqrt(2.0) * s * x
        expected = sum(wi / math.sqrt(math.pi) * grid.curve(float(nu)) for nu, wi in zip(nus, w))
        i = int(np.argmax(expected))
        p_interior = 0.5 * math.erfc((nu_k - m) / (s * math.sqrt(2.0)))
        bayes[tag] = {"post_mean": m, "post_sd": s,
                      "bayes_kelly_H0": math.exp(grid.h_grid[i]),
                      "bayes_kelly_interior": 0 < i < len(expected) - 1,
                      "P_kelly_interior": p_interior}
        print(f"{tag}: posterior N({m:+.3f}, {s:.3f}^2) -> Bayes-Kelly H0={bayes[tag]['bayes_kelly_H0']:.3f} "
              f"(interior={bayes[tag]['bayes_kelly_interior']}), P(nu > nu*_Kelly)={p_interior:.3f}",
              flush=True)
    out["bayes_kelly"] = bayes
    out["mean_variance_thresholds"] = tb["threshold_delta_star"]

    json.dump(out, open(ROOT / "results/kelly.json", "w"), indent=2, default=float)
    print("SAVED results/kelly.json", flush=True)

    # 5. figure ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    H = np.exp(grid.h_grid)
    for nu, col in [(nu_cal, "#444444"), (0.0, "#1b9e77"), (0.1, "#7570b3"), (0.2, "#d95f02")]:
        curve = grid.curve(nu)
        ax[0].plot(H, curve - curve[-1], color=col, lw=1.8, label=fr"$\nu={nu:+.2f}$")
    ax[0].axhline(0, color="grey", lw=0.8, ls=":")
    ax[0].set_xlabel(r"initial health factor $H_0$")
    ax[0].set_ylabel(r"$E[\log W_T]$ relative to $H_0=2$")
    ax[0].set_ylim(-0.05, 0.02)
    ax[0].legend(frameon=False, fontsize=8)
    nus = np.round(np.arange(-0.1, 0.505, 0.01), 3)
    ax[1].plot(nus, [kelly_buffer(grid, float(v))[0] for v in nus], color="black", lw=2, label="Kelly")
    ax[1].plot(nus, [min(half_kelly_buffer(Hk, b), 2.0) if it else 2.0
                     for Hk, it in (kelly_buffer(grid, float(v)) for v in nus)],
               color="black", lw=1.4, ls="--", label="half-Kelly")
    ax[1].axhline(2.0, ls=":", c="grey", lw=1)
    ax[1].set_xlabel(r"spread drift view $\nu=\mu_1-\mu_2$ (per year)")
    ax[1].set_ylabel(r"optimal buffer $H_0^\star$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "latex/fig_kelly.pdf")
    print("saved latex/fig_kelly.pdf", flush=True)


if __name__ == "__main__":
    main()
