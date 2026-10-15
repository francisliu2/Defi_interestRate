"""Break-even drift views and optimal leverage on the full buffer range.

For each calibration (headline 4h; full-history daily; 2024-onward 4h) this job
computes, for mean-variance on terminal wealth (alpha = 0.5, 1, 2; exact overshoot
recovery) and for Kelly (E[log W_T]):

1. the analytic break-even spread drift nu* at zero debt.  With debt x per unit
   equity, W_T = (1+x) e^{X1} - x e^{X2} on survival, and the liquidation
   probability vanishes as x -> 0, so
       dPhi/dx|_0   = (E e^{X1} - E e^{X2}) - alpha Cov(e^{X1}, e^{X1} - e^{X2}),
       dElogW/dx|_0 = 1 - E e^{X2 - X1},
   all closed-form joint exponential moments.  Leverage improves on the unlevered
   position iff the derivative is positive;
2. a numerical check: the argmax over a debt grid from x = 1e-3 (H0 ~ 781) to the
   origination maximum, plus the unlevered point, must leave the unlevered point
   exactly when nu exceeds nu*;
3. the optimal leverage at the sample spread drift of each window;
4. the posterior probability that leverage pays, P(nu > nu*), under the
   carry-anchored prior nu ~ N(c, 0.2^2) updated with the sample drift (Newey-West
   standard error), and the Bayes-Kelly leverage (posterior-averaged E[log W]);
5. Figure 2: optimal initial leverage against the drift view.

Writes results/breakeven_drift.json and latex/fig_buffer_drift.pdf.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimal_long_short.drift import with_price_drift_view  # noqa: E402
from optimal_long_short.job_runners.common import load_calibrated_params  # noqa: E402
from optimal_long_short.kelly import expected_log_wealth  # noqa: E402
from optimal_long_short.model_params import KouParams  # noqa: E402
from optimal_long_short.overshoot_recovery import (exact_wealth_moments,  # noqa: E402
                                                   tilt_exponent_bivariate)

T = 1.0 / 12.0
ALPHAS = (0.5, 1.0, 2.0)
KEYS = ["mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
        "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho"]
PRIOR_SD = 0.20


def view(p, nu, mu2):
    """Parameters with debt-leg price growth mu2 (sample level, incl. borrow APR) and
    spread drift mu1 - mu2 = nu (price growth incl. carry)."""
    return with_price_drift_view(p, mu1=mu2 + nu, mu2=mu2)


def kelly_closed_form(p):
    """Exact Kelly break-even: 1 - E e^{X2-X1} = 0  <=>  Psi_M(-1, 1) = 0, i.e.
    nu* = sigma1^2 - rho sigma1 sigma2 + lambda1 [M1(1) + M1(-1) - 2] (independent of T)."""
    m1 = lambda s: p.p1 / (1 - s * p.eta1_pos) + (1 - p.p1) / (1 + s * p.eta1_neg)  # noqa: E731
    diff = p.sigma1**2 - p.rho * p.sigma1 * p.sigma2
    jump = p.lam1 * (m1(1.0) + m1(-1.0) - 2.0)
    return {"diffusion_part": diff, "jump_part": jump, "nu_star": diff + jump}


def kelly_closed_form_reversed(p):
    """Same for the reversed book (asset 2 as collateral, asset 1 as debt)."""
    m2 = lambda s: p.p2 / (1 - s * p.eta2_pos) + (1 - p.p2) / (1 + s * p.eta2_neg)  # noqa: E731
    diff = p.sigma2**2 - p.rho * p.sigma1 * p.sigma2
    jump = p.lam2 * (m2(1.0) + m2(-1.0) - 2.0)
    return {"diffusion_part": diff, "jump_part": jump, "nu_star": diff + jump}


def marginal(p, nu, obj, mu2):
    q = view(p, nu, mu2)
    E = lambda a, c: math.exp(T * tilt_exponent_bivariate(q, a, c))  # noqa: E731
    if obj == "kelly":
        return 1.0 - E(-1.0, 1.0)
    e1, e2, e11, e12 = E(1, 0), E(0, 1), E(2, 0), E(1, 1)
    return (e1 - e2) - obj * (e11 - e12 - e1 * (e1 - e2))


def breakeven(p, obj, mu2):
    return brentq(lambda nu: marginal(p, nu, obj, mu2), -1.0, 2.0, xtol=1e-6)


class Curve:
    """Objectives on a debt grid x in [1e-3, x_max] plus the unlevered point (x = 0)."""

    def __init__(self, p, b, liq, h0_min, mu2, n=30):
        self.p, self.b, self.liq, self.mu2 = p, b, liq, mu2
        x_max = b / (math.exp(h0_min) - b)
        self.x = np.concatenate([[0.0], np.geomspace(1e-3, x_max, n)])
        self.H0 = np.where(self.x > 0, b + b / np.maximum(self.x, 1e-300), np.inf)
        self.L0 = 1.0 + self.x
        self.cache = {}

    def values(self, nu):
        key = round(float(nu), 5)
        if key not in self.cache:
            q = view(self.p, key, self.mu2)
            mv, kl = [], []
            for x, H in zip(self.x, self.H0):
                if x == 0:  # unlevered: W_T = e^{X1_T}, closed form
                    e1 = math.exp(T * tilt_exponent_bivariate(q, 1.0, 0.0))
                    e11 = math.exp(T * tilt_exponent_bivariate(q, 2.0, 0.0))
                    elog = T * (q.muX1 + q.lam1 * (q.p1 * q.eta1_pos - (1 - q.p1) * q.eta1_neg))
                    mv.append((e1, e11 - e1**2))
                    kl.append(elog)
                    continue
                h = math.log(H)
                w = exact_wealth_moments(q, h, b=self.b, liq_bonus=self.liq, T=T)
                k = expected_log_wealth(q, h, b=self.b, liq_bonus=self.liq, T=T)
                mv.append((w["mean_wealth"], w["var_wealth"]))
                kl.append(k["E_log_W"])
            if not (np.all(np.isfinite([m for pair in mv for m in pair])) and np.all(np.isfinite(kl))):
                raise FloatingPointError(f"non-finite objective on the debt grid at nu={key}")
            self.cache[key] = (np.array(mv), np.array(kl))
        return self.cache[key]

    def argmax(self, nu, obj):
        mv, kl = self.values(nu)
        vals = kl if obj == "kelly" else mv[:, 0] - 0.5 * obj * mv[:, 1]
        i = int(np.argmax(vals))
        return i, float(self.L0[i])


def newey_west(x, lags):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    v = x @ x / n + sum(2 * (1 - L / (lags + 1)) * (x[L:] @ x[:-L]) / n for L in range(1, lags + 1))
    return math.sqrt(v / n)


def sample_drift_headline():
    df = pd.read_parquet(ROOT / "data/AAVE/merged_wbtc_weth_4h.parquet")
    r1, r2 = np.diff(np.log(df.close_btc.values)), np.diff(np.log(df.close_eth.values))
    dt = 4 / (24 * 365)
    g1, g2 = math.log(np.mean(np.exp(r1))) / dt, math.log(np.mean(np.exp(r2))) / dt
    s1 = float(df.supply_apr_btc.mean() / 100.0)
    b2 = float(df.variable_borrow_apr_eth.mean() / 100.0)
    se = newey_west(r1 - r2, 120) / dt  # 120 four-hour lags = 20 days
    return {"nu_price": g1 - g2, "carry": s1 - b2, "nu": g1 - g2 + s1 - b2, "se_nw": se,
            "mu1": g1 + s1, "mu2": g2 + b2, "raw_log_spread": float((r1 - r2).mean() / dt),
            "se_naive": float(np.std(r1 - r2, ddof=1) / math.sqrt(len(r1)) / dt)}


def sample_drift_full():
    from optimal_long_short.full_history import pair_panel
    P = pair_panel(freq="1D", start="2023-01-28")
    r, dt = P.returns, P.dt_years
    g1, g2 = math.log(np.mean(np.exp(r.r1))) / dt, math.log(np.mean(np.exp(r.r2))) / dt
    s1, b2 = float(P.bars.supply1.mean()), float(P.bars.borrow2.mean())
    se = newey_west((r.r1 - r.r2).values, 20) / dt
    return {"nu_price": g1 - g2, "carry": s1 - b2, "nu": g1 - g2 + s1 - b2, "se_nw": se,
            "mu1": g1 + s1, "mu2": g2 + b2, "raw_log_spread": float((r.r1 - r.r2).mean() / dt)}


def main():
    head, market = load_calibrated_params()
    b, liq, h0_min = market["b"], market["liq_bonus"], market["h0_min"]
    hz = json.load(open(ROOT / "results/horizon_analysis.json"))
    cals = {"headline_4h": head}
    for name in ("daily_full", "4h_2024"):
        cals[name] = KouParams(**{k: hz["calibrations"][name][k] for k in KEYS})
    drift = {"headline_4h": sample_drift_headline(), "full_history": sample_drift_full()}
    data_for = {"headline_4h": "headline_4h", "daily_full": "full_history", "4h_2024": "full_history"}
    objs = list(ALPHAS) + ["kelly"]
    out = {"drift_estimates": drift, "calibrations": {}}
    print(json.dumps(drift, indent=1), flush=True)

    for name, p in cals.items():
        d = drift[data_for[name]]
        nu_star = {str(o): breakeven(p, o, d["mu2"]) for o in objs}
        kcf = kelly_closed_form(p)
        assert abs(kcf["nu_star"] - nu_star["kelly"]) < 1e-5, (kcf, nu_star["kelly"])
        curve = Curve(p, b, liq, h0_min, d["mu2"])
        # grid check: just below and above each analytic threshold
        check = {}
        for o in objs:
            ns = nu_star[str(o)]
            below = curve.argmax(ns - 0.01, o)[0] == 0
            above = curve.argmax(ns + 0.01, o)[0] != 0
            check[str(o)] = bool(below and above)
        lev_at_hat = {str(o): curve.argmax(d["nu"], o)[1] for o in objs}
        # posterior with carry-anchored prior
        w0, w1 = 1 / PRIOR_SD**2, 1 / d["se_nw"] ** 2
        m = (w0 * d["carry"] + w1 * d["nu"]) / (w0 + w1)
        s = math.sqrt(1 / (w0 + w1))
        p_lever = {str(o): 0.5 * math.erfc((nu_star[str(o)] - m) / (s * math.sqrt(2))) for o in objs}
        se = d["se_nw"]
        p_lever_flat = {str(o): 0.5 * math.erfc((nu_star[str(o)] - d["nu"]) / (se * math.sqrt(2))) for o in objs}
        bk_rule = nu_star["kelly"] + 0.5 * T * s**2   # exact: lever iff posterior mean exceeds this
        # Bayes-Kelly: maximise the posterior average of E[log W]
        xg, wg = np.polynomial.hermite.hermgauss(9)
        avg = sum(wi / math.sqrt(math.pi) * curve.values(m + math.sqrt(2) * s * xi)[1] for xi, wi in zip(xg, wg))
        ib = int(np.argmax(avg))
        res = {"nu_star": nu_star, "grid_check": check, "nu_hat_used": d["nu"],
               "optimal_leverage_at_nu_hat": lev_at_hat, "posterior": {"mean": m, "sd": s},
               "P_leverage_pays": p_lever, "P_leverage_pays_flat_prior": p_lever_flat,
               "bayes_kelly_leverage": float(curve.L0[ib]),
               "bayes_kelly_closed_rule": {"threshold_on_posterior_mean": bk_rule,
                                           "posterior_mean": m, "levers": bool(m > bk_rule)},
               "kelly_closed_form": kcf, "kelly_closed_form_reversed_book": kelly_closed_form_reversed(p),
               "drift_sample_used": data_for[name]}
        out["calibrations"][name] = res
        print(f"{name}: nu* " + " ".join(f"{k}={v:+.4f}" for k, v in nu_star.items())
              + f" | grid check {check} | L0* at nu_hat={d['nu']:+.3f}: {lev_at_hat}"
              + f" | posterior N({m:+.3f},{s:.3f}) P(lever) {({k: round(v, 3) for k, v in p_lever.items()})}"
              + f" | flat-prior P {({k: round(v, 3) for k, v in p_lever_flat.items()})}"
              + f" | Bayes-Kelly L0={curve.L0[ib]:.2f} (rule m={m:+.4f} vs {bk_rule:+.4f})"
              + f" | Kelly closed form {kcf} | reversed {kelly_closed_form_reversed(p)}", flush=True)
        if name in ("headline_4h", "daily_full"):
            nus = np.round(np.arange(-0.04, 0.305, 0.02), 4)
            res["leverage_curve"] = {"nu": nus.tolist(),
                                     **{str(o): [curve.argmax(v, o)[1] for v in nus] for o in objs}}

    json.dump(out, open(ROOT / "results/breakeven_drift.json", "w"), indent=2, default=float)
    print("SAVED results/breakeven_drift.json", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    styles = {"0.5": ("#1b9e77", "-"), "1.0": ("#7570b3", "-"), "2.0": ("#d95f02", "-"),
              "kelly": ("black", "--")}
    for ax, name, title in ((axes[0], "headline_4h", "headline calibration"),
                            (axes[1], "daily_full", "full-history daily calibration")):
        lc = out["calibrations"][name]["leverage_curve"]
        for o, (col, ls) in styles.items():
            lab = "Kelly" if o == "kelly" else fr"$\alpha={o}$"
            ax.plot(lc["nu"], lc[o], color=col, ls=ls, lw=1.8, label=lab)
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.axvline(out["calibrations"][name]["nu_hat_used"], color="grey", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"spread drift view $\nu=\mu_1-\mu_2$ (per year)")
    axes[0].set_ylabel(r"optimal initial leverage $L_0^\star$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "latex/fig_buffer_drift.pdf")
    print("saved latex/fig_buffer_drift.pdf", flush=True)


if __name__ == "__main__":
    main()
