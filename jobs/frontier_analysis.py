"""
Mean–variance–liquidation efficient frontier.

For three volatility regimes (0.5×, benchmark, 1.5×), sweeps h0 and plots
  x = Var(Pi_T | tau > T),  y = E[Pi_T | tau > T].

Liquidation probability is shown as direct annotations on the benchmark curve
(no colourbar).  Each annotated point carries both h0 and p_liq.

Usage:  python jobs/frontier_analysis.py
Output: latex/fig_frontier.pdf
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_long_short.model_params import KouParams
from optimal_long_short.market_params import MarketParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.moments import ConditionalMoments

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "stix",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.linewidth":    0.7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.fontsize":   8,
    "legend.frameon":    False,
})

# ── Parameters ─────────────────────────────────────────────────────────────────
BASE = dict(
    mu1=0.10, sigma1=0.30, lam1=2.0, p1=0.50, eta1_pos=0.10, eta1_neg=0.08,
    mu2=0.08, sigma2=0.25, lam2=2.0, p2=0.50, eta2_pos=0.09, eta2_neg=0.12,
    rho=0.5,
)
BASE_B = 0.80
BASE_T = 1.0
H0     = np.linspace(0.04, 2.00, 60)

def _p(**kw):
    d = dict(BASE); d.update(kw); return d

# ── Computation ────────────────────────────────────────────────────────────────
def compute_frontier(kou_kw, b=BASE_B, T=BASE_T):
    params = KouParams(**kou_kw)
    market = MarketParams(b=b, S10=1.0, S20=1.0)
    pliq, mean, var = [], [], []
    for h0 in H0:
        try:
            strat = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
            cm    = ConditionalMoments(params=params, strategy=strat)
            ps    = float(np.clip(cm.p_surv(), 0.0, 1.0))
            m1    = float(cm.conditional_moment(1))
            m2    = float(cm.conditional_moment(2))
            pliq.append(1.0 - ps)
            mean.append(m1)
            var.append(max(m2 - m1**2, 0.0))
        except Exception:
            pliq.append(np.nan); mean.append(np.nan); var.append(np.nan)
    return np.array(pliq), np.array(mean), np.array(var)

# ── Frontiers ──────────────────────────────────────────────────────────────────
FRONTIERS = [
    (r"$\sigma\!\times\!0.5$",  _p(sigma1=0.15, sigma2=0.125), "#1f77b4", "--", 1.1),
    (r"Benchmark",               BASE,                           "black",   "-",  2.0),
    (r"$\sigma\!\times\!1.5$",  _p(sigma1=0.45, sigma2=0.375), "#d62728", "--", 1.1),
]

print("Computing frontiers …")
results = []
for lbl, kw, color, ls, lw in FRONTIERS:
    print(f"  {lbl}")
    results.append((lbl, color, ls, lw, compute_frontier(kw)))

# ── Annotation points on the benchmark ────────────────────────────────────────
# (h0 value, label text, xytext offset, ha, va)
# Positions chosen so labels alternate above/below the curve and don't crowd.
# Benchmark curve direction: top-right (low h0) → bottom-left (high h0).
#
#   h0=0.10  (var≈1.82, mean≈2.47)  p_liq=76%
#   h0=0.25  (var≈0.90, mean≈1.73)  p_liq=46%
#   h0=0.50  (var≈0.50, mean≈1.29)  p_liq=17%
#   h0=1.00  (var≈0.30, mean≈1.12)  p_liq=0.9%

ANNOT_H0 = [0.10, 0.25, 0.50, 1.00]

# Offsets (dx, dy) in data coordinates, with alignment
ANNOT_STYLE = [
    ( 0.28, +0.22, "left",  "bottom"),   # h0=0.10: up-right
    (-0.28, +0.18, "right", "bottom"),   # h0=0.25: up-left (alternating)
    ( 0.28, -0.12, "left",  "top"   ),   # h0=0.50: down-right
    ( 0.28, +0.14, "left",  "bottom"),   # h0=1.00: up-right
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.8, 4.0))

# Draw frontiers
for lbl, color, ls, lw, (pliq, mean, var) in results:
    mask = np.isfinite(var) & np.isfinite(mean)
    ax.plot(var[mask], mean[mask],
            color=color, ls=ls, lw=lw,
            label=lbl, zorder=3 if lbl == "Benchmark" else 2)

# Annotate benchmark
bm_pliq, bm_mean, bm_var = results[1][4]
mask = np.isfinite(bm_var)
h0_valid = H0[mask]
var_valid = bm_var[mask]
mean_valid = bm_mean[mask]
pliq_valid = bm_pliq[mask]

for h0_target, (dx, dy, ha, va) in zip(ANNOT_H0, ANNOT_STYLE):
    idx = int(np.argmin(np.abs(h0_valid - h0_target)))
    xi, yi = var_valid[idx], mean_valid[idx]
    pi = pliq_valid[idx]

    # Dot on the curve
    ax.plot(xi, yi, "o", ms=4.5, color="black", zorder=5)

    # Format label
    if pi < 0.001:
        p_str = r"$p_\ell\!<\!0.1\%$"
    elif pi < 0.01:
        p_str = rf"$p_\ell\!=\!{100*pi:.1f}\%$"
    else:
        p_str = rf"$p_\ell\!=\!{100*pi:.0f}\%$"
    label_txt = rf"$h_0\!=\!{h0_target}$" + "\n" + p_str

    ax.annotate(
        label_txt,
        xy=(xi, yi),
        xytext=(xi + dx, yi + dy),
        fontsize=7.5,
        ha=ha, va=va,
        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7,
                        shrinkA=3, shrinkB=3),
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec="none", alpha=0.88),
        zorder=6,
    )

# Direction-of-h0 annotation
idx_a = int(np.argmin(np.abs(h0_valid - 1.30)))
idx_b = int(np.argmin(np.abs(h0_valid - 0.90)))
ax.annotate(
    "",
    xy   =(var_valid[idx_b], mean_valid[idx_b]),
    xytext=(var_valid[idx_a], mean_valid[idx_a]),
    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9,
                    mutation_scale=9),
    zorder=4,
)
ax.text(var_valid[idx_a] + 0.04, mean_valid[idx_a] - 0.04,
        r"$h_0\!\uparrow$", fontsize=7.5, color="0.45", va="top")

# ── Axes ───────────────────────────────────────────────────────────────────────
ax.set_xlabel(r"Conditional variance  $\mathrm{Var}(\Pi_T\!\mid\!\tau>T)$", labelpad=3)
ax.set_ylabel(r"Conditional mean  $\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",       labelpad=3)
ax.set_xlim(left=0.0)
ax.set_ylim(bottom=1.05)
ax.yaxis.grid(True, lw=0.4, alpha=0.25, color="0.65")
ax.set_axisbelow(True)

ax.legend(loc="upper left", handlelength=1.8, labelspacing=0.3)

out = Path(__file__).parent.parent / "latex" / "fig_frontier.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
