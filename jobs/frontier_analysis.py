"""
3-D mean–variance–survival frontier.

  x = Var(Pi_T | tau > T)   — conditional variance
  y = p_surv                 — survival probability
  z = E[Pi_T | tau > T]     — conditional mean  (the vertical axis)

Three volatility regimes are drawn.  The benchmark is annotated at four
representative h0 values; each annotation shows h0 only.
A floor projection (z = 1) and vertical drop lines anchor the 3-D curve
to the variance–survival plane so the reader can also read the 2-D frontier.

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
from mpl_toolkits.mplot3d import Axes3D          # registers the '3d' projection

from optimal_long_short.model_params import KouParams
from optimal_long_short.market_params import MarketParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.moments import ConditionalMoments

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "stix",
    "font.size":         8.5,
    "axes.linewidth":    0.6,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.frameon":    False,
    "legend.fontsize":   8,
})

# ── Baseline parameters ────────────────────────────────────────────────────────
BASE = dict(
    mu1=0.10, sigma1=0.30, lam1=2.0, p1=0.50, eta1_pos=0.10, eta1_neg=0.08,
    mu2=0.08, sigma2=0.25, lam2=2.0, p2=0.50, eta2_pos=0.09, eta2_neg=0.12,
    rho=0.5,
)
BASE_B = 0.80
BASE_T = 1.0

# Fine grid gives a smooth curve in 3-D
H0 = np.linspace(0.04, 2.00, 70)

# ── Core computation ───────────────────────────────────────────────────────────
def compute(kou_kw, b=BASE_B, T=BASE_T):
    """Return (p_surv, cond_mean, cond_var) arrays over H0."""
    params = KouParams(**kou_kw)
    market = MarketParams(b=b, S10=1.0, S20=1.0)
    ps_list, mu_list, var_list = [], [], []
    for h0 in H0:
        try:
            strat  = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
            cm_obj = ConditionalMoments(params=params, strategy=strat)
            ps  = float(np.clip(cm_obj.p_surv(), 0.0, 1.0))
            m1  = float(cm_obj.conditional_moment(1))
            m2  = float(cm_obj.conditional_moment(2))
            ps_list.append(ps)
            mu_list.append(m1)
            var_list.append(max(m2 - m1 ** 2, 0.0))
        except Exception:
            ps_list.append(np.nan)
            mu_list.append(np.nan)
            var_list.append(np.nan)
    return np.array(ps_list), np.array(mu_list), np.array(var_list)

# ── Frontier definitions ───────────────────────────────────────────────────────
FRONTIERS = [
    (r"Benchmark", BASE, "black", "-", 2.2, 1.00),
]

print("Computing frontiers …")
computed = []
for lbl, kw, color, ls, lw, alpha in FRONTIERS:
    print(f"  {lbl}")
    computed.append((lbl, color, ls, lw, alpha, compute(kw)))

# ── Annotation spec on the benchmark ──────────────────────────────────────────
# Axes: x=Var, y=p_surv, z=E[mean]  (z is the visual vertical)
# view_init(elev=24, azim=-52):
#   +x  → right on screen
#   +y  → left / into depth on screen
#   +z  → up on screen
# Labels staggered alternately left-right.

FLOOR_Z = 1.08  # floor of the z (mean) axis for drop lines (just below data min ~1.108)

# Labels sit on the floor (z = FLOOR_Z); offsets are in (dx_var, dy_ps) only.
# view_init(elev=24, azim=-52): +x → right on screen, +y → left/depth on screen.
ANNOTS = [
    (0.10,  ( 0.18, -0.06), "left",  "bottom"),
    (0.25,  (-0.20,  0.05), "right", "bottom"),
    (0.50,  ( 0.16,  0.00), "left",  "top"   ),
    (1.00,  (-0.14, -0.04), "right", "bottom"),
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig  = plt.figure(figsize=(5.6, 4.6))
ax   = fig.add_subplot(111, projection="3d")
ax.view_init(elev=24, azim=-52)

# ── Subtle pane background ─────────────────────────────────────────────────────
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor((0.95, 0.95, 0.95, 0.35))
    pane.set_edgecolor((0.80, 0.80, 0.80, 1.00))
ax.grid(True, lw=0.35, alpha=0.45)

# ── Draw frontiers  (x=var, y=ps, z=mu) ───────────────────────────────────────
bm_ps = bm_mu = bm_var = None
for lbl, color, ls, lw, alpha, (ps, mu, var) in computed:
    good = np.isfinite(var) & np.isfinite(mu) & np.isfinite(ps)
    ax.plot(var[good], ps[good], mu[good],
            color=color, ls=ls, lw=lw * 0.55, alpha=alpha, label=lbl, zorder=3)
    if lbl == "Benchmark":
        bm_ps, bm_mu, bm_var = ps[good], mu[good], var[good]

# ── Shadows on all three pane walls ───────────────────────────────────────────
# Floor  (z = FLOOR_Z): variance × p_surv plane
ax.plot(bm_var, bm_ps, np.full_like(bm_var, FLOOR_Z),
        color="0.60", lw=0.9, ls=":", alpha=0.55, zorder=1)
# Left wall  (x = 0): p_surv × E[mean] plane
ax.plot(np.zeros_like(bm_ps), bm_ps, bm_mu,
        color="0.60", lw=0.9, ls=":", alpha=0.55, zorder=1)
# Back wall  (y = 0): variance × E[mean] plane
ax.plot(bm_var, np.ones_like(bm_var), bm_mu,
        color="0.60", lw=0.9, ls=":", alpha=0.55, zorder=1)

# ── Annotated points ──────────────────────────────────────────────────────────
h0_grid = H0[np.isfinite(bm_var)]    # H0 values corresponding to good bm data

for h0_target, (dx, dy), ha, va in ANNOTS:
    idx = int(np.argmin(np.abs(h0_grid - h0_target)))
    xi  = bm_var[idx]   # variance
    yi  = bm_ps[idx]    # p_surv
    zi  = bm_mu[idx]    # conditional mean  (vertical)

    # Marker on curve
    ax.scatter(xi, yi, zi, s=28, color="black", zorder=6, depthshade=False)

    # Vertical drop line to floor
    ax.plot([xi, xi], [yi, yi], [FLOOR_Z, zi],
            color="0.50", lw=0.7, ls="--", alpha=0.60, zorder=2)

    # Floor marker
    ax.scatter(xi, yi, FLOOR_Z, s=14, color="0.55", marker="x", zorder=2)

    # Label on the floor plane
    txt = rf"$h_0\!=\!{h0_target}$"
    ax.text(xi + dx, yi + dy, FLOOR_Z, txt,
            fontsize=7.5, ha=ha, va=va, color="0.30",
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="white", ec="none", alpha=0.85),
            zorder=7)

# ── Direction indicator (increasing h0) ───────────────────────────────────────
i_tail = int(np.argmin(np.abs(h0_grid - 0.95)))
i_head = int(np.argmin(np.abs(h0_grid - 0.70)))
ax.quiver(bm_var[i_tail], bm_ps[i_tail], bm_mu[i_tail],
          bm_var[i_head] - bm_var[i_tail],
          bm_ps[i_head]  - bm_ps[i_tail],
          bm_mu[i_head]  - bm_mu[i_tail],
          color="0.40", arrow_length_ratio=0.45, lw=1.0, alpha=0.80)
ax.text(bm_var[i_tail] + 0.06, bm_ps[i_tail] + 0.04, bm_mu[i_tail] - 0.03,
        r"$h_0\!\uparrow$", fontsize=7.5, color="0.40", va="top")

# ── Axis labels and limits ────────────────────────────────────────────────────
ax.set_xlabel(r"$\mathrm{Var}(\Pi_T\!\mid\!\tau>T)$",
              labelpad=9, fontsize=8.5)
ax.set_ylabel(r"$p_{\mathrm{surv}}$",
              labelpad=9, fontsize=8.5)
ax.set_zlabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",
              labelpad=7, fontsize=8.5)

pad_x = (bm_var.max() - bm_var.min()) * 0.04
pad_z = (bm_mu.max()  - bm_mu.min())  * 0.04
ax.set_xlim(max(0.0, bm_var.min() - pad_x), bm_var.max() + pad_x)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(FLOOR_Z, bm_mu.max() + pad_z)
ax.tick_params(axis="x", labelsize=7, pad=1)
ax.tick_params(axis="y", labelsize=7, pad=1)
ax.tick_params(axis="z", labelsize=7, pad=1)


out = Path(__file__).parent.parent / "latex" / "fig_frontier.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
