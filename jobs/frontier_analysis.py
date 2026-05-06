"""
Mean–variance–liquidation frontier: 3-D overview and four 2-D projections.

Figure layout (3 rows × 2 columns):
  Row 0 (spanning both columns): 3-D frontier parametrised by h_0
          x = Var(Pi_T | tau > T),  y = p_liq = 1 - p_surv,
          z = E[Pi_T | tau > T]  (vertical)
  Row 1 left  (a): E[Pi_T | tau>T]      vs  p_liq  — mean–liquidation tradeoff
  Row 1 right (b): Var(Pi_T | tau>T)    vs  p_liq  — risk–liquidation tradeoff
  Row 2 left  (c): E[Pi_T | tau>T]      vs  L_0    — mean–leverage tradeoff
  Row 2 right (d): Phi(h_0)             vs  h_0    — penalised objective value
                   Phi = E[..] - (alpha/2) Var[..] - delta * p_liq
                   (star marks optimal h_0^*)

Representative h_0 values are annotated on each 2-D panel so the reader can
link the tradeoff curves back to the health-buffer parameter.

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
from matplotlib.gridspec import GridSpec
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
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.frameon":    False,
    "legend.fontsize":   8,
})

# ── Benchmark parameters ───────────────────────────────────────────────────────
BASE = dict(
    mu1=0.10, sigma1=0.30, lam1=2.0, p1=0.50, eta1_pos=0.10, eta1_neg=0.08,
    mu2=0.08, sigma2=0.25, lam2=2.0, p2=0.50, eta2_pos=0.09, eta2_neg=0.12,
    rho=0.5,
)
BASE_B = 0.80
BASE_T = 1.0

# Objective coefficients for panel (d): Phi = E[..] - (alpha/2) Var[..] - delta * p_liq
ALPHA = 1.0
DELTA = 0.5

# h_0 grid: fine for smooth 3-D curve
H0 = np.linspace(0.04, 2.00, 70)

# h_0 values to annotate on 2-D panels
ANNOT_H0 = [0.25, 0.50, 1.00, 1.50]


# ── Core computation ───────────────────────────────────────────────────────────
def compute(kou_kw, b=BASE_B, T=BASE_T):
    """Return (ps, mu, var, L0) arrays over H0.

    ps  = survival probability P(tau > T)
    mu  = E[Pi_T | tau > T]       (conditional mean)
    var = Var(Pi_T | tau > T)     (conditional variance)
    L0  = exp(h0) / (exp(h0) - b) (initial leverage, analytical)
    """
    params = KouParams(**kou_kw)
    market = MarketParams(b=b, S10=1.0, S20=1.0)
    ps_list, mu_list, var_list = [], [], []
    for h0 in H0:
        try:
            strat  = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
            cm_obj = ConditionalMoments(params=params, strategy=strat)
            ps_list.append(float(np.clip(cm_obj.p_surv(), 0.0, 1.0)))
            mu_list.append(float(cm_obj.conditional_mean()))
            var_list.append(max(float(cm_obj.conditional_variance()), 0.0))
        except Exception:
            ps_list.append(np.nan)
            mu_list.append(np.nan)
            var_list.append(np.nan)
    ps  = np.array(ps_list)
    mu  = np.array(mu_list)
    var = np.array(var_list)
    L0  = np.exp(H0) / (np.exp(H0) - b)
    return ps, mu, var, L0


# ── Compute benchmark frontier ─────────────────────────────────────────────────
print("Computing benchmark frontier …")
ps, mu, var, L0 = compute(BASE)
pliq = 1.0 - ps
phi  = mu - 0.5 * ALPHA * var - DELTA * pliq

good = np.isfinite(var) & np.isfinite(mu) & np.isfinite(ps)
H0g, psg, mug, varg, pliqg, L0g, phig = (
    H0[good], ps[good], mu[good], var[good], pliq[good], L0[good], phi[good]
)


# ── Helper: annotate h_0 values on a 2-D axes ─────────────────────────────────
def mark_h0(ax, xdata, ydata, h0_vals=ANNOT_H0, alternating=True):
    """Place a filled dot and a text label at each specified h_0 value.

    Labels alternate above/below the curve to reduce overlap.
    All coordinates are in data space; text is offset in points.
    """
    for k, h0_t in enumerate(h0_vals):
        idx = int(np.argmin(np.abs(H0g - h0_t)))
        x, y = xdata[idx], ydata[idx]
        ax.scatter(x, y, s=18, color="black", zorder=5, clip_on=False)
        sign  = 1 if (not alternating or k % 2 == 0) else -1
        dy_pt = sign * 7          # offset in points
        va    = "bottom" if sign > 0 else "top"
        ax.annotate(
            rf"$h_0\!=\!{h0_t}$",
            xy=(x, y),
            xytext=(0, dy_pt),
            textcoords="offset points",
            fontsize=5.8, color="0.25", ha="center", va=va,
        )


# ── Figure: 3 rows × 2 columns ────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 9.5))
gs  = GridSpec(3, 2,
               height_ratios=[1.45, 1.0, 1.0],
               hspace=0.55, wspace=0.42,
               left=0.10, right=0.97, top=0.97, bottom=0.06)

ax3d = fig.add_subplot(gs[0, :], projection="3d")
ax_a = fig.add_subplot(gs[1, 0])   # (a) mean  vs p_liq
ax_b = fig.add_subplot(gs[1, 1])   # (b) var   vs p_liq
ax_c = fig.add_subplot(gs[2, 0])   # (c) mean  vs L0
ax_d = fig.add_subplot(gs[2, 1])   # (d) Phi   vs h0

# ══════════════════════════════════════════════════════════════════════════════
# Row 0: 3-D frontier
# ══════════════════════════════════════════════════════════════════════════════
ax3d.view_init(elev=10, azim=-30)

for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.set_facecolor((0.95, 0.95, 0.95, 0.35))
    pane.set_edgecolor((0.80, 0.80, 0.80, 1.00))
ax3d.grid(True, lw=0.35, alpha=0.45)

pad_x  = (varg.max() - varg.min()) * 0.04
pad_z  = (mug.max()  - mug.min())  * 0.04
FLOOR_Z = max(1.0, mug.min() - pad_z)
FLOOR_X = max(0.0, varg.min() - pad_x)

ax3d.set_xlim(FLOOR_X, varg.max() + pad_x)
ax3d.set_ylim(0.0, 1.0)
ax3d.set_zlim(FLOOR_Z, mug.max() + pad_z)
ax3d.tick_params(axis="x", labelsize=7, pad=1)
ax3d.tick_params(axis="y", labelsize=7, pad=1)
ax3d.tick_params(axis="z", labelsize=7, pad=1)

# Main curve (x=var, y=p_liq, z=mean)
ax3d.plot(varg, pliqg, mug, color="black", lw=1.2, alpha=1.0, zorder=3)

# Shadow projections on the three coordinate planes
ax3d.plot(varg, pliqg, np.full_like(varg, FLOOR_Z),         # floor
          color="0.60", lw=0.9, ls="-", alpha=0.85, zorder=1)
ax3d.plot(np.full_like(pliqg, FLOOR_X), pliqg, mug,         # left wall
          color="0.60", lw=0.9, ls="-", alpha=0.85, zorder=1)
ax3d.plot(varg, np.ones_like(varg), mug,                    # back wall (p_liq=1)
          color="0.60", lw=0.9, ls="-", alpha=0.85, zorder=1)

# Annotated h_0 points on 3-D curve (dots + vertical drop lines)
for h0_tgt in ANNOT_H0:
    idx = int(np.argmin(np.abs(H0g - h0_tgt)))
    xi, yi, zi = varg[idx], pliqg[idx], mug[idx]
    ax3d.scatter(xi, yi, zi, s=10, color="black", zorder=6, depthshade=False)
    ax3d.plot([xi, xi], [yi, yi], [FLOOR_Z, zi],
              color="0.50", lw=0.5, ls="--", alpha=0.60, zorder=2)

ax3d.set_xlabel(r"$\mathrm{Var}(\Pi_T\!\mid\!\tau>T)$", labelpad=9, fontsize=8.5)
ax3d.set_ylabel(r"$p_{\mathrm{liq}}$",                  labelpad=9, fontsize=8.5)
ax3d.set_zlabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",   labelpad=7, fontsize=8.5)

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 left — (a) Conditional mean vs liquidation probability
# ══════════════════════════════════════════════════════════════════════════════
ax_a.plot(pliqg, mug, color="black", lw=1.5)
ax_a.set_xlabel(r"$p_{\mathrm{liq}}$",                 labelpad=2)
ax_a.set_ylabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",  labelpad=2)
ax_a.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
ax_a.set_axisbelow(True)
ax_a.text(0.04, 0.96, r"$\leftarrow$ higher liquidation risk",
          transform=ax_a.transAxes, ha="left", va="top",
          fontsize=6.0, color="0.45", style="italic")
mark_h0(ax_a, pliqg, mug)

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 right — (b) Conditional variance vs liquidation probability
# ══════════════════════════════════════════════════════════════════════════════
ax_b.plot(pliqg, varg, color="black", lw=1.5)
ax_b.set_xlabel(r"$p_{\mathrm{liq}}$",                    labelpad=2)
ax_b.set_ylabel(r"$\mathrm{Var}(\Pi_T\!\mid\!\tau>T)$",   labelpad=2)
ax_b.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
ax_b.set_axisbelow(True)
ax_b.text(0.04, 0.96, r"$\leftarrow$ higher liquidation risk",
          transform=ax_b.transAxes, ha="left", va="top",
          fontsize=6.0, color="0.45", style="italic")
mark_h0(ax_b, pliqg, varg)

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 left — (c) Conditional mean vs initial leverage
# ══════════════════════════════════════════════════════════════════════════════
ax_c.plot(L0g, mug, color="black", lw=1.5)
ax_c.set_xlabel(r"Initial leverage $L_0$",              labelpad=2)
ax_c.set_ylabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",  labelpad=2)
ax_c.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
ax_c.set_axisbelow(True)
ax_c.text(0.96, 0.96, r"higher leverage $\rightarrow$",
          transform=ax_c.transAxes, ha="right", va="top",
          fontsize=6.0, color="0.45", style="italic")
mark_h0(ax_c, L0g, mug)

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 right — (d) Objective Phi(h0) vs h0
# ══════════════════════════════════════════════════════════════════════════════
ax_d.plot(H0g, phig, color="black", lw=1.5)
ax_d.set_xlabel(r"$h_0$  (log health buffer)", labelpad=2)
ax_d.set_ylabel(
    rf"$\Phi(h_0)$  ($\alpha\!=\!{ALPHA:.0f},\ \delta\!=\!{DELTA:.1f}$)",
    labelpad=2)
ax_d.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
ax_d.set_axisbelow(True)

# Mark optimal h_0
i_opt   = int(np.nanargmax(phig))
h0_opt  = H0g[i_opt]
phi_opt = phig[i_opt]
ax_d.axvline(h0_opt, color="0.55", lw=0.9, ls="--", alpha=0.80)
ax_d.scatter(h0_opt, phi_opt, s=28, color="black", zorder=5)
ax_d.annotate(
    rf"$h_0^\star\!=\!{h0_opt:.2f}$",
    xy=(h0_opt, phi_opt),
    xytext=(8, -4),
    textcoords="offset points",
    fontsize=6.5, color="0.20", ha="left", va="top",
)
ax_d.text(0.04, 0.06, r"$\leftarrow$ high leverage",
          transform=ax_d.transAxes, ha="left", va="bottom",
          fontsize=6.0, color="0.45", style="italic")
ax_d.text(0.96, 0.06, r"safe buffer $\rightarrow$",
          transform=ax_d.transAxes, ha="right", va="bottom",
          fontsize=6.0, color="0.45", style="italic")

# ── Panel letters (a)–(d) on 2-D panels ───────────────────────────────────────
for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], [ax_a, ax_b, ax_c, ax_d]):
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            va="top", fontsize=8, fontweight="bold", color="0.4")

out = Path(__file__).parent.parent / "latex" / "fig_frontier.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
