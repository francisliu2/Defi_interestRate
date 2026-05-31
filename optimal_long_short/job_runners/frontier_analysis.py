"""
Mean–variance–liquidation frontier: 3-D overview and four 2-D projections.
Parameters loaded from results/params_WBTC_WETH.json (jobs/calibrate_btc_eth.py).

Figure layout (3 rows × 2 columns):
  Row 0 (spanning both columns): 3-D frontier parametrised by H_0 = exp(h_0)
          x = Var(Pi_T | tau > T),  y = p_liq = 1 - p_surv,
          z = E[Pi_T | tau > T]  (vertical)
  Row 1 left  (a): E[Pi_T | tau>T]      vs  p_liq  — mean–liquidation tradeoff
  Row 1 right (b): Var(Pi_T | tau>T)    vs  p_liq  — risk–liquidation tradeoff
  Row 2 left  (c): E[Pi_T | tau>T]      vs  L_0    — mean–leverage tradeoff
  Row 2 right (d): Phi(H_0)             vs  H_0    — penalised objective value
                   Phi = E[..] - (alpha/2) Var[..] - delta * p_liq
                   (star marks optimal H_0^*)

Representative H_0 values are annotated on each 2-D panel so the reader can
link the tradeoff curves back to the health-buffer parameter.

Usage:  python jobs/frontier_analysis.py
        python jobs/frontier_analysis.py --H0-max 2.0 --delta-mu1 0.02 --delta-mu2 -0.01
Output: latex/fig_frontier.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D          # registers the '3d' projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from optimal_long_short.drift import drift_summary
from optimal_long_short.job_runners.common import (
    LATEX_DIR,
    load_calibrated_params,
)
from optimal_long_short.risk_report import h0_liquidation_moment_report

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

BASE_T = 1.0 / 12.0

# Objective coefficients for panel (d): Phi = E[..] - (alpha/2) Var[..] - delta * p_liq
ALPHA = 1.0
DELTA = 0.5

# Initial health-factor locations to annotate on 2-D panels.
ANNOT_H0 = [1.10, 1.20, 1.50]


def parse_args() -> argparse.Namespace:
    """Parse optional drift-view inputs for the frontier figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=LATEX_DIR / "fig_frontier.pdf")
    parser.add_argument("--T", type=float, default=BASE_T)
    parser.add_argument("--H0-max", type=float, default=2.0)
    parser.add_argument("--H0-count", type=int, default=180)
    parser.add_argument("--mu1", type=float, default=None, help="Absolute annual price-growth drift for WBTC.")
    parser.add_argument("--mu2", type=float, default=None, help="Absolute annual price-growth drift for WETH.")
    parser.add_argument("--delta-mu1", type=float, default=0.0, help="Additive annual price-growth drift view for WBTC.")
    parser.add_argument("--delta-mu2", type=float, default=0.0, help="Additive annual price-growth drift view for WETH.")
    return parser.parse_args()


def price_drift_view(args: argparse.Namespace) -> dict[str, float]:
    """Return a loader-compatible price-drift view dict."""
    view = {
        "delta_mu1": args.delta_mu1,
        "delta_mu2": args.delta_mu2,
    }
    if args.mu1 is not None:
        view["mu1"] = args.mu1
    if args.mu2 is not None:
        view["mu2"] = args.mu2
    return view


# ── Core computation ───────────────────────────────────────────────────────────
def compute(params, h0_grid, b, T, S10=1.0, S20=1.0):
    """Return (ps, mu, var, L0) arrays over h0.

    ps  = survival probability P(tau > T)
    mu  = E[Pi_T | tau > T]       (conditional mean)
    var = Var(Pi_T | tau > T)     (conditional variance)
    L0  = exp(h0) / (exp(h0) - b) (initial leverage, analytical)
    """
    rows = h0_liquidation_moment_report(
        params,
        h0_grid,
        b=b,
        T=T,
        S10=S10,
        S20=S20,
        max_moment_order=2,
    )
    ps = np.array([row["p_surv"] for row in rows])
    mu = np.array([row["conditional_mean"] for row in rows])
    var = np.array([row["conditional_variance"] for row in rows])
    L0 = np.array([row["initial_leverage"] for row in rows])
    return ps, mu, var, L0



def main() -> None:
    args = parse_args()
    params, constraint = load_calibrated_params(
        args.params or LATEX_DIR.parent / "results" / "params_WBTC_WETH.json",
        price_drift_view=price_drift_view(args),
    )
    base_b = constraint["b"]
    S10 = constraint.get("S10", 1.0)
    S20 = constraint.get("S20", 1.0)
    h0_min = constraint["h0_min"]   # AAVE minimum h0 at origination
    H0_min = np.exp(h0_min)
    if args.H0_max <= H0_min:
        raise ValueError(f"--H0-max must exceed AAVE feasible minimum {H0_min:.6f}.")
    health_grid = np.linspace(H0_min, args.H0_max, args.H0_count)
    h0_grid = np.log(health_grid)

    # ── Compute benchmark frontier ─────────────────────────────────────────────────
    print(
        f"Loaded WBTC/WETH calibrated params  "
        f"(b={base_b:.2f}, h0_min={h0_min:.4f}, H0_min={H0_min:.4f}, "
        f"H0_max={args.H0_max:.4f})"
    )
    print(f"Initial prices: WBTC S10={S10:.6f}, WETH S20={S20:.6f}")
    ds = drift_summary(params)
    print(
        "Drift convention: params.mu is annualized expected price growth; "
        "muX is used inside Psi/moments."
    )
    print(
        "Spread drifts: "
        f"mu1-mu2={ds['spread']['mu_price_growth_1_minus_2']:.4f}, "
        f"muX1-muX2={ds['spread']['muX_1_minus_2']:.4f}"
    )
    print("Computing benchmark frontier …")
    ps, mu, var, L0 = compute(params, h0_grid, base_b, args.T, S10=S10, S20=S20)
    pliq = 1.0 - ps
    phi  = mu - 0.5 * ALPHA * var - DELTA * pliq

    good = np.isfinite(var) & np.isfinite(mu) & np.isfinite(ps)
    h0g, psg, mug, varg, pliqg, L0g, phig = (
        h0_grid[good], ps[good], mu[good], var[good], pliq[good], L0[good], phi[good]
    )
    health_g = np.exp(h0g)

    # Panel (a) is a frontier view: show the increasing branch and omit the
    # dominated low-risk segment that causes the raw projection to dip first.
    min_mean_idx = int(np.nanargmin(mug))
    branch_mask = pliqg >= pliqg[min_mean_idx]
    branch_order = np.argsort(pliqg[branch_mask])
    pliq_branch = pliqg[branch_mask][branch_order]
    mean_branch = mug[branch_mask][branch_order]
    health_branch = health_g[branch_mask][branch_order]


    # ── Helper: annotate H_0 values on a 2-D axes ─────────────────────────────────
    def mark_health(ax, xdata, ydata, H0_vals=ANNOT_H0, alternating=True):
        """Place a filled dot and a text label at each specified H0 value.

        Labels alternate above/below the curve to reduce overlap.
        All coordinates are in data space; text is offset in points.
        """
        for k, H0_tgt in enumerate(H0_vals):
            if H0_tgt < health_g.min() or H0_tgt > health_g.max():
                continue
            idx = int(np.argmin(np.abs(health_g - H0_tgt)))
            x, y = xdata[idx], ydata[idx]
            H0_t = np.exp(h0g[idx])
            ax.scatter(x, y, s=18, color="black", zorder=5, clip_on=False)
            sign  = 1 if (not alternating or k % 2 == 0) else -1
            dy_pt = sign * 7          # offset in points
            va    = "bottom" if sign > 0 else "top"
            ax.annotate(
                rf"$H_0\!=\!{H0_t:.2f}$",
                xy=(x, y),
                xytext=(0, dy_pt),
                textcoords="offset points",
                fontsize=5.8, color="0.25", ha="center", va=va,
            )


    # ── Figure: 3 rows × 2 columns ────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.4, 10.4))
    gs  = GridSpec(3, 2,
                   height_ratios=[2.25, 1.0, 1.0],
                   hspace=0.50, wspace=0.42,
                   left=0.095, right=0.975, top=0.985, bottom=0.055)

    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    ax_a = fig.add_subplot(gs[1, 0])   # (a) mean  vs p_liq
    ax_b = fig.add_subplot(gs[1, 1])   # (b) var   vs p_liq
    ax_c = fig.add_subplot(gs[2, 0])   # (c) mean  vs L0
    ax_d = fig.add_subplot(gs[2, 1])   # (d) Phi   vs H0

    # ══════════════════════════════════════════════════════════════════════════════
    # Row 0: 3-D frontier
    # ══════════════════════════════════════════════════════════════════════════════
    ax3d.view_init(elev=24, azim=-62)
    ax3d.set_proj_type("ortho")
    ax3d.set_box_aspect((1.45, 1.0, 0.82), zoom=1.20)

    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor((1.00, 1.00, 1.00, 0.0))
        pane.set_edgecolor((0.78, 0.78, 0.78, 1.00))
    ax3d.grid(True, lw=0.35, alpha=0.28)

    pad_x  = (varg.max() - varg.min()) * 0.08
    pad_y  = (pliqg.max() - pliqg.min()) * 0.08
    pad_z  = (mug.max()  - mug.min())  * 0.08
    floor_x = max(0.0, varg.min() - pad_x)

    ax3d.set_xlim(floor_x, varg.max() + pad_x)
    ax3d.set_ylim(max(0.0, pliqg.min() - pad_y), pliqg.max() + pad_y)
    ax3d.set_zlim(mug.min() - pad_z, mug.max() + pad_z)
    ax3d.tick_params(axis="x", labelsize=7.2, pad=0)
    ax3d.tick_params(axis="y", labelsize=7.2, pad=0)
    ax3d.tick_params(axis="z", labelsize=7.2, pad=1)

    # Main curve (x=var, y=p_liq, z=mean), colored by initial health factor.
    points3 = np.column_stack([varg, pliqg, mug])
    segments3 = np.stack([points3[:-1], points3[1:]], axis=1)
    norm = Normalize(vmin=health_g.min(), vmax=health_g.max())
    lc3 = Line3DCollection(segments3, cmap="viridis", norm=norm, linewidth=2.4)
    lc3.set_array(health_g[:-1])
    ax3d.add_collection3d(lc3)

    # Orthogonal wall projections.  Keep them quiet so the frontier remains dominant.
    floor_z = mug.min() - 0.65 * pad_z
    wall_x = floor_x
    wall_y = pliqg.max() + 0.55 * pad_y
    ax3d.plot(varg, pliqg, np.full_like(varg, floor_z),
              color="0.64", lw=1.0, ls="-", alpha=0.62, zorder=1)
    ax3d.plot(np.full_like(pliqg, wall_x), pliqg, mug,
              color="0.64", lw=1.0, ls="-", alpha=0.62, zorder=1)
    ax3d.plot(varg, np.full_like(varg, wall_y), mug,
              color="0.64", lw=1.0, ls="-", alpha=0.62, zorder=1)

    projection_style = dict(color="0.70", lw=0.45, ls=":", alpha=0.38, zorder=0)
    for xi, yi, zi in zip(varg[::18], pliqg[::18], mug[::18]):
        ax3d.plot([xi, xi], [yi, yi], [floor_z, zi], **projection_style)
        ax3d.plot([wall_x, xi], [yi, yi], [zi, zi], **projection_style)
        ax3d.plot([xi, xi], [yi, wall_y], [zi, zi], **projection_style)

    # Annotated H_0 points on 3-D curve (dots + vertical drop lines)
    for H0_tgt in ANNOT_H0:
        if H0_tgt < health_g.min() or H0_tgt > health_g.max():
            continue
        idx = int(np.argmin(np.abs(health_g - H0_tgt)))
        xi, yi, zi = varg[idx], pliqg[idx], mug[idx]
        ax3d.scatter(xi, yi, zi, s=22, color="black", zorder=6, depthshade=False)
        ax3d.text(
            xi, yi, zi,
            rf"  $H_0={health_g[idx]:.2f}$",
            fontsize=6.2,
            color="0.25",
            zorder=7,
        )

    ax3d.set_xlabel(r"$\mathrm{Var}(\Pi_T\!\mid\!\tau>T)$", labelpad=7, fontsize=9.0)
    ax3d.set_ylabel(r"$p_{\mathrm{liq}}$",                  labelpad=7, fontsize=9.0)
    ax3d.set_zlabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",   labelpad=7, fontsize=9.0)
    ax3d.text2D(0.02, 0.95, r"3D frontier, coloured by $H_0$",
                transform=ax3d.transAxes, fontsize=8.0, color="0.35")

    # ══════════════════════════════════════════════════════════════════════════════
    # Row 1 left — (a) Conditional mean vs liquidation probability
    # ══════════════════════════════════════════════════════════════════════════════
    ax_a.plot(pliq_branch, mean_branch, color="black", lw=1.5)
    ax_a.set_xlabel(r"$p_{\mathrm{liq}}$",                 labelpad=2)
    ax_a.set_ylabel(r"$\mathrm{E}[\Pi_T\!\mid\!\tau>T]$",  labelpad=2)
    ax_a.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
    ax_a.set_axisbelow(True)
    ax_a.text(0.04, 0.96, r"$\leftarrow$ higher liquidation risk",
              transform=ax_a.transAxes, ha="left", va="top",
              fontsize=6.0, color="0.45", style="italic")
    for H0_tgt in [1.10, 1.20, 1.30]:
        if H0_tgt < health_branch.min() or H0_tgt > health_branch.max():
            continue
        idx = int(np.argmin(np.abs(health_branch - H0_tgt)))
        x, y = pliq_branch[idx], mean_branch[idx]
        ax_a.scatter(x, y, s=18, color="black", zorder=5, clip_on=False)
        ax_a.annotate(
            rf"$H_0\!=\!{health_branch[idx]:.2f}$",
            xy=(x, y),
            xytext=(0, 7),
            textcoords="offset points",
            fontsize=5.8,
            color="0.25",
            ha="center",
            va="bottom",
        )

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
    mark_health(ax_b, pliqg, varg)

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
    mark_health(ax_c, L0g, mug)

    # ══════════════════════════════════════════════════════════════════════════════
    # Row 2 right — (d) Objective Phi(H0) vs H0
    # ══════════════════════════════════════════════════════════════════════════════
    ax_d.plot(health_g, phig, color="black", lw=1.5)
    ax_d.set_xlabel(r"$H_0$  (initial health factor)", labelpad=2)
    ax_d.set_ylabel(
        rf"$\Phi(H_0)$  ($\alpha\!=\!{ALPHA:.0f},\ \delta\!=\!{DELTA:.1f}$)",
        labelpad=2)
    ax_d.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
    ax_d.set_axisbelow(True)

    # Mark optimal H_0
    i_opt   = int(np.nanargmax(phig))
    H0_opt  = health_g[i_opt]
    phi_opt = phig[i_opt]
    ax_d.axvline(H0_opt, color="0.55", lw=0.9, ls="--", alpha=0.80)
    ax_d.scatter(H0_opt, phi_opt, s=28, color="black", zorder=5)
    ax_d.annotate(
        rf"$H_0^\star\!=\!{H0_opt:.2f}$",
        xy=(H0_opt, phi_opt),
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

    # AAVE minimum H0 constraint
    ax_d.axvline(H0_min, color="0.40", lw=0.9, ls=":", alpha=0.80)
    ax_d.annotate(
        rf"$H_0^{{\min}}\!=\!{H0_min:.3f}$" + "\n(AAVE max LTV)",
        xy=(H0_min, phig.min() + 0.05 * (phig.max() - phig.min())),
        xytext=(5, 0), textcoords="offset points",
        fontsize=5.5, color="0.30", ha="left", va="bottom",
    )

    # ── Panel letters (a)–(d) on 2-D panels ───────────────────────────────────────
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], [ax_a, ax_b, ax_c, ax_d]):
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                va="top", fontsize=8, fontweight="bold", color="0.4")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
