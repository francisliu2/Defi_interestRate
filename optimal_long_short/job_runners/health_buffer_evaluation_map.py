"""
Generate the three-dimensional health-buffer evaluation map.

The curve is parametrised by H_0 = exp(h_0), with conditional payoff variance,
liquidation probability, and conditional payoff mean on the three axes.
Parameters are loaded from results/params_empirical_showcase.json.

Usage:  python jobs/health_buffer_evaluation_map.py
        python jobs/health_buffer_evaluation_map.py --H0-max 2.0 --delta-mu1 0.02 --delta-mu2 -0.01
Output: latex/fig_health_buffer_evaluation_map.pdf
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from optimal_long_short.drift import drift_summary
from optimal_long_short.job_runners.common import (
    DEFAULT_EMPIRICAL_PARAMS,
    LATEX_DIR,
    load_calibrated_params,
)
from optimal_long_short.risk_report import h0_liquidation_moment_report

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "mathtext.fontset":  "stix",
    "font.size":         11,
    "axes.linewidth":    0.8,
})

BASE_T = 1.0 / 12.0

# Initial health-factor locations to annotate on the 3-D curve.
ANNOT_H0 = [1.10, 1.20, 1.50]


def parse_args() -> argparse.Namespace:
    """Parse optional drift-view inputs for the evaluation-map figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=LATEX_DIR / "fig_health_buffer_evaluation_map.pdf")
    parser.add_argument("--T", type=float, default=BASE_T)
    parser.add_argument("--H0-max", type=float, default=2.0)
    parser.add_argument("--H0-count", type=int, default=180)
    parser.add_argument("--mu1", type=float, default=None, help="Absolute annual price-growth drift for the selected long asset.")
    parser.add_argument("--mu2", type=float, default=None, help="Absolute annual price-growth drift for the selected short asset.")
    parser.add_argument("--delta-mu1", type=float, default=0.0, help="Additive annual price-growth drift view for the selected long asset.")
    parser.add_argument("--delta-mu2", type=float, default=0.0, help="Additive annual price-growth drift view for the selected short asset.")
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
def compute(params, h0_grid, b, T, S10=1.0, S20=1.0, ltv_max=None):
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
        ltv_max=ltv_max,
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
        args.params or DEFAULT_EMPIRICAL_PARAMS,
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

    # ── Compute benchmark evaluation map ───────────────────────────────────────────
    print(
        f"Loaded {constraint.get('asset1', 'asset 1')}/{constraint.get('asset2', 'asset 2')} empirical params  "
        f"(b={base_b:.2f}, h0_min={h0_min:.4f}, H0_min={H0_min:.4f}, "
        f"H0_max={args.H0_max:.4f})"
    )
    print(
        f"Initial prices: {constraint.get('asset1', 'asset 1')} S10={S10:.6f}, "
        f"{constraint.get('asset2', 'asset 2')} S20={S20:.6f}"
    )
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
    print("Computing health-buffer evaluation map …")
    ps, mu, var, _ = compute(
        params,
        h0_grid,
        base_b,
        args.T,
        S10=S10,
        S20=S20,
        ltv_max=constraint["ltv_max"],
    )
    pliq = 1.0 - ps

    good = np.isfinite(var) & np.isfinite(mu) & np.isfinite(ps)
    h0g, mug, varg, pliqg = h0_grid[good], mu[good], var[good], pliq[good]
    health_g = np.exp(h0g)

    # ── Enlarged 3-D evaluation map ──────────────────────────────────────────────
    fig = plt.figure(figsize=(9.4, 6.6), dpi=200)
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.set_position([0.00, 0.01, 0.97, 0.97])

    ax3d.view_init(elev=24, azim=-62)
    ax3d.set_proj_type("ortho")
    ax3d.set_box_aspect((1.48, 1.0, 0.88), zoom=1.35)

    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor((1.00, 1.00, 1.00, 0.0))
        pane.set_edgecolor((0.76, 0.76, 0.76, 1.00))
    ax3d.grid(True, lw=0.45, alpha=0.30)

    pad_x = (varg.max() - varg.min()) * 0.08
    pad_y = (pliqg.max() - pliqg.min()) * 0.08
    pad_z = (mug.max() - mug.min()) * 0.08
    floor_x = max(0.0, varg.min() - pad_x)
    floor_z = mug.min() - 0.65 * pad_z
    wall_x = floor_x
    wall_y = pliqg.max() + 0.55 * pad_y

    ax3d.set_xlim(floor_x, varg.max() + pad_x)
    ax3d.set_ylim(max(0.0, pliqg.min() - pad_y), pliqg.max() + pad_y)
    ax3d.set_zlim(floor_z, mug.max() + pad_z)

    # Main curve, coloured by initial health factor.
    points3 = np.column_stack([varg, pliqg, mug])
    segments3 = np.stack([points3[:-1], points3[1:]], axis=1)
    norm = Normalize(vmin=health_g.min(), vmax=health_g.max())
    lc3 = Line3DCollection(segments3, cmap="viridis", norm=norm, linewidth=3.8)
    lc3.set_array(health_g[:-1])
    ax3d.add_collection3d(lc3)

    # Quiet orthogonal guides keep the spatial relation between the three axes legible.
    ax3d.plot(
        varg,
        pliqg,
        np.full_like(varg, floor_z),
        color="0.62",
        lw=1.2,
        alpha=0.62,
        zorder=1,
    )
    ax3d.plot(
        np.full_like(pliqg, wall_x),
        pliqg,
        mug,
        color="0.62",
        lw=1.2,
        alpha=0.62,
        zorder=1,
    )
    ax3d.plot(
        varg,
        np.full_like(varg, wall_y),
        mug,
        color="0.62",
        lw=1.2,
        alpha=0.62,
        zorder=1,
    )

    projection_style = dict(color="0.70", lw=0.55, ls=":", alpha=0.38, zorder=0)
    for xi, yi, zi in zip(varg[::18], pliqg[::18], mug[::18]):
        ax3d.plot([xi, xi], [yi, yi], [floor_z, zi], **projection_style)
        ax3d.plot([wall_x, xi], [yi, yi], [zi, zi], **projection_style)
        ax3d.plot([xi, xi], [yi, wall_y], [zi, zi], **projection_style)

    for H0_tgt in ANNOT_H0:
        if H0_tgt < health_g.min() or H0_tgt > health_g.max():
            continue
        idx = int(np.argmin(np.abs(health_g - H0_tgt)))
        xi, yi, zi = varg[idx], pliqg[idx], mug[idx]
        ax3d.scatter(xi, yi, zi, s=46, color="black", zorder=6, depthshade=False)
        if H0_tgt < 1.15:
            ax3d.text(
                xi - 0.003,
                yi - 0.020,
                zi,
                rf"$H_0={health_g[idx]:.2f}$",
                fontsize=9.5,
                color="0.20",
                ha="right",
                zorder=7,
            )
        else:
            ax3d.text(
                xi,
                yi,
                zi,
                rf"  $H_0={health_g[idx]:.2f}$",
                fontsize=9.5,
                color="0.20",
                zorder=7,
            )

    ax3d.set_xlabel(r"$\mathrm{Var}(\Pi_T\mid\tau>T)$", labelpad=16, fontsize=12.5)
    ax3d.set_ylabel(r"$p_{\mathrm{liq}}$", labelpad=16, fontsize=12.5)
    ax3d.set_zlabel("")
    ax3d.tick_params(axis="x", labelsize=9.5, pad=2)
    ax3d.tick_params(axis="y", labelsize=9.5, pad=2)
    ax3d.tick_params(axis="z", labelsize=9.5, pad=4)
    ax3d.text2D(
        0.03,
        0.94,
        r"Health-buffer evaluation map, coloured by $H_0$",
        transform=ax3d.transAxes,
        fontsize=12,
        color="0.30",
    )
    ax3d.text2D(
        1.01,
        0.52,
        r"$\mathrm{E}[\Pi_T\mid\tau>T]$",
        transform=ax3d.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=12.5,
    )

    cbar_ax = fig.add_axes([0.70, 0.86, 0.22, 0.025])
    cbar = fig.colorbar(lc3, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"$H_0$", fontsize=11.5)
    cbar.ax.tick_params(labelsize=9, pad=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
