"""Generate a 3D-only mean-variance-liquidation frontier figure."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from optimal_long_short.job_runners.common import LATEX_DIR, load_calibrated_params
from optimal_long_short.job_runners.frontier_analysis import BASE_T, compute


OUT = LATEX_DIR / "fig_frontier_3d.pdf"


def main() -> None:
    params, constraint = load_calibrated_params()
    b = constraint["b"]
    h0_min = constraint["h0_min"]
    H0_min = float(np.exp(h0_min))
    H0_max = 2.0
    health = np.linspace(H0_min, H0_max, 260)
    h0_grid = np.log(health)

    ps, mu, var, _ = compute(
        params,
        h0_grid,
        b,
        BASE_T,
        S10=constraint.get("S10", 1.0),
        S20=constraint.get("S20", 1.0),
    )
    pliq = 1.0 - ps

    good = np.isfinite(var) & np.isfinite(mu) & np.isfinite(pliq)
    health, var, pliq, mu = health[good], var[good], pliq[good], mu[good]

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
    })
    fig = plt.figure(figsize=(12.8, 7.2), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00)

    ax.view_init(elev=24, azim=-62)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.55, 1.0, 0.90), zoom=1.64)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((0.74, 0.74, 0.74, 1.0))
    ax.grid(True, lw=0.45, alpha=0.30)

    pad_x = (var.max() - var.min()) * 0.09
    pad_y = (pliq.max() - pliq.min()) * 0.09
    pad_z = (mu.max() - mu.min()) * 0.09
    floor_x = max(0.0, var.min() - pad_x)
    floor_z = mu.min() - 0.60 * pad_z
    wall_x = floor_x
    wall_y = pliq.max() + 0.55 * pad_y

    ax.set_xlim(floor_x, var.max() + pad_x)
    ax.set_ylim(max(0.0, pliq.min() - pad_y), pliq.max() + pad_y)
    ax.set_zlim(floor_z, mu.max() + pad_z)

    points = np.column_stack([var, pliq, mu])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=health.min(), vmax=health.max())
    line = Line3DCollection(segments, cmap="viridis", norm=norm, linewidth=4.2)
    line.set_array(health[:-1])
    ax.add_collection3d(line)

    ax.plot(var, pliq, np.full_like(var, floor_z), color="0.58", lw=1.6, alpha=0.62)
    ax.plot(np.full_like(pliq, wall_x), pliq, mu, color="0.58", lw=1.6, alpha=0.62)
    ax.plot(var, np.full_like(var, wall_y), mu, color="0.58", lw=1.6, alpha=0.62)

    for H0_tgt in [1.10, 1.20, 1.50]:
        if H0_tgt < health.min() or H0_tgt > health.max():
            continue
        idx = int(np.argmin(np.abs(health - H0_tgt)))
        xi, yi, zi = var[idx], pliq[idx], mu[idx]
        ax.scatter(xi, yi, zi, s=54, color="black", depthshade=False, zorder=6)
        if H0_tgt < 1.15:
            ax.text(xi - 0.0030, yi - 0.020, zi, rf"$H_0={health[idx]:.2f}$",
                    fontsize=13, color="0.20", ha="right")
        else:
            ax.text(xi, yi, zi, rf"  $H_0={health[idx]:.2f}$", fontsize=13, color="0.20")

    ax.set_xlabel(r"$\mathrm{Var}(\Pi_T\mid\tau>T)$", labelpad=22, fontsize=17)
    ax.set_ylabel(r"$p_{\mathrm{liq}}$", labelpad=22, fontsize=17)
    ax.set_zlabel(r"$\mathrm{E}[\Pi_T\mid\tau>T]$", labelpad=22, fontsize=17)
    ax.tick_params(axis="x", labelsize=12, pad=3)
    ax.tick_params(axis="y", labelsize=12, pad=3)
    ax.tick_params(axis="z", labelsize=12, pad=6)
    ax.text2D(0.06, 0.92, r"Efficient frontier, coloured by $H_0$",
              transform=ax.transAxes, fontsize=17, color="0.30")

    cbar = fig.colorbar(line, ax=ax, shrink=0.58, pad=0.02, fraction=0.035)
    cbar.set_label(r"$H_0$", fontsize=15)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
