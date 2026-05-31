"""
Parameter sensitivity — production figure for the paper.

Layout: 5 rows × 2 columns.
  Rows 1–4  (λ, ρ, σ, T)  left = p_surv,   right = conditional mean
  Row  5    (b)            left = leverage L0 vs H0,  right = conditional mean
             (p_surv is independent of b at fixed H0, so we plot L0 instead)

Visual encoding:
  • 4 qualitative colours + 4 line styles (colour-blind and greyscale safe)
  • Index 2 in every sweep = benchmark value → solid, heavier weight
  • Benchmark label carries "(bm)" to identify it in the legend

Annotations on p_surv panels:
  • Horizontal threshold lines at p_surv = 0.90 (10 % liq) and 0.75 (25 % liq)
  • Region labels "← high leverage" and "safe buffer →" at left / right of x-axis

Usage:  python jobs/sensitivity_analysis.py
Output: latex/fig_sensitivity.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from optimal_long_short.job_runners.common import (
    LATEX_DIR,
    frontier_moments,
    load_calibrated_params,
)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "serif",
    "mathtext.fontset":     "stix",
    "font.size":            8.5,
    "axes.labelsize":       8.5,
    "axes.titlesize":       8.5,
    "axes.linewidth":       0.65,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.major.width":    0.65,
    "ytick.major.width":    0.65,
    "legend.fontsize":      7.0,
    "legend.frameon":       False,
    "legend.handlelength":  2.0,
    "legend.handletextpad": 0.5,
    "legend.labelspacing":  0.3,
})

# ── Double encoding: colour + line style ───────────────────────────────────────
# Index 2 in every sweep variant list is the benchmark value.
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # red, blue, green, purple
LS     = ["--",      ":",       "-",       "-."]
LW     = [1.1,       1.1,       2.2,       1.1]

# Horizontal threshold lines on p_surv panels
P_SURV_THRESHOLDS = [
    (0.90, r"$p_{\mathrm{liq}}\!=\!10\%$"),
    (0.75, r"$p_{\mathrm{liq}}\!=\!25\%$"),
]

# ── Benchmark parameters (loaded from calibration JSON) ───────────────────────
_PARAMS_OBJ, _CONSTRAINT = load_calibrated_params()
BASE = dict(
    mu1=_PARAMS_OBJ.mu1,       sigma1=_PARAMS_OBJ.sigma1,
    lam1=_PARAMS_OBJ.lam1,     p1=_PARAMS_OBJ.p1,
    eta1_pos=_PARAMS_OBJ.eta1_pos, eta1_neg=_PARAMS_OBJ.eta1_neg,
    mu2=_PARAMS_OBJ.mu2,       sigma2=_PARAMS_OBJ.sigma2,
    lam2=_PARAMS_OBJ.lam2,     p2=_PARAMS_OBJ.p2,
    eta2_pos=_PARAMS_OBJ.eta2_pos, eta2_neg=_PARAMS_OBJ.eta2_neg,
    rho=_PARAMS_OBJ.rho,
)
BASE_B = _CONSTRAINT["b"]
BASE_S10 = _CONSTRAINT.get("S10", 1.0)
BASE_S20 = _CONSTRAINT.get("S20", 1.0)
BASE_T = 1.0 / 12.0
H0_LOG_MIN = _CONSTRAINT["h0_min"]
HEALTH_MIN = np.exp(H0_LOG_MIN)
# Start exactly at the AAVE feasible minimum H0 = b / max_ltv.
HEALTH_GRID = np.linspace(HEALTH_MIN, 2.00, 40)
H0_LOG_GRID = np.log(HEALTH_GRID)
HEALTH_TICKS = [1.1, 1.25, 1.5, 1.75, 2.0]


def _p(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


# ── Core computation ───────────────────────────────────────────────────────────
def compute(kou_kw, b, T):
    """Return (p_surv array, cond_mean array) over the log-h0 grid."""
    return frontier_moments(
        kou_kw,
        H0_LOG_GRID,
        b,
        T,
        include_variance=False,
        S10=BASE_S10,
        S20=BASE_S20,
    )


# ── Parameter sweeps ───────────────────────────────────────────────────────────
# Each variant list has exactly 4 entries; index 2 is always the benchmark value
# for that row (drawn solid + heavy via COLORS/LS/LW defined above).
# Variant tuple: (legend_label, kou_kw, b, T)

_s1 = BASE["sigma1"]
_s2 = BASE["sigma2"]
_l1 = BASE["lam1"]
_l2 = BASE["lam2"]
_bm_lam_label = rf"$\lambda\!\times\!1.0\ \rm(bm)$"

SWEEPS = [
    (
        r"Jump intensity $(\lambda_1,\lambda_2)$",
        # Sweep as multiples of calibrated benchmark
        [(rf"$\lambda\!\times\!0.1$",      _p(lam1=_l1*0.1, lam2=_l2*0.1), BASE_B, BASE_T),
         (rf"$\lambda\!\times\!0.5$",      _p(lam1=_l1*0.5, lam2=_l2*0.5), BASE_B, BASE_T),
         (_bm_lam_label,                   _p(lam1=_l1,      lam2=_l2),     BASE_B, BASE_T),
         (rf"$\lambda\!\times\!2.0$",      _p(lam1=_l1*2.0, lam2=_l2*2.0), BASE_B, BASE_T)],
    ),
    (
        r"Brownian correlation $\rho$",
        [(r"$\rho\!=\!0.50$",                   _p(rho=0.50), BASE_B, BASE_T),
         (r"$\rho\!=\!0.70$",                   _p(rho=0.70), BASE_B, BASE_T),
         (rf"$\rho\!=\!{BASE['rho']:.2f}\ \rm(bm)$", BASE,   BASE_B, BASE_T),
         (r"$\rho\!=\!0.99$",                   _p(rho=0.99), BASE_B, BASE_T)],
    ),
    (
        r"Volatility $(\sigma_1,\sigma_2)$",
        [(r"$\sigma\!\times\!0.3$", _p(sigma1=_s1*0.3, sigma2=_s2*0.3), BASE_B, BASE_T),
         (r"$\sigma\!\times\!0.6$", _p(sigma1=_s1*0.6, sigma2=_s2*0.6), BASE_B, BASE_T),
         (r"$\sigma\!\times\!1.0\ \rm(bm)$", BASE,                       BASE_B, BASE_T),
         (r"$\sigma\!\times\!2.0$", _p(sigma1=_s1*2.0, sigma2=_s2*2.0), BASE_B, BASE_T)],
    ),
    (
        r"Horizon $T$",
        [(r"$T\!=\!1\ \rm wk$",          BASE, BASE_B, 7.0 / 365.0),
         (r"$T\!=\!2\ \rm wk$",          BASE, BASE_B, 14.0 / 365.0),
         (r"$T\!=\!1/12\ \rm yr\ \rm(bm)$", BASE, BASE_B, BASE_T),
         (r"$T\!=\!3\ \rm mo$",          BASE, BASE_B, 90.0 / 365.0)],
    ),
    (
        r"Liquidation threshold $b$",
        [(r"$b\!=\!0.50$",          BASE, 0.50, BASE_T),
         (r"$b\!=\!0.65$",          BASE, 0.65, BASE_T),
         (rf"$b\!=\!{BASE_B:.2f}\ \rm(bm)$", BASE, BASE_B, BASE_T),
         (r"$b\!=\!0.90$",          BASE, 0.90, BASE_T)],
    ),
]

B_VALS = [0.50, 0.65, BASE_B, 0.90]   # for the leverage panel (row 5, left)



def main() -> None:
    # ── Run all computations ───────────────────────────────────────────────────────
    print("Computing sensitivity curves …")
    print(
        f"Benchmark horizon T={BASE_T:.6f} yr (1 month = 1/12); "
        f"H0 starts at AAVE feasible min {HEALTH_MIN:.6f} "
        f"(h0={H0_LOG_MIN:.6f}); "
        f"S10={BASE_S10:.6f}, S20={BASE_S20:.6f}"
    )
    DATA = []
    for title, variants in SWEEPS:
        print(f"  {title}")
        DATA.append((title, [(lbl, compute(kw, b, T)) for lbl, kw, b, T in variants]))

    # ── Figure ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(6.8, 9.8),
                             gridspec_kw={"hspace": 0.60, "wspace": 0.30})

    for row, (title, variants) in enumerate(DATA):
        ax_l = axes[row, 0]
        ax_r = axes[row, 1]

        for i, (lbl, (ps, mu)) in enumerate(variants):
            kw = dict(color=COLORS[i], ls=LS[i], lw=LW[i], label=lbl)
            if row < 4:
                ax_l.plot(HEALTH_GRID, ps, **kw)
            ax_r.plot(HEALTH_GRID, mu, **kw)

        # ── Row 5 (b): left panel shows leverage L0 instead of p_surv ───────────
        if row == 4:
            for i, b in enumerate(B_VALS):
                L0  = HEALTH_GRID / (HEALTH_GRID - b)
                lbl = SWEEPS[4][1][i][0]
                ax_l.plot(HEALTH_GRID, L0, color=COLORS[i], ls=LS[i], lw=LW[i], label=lbl)
            ax_l.set_ylabel(r"Initial leverage $L_0$", labelpad=2)
            ax_l.set_yscale("log")
            ax_l.set_ylim(0.9, 15)
            ax_l.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax_l.text(0.97, 0.97,
                      r"($p_{\mathrm{surv}}$ independent of $b$ at fixed $H_0$)",
                      transform=ax_l.transAxes, ha="right", va="top",
                      fontsize=6.0, color="0.45", style="italic")
            # Region labels for leverage panel
            ax_l.text(0.04, 0.95, r"high leverage $\uparrow$",
                      transform=ax_l.transAxes, ha="left", va="top",
                      fontsize=6.0, color="0.45", style="italic")
            ax_l.text(0.96, 0.35, r"$\downarrow$ low leverage",
                      transform=ax_l.transAxes, ha="right", va="bottom",
                      fontsize=6.0, color="0.45", style="italic")
        else:
            ax_l.set_ylabel(r"$p_{\mathrm{surv}}$", labelpad=2)
            ax_l.set_ylim(-0.02, 1.05)
            ax_l.set_yticks([0.0, 0.5, 1.0])
            ax_l.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
            ax_l.set_axisbelow(True)

            # Liquidation probability threshold lines
            for p_thresh, thresh_lbl in P_SURV_THRESHOLDS:
                ax_l.axhline(p_thresh, color="0.50", lw=0.8, ls=":", alpha=0.75, zorder=0)
                ax_l.text(HEALTH_GRID[-1] * 0.97, p_thresh + 0.03, thresh_lbl,
                          ha="right", va="bottom", fontsize=5.8, color="0.40",
                          style="italic")

            # Region labels: left = high leverage, right = safe buffer
            ax_l.text(0.04, 0.06, r"$\leftarrow$ high leverage",
                      transform=ax_l.transAxes, ha="left", va="bottom",
                      fontsize=6.0, color="0.45", style="italic")
            ax_l.text(0.96, 0.06, r"safe buffer $\rightarrow$",
                      transform=ax_l.transAxes, ha="right", va="bottom",
                      fontsize=6.0, color="0.45", style="italic")

        # ── Conditional mean panel: region labels ────────────────────────────────
        ax_r.text(0.04, 0.95, r"$\leftarrow$ high leverage",
                  transform=ax_r.transAxes, ha="left", va="top",
                  fontsize=6.0, color="0.45", style="italic")
        ax_r.text(0.96, 0.95, r"safe buffer $\rightarrow$",
                  transform=ax_r.transAxes, ha="right", va="top",
                  fontsize=6.0, color="0.45", style="italic")

        # ── Common axis formatting ───────────────────────────────────────────────
        for ax in (ax_l, ax_r):
            ax.set_xlim(HEALTH_GRID[0], HEALTH_GRID[-1])
            ax.set_xticks(HEALTH_TICKS)
            ax.set_xlabel(r"$H_0$  (initial health factor)", labelpad=2)

        ax_r.set_ylabel(r"$\mathrm{E}[\Pi_T\mid\tau>T]$", labelpad=2)

        # ── Row title on left panel ──────────────────────────────────────────────
        ax_l.set_title(title, loc="left", pad=4, fontweight="bold")

        # ── Legend on the left panel ─────────────────────────────────────────────
        ax_l.legend(loc="lower right" if row < 4 else "upper right",
                    ncol=1, borderpad=0.4)

    # ── Panel letters (a)–(j) ─────────────────────────────────────────────────────
    for k, ax in enumerate(axes.flat):
        ax.text(0.03, 0.98, f"({chr(97 + k)})", transform=ax.transAxes,
                va="top", fontsize=8, fontweight="bold", color="0.4")

    out = LATEX_DIR / "fig_sensitivity.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
