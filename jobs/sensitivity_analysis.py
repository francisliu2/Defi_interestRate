"""
Parameter sensitivity — production figure for the paper.

Layout: 5 rows × 2 columns.
  Rows 1–4  (λ, ρ, σ, T)  left = p_surv,   right = conditional mean
  Row  5    (b)            left = leverage L0 vs h0,  right = conditional mean
             (p_surv is independent of b at fixed h0, so we plot L0 instead)

Encoding: 4 qualitative colors + 4 line styles so the figure is legible
both in colour and in greyscale/print.  The benchmark curve (index 2) is
drawn solid with a heavier weight; the others use dashed variants.

Usage:  python jobs/sensitivity_analysis.py
Output: latex/fig_sensitivity.pdf
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
    "font.family":         "serif",
    "mathtext.fontset":    "stix",
    "font.size":           8.5,
    "axes.labelsize":      8.5,
    "axes.titlesize":      8.5,
    "axes.linewidth":      0.65,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.major.width":   0.65,
    "ytick.major.width":   0.65,
    "legend.fontsize":     7.5,
    "legend.frameon":      False,
    "legend.handlelength": 2.0,
    "legend.handletextpad":0.5,
    "legend.labelspacing": 0.3,
})

# ── Double encoding: colour + line style ───────────────────────────────────────
# Four qualitatively distinct colours (perceptually separated, not a gradient)
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # red, blue, green, purple
LS     = ["--",      ":",       "-",       "-."]        # index 2 = solid = benchmark
LW     = [1.1,       1.1,       2.2,       1.1]

# ── Benchmark parameters ───────────────────────────────────────────────────────
BASE = dict(
    mu1=0.10, sigma1=0.30, lam1=2.0, p1=0.50, eta1_pos=0.10, eta1_neg=0.08,
    mu2=0.08, sigma2=0.25, lam2=2.0, p2=0.50, eta2_pos=0.09, eta2_neg=0.12,
    rho=0.5,
)
BASE_B = 0.80
BASE_T = 1.0
H0 = np.linspace(0.05, 2.00, 40)

def _p(**kw):
    d = dict(BASE); d.update(kw); return d

# ── Core computation ───────────────────────────────────────────────────────────
def compute(kou_kw, b, T):
    """Return (p_surv array, cond_mean array) over H0."""
    params = KouParams(**kou_kw)
    market = MarketParams(b=b, S10=1.0, S20=1.0)
    ps, mu = [], []
    for h0 in H0:
        try:
            strat = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
            obj   = ConditionalMoments(params=params, strategy=strat)
            ps.append(float(np.clip(obj.p_surv(), 0.0, 1.0)))
            mu.append(float(obj.conditional_mean()))
        except Exception:
            ps.append(np.nan); mu.append(np.nan)
    return np.array(ps), np.array(mu)

# ── Parameter sweeps ───────────────────────────────────────────────────────────
# Wide ranges chosen so the four curves are visually separated.

SWEEPS = [
    # (row_title, [(legend_label, kou_kw, b, T), ...])
    (
        r"Jump intensity $\lambda_1\!=\!\lambda_2$",
        [(r"$\lambda=0.5$",  _p(lam1=0.5,  lam2=0.5),  BASE_B, BASE_T),
         (r"$\lambda=2.0$",  _p(lam1=2.0,  lam2=2.0),  BASE_B, BASE_T),
         (r"$\lambda=5.0$",  _p(lam1=5.0,  lam2=5.0),  BASE_B, BASE_T),
         (r"$\lambda=10.0$", _p(lam1=10.0, lam2=10.0), BASE_B, BASE_T)],
    ),
    (
        r"Brownian correlation $\rho$",
        [(r"$\rho=-0.80$", _p(rho=-0.80), BASE_B, BASE_T),
         (r"$\rho=0.00$",  _p(rho= 0.00), BASE_B, BASE_T),
         (r"$\rho=0.50$",  _p(rho= 0.50), BASE_B, BASE_T),
         (r"$\rho=0.90$",  _p(rho= 0.90), BASE_B, BASE_T)],
    ),
    (
        r"Volatility $(\sigma_1,\sigma_2)$",
        [(r"$\sigma\!\times\!0.3$", _p(sigma1=0.09,  sigma2=0.075), BASE_B, BASE_T),
         (r"$\sigma\!\times\!0.6$", _p(sigma1=0.18,  sigma2=0.150), BASE_B, BASE_T),
         (r"$\sigma\!\times\!1.0$", _p(sigma1=0.30,  sigma2=0.250), BASE_B, BASE_T),
         (r"$\sigma\!\times\!2.0$", _p(sigma1=0.60,  sigma2=0.500), BASE_B, BASE_T)],
    ),
    (
        r"Horizon $T$",
        [(r"$T=0.1$", BASE, BASE_B, 0.1),
         (r"$T=0.5$", BASE, BASE_B, 0.5),
         (r"$T=1.0$", BASE, BASE_B, 1.0),
         (r"$T=3.0$", BASE, BASE_B, 3.0)],
    ),
    # Row 5: b — p_surv is independent of b at fixed h0, so left panel shows L0.
    (
        r"Collateral factor $b$",
        [(r"$b=0.30$", BASE, 0.30, BASE_T),
         (r"$b=0.50$", BASE, 0.50, BASE_T),
         (r"$b=0.80$", BASE, 0.80, BASE_T),
         (r"$b=0.95$", BASE, 0.95, BASE_T)],
    ),
]

B_VALS = [0.30, 0.50, 0.80, 0.95]   # for the leverage panel (row 5 left)

# ── Run all computations ───────────────────────────────────────────────────────
print("Computing sensitivity curves …")
DATA = []
for title, variants in SWEEPS:
    print(f"  {title}")
    DATA.append((title, [(lbl, compute(kw, b, T)) for lbl, kw, b, T in variants]))

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 2, figsize=(6.8, 9.2),
                         gridspec_kw={"hspace": 0.52, "wspace": 0.30})

for row, (title, variants) in enumerate(DATA):
    ax_l = axes[row, 0]
    ax_r = axes[row, 1]

    for i, (lbl, (ps, mu)) in enumerate(variants):
        kw = dict(color=COLORS[i], ls=LS[i], lw=LW[i], label=lbl)
        # Row 4 (b): left panel shows leverage, not p_surv
        if row < 4:
            ax_l.plot(H0, ps, **kw)
        ax_r.plot(H0, mu, **kw)

    # ── Row 5 (b): replace p_surv with leverage L0 ──────────────────────────
    if row == 4:
        for i, b in enumerate(B_VALS):
            L0 = np.exp(H0) / (np.exp(H0) - b)
            lbl = SWEEPS[4][1][i][0]
            ax_l.plot(H0, L0, color=COLORS[i], ls=LS[i], lw=LW[i], label=lbl)
        ax_l.set_ylabel(r"Initial leverage $L_0$", labelpad=2)
        ax_l.set_yscale("log")
        ax_l.set_ylim(0.9, 15)
        ax_l.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        note = "($p_{\\mathrm{surv}}$ is independent of $b$ at fixed $h_0$)"
        ax_l.text(0.97, 0.96, note, transform=ax_l.transAxes,
                  ha="right", va="top", fontsize=6.5, color="0.45",
                  style="italic")
    else:
        ax_l.set_ylabel(r"$p_{\mathrm{surv}}$", labelpad=2)
        ax_l.set_ylim(-0.02, 1.05)
        ax_l.set_yticks([0.0, 0.5, 1.0])
        ax_l.yaxis.grid(True, lw=0.4, alpha=0.35, color="0.65")
        ax_l.set_axisbelow(True)

    # ── Common axis formatting ───────────────────────────────────────────────
    for ax in (ax_l, ax_r):
        ax.set_xlim(H0[0], H0[-1])
        ax.set_xticks([0.5, 1.0, 1.5, 2.0])
        ax.set_xlabel(r"$h_0$  (log health buffer)", labelpad=2)

    ax_r.set_ylabel(r"$\mathrm{E}[\Pi_T\mid\tau>T]$", labelpad=2)

    # ── Row title spanning both panels ───────────────────────────────────────
    ax_l.set_title(title, loc="left", pad=4, fontweight="bold")

    # ── Legend on p_surv / leverage panel ───────────────────────────────────
    ax_l.legend(loc="lower right" if row < 4 else "upper right",
                ncol=1, borderpad=0.4)

# ── Panel letters (a)–(j) ─────────────────────────────────────────────────────
for k, ax in enumerate(axes.flat):
    ax.text(0.03, 0.98, f"({chr(97+k)})", transform=ax.transAxes,
            va="top", fontsize=8, fontweight="bold", color="0.4")

out = Path(__file__).parent.parent / "latex" / "fig_sensitivity.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
