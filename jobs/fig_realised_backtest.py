"""Figure for the realised backtest (reads results/realised_backtest.json).

Left: excess log growth over the unlevered book against initial leverage, 30-day
rolls, Aave liquidation mechanics, one line per pair. Right: realised drift nu_hat
against the realised Kelly break-even nu_star_hat; above the diagonal a little
leverage paid. Writes latex/fig_realised_backtest.pdf.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STYLE = {"cross-asset": ("#2a78d6", "o"), "same-underlying": ("#eb6834", "s"),
         "stablecoin": ("#1baf7a", "^")}
LS = [1.25, 1.5, 2.0, 2.5, 3.0]
YMIN = -1.6
OFFSETS = {("WBTC", "USDT"): (4, -9), ("cbBTC", "WETH"): (-22, -13)}


def main():
    J = json.load(open(ROOT / "results/realised_backtest.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    for p in J["pairs"]:
        col, mk = STYLE[p["group"]]
        ex = {r["L0"]: r["excess"] for r in p["results"]
              if r["mode"] == "aave" and r["cost_bps"] == 0 and r["horizon"] == 30}
        y = [0.0] + [max(ex[L], YMIN) for L in LS if L in ex]
        x = [1.0] + [L for L in LS if L in ex]
        a.plot(x, y, color=col, lw=0.9, alpha=0.75, marker=mk, ms=3)
    a.axhline(0, color="0.3", lw=0.8)
    a.set_ylim(YMIN - 0.05, 0.25)
    a.set_xlabel("initial leverage $L_0$ (30-day rolls)")
    a.set_ylabel("excess log growth over unlevered, per year")
    a.text(1.02, YMIN + 0.02, f"values below {YMIN} drawn at {YMIN}", fontsize=7, color="0.35")

    lim = (-0.5, 0.5)
    b.plot(lim, lim, color="0.3", lw=0.8)
    for p in J["pairs"]:
        col, mk = STYLE[p["group"]]
        rb = p["realised_breakeven"]
        b.scatter(rb["nu_star_hat"], rb["nu_hat"], color=col, marker=mk, s=26,
                  edgecolor="white", linewidth=0.6, zorder=3)
        if rb["margin"] > 0:
            b.annotate(f"{p['collateral']}/{p['debt']}", (rb["nu_star_hat"], rb["nu_hat"]),
                       xytext=OFFSETS.get((p["collateral"], p["debt"]), (4, 3)),
                       textcoords="offset points", fontsize=7, color="0.2")
    b.set_xlim(-0.12, 0.52)
    b.set_ylim(*lim)
    b.set_xlabel(r"realised Kelly break-even $\hat\nu^\star$ (variance minus covariance)")
    b.set_ylabel(r"realised spread drift $\hat\nu$")
    b.text(-0.1, 0.44, "above the diagonal:\nleverage paid", fontsize=7.5, color="0.35", va="top")
    for g, (col, mk) in STYLE.items():
        b.scatter([], [], color=col, marker=mk, s=26, label=g)
    b.legend(fontsize=7.5, frameon=False, loc="lower right")
    for ax in (a, b):
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "latex/fig_realised_backtest.pdf")
    fig.savefig(Path("/private/tmp/claude-501/-Users-tedig-Defi-interestRate/"
                     "24298d8c-dc1d-4d7e-98f9-1cbf1183f1e1/scratchpad/fig_rb.png"), dpi=130)
    print("saved latex/fig_realised_backtest.pdf")


if __name__ == "__main__":
    main()
