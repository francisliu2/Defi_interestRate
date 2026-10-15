"""Carry-adjusted frontier and depeg-scenario figures.

Demonstrates the carry integration of ``optimal_long_short.carry``: deterministic
AAVE interest accrual enters as a per-leg price-growth drift shift, so the
semi-analytic Kou / Laplace-resolvent machinery sizes an interest-rate carry
book with no change to the core model.

Two illustrative, data-grounded pairs (collateral = asset 1 / long,
debt = asset 2 / short):

* ``USDC/WETH`` carry  — uncorrelated legs, positive net carry.  Realized AAVE v3
  rates: USDC supply 4.98%, WETH borrow 2.86% (net +2.12%).  WETH ann. vol 0.70
  from ETH price history.
* ``stETH/ETH`` leveraged staking — near-perfectly correlated legs
  (rho = 0.99), positive net carry (stETH staking+supply 3.8%, ETH borrow 2.5%).

Both use *illustrative* Kou parameters (documented in ``PAIRS``); swap in a
calibrated fit via ``--params`` conventions if desired.

Outputs
-------
latex/fig_carry_frontier.pdf : p_liq(L0) and Phi(L0) with/without carry, both pairs.
latex/fig_carry_depeg.pdf    : liquidation probability vs collateral-depeg haircut.

Usage
-----
    python jobs/carry_frontier.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from optimal_long_short.carry import (
    CarryLeg,
    CarrySpec,
    apply_carry,
    depeg_scenario,
    objective_phi,
    size_for_target_pliq,
)
from optimal_long_short.utils.helpers import LATEX_DIR
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.model.risk_report import h0_liquidation_moment_report

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
})

# Objective coefficients (shared with the paper's numerical section).
ALPHA = 1.0
DELTA = 0.5
TARGET_PLIQ = 0.10


@dataclass(frozen=True)
class Pair:
    name: str
    params: KouParams
    spec: CarrySpec
    b: float
    T: float
    H0_max: float


def demo_pairs() -> list[Pair]:
    """Two illustrative, data-grounded carry pairs.

    Reserve thresholds (b) and realized supply/borrow APRs are taken from the
    Aave v3 Ethereum snapshot of 2026-08-04 fetched via ``get_aave_hourly.py``
    (reserve configs + trailing-mean hourly rates over 2026-03-17..2026-08-04).
    The WETH diffusion volatility (0.52) is the realized annualized value of the
    oracle-implied WETH price series over the same window.
    """
    usdc_weth = Pair(
        name="USDC/WETH carry",
        params=KouParams(
            mu1=0.0, sigma1=0.05, lam1=0.5, p1=0.10, eta1_pos=0.005, eta1_neg=0.010,
            mu2=0.0, sigma2=0.52, lam2=1.5, p2=0.50, eta2_pos=0.19, eta2_neg=0.16,
            rho=0.0,
        ),
        # realized Aave v3 rates (trailing mean): USDC supply 3.69%, WETH borrow 2.51%
        spec=CarrySpec(CarryLeg("USDC", 0.0369), CarryLeg("WETH", 0.0251)),
        b=0.78, T=0.25, H0_max=3.0,
    )
    steth_eth = Pair(
        name="stETH/ETH staking",
        # sigma here is an illustrative outright-leg vol; the near-unit rho makes
        # the stETH-ETH *spread* the binding risk (diffusion + independent depeg
        # jumps), which is what a leveraged-staking book actually faces.
        params=KouParams(
            mu1=0.0, sigma1=0.70, lam1=1.5, p1=0.50, eta1_pos=0.19, eta1_neg=0.16,
            mu2=0.0, sigma2=0.70, lam2=1.5, p2=0.50, eta2_pos=0.19, eta2_neg=0.16,
            rho=0.99,
        ),
        # stETH staking yield 3.0% (off-protocol) vs realized WETH borrow 2.51%
        spec=CarrySpec(CarryLeg("stETH", 0.030), CarryLeg("ETH", 0.0251)),
        b=0.81, T=1.0 / 12.0, H0_max=2.2,
    )
    return [usdc_weth, steth_eth]


def frontier(pair: Pair, *, carry: bool, n: int = 160) -> dict[str, np.ndarray]:
    """p_liq, conditional mean, Phi, and leverage over an h0 grid."""
    params = apply_carry(pair.params, pair.spec) if carry else pair.params
    H0_min = pair.b + 1e-3 if pair.b >= 1.0 else max(1.0 + 1e-3, pair.b + 1e-3)
    health = np.linspace(H0_min, pair.H0_max, n)
    h0_grid = np.log(health)
    rows = h0_liquidation_moment_report(
        params, h0_grid, b=pair.b, T=pair.T, max_moment_order=2
    )
    for r in rows:
        r["phi"] = objective_phi(r, alpha=ALPHA, delta=DELTA)
    return {
        "L0": np.array([r["initial_leverage"] for r in rows]),
        "p_liq": np.array([r["p_liq"] for r in rows]),
        "mean": np.array([r["conditional_mean"] for r in rows]),
        "phi": np.array([r["phi"] for r in rows]),
    }


def build_metrics(pairs: list[Pair]) -> dict:
    """Numbers embedded in the paper's carry section."""
    out: dict = {}
    for pair in pairs:
        base = size_for_target_pliq(pair.params, CarrySpec(CarryLeg("-", 0.0), CarryLeg("-", 0.0)),
                                    TARGET_PLIQ, b=pair.b, T=pair.T)
        carried = size_for_target_pliq(pair.params, pair.spec, TARGET_PLIQ, b=pair.b, T=pair.T)
        dp = depeg_scenario(pair.params, pair.spec, carried["h0"],
                            [0.0, 0.02, 0.05, 0.10], b=pair.b, T=pair.T)
        out[pair.name] = {
            "net_carry": pair.spec.net_carry,
            "b": pair.b, "T": pair.T,
            "no_carry": {k: base[k] for k in ("h0", "H0", "initial_leverage", "conditional_mean", "conditional_sharpe")},
            "carry": {k: carried[k] for k in ("h0", "H0", "initial_leverage", "conditional_mean", "conditional_sharpe")},
            "depeg": [(r["depeg_fraction"], r["p_liq"]) for r in dp],
        }
    return out


def plot_frontier(pairs: list[Pair]) -> None:
    fig, axes = plt.subplots(2, len(pairs), figsize=(7.2, 5.4))
    for j, pair in enumerate(pairs):
        f0 = frontier(pair, carry=False)
        f1 = frontier(pair, carry=True)
        pt = size_for_target_pliq(pair.params, pair.spec, TARGET_PLIQ, b=pair.b, T=pair.T)

        ax = axes[0, j]
        ax.plot(f0["L0"], f0["p_liq"], color="0.6", lw=1.2, label="no carry")
        ax.plot(f1["L0"], f1["p_liq"], color="C0", lw=1.4, label=f"+carry ({pair.spec.net_carry:+.2%})")
        ax.axhline(TARGET_PLIQ, color="C3", ls=":", lw=0.9)
        ax.scatter([pt["initial_leverage"]], [pt["p_liq"]], s=22, color="C3", zorder=5)
        ax.annotate(rf"$L_0^\dagger={pt['initial_leverage']:.2f}$",
                    xy=(pt["initial_leverage"], pt["p_liq"]),
                    xytext=(4, 6), textcoords="offset points", fontsize=6.5, color="C3")
        ax.set_title(pair.name, fontsize=8.5)
        ax.set_ylabel(r"$p_{\mathrm{liq}}$")
        ax.set_xlabel(r"initial leverage $L_0$")
        if j == 0:
            ax.legend(loc="upper left")

        ax = axes[1, j]
        ax.plot(f0["L0"], f0["phi"], color="0.6", lw=1.2)
        ax.plot(f1["L0"], f1["phi"], color="C0", lw=1.4)
        ax.set_ylabel(r"$\Phi(L_0)$")
        ax.set_xlabel(r"initial leverage $L_0$")
    fig.tight_layout()
    out = LATEX_DIR / "fig_carry_frontier.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_depeg(pairs: list[Pair]) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    haircuts = np.linspace(0.0, 0.12, 25)
    for pair, color in zip(pairs, ["C0", "C3"]):
        pt = size_for_target_pliq(pair.params, pair.spec, TARGET_PLIQ, b=pair.b, T=pair.T)
        dp = depeg_scenario(pair.params, pair.spec, pt["h0"], haircuts, b=pair.b, T=pair.T)
        ax.plot([r["depeg_fraction"] for r in dp], [r["p_liq"] for r in dp],
                color=color, lw=1.5, label=f"{pair.name} ($L_0={pt['initial_leverage']:.2f}$)")
    ax.axhline(TARGET_PLIQ, color="0.5", ls=":", lw=0.9)
    ax.set_xlabel("collateral depeg haircut")
    ax.set_ylabel(r"$p_{\mathrm{liq}}$ (90d)")
    ax.set_title("Depeg sensitivity of the target-sized book", fontsize=8.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = LATEX_DIR / "fig_carry_depeg.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    matplotlib.use("Agg")  # headless figure generation for the CLI entry point
    pairs = demo_pairs()
    metrics = build_metrics(pairs)
    for name, m in metrics.items():
        print(f"\n{name}  (net carry {m['net_carry']:+.4f}, b={m['b']}, T={m['T']:.4f})")
        print(f"  sized to p_liq={TARGET_PLIQ:.0%}: "
              f"no-carry L0={m['no_carry']['initial_leverage']:.3f} "
              f"mean={m['no_carry']['conditional_mean']:.4f}  |  "
              f"+carry L0={m['carry']['initial_leverage']:.3f} "
              f"mean={m['carry']['conditional_mean']:.4f}")
        print("  depeg:", "  ".join(f"{d:.0%}->{p:.3f}" for d, p in m["depeg"]))
    plot_frontier(pairs)
    plot_depeg(pairs)


if __name__ == "__main__":
    main()
