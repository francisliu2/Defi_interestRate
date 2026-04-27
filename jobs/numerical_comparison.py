"""
Reproduce the numerical results in Section 7 of the paper.

Computes conditional mean, variance, skewness, and excess kurtosis of Pi_T
for both the Laplace--resolvent method and Monte Carlo simulation, across a
range of initial log-health values h0. Also verifies the analytical
no-shorting limit at h0 = 100.

Usage:
    python jobs/numerical_comparison.py

Output:
    Prints two LaTeX-ready tables and the no-shorting validation block.
    Timing figures use wall-clock seconds on the host machine.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from optimal_long_short.model_params import KouParams
from optimal_long_short.market_params import MarketParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.monte_carlo import MonteCarlo
from optimal_long_short.kou_model import BivariateKouModel

# ── Parameters ────────────────────────────────────────────────────────────────

PARAMS = KouParams(
    mu1=0.10, sigma1=0.30, lam1=2.0, p1=0.50, eta1_pos=0.10, eta1_neg=0.08,
    mu2=0.08, sigma2=0.25, lam2=2.0, p2=0.50, eta2_pos=0.09, eta2_neg=0.12,
    rho=0.5,
)
MARKET = MarketParams(b=0.80, S10=1.0, S20=1.0)
T      = 1.0

H0_GRID    = [0.01, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
H0_NOSHORT = 100.0          # proxy for h0 -> infinity

MC_PATHS = 100_000
MC_STEPS = 252
MC_SEED  = 42

# ── Helper ────────────────────────────────────────────────────────────────────

def standardised(r1, r2, r3, r4):
    """
    Convert the first four raw conditional moments into
    (mean, variance, skewness, excess kurtosis).
    """
    mean = r1
    var  = r2 - r1 ** 2
    std  = var ** 0.5
    skew = (r3 - 3 * r1 * r2 + 2 * r1 ** 3) / std ** 3
    kurt = (r4 - 4 * r1 * r3 + 6 * r1 ** 2 * r2 - 3 * r1 ** 4) / var ** 2 - 3
    return mean, var, skew, kurt


def laplace_row(h0):
    """Return (p_surv, mean, var, skew, kurt, elapsed) via Laplace inversion."""
    strategy = UnitExposureLongShortStrategy(h0=h0, market=MARKET, T=T)
    cm = ConditionalMoments(params=PARAMS, strategy=strategy)
    t0 = time.perf_counter()
    p_surv = cm.p_surv()
    raw    = [cm.conditional_moment(k) for k in [1, 2, 3, 4]]
    elapsed = time.perf_counter() - t0
    return (p_surv, *standardised(*raw), elapsed)


def mc_row(h0):
    """Return (p_surv, mean, var, skew, kurt, elapsed) via Monte Carlo."""
    strategy = UnitExposureLongShortStrategy(h0=h0, market=MARKET, T=T)
    mc = MonteCarlo(
        params=PARAMS,
        strategy=strategy,
        n_paths=MC_PATHS,
        n_steps=MC_STEPS,
        seed=MC_SEED,
    )
    t0 = time.perf_counter()
    res = mc.run()
    elapsed = time.perf_counter() - t0
    sp = res.payoff[res.survived]
    r1 = sp.mean()
    r2 = (sp ** 2).mean()
    r3 = (sp ** 3).mean()
    r4 = (sp ** 4).mean()
    return (res.p_surv, *standardised(r1, r2, r3, r4), elapsed)

# ── Main computation ──────────────────────────────────────────────────────────

print("Computing Laplace rows ...")
lap_rows = {h0: laplace_row(h0) for h0 in H0_GRID}

print("Computing Monte Carlo rows ...")
mc_rows  = {h0: mc_row(h0)     for h0 in H0_GRID}

# ── Table 1: p_surv, mean, variance ──────────────────────────────────────────

print("\n" + "=" * 78)
print("TABLE 1  —  Survival probability, conditional mean and variance")
print("=" * 78)

hdr = (
    f"{'h0':>5}  "
    f"{'pS(Lap)':>9}  {'pS(MC)':>9}  "
    f"{'mu(Lap)':>9}  {'mu(MC)':>9}  "
    f"{'var(Lap)':>9}  {'var(MC)':>9}"
)
print(hdr)
print("-" * 78)
for h0 in H0_GRID:
    pL, mL, vL, _, _, tL = lap_rows[h0]
    pM, mM, vM, _, _, tM = mc_rows[h0]
    print(
        f"{h0:>5.2f}  "
        f"{pL:>9.5f}  {pM:>9.5f}  "
        f"{mL:>9.5f}  {mM:>9.5f}  "
        f"{vL:>9.5f}  {vM:>9.5f}"
    )

# ── Table 2: skewness, excess kurtosis, CPU time ─────────────────────────────

print("\n" + "=" * 78)
print("TABLE 2  —  Skewness, excess kurtosis, and CPU time")
print("=" * 78)

hdr2 = (
    f"{'h0':>5}  "
    f"{'sk(Lap)':>9}  {'sk(MC)':>9}  "
    f"{'kt(Lap)':>9}  {'kt(MC)':>9}  "
    f"{'t_Lap(s)':>10}  {'t_MC(s)':>9}"
)
print(hdr2)
print("-" * 78)
for h0 in H0_GRID:
    pL, mL, vL, sL, kL, tL = lap_rows[h0]
    pM, mM, vM, sM, kM, tM = mc_rows[h0]
    print(
        f"{h0:>5.2f}  "
        f"{sL:>9.5f}  {sM:>9.5f}  "
        f"{kL:>9.4f}  {kM:>9.4f}  "
        f"{tL:>10.3f}  {tM:>9.3f}"
    )

# ── No-shorting limit ─────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("NO-SHORTING LIMIT  —  h0 = 100 vs. analytical E[exp(k X_{1,T})]")
print("=" * 78)

model = BivariateKouModel(params=PARAMS)
r_anal = [np.exp(T * model.levy_khintchine(-1j * k, 0)).real for k in [1, 2, 3, 4]]
mn_a, vr_a, sk_a, kt_a = standardised(*r_anal)

strategy_ns = UnitExposureLongShortStrategy(h0=H0_NOSHORT, market=MARKET, T=T)
cm_ns = ConditionalMoments(params=PARAMS, strategy=strategy_ns)
r_lap = [cm_ns.conditional_moment(k) for k in [1, 2, 3, 4]]
mn_l, vr_l, sk_l, kt_l = standardised(*r_lap)

rows = [
    ("Mean",           mn_l, mn_a),
    ("Variance",       vr_l, vr_a),
    ("Skewness",       sk_l, sk_a),
    ("Excess kurtosis", kt_l, kt_a),
]
print(f"{'Statistic':<18}  {'Laplace':>14}  {'Analytical':>14}  {'|rel.err|':>12}")
print("-" * 65)
for name, lap, anal in rows:
    err = abs(lap - anal) / abs(anal)
    print(f"{name:<18}  {lap:>14.8f}  {anal:>14.8f}  {err:>12.2e}")
