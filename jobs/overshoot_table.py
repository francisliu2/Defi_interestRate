"""T10 (variance-preserving overshoot), T7 (buffer vs drift curve + figure),
T8 (Talbot node sensitivity)."""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimal_long_short.model.model_params import KouParams
from optimal_long_short.overshoot_recovery import exact_wealth_moments
from optimal_long_short.laplace.inversion import TalbotInverter
from optimal_long_short.model.moments import ConditionalMoments
from optimal_long_short.model.strategy import UnitExposureLongShortStrategy
from optimal_long_short.model.market_params import MarketParams
from optimal_long_short.drift_bayes import _BufferGrid, optimal_buffer

blob = json.load(open("results/params_WBTC_WETH_headline.json")); P = blob["params"]
KEYS = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
base = KouParams(**{k: P[k] for k in KEYS})
b = blob["aave_constraint"]["b"]; liq = blob["aave_constraint"]["liq_bonus"]; T = 1/12
h0 = math.log(1.16)
out = {}

# ---- T10: two designs ----
def scaled(pp, s_size, s_int):
    d = {k: getattr(pp, k) for k in KEYS}
    for k in ("eta1_pos","eta1_neg","eta2_pos","eta2_neg"): d[k] *= s_size
    d["lam1"] *= s_int; d["lam2"] *= s_int
    return KouParams(**d)

def jvarZ(pp):
    def jv(lam,p,ep,en): return lam*2*(p*ep**2+(1-p)*en**2)
    return jv(pp.lam1,pp.p1,pp.eta1_pos,pp.eta1_neg)+jv(pp.lam2,pp.p2,pp.eta2_pos,pp.eta2_neg)

rows = []
for s in (1.0, 3.0, 10.0, 20.0):
    # Design A: size x s, intensity fixed (jump variance x s^2)
    pa = scaled(base, s, 1.0)
    wa = exact_wealth_moments(pa, h0, b=b, liq_bonus=liq, T=T)
    # Design B: size x s, intensity x 1/s^2 (jump variance ~fixed)
    pb = scaled(base, s, 1.0/s**2)
    wb = exact_wealth_moments(pb, h0, b=b, liq_bonus=liq, T=T)
    rows.append({"s": s,
        "A_jvarZ": jvarZ(pa), "A_p_liq": wa["p_liq"], "A_Rcc": wa["E_recovery_cc"], "A_Rex": wa["E_recovery"],
        "A_rel": (wa["E_recovery"]-wa["E_recovery_cc"])/wa["E_recovery_cc"],
        "B_jvarZ": jvarZ(pb), "B_lam2": pb.lam2, "B_p_liq": wb["p_liq"], "B_Rcc": wb["E_recovery_cc"], "B_Rex": wb["E_recovery"],
        "B_rel": (wb["E_recovery"]-wb["E_recovery_cc"])/wb["E_recovery_cc"]})
out["overshoot_two_designs"] = rows
print("=== T10 two designs (H0=1.16) ===")
print("s   | A:jvarZ  p_liq  Rcc     Rex     rel   | B:jvarZ  lam2   p_liq  Rcc     Rex     rel")
for r in rows:
    print(f"{r['s']:4.0f}| {r['A_jvarZ']:.4f} {r['A_p_liq']:.3f} {r['A_Rcc']:+.3f} {r['A_Rex']:+.3f} {r['A_rel']*100:+5.0f}%| "
          f"{r['B_jvarZ']:.4f} {r['B_lam2']:6.1f} {r['B_p_liq']:.3f} {r['B_Rcc']:+.3f} {r['B_Rex']:+.3f} {r['B_rel']*100:+5.0f}%")

# ---- T7: buffer vs drift curve ----
grid = _BufferGrid(params=base, b=b, liq_bonus=liq, T=T)
nu = np.round(np.arange(-0.30, 0.505, 0.01), 4)
curve = {a: [] for a in (0.5,1.0,2.0)}
for a in curve:
    for v in nu:
        H, interior = optimal_buffer(grid, float(v), a)
        curve[a].append((float(v), H, interior))
out["buffer_vs_drift"] = {str(a): curve[a] for a in curve}

# Figure 2 (latex/fig_buffer_drift.pdf) is written by jobs/drift_bayes.py only.

# ---- T8: Talbot node sensitivity ----
print("=== T8 Talbot node sensitivity (h0=0.07) ===")
strat = UnitExposureLongShortStrategy(h0=0.07, market=MarketParams(b=b,S10=1,S20=1), T=T)
talbot = {}
for M in (12,16,24,32,48,64):
    cm = ConditionalMoments(params=base, strategy=strat, inverter=TalbotInverter(M=M))
    ps = cm.p_surv(); k1 = cm.killed_moment(1)
    talbot[M] = {"p_surv": ps, "killed_1": k1}
    print(f"  M={M:3d}  p_surv={ps:.10f}  killed_1={k1:.10f}")
ref = talbot[64]["p_surv"]
for M in talbot: talbot[M]["p_surv_absdiff_vs_M64"] = abs(talbot[M]["p_surv"]-ref)
out["talbot_sensitivity"] = talbot

json.dump(out, open("results/revision_moderate.json","w"), indent=2, default=float)
print("SAVED results/revision_moderate.json")
