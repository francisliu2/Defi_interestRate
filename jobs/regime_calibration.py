"""T9: calibrate the two-state carry regime from the WETH utilization series and
rerun the regime resolvent at the benchmark WBTC/WETH parameters, T=1/12."""
import json, math
import numpy as np, pandas as pd
from optimal_long_short.model_params import KouParams
from optimal_long_short.kou_model import KouZTiltedDynamics
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.market_params import MarketParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.inversion import TalbotInverter
from optimal_long_short.regime_switching import (
    RegimeSpec, RegimeSwitchingDynamics, RegimeResolvent)

blob = json.load(open("results/params_WBTC_WETH.json")); P = blob["params"]
KEYS = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
# A2: the stored benchmark drifts already fold in the supply/borrow APRs. The regime
# model adds a state-dependent net carry on top, so it must start from the PRICE-ONLY
# drifts (APR adjustment removed) to avoid double-counting carry.
supply_btc = blob["aave_rates"]["supply_btc"]
borrow_eth = blob["aave_rates"]["borrow_eth"]
Ppo = dict(P); Ppo["mu1"] = P["mu1"] - supply_btc; Ppo["mu2"] = P["mu2"] - borrow_eth
base = KouParams(**{k: Ppo[k] for k in KEYS})   # price-only base for the regime runs
print("price-only spread mu1-mu2 = %.4f (APR-adjusted was %.4f)"
      % (base.mu1 - base.mu2, P["mu1"] - P["mu2"]))
b = blob["aave_constraint"]["b"]; T = 1/12
U_OPT = 0.90  # WETH optimalUtilisationRate

w = pd.read_parquet("data/AAVE/hourly_weth.parquet")
win = w.loc["2026-03-17 18:00:00":"2026-08-04 16:00:00"].copy()
u = win["utilization"].astype(float).to_numpy()
if np.nanmax(u) > 1.5:  # percent -> fraction
    u = u/100.0
br = win["borrower_variable_apr"].astype(float).to_numpy()
if np.nanmax(br) > 1.0:  # percent -> decimal
    br = br/100.0
print(f"n hours={len(u)}  U range [{np.nanmin(u):.3f},{np.nanmax(u):.3f}]  mean {np.nanmean(u):.3f}")
print(f"borrow apr range [{np.nanmin(br):.4f},{np.nanmax(br):.4f}] mean {np.nanmean(br):.4f}")

stressed = u > U_OPT
frac = float(np.nanmean(stressed))
# mean duration of stressed runs (hours -> years)
runs=[]; c=0
for s in stressed:
    if s: c+=1
    elif c>0: runs.append(c); c=0
if c>0: runs.append(c)
mean_dur_hours = float(np.mean(runs)) if runs else 1.0
mean_dur_years = mean_dur_hours/8760.0
print(f"frac_stressed={frac:.4f}  n_runs={len(runs)}  mean_dur={mean_dur_hours:.1f}h ({mean_dur_years:.5f}yr)")

# realized net carry per state: c = supply_btc - borrow_eth(state); supply_btc tiny
supply_btc = json.load(open("data/AAVE/realized_rates_aug2026.json"))["wbtc"]["supply_apr_pct_mean"]/100.0
borrow_normal = float(np.nanmean(br[~stressed])); borrow_stress = float(np.nanmean(br[stressed]))
c1 = supply_btc - borrow_normal; c2 = supply_btc - borrow_stress
print(f"supply_btc={supply_btc:.5f}  borrow_normal={borrow_normal:.4f}  borrow_stressed={borrow_stress:.4f}")
print(f"c1(normal)={c1:+.4f}  c2(stressed)={c2:+.4f}")

spec = RegimeSpec.from_occupation(frac_stressed=frac, mean_duration_stressed=mean_dur_years,
                                  c_normal=c1, c_stressed=c2)
print(f"q1={spec.q1:.3f} q2={spec.q2:.3f}  stationary={spec.stationary}  mean_carry={spec.mean_carry:+.4f}")

dyn0 = KouZTiltedDynamics(params=base, k=0)
rdyn = RegimeSwitchingDynamics(base=dyn0, spec=spec)
inv = TalbotInverter(M=32)

c_bar = float(spec.mean_carry)   # stationary mean net carry
# deterministic-mean-carry base: shift the Z-drift by c_bar (Prop. carry), i.e. mu2 -= c_bar
Pdc = dict(Ppo); Pdc["mu2"] = Ppo["mu2"] - c_bar
base_dc = KouParams(**{k: Pdc[k] for k in KEYS})
print(f"c_bar(stationary mean carry)={c_bar:+.4f}")

rows=[]
for h0 in (0.10, 0.25, 0.50):
    strat = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=b,S10=1,S20=1), T=T)
    # (1) price-only single regime, no carry
    ps_price = ConditionalMoments(params=base, strategy=strat, inverter=inv).p_surv()
    # (2) deterministic mean-carry single regime (Prop. carry with c = c_bar)
    ps_detc = ConditionalMoments(params=base_dc, strategy=strat, inverter=inv).p_surv()
    # (3,4) regime resolvent (k=0 -> survival), per opening regime, price-only base + regime carry
    res = RegimeResolvent(dynamics=rdyn, h0=h0, b=b)
    ps_normal = float(inv.invert(lambda q: res.transform(q)[0], T))
    ps_stress = float(inv.invert(lambda q: res.transform(q)[1], T))
    rows.append({"h0":h0,"p_price_only":float(ps_price),"p_det_mean_carry":float(ps_detc),
                 "p_open_normal":ps_normal,"p_open_stressed":ps_stress,
                 "open_diff_pp":100*(ps_stress-ps_normal),
                 "carry_level_pp":100*(ps_detc-ps_price),
                 "jensen_pp":100*(0.5*(ps_normal+ps_stress)-ps_detc)})
    print(f"h0={h0:.2f}: price={ps_price:.4f} detc={ps_detc:.4f} normal={ps_normal:.4f} "
          f"stressed={ps_stress:.4f} | open_diff={100*(ps_stress-ps_normal):+.2f}pp "
          f"carry_level={100*(ps_detc-ps_price):+.2f}pp")

out={"U_opt":U_OPT,"frac_stressed":frac,"mean_dur_hours":mean_dur_hours,
     "q1":spec.q1,"q2":spec.q2,"c1":c1,"c2":c2,"stationary_stressed":spec.stationary[1],
     "mean_carry":c_bar,"borrow_normal":borrow_normal,"borrow_stressed":borrow_stress,
     "price_only_spread":float(base.mu1-base.mu2),"adj_spread":float(P["mu1"]-P["mu2"]),
     "T":T,"rows":rows}
json.dump(out, open("results/regime_calibrated.json","w"), indent=2)
print("SAVED results/regime_calibrated.json")
