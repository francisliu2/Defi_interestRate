"""T12/B1: recalibrate on the extended 1.6-year WBTC/WETH sample and make it a
first-class robustness column.

Part 1 (recalibration) is preserved from the original script: it fits the bivariate
Kou model on 2025-01..2026-08 (4h) and reports the flipped spread drift.

Part 2 (B1) evaluates the sizing framework at the extended-sample parameters:
  * the survival + conditional-moment table (companion to tab:results);
  * the interior-optimum drift thresholds nu*(alpha) under unconditional-wealth
    mean-variance, to test whether the unlevered corner solution survives the
    change of estimation window.
"""
import json, math
import numpy as np, pandas as pd
from optimal_long_short.calibration import calibrate_ecf
from optimal_long_short.model_params import KouParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.market_params import MarketParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.inversion import TalbotInverter
from optimal_long_short.drift import with_price_drift_view
from optimal_long_short.overshoot_recovery import exact_wealth_moments
from optimal_long_short.job_runners.calibrate_btc_eth import CALIB_BOUNDS

RECAL = "--recal" in __import__("sys").argv
if RECAL:
    wb = pd.read_parquet("data/AAVE/hourly_wbtc.parquet")["price_usd"].astype(float)
    we = pd.read_parquet("data/AAVE/hourly_weth.parquet")["price_usd"].astype(float)
    px = pd.concat([wb.rename("btc"), we.rename("eth")], axis=1).dropna()
    px = px.resample("4h").last().dropna()
    lb = np.diff(np.log(px["btc"].to_numpy())); le = np.diff(np.log(px["eth"].to_numpy()))
    lb = lb - pd.Series(lb).ewm(span=30).mean().to_numpy(); le = le - pd.Series(le).ewm(span=30).mean().to_numpy()
    Z = lb - le; DT = 4/(24*365)
    res = calibrate_ecf(lb, le, DT, n_starts=12, seed=0, bounds=CALIB_BOUNDS)  # headline bounds
    ep = res.params
    ext = {k: float(getattr(ep, k)) for k in
           ["sigma1","sigma2","lam1","lam2","p1","p2","eta1_pos","eta1_neg","eta2_pos","eta2_neg","rho","mu1","mu2"]}
    _prev = json.load(open("results/recalib_extended.json"))
    _prev.update({"extended": ext, "bounds": "CALIB_BOUNDS (lambda_max=5000)",
                  "spread_drift_extended": ext["mu1"] - ext["mu2"]})
    json.dump(_prev, open("results/recalib_extended.json", "w"), indent=2)
    print("SAVED results/recalib_extended.json")
else:
    ext = json.load(open("results/recalib_extended.json"))["extended"]

blob = json.load(open("results/params_WBTC_WETH.json"))
b = blob["aave_constraint"]["b"]; liq = blob["aave_constraint"]["liq_bonus"]; T = 1/12
rates = blob["aave_rates"]; carry = rates["supply_btc"] - rates["borrow_eth"]

KEYS = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
pe = KouParams(**{k: ext[k] for k in KEYS})
sigZ = math.sqrt(pe.sigma1**2 + pe.sigma2**2 - 2*pe.rho*pe.sigma1*pe.sigma2)
spread_raw = pe.mu1 - pe.mu2
spread_adj = spread_raw + carry   # APR-adjusted, comparable to the benchmark -0.858
print(f"extended sigma_Z={sigZ:.4f}  spread raw={spread_raw:+.4f}  APR-adj={spread_adj:+.4f}")

# --- companion tab:results at extended params ---
inv = TalbotInverter(M=32)
def cstats(cm):
    m1=cm.conditional_moment(1); m2=cm.conditional_moment(2)
    m3=cm.conditional_moment(3); m4=cm.conditional_moment(4)
    var=m2-m1**2
    return m1,var,(m3-3*m1*m2+2*m1**3)/var**1.5,(m4-4*m1*m3+6*m1**2*m2-3*m1**4)/var**2-3
res_rows=[]
for h0 in (0.07,0.10,0.25,0.50,0.75,1.00,1.50,2.00):
    strat=UnitExposureLongShortStrategy(h0=h0,market=MarketParams(b=b,S10=1,S20=1),T=T)
    cm=ConditionalMoments(params=pe,strategy=strat,inverter=inv)
    ps=cm.p_surv(); m1,var,sk,ku=cstats(cm)
    res_rows.append(dict(h0=h0,H0=math.exp(h0),psurv=ps,mean=m1,var=var,skew=sk,kurt=ku))
    print(f"h0={h0:.2f} psurv={ps:.4f} mean={m1:.4f} var={var:.4f} skew={sk:.4f} kurt={ku:.4f}")

# --- interior thresholds nu*(alpha) at extended params ---
H_lo, H_hi = blob["aave_constraint"]["H0_min"], 2.0
hgrid = np.linspace(math.log(H_lo), math.log(H_hi), 40)
DELTA = np.round(np.arange(-0.30, 0.605, 0.02), 4)
def h0star(delta, alpha):
    pv = with_price_drift_view(pe, delta_mu1=(float(delta) - (pe.mu1 - pe.mu2)))
    vals=[]
    for h in hgrid:
        wm = exact_wealth_moments(pv, h, b=b, liq_bonus=liq, T=T)
        vals.append(wm["mean_wealth"] - 0.5*alpha*wm["var_wealth"])
    i=int(np.argmax(vals)); return math.exp(hgrid[i]), (0 < i < len(hgrid)-1)
alphas=[0.5,1.0,2.0]; thr={}
for a in alphas:
    t=None
    for d in DELTA:
        _,it=h0star(d,a)
        if it: t=float(d); break
    thr[a]=t
    print(f"alpha={a}: nu*={t}")

survives = all((thr[a] is None) or (spread_raw < thr[a] and spread_adj < thr[a]) for a in alphas)
out={"extended_params":ext,"sigma_Z":sigZ,"spread_raw":spread_raw,"spread_adj":spread_adj,
     "carry":carry,"results_rows":res_rows,"nu_star":thr,"corner_survives":bool(survives),
     "benchmark_spread_adj":blob["params"]["mu1"]-blob["params"]["mu2"]}
json.dump(out, open("results/recalib_extended_sizing.json","w"), indent=2)
print("corner_survives=",survives)
print("SAVED results/recalib_extended_sizing.json")
