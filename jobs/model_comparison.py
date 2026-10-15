"""T11/B3: nested-model comparison on the liquidation-relevant spread Z = r1 - r2.

Fit GBM (normal), Merton (normal + Gaussian jumps) and Kou (normal + double-
exponential jumps) to the ECF of the 4h demeaned spread returns and compare on:
  (1) in-sample ECF objective and a parameter-count-penalised (BIC-style) criterion;
  (2) OUT-OF-SAMPLE ECF objective: fit on the first 60% of the window, evaluate the
      ECF distance on the last 40%;
  (3) a LIQUIDATION-RELEVANT barrier-crossing diagnostic: for several h0 the empirical
      frequency that the spread path crosses -h0 within one month (rolling 4h windows),
      against each fitted model's simulated one-month crossing probability under the
      same discrete monitoring.
"""
import json, math
import numpy as np, pandas as pd
from scipy.optimize import least_squares

df = pd.read_parquet("data/AAVE/merged_wbtc_weth_4h.parquet")
lb = np.diff(np.log(df["close_btc"].to_numpy())); le = np.diff(np.log(df["close_eth"].to_numpy()))
lb = lb - pd.Series(lb).ewm(span=30).mean().to_numpy(); le = le - pd.Series(le).ewm(span=30).mean().to_numpy()
Z = lb - le
n = len(Z); sd = Z.std(); print("n spread returns", n, "std", sd)

# ECF grid + Gaussian weights, matching the paper's spread-direction weighting
def make_grid(scale):
    U = np.array([0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0]) / scale
    w = np.exp(-1.0*(U*scale)**2)
    return U, w

def gbm_cf(th, u):
    mu, ls = th; s = math.exp(ls); return np.exp(1j*mu*u - 0.5*s**2*u**2)
def merton_cf(th, u):
    mu, ls, ll, mj, lsj = th; s=math.exp(ls); lam=math.exp(ll); sj=math.exp(lsj)
    return np.exp(1j*mu*u - 0.5*s**2*u**2 + lam*(np.exp(1j*mj*u - 0.5*sj**2*u**2)-1))
def kou_cf(th, u):
    mu, ls, ll, sp, lep, lem = th
    s=math.exp(ls); lam=math.exp(ll); p=1/(1+math.exp(-sp)); ep=math.exp(lep); em=math.exp(lem)
    phiJ = p/(1-1j*u*ep) + (1-p)/(1+1j*u*em)
    return np.exp(1j*mu*u - 0.5*s**2*u**2 + lam*(phiJ-1))
CF = {"GBM":gbm_cf, "Merton":merton_cf, "Kou":kou_cf}
NP = {"GBM":2, "Merton":5, "Kou":6}
def x0(name, scale):
    if name=="GBM":    return [Z.mean(), math.log(scale)]
    if name=="Merton": return [Z.mean(), math.log(scale*0.7), math.log(50), 0.0, math.log(scale*0.5)]
    return [Z.mean(), math.log(scale*0.7), math.log(50), 0.0, math.log(scale*0.5), math.log(scale*0.5)]

def fit(data):
    scale = data.std()
    U, w = make_grid(scale)
    phi_emp = np.array([np.mean(np.exp(1j*u*data)) for u in U])
    def resid_for(cf):
        def r(th):
            d = (np.array([cf(th,u) for u in U]) - phi_emp)*np.sqrt(w)
            return np.concatenate([d.real, d.imag])
        return r
    res = {}
    for name,cf in CF.items():
        res[name] = least_squares(resid_for(cf), x0=x0(name,scale))
    return res, (U, w, phi_emp)

def ecf_obj(cf, th, data):
    scale = data.std(); U, w = make_grid(scale)
    phi_emp = np.array([np.mean(np.exp(1j*u*data)) for u in U])
    d = (np.array([cf(th,u) for u in U]) - phi_emp)*np.sqrt(w)
    return float(0.5*np.sum(d.real**2 + d.imag**2))

# ---- (1) in-sample fit + BIC-style penalty ----
fits, _ = fit(Z)
qs=[0.5,1,5,95,99,99.5]; emp_q=np.percentile(Z,qs)
out={"n":n,"Z_std":float(sd),"emp_quantiles_pct":{str(q):float(v) for q,v in zip(qs,emp_q)},
     "emp_kurtosis":float(pd.Series(Z).kurt()),"models":{}}
for name in ("GBM","Merton","Kou"):
    f=fits[name]; obj=0.5*np.sum(f.fun**2)
    bic = n*math.log(obj) + NP[name]*math.log(n)   # BIC-style: penalise params
    out["models"][name]={"ecf_obj_insample":float(obj),"n_params":NP[name],"bic":float(bic),
                          "params":[float(x) for x in f.x]}
    print(f"{name:7s} in-sample obj={obj:.3e} bic={bic:.1f}")

# ---- (2) out-of-sample ECF ----
cut=int(0.6*n); Ztr, Zte = Z[:cut], Z[cut:]
fits_tr,_=fit(Ztr)
print("--- out-of-sample (fit 60%, eval 40%) ---")
for name in ("GBM","Merton","Kou"):
    oos=ecf_obj(CF[name], fits_tr[name].x, Zte)
    ins=0.5*np.sum(fits_tr[name].fun**2)
    out["models"][name]["ecf_obj_oos"]=float(oos)
    out["models"][name]["ecf_obj_train"]=float(ins)
    print(f"{name:7s} train={ins:.3e} oos={oos:.3e}")

# ---- (3) barrier-crossing diagnostic ----
STEPS_1M = int(round((1/12)/(4/(24*365))))   # 4h steps in one month
Zc = np.cumsum(Z)                             # spread level path
h0s=[0.05,0.07,0.10,0.15,0.20]
emp_cross={}
for h0 in h0s:
    hits=0; tot=0
    for s0 in range(0, n-STEPS_1M):
        seg = Zc[s0:s0+STEPS_1M+1] - Zc[s0]
        tot+=1
        if seg.min() < -h0: hits+=1
    emp_cross[h0]=hits/tot if tot else float("nan")
print("--- barrier crossing (empirical rolling 1-month, discrete 4h monitoring) ---")
print("STEPS_1M=",STEPS_1M)

def draw_jumps(name, th, shape, rng):
    N,steps=shape
    mu,ls,ll = th[0],th[1],th[2]; s=math.exp(ls); lam=math.exp(ll)
    x=mu+s*rng.standard_normal((N,steps))
    nj=rng.poisson(lam,(N,steps)); tot=int(nj.sum())
    if tot==0: return x
    flat=np.zeros(nj.size)
    if name=="Merton":
        mj,sj=th[3],math.exp(th[4])
        js=rng.normal(mj,sj,tot)
    else:  # Kou
        sp,ep,em=th[3],math.exp(th[4]),math.exp(th[5]); p=1/(1+math.exp(-sp))
        up=rng.random(tot)<p
        js=np.where(up, rng.exponential(ep,tot), -rng.exponential(em,tot))
    # scatter jump sums into cells
    idx=np.repeat(np.arange(nj.size), nj.ravel())
    np.add.at(flat, idx, js)
    return x + flat.reshape(N,steps)

rng=np.random.default_rng(7); NSIM=200_000
model_cross={m:{} for m in CF}
for name in ("GBM","Merton","Kou"):
    th=fits[name].x
    if name=="GBM":
        mu,ls=th; s=math.exp(ls); inc=mu+s*rng.standard_normal((NSIM,STEPS_1M))
    else:
        inc=draw_jumps(name,th,(NSIM,STEPS_1M),rng)
    path=np.cumsum(inc,axis=1); mn=path.min(axis=1)
    for h0 in h0s:
        model_cross[name][h0]=float((mn < -h0).mean())

out["barrier_crossing"]={"steps_1m":STEPS_1M,"h0":h0s,
    "empirical":{str(h):emp_cross[h] for h in h0s},
    "model":{m:{str(h):model_cross[m][h] for h in h0s} for m in CF}}
print(f"{'h0':>6}{'emp':>9}{'GBM':>9}{'Merton':>9}{'Kou':>9}")
for h0 in h0s:
    print(f"{h0:6.2f}{emp_cross[h0]:9.4f}{model_cross['GBM'][h0]:9.4f}{model_cross['Merton'][h0]:9.4f}{model_cross['Kou'][h0]:9.4f}")

json.dump(out, open("results/model_comparison.json","w"), indent=2)
print("SAVED results/model_comparison.json")
