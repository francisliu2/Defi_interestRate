"""T8: Brownian-bridge-corrected Monte Carlo for survival at h0=0.07.

Between grid points the continuous part of Z=X1-X2 is a BM with drift; the
probability it dips below the barrier within a step is added analytically
(the Brownian-bridge / Gobet correction), which removes most of the discretisation
bias that a naive grid-crossing check leaves. Jumps are applied at step ends and
checked directly. We report survival vs step count, with and without the bridge
correction, and a fair wall-clock comparison against the Laplace resolvent."""
import json, math, time
import numpy as np
from optimal_long_short.model_params import KouParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.market_params import MarketParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.inversion import TalbotInverter

blob = json.load(open("results/params_WBTC_WETH.json")); P = blob["params"]
KEYS = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
p = KouParams(**{k: P[k] for k in KEYS})
b = blob["aave_constraint"]["b"]; T = 1/12; h0 = 0.07; m = -h0

sigZ = math.sqrt(p.sigma1**2 + p.sigma2**2 - 2*p.rho*p.sigma1*p.sigma2)

# analytic reference
t0=time.time()
strat = UnitExposureLongShortStrategy(h0=h0, market=MarketParams(b=b,S10=1,S20=1), T=T)
ps_lap = ConditionalMoments(params=p, strategy=strat, inverter=TalbotInverter(M=32)).p_surv()
lap_time = time.time()-t0
print(f"Laplace p_surv(h0=0.07)={ps_lap:.6f}  ({lap_time*1000:.0f} ms)", flush=True)

def kou_jump(rng, lam, prob, ep, en, n, dt):
    nj = rng.poisson(lam*dt, size=n); nu = rng.binomial(nj, prob); nd = nj-nu
    up = rng.gamma(np.maximum(nu,0)+1e-300, ep); dn = rng.gamma(np.maximum(nd,0)+1e-300, en)
    up[nu==0]=0.0; dn[nd==0]=0.0
    return up-dn

def mc(n_paths, n_steps, bridge, seed):
    rng = np.random.default_rng(seed)
    dt = T/n_steps; sdt=math.sqrt(dt); sq=math.sqrt(max(1-p.rho**2,0.0))
    X1=np.zeros(n_paths); X2=np.zeros(n_paths); alive=np.ones(n_paths,bool)
    for _ in range(n_steps):
        W1=rng.standard_normal(n_paths); W2=rng.standard_normal(n_paths)
        Zprev = X1 - X2
        dX1 = p.muX1*dt + p.sigma1*sdt*W1
        dX2 = p.muX2*dt + p.sigma2*sdt*(p.rho*W1+sq*W2)
        Zmid = (X1+dX1) - (X2+dX2)          # after diffusion, before jumps
        # diffusion grid crossing
        cross = Zmid <= m
        if bridge:
            # bridge crossing prob for paths with both endpoints above barrier
            safe = (~cross) & alive & (Zprev>m) & (Zmid>m)
            pcr = np.zeros(n_paths)
            pcr[safe] = np.exp(-2.0*(Zprev[safe]-m)*(Zmid[safe]-m)/(sigZ**2*dt))
            bridge_hit = rng.random(n_paths) < pcr
            cross = cross | bridge_hit
        J1 = kou_jump(rng,p.lam1,p.p1,p.eta1_pos,p.eta1_neg,n_paths,dt)
        J2 = kou_jump(rng,p.lam2,p.p2,p.eta2_pos,p.eta2_neg,n_paths,dt)
        X1 = X1+dX1+J1; X2 = X2+dX2+J2
        cross = cross | ((X1-X2) <= m)      # jump crossing at step end
        alive &= ~cross
    ps = alive.mean(); se = math.sqrt(ps*(1-ps)/n_paths)
    return ps, se

NP = 200_000
print("=== convergence (Brownian-bridge corrected) ===", flush=True)
conv=[]
for ns in (125, 250, 500, 1000, 2000, 4000):
    t0=time.time(); ps,se = mc(NP, ns, True, seed=11); el=time.time()-t0
    conv.append({"n_steps":ns,"bridge":True,"p_surv":ps,"se":se,"z_vs_lap":(ps_lap-ps)/se,"secs":el})
    print(f"  bridge  steps={ns:5d}  p_surv={ps:.5f}+-{se:.5f}  z={(ps_lap-ps)/se:+.2f}  ({el:.1f}s)", flush=True)
print("=== convergence (naive grid check, no bridge) ===", flush=True)
for ns in (125, 250, 500, 1000, 2000, 4000):
    t0=time.time(); ps,se = mc(NP, ns, False, seed=11); el=time.time()-t0
    conv.append({"n_steps":ns,"bridge":False,"p_surv":ps,"se":se,"z_vs_lap":(ps_lap-ps)/se,"secs":el})
    print(f"  naive   steps={ns:5d}  p_surv={ps:.5f}+-{se:.5f}  z={(ps_lap-ps)/se:+.2f}  ({el:.1f}s)", flush=True)

out={"h0":h0,"laplace_p_surv":float(ps_lap),"laplace_ms":lap_time*1000,"sigma_Z":sigZ,
     "n_paths":NP,"convergence":conv}
json.dump(out, open("results/mc_convergence.json","w"), indent=2, default=float)
print("SAVED results/mc_convergence.json", flush=True)
