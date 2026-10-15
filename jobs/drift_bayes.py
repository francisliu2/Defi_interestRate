"""Task 4 (correct objective): unconditional wealth W_T = Pi_T + R_tau
mean-variance, matching the paper's Section 8 SSD analysis.  Build h0*(delta),
the interior threshold delta*(alpha), and the Bayesian posterior over delta ->
distribution over H0*."""
import json, math
import numpy as np
from optimal_long_short.model_params import KouParams
from optimal_long_short.drift import with_price_drift_view
from optimal_long_short.drift_bayes import _BufferGrid

blob = json.load(open("results/params_WBTC_WETH.json")); pd = blob["params"]
KEYS = ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg","mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]
params = KouParams(**{k: pd[k] for k in KEYS})
b = blob["aave_constraint"]["b"]; liq = blob["aave_constraint"]["liq_bonus"]; T = 1/12
rates = blob["aave_rates"]

H_lo, H_hi = 1.0684931506849316, 2.0
hgrid = np.linspace(math.log(H_lo), math.log(H_hi), 40)
DELTA = np.round(np.arange(-0.30, 0.605, 0.02), 4)

# (mean_wealth, var_wealth) along hgrid for each delta, with exact jump-overshoot
# recovery (Section 6.3); the grid caches per delta and is reused for the figure.
grid = _BufferGrid(params=params, b=b, liq_bonus=liq, T=T, H0_lo=H_lo, H0_hi=H_hi,
                   n_grid=len(hgrid), recovery="exact")

def h0star(delta, alpha):
    d = float(DELTA[int(np.argmin(np.abs(DELTA - delta)))])
    rows = grid.curve(d)
    vals = [m - 0.5*alpha*v for (m, v) in rows]
    i = int(np.argmax(vals))
    interior = 0 < i < len(hgrid)-1
    return math.exp(hgrid[i]), interior

alphas = [0.5, 1.0, 2.0]
# threshold delta*(alpha): smallest delta giving an interior optimum
threshold = {}
for a in alphas:
    t = None
    for d in DELTA:
        H, it = h0star(d, a)
        if it:
            t = float(d); break
    threshold[a] = t

# drift posterior inputs
delta_ecf = params.mu1 - params.mu2
emp = blob["meta"]["empirical_first_moments"]["demeaned_price_growth_mu_annualized"]
delta_emp = emp["WBTC"] - emp["WETH"]
sigZ2 = params.sigma1**2 + params.sigma2**2 - 2*params.rho*params.sigma1*params.sigma2
def jvar(lam,p,ep,en): return lam*2.0*(p*ep**2+(1-p)*en**2)
varZ = sigZ2 + jvar(params.lam1,params.p1,params.eta1_pos,params.eta1_neg) \
             + jvar(params.lam2,params.p2,params.eta2_pos,params.eta2_neg)
T_obs = blob["meta"]["n_obs"]*blob["meta"]["dt_years"]
se_realised = math.sqrt(varZ/T_obs)     # realised-variance SE (old headline, 0.54)
se_diff = math.sqrt(sigZ2/T_obs)
# A4: use the bootstrap SE of the spread drift mu1-mu2 (headline estimator) as the
# likelihood SE, if available. It is the estimator's own finite-sample SE and is the
# more honest measure of how weakly the drift is pinned down.
try:
    se_boot = float(json.load(open("results/bootstrap_se.json"))["spread"]["se"])
except Exception:
    se_boot = se_realised
se_total = se_boot   # likelihood SE used in the posterior below
carry = rates["supply_btc"] - rates["borrow_eth"]

def post(m0,s0,data,se):
    t0,td = 1/s0**2, 1/se**2
    return (t0*m0+td*data)/(t0+td), math.sqrt(1/(t0+td))

def norm_sf(x): return 0.5*math.erfc(x/math.sqrt(2))

def dist(m,s,alpha,n=20000,seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.normal(m,s,size=n)
    H = np.empty(n); it = np.empty(n,bool)
    lut = {}
    for j,d in enumerate(draws):
        key = float(DELTA[int(np.argmin(np.abs(DELTA-d)))])
        if key not in lut: lut[key] = h0star(key,alpha)
        H[j],it[j] = lut[key]
    r = {"p_interior": float(it.mean()),
         "H0_median": float(np.median(H)),
         "H0_q25": float(np.quantile(H,0.25)),
         "H0_q75": float(np.quantile(H,0.75))}
    if it.any():
        r["H0_int_median"] = float(np.median(H[it]))
        r["H0_int_q25"] = float(np.quantile(H[it],0.25))
        r["H0_int_q75"] = float(np.quantile(H[it],0.75))
    return r

out = {
  "objective": "unconditional wealth W_T mean-variance (exact overshoot recovery)",
  "note": "threshold_delta_star = smallest drift with an optimum below H0 = 2 (capped grid), NOT the break-even for leverage; see results/breakeven_drift.json",
  "anchor_check_delta+0.2": {str(a): h0star(0.2,a) for a in alphas},
  "benchmark_delta-0.858": {str(a): h0star(delta_ecf,a) for a in alphas},
  "threshold_delta_star": {str(a): threshold[a] for a in alphas},
  "drift_inputs": {"delta_ecf":delta_ecf,"delta_emp":delta_emp,"carry":carry,
     "sigma_Z":math.sqrt(sigZ2),"se_total":se_total,"se_realised":se_realised,
     "se_boot":se_boot,"se_diff":se_diff,"T_obs":T_obs,
     "ecf_sigmas_from_zero": abs(delta_ecf)/se_total,
     "ecf_sigmas_from_zero_realised": abs(delta_ecf)/se_realised},
  "posteriors": {},
}
for (m0,s0,tag) in [(carry,0.20,"carry_s.20"),(carry,0.10,"carry_s.10"),(carry,0.40,"carry_s.40")]:
    for data,dt in [(delta_ecf,"ecf"),(delta_emp,"emp")]:
        m,s = post(m0,s0,data,se_total)
        row = {"prior_mean":m0,"prior_sd":s0,"data":data,"se":se_total,
               "post_mean":m,"post_sd":s,"P_delta_gt_0":norm_sf(-m/s)}
        for a in alphas:
            row[f"P_interior_a{a}"] = norm_sf((threshold[a]-m)/s) if threshold[a] is not None else 0.0
            row[f"dist_a{a}"] = dist(m,s,a)
        out["posteriors"][f"{tag}__{dt}"] = row

json.dump(out, open("results/task4_numbers.json","w"), indent=2)
print(json.dumps(out, indent=2))

# Figure 2 (latex/fig_buffer_drift.pdf) is written by jobs/breakeven_drift.py, which
# searches the full buffer range; the thresholds here are on the capped grid H0 <= 2.
