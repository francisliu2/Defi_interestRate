"""Generate notebooks/05_integrated_framework.ipynb.

Kept as a script so the notebook is reproducible from source rather than
hand-edited JSON.  Run from the repository root:

    python notebooks/build_05.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/05_integrated_framework.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# Integrated Framework: DeFi Long--Short Sizing under Liquidation, Jumps, and Rate Convexity

This notebook is the single executable overview of the whole project. It runs every layer of
the stack end to end, states the mathematics in LaTeX alongside the code that implements it,
and points at the source file behind each result.

**What the project computes.** A leveraged position on a DeFi lending protocol holds collateral
$S_1$ and owes $S_2$. Solvency is governed by the health factor

$$H_t \;=\; \frac{b\,V_{1,t}}{V_{2,t}} \;=\; \frac{b\,n_1S_{1,t}}{n_2S_{2,t}},
\qquad h_t=\log H_t,\qquad \tau=\inf\{t\ge0:\ h_t<0\},$$

and the whole problem collapses onto the one-dimensional log price ratio $Z=X_1-X_2$, since
$h_t=h_0+Z_t$. The framework evaluates, semi-analytically and for every admissible integer
moment order $k$, the survival probability and the killed moments $\mathbb E[\Pi_T^k]$ as
functions of the single initial buffer $h_0$.

**What this notebook adds beyond the base resolvent.** In order of how much new machinery they
need:

1. a **protocol data layer** with a pre-registered asset screen and a 37-pair cross-section;
2. a **utilization / rate-curve model** replacing the pathological CIR fits;
3. **liquidation recovery** $R_\tau$, from a first-passage transform of the same barrier system,
   including the **exact jump-overshoot** correction, the **overshoot law** and the exact
   **bad-debt probability**;
4. the **expected log of wealth**, so Kelly sizing is semi-analytic;
5. **regime-switching carry**, which enlarges the barrier system from $3\times3$ to $6\times6$;

and the sizing results of paper §8: a closed-form **break-even drift** above which leverage pays,
a sample drift too noisy to say whether it is exceeded, and a **model-free realised backtest** of
all 37 pairs showing that leverage rarely paid and that the model's independent jumps bias it.
""")

# ---------------------------------------------------------------- component map
md(r"""
---
## 1. Component map

Layers run bottom-up; each depends only on those above it.

| Layer | Module | Purpose | Key objects |
|---|---|---|---|
| **Data** | `optimal_long_short/aave_hourly.py` | Aave v3 subgraph fetcher; hourly reserve history and published rate-curve parameters | `fetch_hourly`, `fetch_reserve_configs`, `RateCurve` |
| **Screen** | `optimal_long_short/screen.py` | Pre-registered liquidity/admissibility screen; all admissible ordered pairs | `apply_screen`, `pair_cross_section` |
| **Rates** | `optimal_long_short/rate_curve.py` | Invert $r_b(U)$ for utilization; Jacobi diffusion on $[0,1]$; regime occupancy | `implied_utilization`, `fit_jacobi`, `regime_summary` |
| **Model** | `optimal_long_short/model_params.py`, `kou_model.py` | Bivariate Kou parameters and $k$-tilted dynamics of $Z$ | `KouParams`, `KouZTiltedDynamics` |
| **Core** | `optimal_long_short/root_finder.py`, `laplace_resolvent.py`, `moments.py` | Degree-6 characteristic roots; $3\times3$ barrier system; killed moments | `CharacteristicRootFinder`, `GeneralSolution`, `ConditionalMoments` |
| **Inversion** | `optimal_long_short/inversion.py` | Numerical Laplace inversion | `TalbotInverter` |
| **Recovery** | `optimal_long_short/recovery.py`, `overshoot_recovery.py` | First-passage transform; wealth moments $W_T=\Pi_T+R_\tau\mathbf 1$, continuous-crossing and exact overshoot | `wealth_moments`, `exact_wealth_moments` |
| **Regimes** | `optimal_long_short/regime_switching.py` | Markov-modulated carry; degree-12 roots; $6\times6$ barrier system | `RegimeSpec`, `RegimeResolvent` |
| **Log wealth** | `optimal_long_short/kelly.py` | Killed joint moments, overshoot law, bad-debt probability, $\mathbb E[\log W_T]$ | `overshoot_law`, `expected_log_wealth` |
| **Sizing** | `jobs/breakeven_drift.py`, `optimal_long_short/drift_bayes.py` | Closed-form break-even drift; sample drifts; posteriors and the Bayes--Kelly rule; buffer grid on unconditional wealth | `breakeven`, `kelly_closed_form`, `_BufferGrid` |
| **Realised** | `optimal_long_short/full_history.py`, `jobs/realised_backtest.py` | Stale-aware full-history panels; hourly replay of every pair under Aave liquidation rules | `pair_panel`, `simulate`, `load_pair` |
| **Strategy** | `optimal_long_short/barbell.py` | Supplier leg (long the kink), keeper leg (long the barrier) | `supplier_leg`, `keeper_leg` |

**Reproduction scripts** (`jobs/`): data and calibration --- `fetch_universe.py`,
`build_merged_4h.py`, `calibrate_btc_eth.py --merged`; paper tables --- `numerical_comparison.py`,
`mc_convergence.py`, `overshoot_table.py`, `regime_calibration.py`, `breakeven_drift.py`,
`kelly_sizing.py`, `bootstrap_se.py`, `model_comparison.py`; full history and backtests ---
`horizon_analysis.py`, `rolling_backtest.py`, `backtest_bias.py`, `realised_backtest.py`,
`fig_realised_backtest.py`; validation --- `validate_recovery.py`, `validate_regime_switching.py`.
The README maps each to its table.

```
        Aave v3 subgraph
               |
      aave_hourly.py ......... hourly history + published curve params
         |          \
   screen.py      rate_curve.py ... U = r_b^{-1}(.), Jacobi on [0,1]
         |                 |
   cross-section     regime calibration
                           |
   model_params -> kou_model -> root_finder -> laplace_resolvent -> moments
                                    |                |
                                    |          recovery.py ..... R_tau, W_T (continuous crossing)
                                    |          overshoot_recovery.py ... exact overshoot
                                    |                |
                                    |          kelly.py ........ overshoot law, E[log W]
                                    |                |
                              regime_switching.py (6x6)      breakeven_drift.py (sizing)
                                          |                       |
                                     barbell.py        realised_backtest.py (data, no model)
```
""")

code(r"""
import sys, json, math, warnings
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
print("repository root:", ROOT)
""")

# ---------------------------------------------------------------- data layer
md(r"""
---
## 2. Data layer

`optimal_long_short/aave_hourly.py` replaces the ad-hoc script that used to live at
`data/OLD_AAVE/get_aave_hourly.py`. Three defects were fixed:

* the `.env` file declared `THE_GRAPH_API_KEY ` **with a trailing space**, so `os.environ.get`
  returned `None` and every historical fetch silently ran unauthenticated;
* the hourly path hard-coded a five-coin dictionary, so wrapped-BTC and staking reserves could
  never be fetched even though they were listed for the config snapshot;
* utilization, the quantity the rate model actually needs, was never derived.

Hourly history is stored at `data/AAVE/hourly_<token>.parquet` (12 reserves, 14,321 hours each,
2025-01-01 to 2026-08-20) and reserve configuration at `data/AAVE/reserve_configs_v3_eth.csv`.

**Prices.** The oracle-implied USD price is $\texttt{TVL\_USD}/(\texttt{token\_balance}/10^{\text{decimals}})$.
The fetcher now applies the $10^{\text{decimals}}$ scaling; parquets written before that fix hold
`price_usd` off by exactly that constant, which cancels in log returns. The headline WBTC/WETH
panel is rebuilt from the hourly files by `jobs/build_merged_4h.py` (4-hour bars, last value in
each bin), and `jobs/calibrate_btc_eth.py --merged` then reproduces `results/params_WBTC_WETH_headline.json`
bit for bit. The cell below checks the first of those claims.
""")

code(r"""
from jobs.build_merged_4h import build as build_merged

merged = pd.read_parquet(ROOT / "data/AAVE/merged_wbtc_weth_4h.parquet")
rebuilt = build_merged("2026-03-17 16:00:00", "2026-08-04 16:00:00")
for c in ("close_btc", "close_eth"):
    print(f"{c}: {len(merged)} bars, max relative difference stored vs rebuilt = "
          f"{np.max(np.abs(rebuilt[c].values / merged[c].values - 1)):.1e}")
print()

configs = pd.read_csv(ROOT / "data/AAVE/reserve_configs_v3_eth.csv", index_col=0)
universe = ["WETH", "wstETH", "weETH", "cbBTC", "tBTC", "WBTC",
            "USDC", "USDT", "DAI", "GHO", "USDe", "LINK"]
cols = ["baseLTVasCollateral", "reserveLiquidationThreshold", "liquidationBonus",
        "reserveFactor", "borrowingEnabled", "usageAsCollateralEnabled",
        "optimalUtilisationRate", "variableRateSlope1", "variableRateSlope2"]
print(f"{len(configs)} reserves fetched; universe rows:")
configs.loc[universe, cols]
""")

md(r"""
### 2.1 The rate curve is a kink

Aave's variable borrow rate is *not* a free stochastic process. It is a deterministic, strictly
increasing, piecewise-linear function of utilization $U$:

$$r_b(U)=\begin{cases}
r_0 + s_1\,\dfrac{U}{U_{\mathrm{opt}}}, & U\le U_{\mathrm{opt}},\\[8pt]
r_0 + s_1 + s_2\,\dfrac{U-U_{\mathrm{opt}}}{1-U_{\mathrm{opt}}}, & U> U_{\mathrm{opt}},
\end{cases}$$

with $(U_{\mathrm{opt}}, r_0, s_1, s_2)$ published on-chain per reserve. The slope ratio
$\frac{s_2/(1-U_{\mathrm{opt}})}{s_1/U_{\mathrm{opt}}}$ measures how violent the kink is — this
is the convexity a **supplier is long** and a **borrower is short**.
""")

code(r"""
from optimal_long_short.aave_hourly import published_curve

rows = []
for sym in universe:
    cv = published_curve(configs, sym)
    slope_lo = cv.slope1 / cv.u_opt
    slope_hi = cv.slope2 / (1 - cv.u_opt) if cv.u_opt < 1 else np.nan
    rows.append({
        "symbol": sym, "U_opt": cv.u_opt,
        "r_b(U_opt)": cv.base + cv.slope1,
        "r_b(1.0)": cv.base + cv.slope1 + cv.slope2,
        "slope below": slope_lo, "slope above": slope_hi,
        "kink ratio": slope_hi / slope_lo if slope_lo else np.nan,
    })
pd.DataFrame(rows).set_index("symbol").round(4)
""")

md(r"""
### 2.2 Utilization must be *inverted*, not measured

Because $r_b$ is strictly increasing it is invertible, so **the observed borrow rate is a
sufficient statistic for utilization**. This matters, because the naive subgraph ratio
$U^{\text{raw}} = \texttt{totalBorrowBalanceUSD}/\texttt{totalDepositBalanceUSD}$ is badly
biased — it exceeds 1 for several reserves.

The check below is the decisive one: in the *flat* segment the two measures must agree exactly,
and for the wrapped-BTC reserves they do (correlation $1.0000$). Where the reserve actually
visits its kink, they diverge sharply, and the raw measure misclassifies the regime.

Source: `optimal_long_short/rate_curve.py::implied_utilization`.
""")

code(r"""
from optimal_long_short.rate_curve import implied_utilization

TOKEN = {"weth": "WETH", "usdc": "USDC", "usdt": "USDT", "wbtc": "WBTC",
         "cbbtc": "cbBTC", "usde": "USDe", "wsteth": "wstETH", "dai": "DAI"}
panel, rows = {}, []
for tok, sym in TOKEN.items():
    d = pd.read_parquet(ROOT / f"data/AAVE/hourly_{tok}.parquet")
    panel[sym] = d
    cv = published_curve(configs, sym)
    u_raw = (d.borrowed_USD / d.supplied_USD).clip(0, 1).values
    u_imp = implied_utilization(cv, d.borrower_variable_apr.values / 100.0)
    rows.append({
        "symbol": sym, "U_opt": cv.u_opt,
        "raw mean": np.nanmean(u_raw), "implied mean": np.nanmean(u_imp),
        "raw max": np.nanmax(u_raw), "implied max": np.nanmax(u_imp),
        "corr": np.corrcoef(u_raw, u_imp)[0, 1],
        "raw % stressed": 100 * np.mean(u_raw > cv.u_opt),
        "true % stressed": 100 * np.mean(u_imp > cv.u_opt),
    })
pd.DataFrame(rows).set_index("symbol").round(4)
""")

md(r"""
### 2.3 Why the old CIR/JCIR rate fits failed

`results/cir_usdc_weth_aave_v3.csv` and `results/jcir_usdc_weth_aave_v3.csv` report
$\kappa \approx 2089$ and $1399$ per year — a mean-reversion half-life of about **three hours** —
with $\theta$ pinned at its optimizer bound, and the jump variant violating the Feller condition
($2\kappa\theta = 13.9$ against $\sigma^2 = 72.3$).

The diagnosis is structural, not numerical: fitting a smooth affine mean-reverting process to a
*kinked deterministic transform of a bounded state* manufactures both the apparent jumps (the
kink) and the apparent extreme mean reversion (the flat segment). The fix is to model $U$ on
$[0,1]$ with a **Jacobi (Wright--Fisher) diffusion**

$$dU_t=\kappa(\theta-U_t)\,dt+\sigma\sqrt{U_t(1-U_t)}\,dW_t,$$

whose stationary law is $\mathrm{Beta}(a,b)$ with $a=2\kappa\theta/\sigma^2$,
$b=2\kappa(1-\theta)/\sigma^2$, and whose boundaries are unattainable when
$2\kappa\theta\ge\sigma^2$ and $2\kappa(1-\theta)\ge\sigma^2$. Unlike CIR, it respects $U\le1$.

Source: `optimal_long_short/rate_curve.py::fit_jacobi`.
""")

code(r"""
from optimal_long_short.rate_curve import fit_jacobi, regime_summary

DT = 1 / (365 * 24)
W0, W1 = "2026-03-17", "2026-08-04"      # canonical estimation window
rows = []
for sym in ["WETH", "USDC", "USDT", "USDe", "WBTC"]:
    d = panel[sym].loc[W0:W1]
    cv = published_curve(configs, sym)
    u = implied_utilization(cv, d.borrower_variable_apr.values / 100.0)
    jp = fit_jacobi(u, DT)
    lo, hi = jp.boundaries_unattainable
    rs = regime_summary(u, cv, reserve_factor=float(configs.loc[sym, "reserveFactor"]))
    rows.append({
        "symbol": sym, "kappa": jp.kappa, "theta": jp.theta, "sigma": jp.sigma,
        "half-life (d)": math.log(2) / jp.kappa * 365 if jp.kappa > 0 else np.nan,
        "0 unattainable": lo, "1 unattainable": hi,
        "% stressed": 100 * rs.frac_stressed, "convexity": rs.convexity_ratio,
    })
print("Compare: old CIR gave kappa = 2089/yr (half-life 3 hours), theta at bound.\n")
pd.DataFrame(rows).set_index("symbol").round(4)
""")

# ---------------------------------------------------------------- screen
md(r"""
---
## 3. Pre-registered screen and the pair cross-section

To keep the empirical section free of selection bias the screen is **fixed before any model
output is examined** and selects on liquidity and protocol admissibility only:

1. universe fixed in advance — blue-chip majors and canonical wrappers;
2. not frozen, not paused;
3. listed for the full estimation window;
4. ranked by USD TVL **at the window start** (information available when the window opens);
5. collateral needs $\mathrm{LTV}>0$ and collateral-enabled; debt needs borrowing-enabled.

Every admissible ordered pair is then reported. **No pair is promoted to a headline on the
basis of its numbers.**

Source: `optimal_long_short/screen.py`.
""")

code(r"""
from optimal_long_short.screen import load_panel, apply_screen, pair_cross_section

full_panel = load_panel(ROOT / "data/AAVE")
screen = apply_screen(full_panel, configs)
screen[["tvl_usd_at_start", "ltv", "liq_threshold", "liq_bonus", "u_opt",
        "rank", "in_universe", "admissible_collateral", "admissible_debt"]]
""")

md(r"""
### 3.1 The carry-to-barrier diagnostic

Deterministic carry enters the framework as a per-leg drift shift (Proposition *Carry is an
effective drift*): $\mu_1\mapsto\mu_1+r_s$, $\mu_2\mapsto\mu_2+r_b$, leaving $\sigma$, the jump
law, the phase rates and the admissibility bounds untouched. So carry is **free** to incorporate.
That does not make it **material**. The diagnostic is

$$\mathrm{CB}\;=\;\frac{c\,T}{\sigma_Z\sqrt{T}},\qquad c=r_s-r_b,\qquad
\sigma_Z^2=\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2,$$

the number of barrier standard deviations the carry drift is worth over the horizon.
""")

code(r"""
xs = pair_cross_section(full_panel, configs, screen)
xs.to_csv(ROOT / "results/pair_cross_section.csv", index=False)
print(f"{len(xs)} admissible ordered pairs\n")

view = xs.copy()
view["stable-stable"] = view.sigma_Z < 0.05
print("Most carry-sensitive (|CB| largest):")
cols = ["collateral", "debt", "sigma_Z", "net_carry", "carry_over_barrier",
        "debt_frac_stressed", "debt_convexity"]
display(view.reindex(view.carry_over_barrier.abs().sort_values(ascending=False).index)[cols].head(6).round(4))
print("\nLeast carry-sensitive:")
display(view.reindex(view.carry_over_barrier.abs().sort_values().index)[cols].head(6).round(4))
""")

code(r"""
vol = xs[xs.sigma_Z >= 0.05]
stb = xs[xs.sigma_Z < 0.05]
print(f"Volatile pairs (sigma_Z >= 0.05):  n={len(vol):2d}  "
      f"CB in [{vol.carry_over_barrier.min():+.4f}, {vol.carry_over_barrier.max():+.4f}]")
print(f"Stable-stable  (sigma_Z <  0.05):  n={len(stb):2d}  "
      f"CB in [{stb.carry_over_barrier.min():+.4f}, {stb.carry_over_barrier.max():+.4f}]")
print(f"\nPairs with NEGATIVE net carry: {(xs.net_carry < 0).sum()} of {len(xs)}")
print("\nCollateral-leg supply rates (why the levered long-crypto book pays carry):")
display(xs.groupby("collateral").r_s.first().mul(100).round(3).rename("supply APR %").to_frame())
""")

md(r"""
**Two findings, both cutting against how carry is usually discussed.**

1. *For every volatile pair the carry drift is negligible* — $\mathrm{CB}\in[-0.03,+0.02]$. Carry
   cannot change a sizing decision for such a book, whatever its sign. The exception is exactly
   where theory predicts: stable-versus-stable pairs collapse $\sigma_Z$ to $0.0024$–$0.0043$ and
   reach $|\mathrm{CB}|\approx0.96$, a full barrier standard deviation.
2. *The leveraged long-crypto book pays carry rather than earning it.* Every ETH- and
   BTC-collateral pair has negative net carry: the wrapped-BTC reserves pay $0.00$–$0.01\%$ as
   collateral while debt costs $2.5$–$4.5\%$.

A caution on the stable-stable pairs: at $\sigma_Z=0.0043$ with $h_0^{\min}=0.039$ the barrier
sits roughly **31 diffusive standard deviations** away, so essentially all of their liquidation
risk is carried by the jump component and by depeg events. They look riskless under any Gaussian
reading and are pure tail bets.
""")

# ---------------------------------------------------------------- core
md(r"""
---
## 4. Core: the $3\times3$ killed resolvent

Recap of the base construction (paper Sections 3–5 and Appendix A). Under the $k$-tilted measure
$d\mathbb P^{(k)}/d\mathbb P\propto e^{kX_{2,t}}$, the ratio $Z$ remains Lévy with exponent

$$\psi_Z^{(k)}(s)=\mu_Z^{(k)}s+\tfrac12\sigma_Z^2s^2+\lambda_1\bigl(M_1(s)-1\bigr)
+\lambda_2\bigl(M_2(k-s)-M_2(k)\bigr),$$

and each killed moment reduces to a one-dimensional killed expectation. The forcing
$g_k(x)=(e^x-b)^k=\sum_j c_{k,j}e^{jx}$ expands in exponentials, the generator acts diagonally on
them ($\mathcal L^{(k)}e^{sz}=\psi_Z^{(k)}(s)e^{sz}$), and the resolvent is

$$\widehat U_k(q,x)=\sum_{j=0}^{k}\frac{c_{k,j}}{q-\psi_Z^{(k)}(j)}e^{jx}
+\sum_{m=1}^{3}B_m^{(k)}(q)\,e^{\gamma^{(k)}_{m+3}(q)x}.$$

$\psi_Z^{(k)}$ has **four simple poles**, so clearing denominators gives a degree-6
characteristic polynomial: three decaying roots, matched by three boundary conditions
(absorption plus two downward-phase cancellations). Hence $3\times3$, **independently of $k$**.

Sources: `root_finder.py`, `laplace_resolvent.py`, `moments.py`, `inversion.py`.
""")

code(r"""
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.model.market_params import MarketParams
from optimal_long_short.model.strategy import UnitExposureLongShortStrategy
from optimal_long_short.model.kou_model import KouZTiltedDynamics
from optimal_long_short.laplace.root_finder import CharacteristicRootFinder
from optimal_long_short.model.risk_report import h0_liquidation_moment_report

# The merged paper's benchmark: ECF shape parameters with the window's sample drifts
# (price growth plus Aave rates), built by jobs/build_benchmark_params.py. The ECF drift
# parameters are normalizers, not forecasts, so the drift is kept outside the fit.
CAL = json.load(open(ROOT / "results/params_WBTC_WETH_benchmark.json"))
p, C = CAL["params"], CAL["aave_constraint"]
PARAMS = KouParams(**{k: p[k] for k in
    ["mu1","sigma1","lam1","p1","eta1_pos","eta1_neg",
     "mu2","sigma2","lam2","p2","eta2_pos","eta2_neg","rho"]})
b, T = C["b"], 1/12
print(f"calibration: asset1={CAL['meta']['asset1']} (collateral), "
      f"asset2={CAL['meta']['asset2']} (debt)")
print(f"window {CAL['meta']['date_range'][0][:10]} .. {CAL['meta']['date_range'][1][:10]}, "
      f"n={CAL['meta']['n_obs']}, b={b}, LTV_max={C['ltv_max']}, h0_min={C['h0_min']:.4f}")
sig_z = math.sqrt(p["sigma1"]**2 + p["sigma2"]**2 - 2*p["rho"]*p["sigma1"]*p["sigma2"])
print(f"benchmark drifts mu1={p['mu1']:+.4f}, mu2={p['mu2']:+.4f} (sample, incl. carry); "
      f"spread nu={p['mu1']-p['mu2']:+.4f}/yr")
print(f"sigma_Z = {sig_z:.4f};  barrier scale over T=1/12 is sigma_Z*sqrt(T) = {sig_z*math.sqrt(T):.4f}")

roots = CharacteristicRootFinder(KouZTiltedDynamics(params=PARAMS, k=0)).find(1.0)
print(f"\ndegree-6 roots at q=1: {len(roots.positive)} growing, {len(roots.negative)} decaying")
for g in roots.negative:
    print(f"   {g.real:+.6f} {g.imag:+.6f}j")
""")

code(r"""
H_grid = np.linspace(math.exp(C["h0_min"]), 2.0, 25)
h_grid = np.log(H_grid)
report = pd.DataFrame(h0_liquidation_moment_report(
    PARAMS, h_grid, b=b, T=T, max_moment_order=2))
report[["H0","initial_leverage","p_liq","conditional_mean","conditional_variance"]].iloc[::4].round(5)
""")

# ---------------------------------------------------------------- recovery
md(r"""
---
## 5. Liquidation recovery $R_\tau$

The killed payoff sets wealth to zero after liquidation. That is deliberately severe, and the
repository has always held the data to do better — `liquidationBonus` sits in every calibration
JSON and was used by nothing. Investor wealth is

$$W_T=\Pi_T+R_\tau\mathbf 1_{\{\tau\le T\}}.$$

A liquidator repays debt and seizes collateral worth $(1+\lambda)$ times the repayment. On a
*continuous* crossing $H_\tau=1$, so $V_{2,\tau}=b\,V_{1,\tau}$ and

$$R_\tau=V_{1,\tau}-(1+\lambda)V_{2,\tau}=V_{2,\tau}\,\frac{1-b(1+\lambda)}{b},$$

positive iff $\lambda<1/b-1$ (namely $0.282$ at $b=0.78$). Expressing recovery through the
**debt** leg is what keeps it inside the existing machinery: it makes $R_\tau\propto e^{X_2}$ at
$\tau$, and $X_2$ is the leg already tilted.

Since $M_t=\exp(kX_{2,t}-t\Psi(0,-ik))$ is a martingale and $\tau$ a stopping time, changing
measure on $\mathcal F_\tau$ gives

$$\mathbb E\bigl[e^{kX_{2,\tau}}\mathbf 1_{\{\tau\le T\}}\bigr]
=\mathbb E^{(k)}\bigl[e^{\alpha_k\tau}\mathbf 1_{\{\tau\le T\}}\bigr],\qquad \alpha_k=\Psi(0,-ik),$$

with time-Laplace transform

$$\frac1q\Bigl(1-(q-\alpha_k)\,\widehat p^{\,(k)}_{\mathrm{surv}}(q-\alpha_k)\Bigr). \tag{FP}$$

One extra inversion per moment order; **same $3\times3$ system**. Because $\Pi_T$ and $R_\tau$
live on disjoint events, $\mathbb E[W_T^2]=\mathbb E[\Pi_T^2]+\mathbb E[R_\tau^2\mathbf 1]$.

> **Continuous crossing is an upper bound.** A downward jump can push $H$ strictly below 1,
> leaving less recovery than the formula returns — possibly negative (bad debt). The exact
> residual equity uses the actual leg values at $\tau$,
> $R_\tau=A\,e^{X_{1,\tau}}-B\,e^{X_{2,\tau}}$ with $A=H_0/(H_0-b)$, $B=(1+\lambda)b/(H_0-b)$,
> whose moments are joint hitting moments $\mathbb E[e^{a_1X_{1,\tau}+a_2X_{2,\tau}}\mathbf 1]$
> obtained by the same transform (FP) under a *bivariate* tilt — still a $3\times3$ system.
> This is implemented (paper §7.3) and is what the sizing analysis of Section 8 uses.

Sources: `optimal_long_short/recovery.py` (continuous crossing), `overshoot_recovery.py` (exact),
validated by `jobs/validate_recovery.py` and `tests/test_overshoot_and_bayes.py`.
""")

code(r"""
from jobs.validate_recovery import check_k0, BENCH as RECOVERY_BENCH
from optimal_long_short.recovery import tilt_exponent

print("Check A -- at k=0 the transform (FP) must equal 1 - p_surv from the survival resolvent:")
display(pd.DataFrame(check_k0()).round(11))
print(f"Check A' -- Psi(0,-i) = {tilt_exponent(RECOVERY_BENCH, 1):.10f} "
      f"must equal mu2 = {RECOVERY_BENCH.mu2}  ->  confirms the drift normalisation")
print("\nCheck B (Monte Carlo, k=1) is slow; run `python jobs/validate_recovery.py`.")
""")

code(r"""
from optimal_long_short.recovery import wealth_moments
from optimal_long_short.overshoot_recovery import exact_wealth_moments

lam = C["liq_bonus"]
rows = [exact_wealth_moments(PARAMS, math.log(H), b=b, liq_bonus=lam, T=T)
        for H in (1.068, 1.10, 1.16, 1.20, 1.50, 2.00)]
W = pd.DataFrame(rows)
W["initial_leverage"] = W.H0 / (W.H0 - b)
W["overshoot corr."] = W.E_recovery / W.E_recovery_cc - 1
W = W[["H0", "initial_leverage", "p_liq", "mean_killed", "var_killed",
       "mean_wealth_cc", "mean_wealth", "var_wealth_cc", "var_wealth", "overshoot corr."]]
print(f"liquidation bonus lambda = {lam};  b(1+lambda) = {b*(1+lam):.4f} < 1, so recovery > 0")
print("columns: killed (zero recovery) | _cc continuous crossing | unsuffixed = exact overshoot\n")
W.round(5)
""")

md(r"""
Recovery is quantitatively large: at $H_0=1.16$ it raises the mean of terminal wealth from
$0.927$ (killed) to $0.980$ (exact) and cuts the variance from $0.127$ to $0.054$, by removing the
total-loss atom the killed-payoff convention places at zero. The overshoot correction to recovery
itself is small at the fitted sub-percent jumps (about $-3\%$ at $H_0=1.16$); paper Table 5 shows
it growing to bad debt once jumps are scaled up with their variance. Recovery matters for the
levered buffers; the break-even drift of Section 8 is set at zero debt, where liquidation vanishes.
""")

# ---------------------------------------------------------------- regimes
md(r"""
---
## 6. Regime-switching carry: the $6\times6$ resolvent

Section 3 ruled out the *level* of the rate. What remains is its *shape*. Above $U_{\mathrm{opt}}$
the borrow rate climbs a slope of $0.60$–$3.00$, so a debt leg that normally costs $2.5\%$ can
cost tens of percent while the pool is stressed — and the stressed state is exactly the
deleveraging that also moves $Z$. So let the carry switch.

Let $J_t\in\{1,2\}$ be a Markov chain with generator
$Q=\begin{pmatrix}-q_1&q_1\\q_2&-q_2\end{pmatrix}$ and net carry $c_i$ in state $i$. **Only the
drift switches**, so

$$\psi^i(s)=\psi_Z^{(k)}(s)+c_i\,s,$$

leaving $\sigma_Z$, the jump law, the phase rates $r_{1,-},r_{2,-}$ and the admissibility bounds
$k\eta_{1,+}<1$, $k\eta_{2,+}<1$ untouched. This is the deterministic-carry proposition applied
state by state, and it is what makes the extension cheap.

The backward equations couple through $Q$:

$$(q+q_1-\mathcal L^{1})\widehat U^{1}-q_1\widehat U^{2}=g_k,\qquad
-q_2\widehat U^{1}+(q+q_2-\mathcal L^{2})\widehat U^{2}=g_k.$$

A mode $B\,v(\gamma)e^{\gamma x}$ solves the homogeneous system iff $v\in\ker A(\gamma)$ with

$$A(\gamma)=\begin{pmatrix}q+q_1-\psi^1(\gamma)&-q_1\\-q_2&q+q_2-\psi^2(\gamma)\end{pmatrix},$$

i.e. iff $\Delta(\gamma):=\det A(\gamma)=0$.

### Root count and system size

Both $\psi^i$ share the same four poles, so with $\psi^i=N^i/D$, $\deg D=4$, $\deg N^i=6$:

$$\Delta(s)D(s)^2=\widetilde N^1(s)\widetilde N^2(s)-q_1q_2D(s)^2,
\qquad \widetilde N^i=(q+q_i)D-N^i,$$

**degree 12** — six decaying roots instead of three. The boundary conditions also number six:
absorption in each state, plus two downward-phase cancellations in each state. The coupling
terms $-q_i\widehat U^j$ carry no generator and so contribute **no** zero-extension residual.
Hence exactly $6\times6$, and $q_1=q_2=0$ recovers two decoupled copies of the $3\times3$.

Two implementation points: the kernel vector is taken from the **adjugate** of $A(\gamma)$, since
in the decoupled limit one adjugate column vanishes identically while the other correctly selects
the state a root belongs to; and the non-resonance condition $q\neq\psi_Z^{(k)}(j)$ generalises to
$\Delta(j)\neq0$, because $\det A(j)=\Delta(j)$ is exactly the determinant of the $2\times2$
particular-solution solve.

**Invertibility.** $\det \mathbf M_{\mathrm{bar}}$ is analytic in $(q_1,q_2)$ and factorises at
$q_1=q_2=0$ into the product of the two single-regime barrier determinants, each nonzero. A
non-vanishing analytic function is nonzero off a measure-zero set. This is weaker than a
pointwise theorem and is stated as such.

Sources: `optimal_long_short/regime_switching.py`, validated by `jobs/validate_regime_switching.py`.
""")

code(r"""
from optimal_long_short.regime_switching import (
    RegimeSpec, RegimeSwitchingDynamics, RegimeRootFinder, RegimeResolvent)

spec = RegimeSpec(q1=2.0, q2=25.0, c1=0.02, c2=-0.60)
pi_n, pi_s = spec.stationary
print(f"spec: q1={spec.q1}, q2={spec.q2}, c1={spec.c1:+.2%}, c2={spec.c2:+.2%}")
print(f"stationary: normal {pi_n:.4f}, stressed {pi_s:.4f}")
print(f"stationary mean carry = {spec.mean_carry:+.4%}   <- all a deterministic model can see")

dyn = RegimeSwitchingDynamics(base=KouZTiltedDynamics(params=PARAMS, k=0), spec=spec)
poly = RegimeRootFinder(dyn).polynomial(1.0)
print(f"\ncharacteristic polynomial degree = {len(poly) - 1}  (single regime: 6)")
allroots = np.roots(poly)
print(f"total roots = {len(allroots)}, decaying = {(allroots.real < 0).sum()}")
""")

code(r"""
from jobs.validate_regime_switching import (
    check_constant_carry, check_decoupled, check_conditioning)

print("Test A -- constant carry c1==c2 with q1!=q2 (coupling ACTIVE, economically irrelevant).")
print("          Both states must reproduce the single-regime 3x3 answer.")
display(pd.DataFrame(check_constant_carry()))

print("Test B -- decoupled limit q1=q2=0, c1!=c2. Each state matches its own single-regime answer.")
display(pd.DataFrame(check_decoupled()))

print("Conditioning of the 6x6 barrier matrix (no blow-up):")
display(pd.DataFrame(check_conditioning()))
""")

code(r"""
from optimal_long_short.laplace.inversion import TalbotInverter

talbot = TalbotInverter(M=32)
BENCH_RS = KouParams(mu1=0.05, sigma1=0.20, lam1=1.0, p1=0.5, eta1_pos=0.10, eta1_neg=0.125,
                     mu2=0.03, sigma2=0.15, lam2=0.8, p2=0.5, eta2_pos=1/12, eta2_neg=1/9,
                     rho=0.3)
dyn_rs = RegimeSwitchingDynamics(base=KouZTiltedDynamics(params=BENCH_RS, k=0), spec=spec)

rows = []
for h0 in (0.10, 0.25, 0.50):
    R = RegimeResolvent(dyn_rs, h0=h0, b=0.78)
    surv = [talbot.invert(lambda q, i=i: R.transform(q)[i], 0.25) for i in (0, 1)]
    rows.append({"h0": h0, "open normal": surv[0], "open stressed": surv[1],
                 "difference (pp)": 100 * (surv[1] - surv[0])})
print("Survival probability by OPENING REGIME (T = 1/4):")
pd.DataFrame(rows).set_index("h0").round(4)
""")

md(r"""
With this **illustrative** specification (a stressed carry of $-60\%$, a toy parameter set and
a three-month horizon) the opening regime moves survival by about ten percentage points, which a
deterministic model using the stationary mean carry cannot represent at all. That shows what the
mechanism *can* do; the next cell shows what it *does* on the calibrated book.

Monte Carlo validation with switching active is slow; run
`python jobs/validate_regime_switching.py`. Its residual is discretisation bias and falls from
$+0.0113$ at $10^3$ steps to $+0.0004$ at $4\times10^3$.
""")

code(r"""
RC = json.load(open(ROOT / "results/merged/regime_calibrated.json"))  # benchmark drift
print(f"calibrated from WETH utilization at U_opt={RC['U_opt']}: stressed {RC['frac_stressed']:.1%}, "
      f"mean spell {RC['mean_dur_hours']:.0f} h")
print(f"q1={RC['q1']:.1f}, q2={RC['q2']:.1f} /yr;  c1={RC['c1']:+.2%}, c2={RC['c2']:+.2%};  "
      f"stationary mean carry {RC['mean_carry']:+.2%}\n")
print("Survival at T = 1/12 on the benchmark WBTC/WETH dynamics (paper Table 13):")
pd.DataFrame(RC["rows"]).set_index("h0").round(4)
""")

md(r"""
**Calibrated, the regime channel is inert at a monthly horizon.** The deterministic mean carry
lowers survival by about $0.8$ pp near the barrier; the *opening regime* adds at most $0.12$ pp and
the convexity correction under $0.05$ pp. Stressed spells of a few days wash out over a month,
and a $4\%$ stressed borrow rate barely moves the health-factor drift.
""")

# ---------------------------------------------------------------- barbell
md(r"""
---
## 7. The barbell: long the two convexities the levered book is short

The cross-section says the leveraged long-short book collects **no** premium (negative net carry)
while holding **two** short-convexity positions:

1. **short the barrier** — the payoff is killed at liquidation;
2. **short the utilization kink** — the borrower pays $r_b(U)$, convex in $U$.

The barbell inverts both. The safe leg is the $h_0\to\infty$ limit, where the payoff converges to
a unit long and the barrier becomes unreachable. The convex leg buys:

**Supplier leg.** A pure supplier (no borrow, hence no barrier) earns
$(1-\text{rf})\,U\,r_b(U)$, convex in $U$. Convexity is paid by Jensen:

$$\underbrace{\mathbb E\bigl[(1-\text{rf})\,U\,r_b(U)\bigr]}_{\text{actual}}
\;>\;\underbrace{(1-\text{rf})\,\mathbb E[U]\,r_b(\mathbb E[U])}_{\text{yield at mean }U},$$

and the gap is the value of the kink — invisible to any deterministic-rate model.

**Keeper leg.** The liquidator collects the bonus on seized collateral, zero except on
$\{\tau\le T\}$. This is **the same transform (FP)** that recovery uses: what the borrower
forfeits is what the keeper collects. The two legs are priced by one object with opposite signs.

Both legs are on-protocol; no options market required.

Source: `optimal_long_short/barbell.py`.
""")

code(r"""
from optimal_long_short.barbell import supplier_leg, keeper_leg

rows = []
for sym in ["USDC", "USDT", "USDe", "WETH"]:
    d = panel[sym].loc[W0:W1]
    cv = published_curve(configs, sym)
    rf = float(configs.loc[sym, "reserveFactor"])
    u = implied_utilization(cv, d.borrower_variable_apr.values / 100.0)
    s = supplier_leg(cv, fit_jacobi(u, DT), rf)
    rows.append({"symbol": sym, "E[yield]": s.expected_yield,
                 "yield at E[U]": s.yield_at_mean, "Jensen gap": s.jensen_gap,
                 "normal": s.yield_normal, "stressed": s.yield_stressed,
                 "convexity x": s.convexity_ratio})
sup = pd.DataFrame(rows).set_index("symbol")
print("CONVEX LEG 1 -- supplier long the utilization kink:\n")
display(sup.style.format({c: "{:.4%}" for c in sup.columns if c != "convexity x"} |
                         {"convexity x": "{:.2f}"}))
""")

code(r"""
rows = []
for H0 in (1.068, 1.10, 1.20, 1.50, 2.00):
    h = math.log(H0)
    k = keeper_leg(PARAMS, h, b=b, liq_bonus=lam, T=T, close_factor=0.5)
    w = wealth_moments(PARAMS, h, b=b, liq_bonus=lam, T=T)
    rows.append({"H0": H0, "L0": w["initial_leverage"], "p_liq": w["p_liq"],
                 "keeper revenue": k["expected_revenue"],
                 "borrower loses": 1 - w["mean_wealth"],
                 "keeper share": k["expected_revenue"] / max(1 - w["mean_wealth"], 1e-12)})
print("CONVEX LEG 2 -- keeper long the barrier (close factor 0.5),")
print("per unit of the BORROWER's initial equity:\n")
pd.DataFrame(rows).set_index("H0").round(5)
""")

md(r"""
> **Normalisation caveat.** The keeper leg is quoted per unit of the *borrower's* initial equity,
> not as a return on the keeper's own capital — with flash loans that capital is near zero, so the
> denominator is ambiguous. The two legs are therefore **not** on a common denominator, and no
> single combined "barbell return" is reported here. Doing so would be misleading.
""")

# ---------------------------------------------------------------- sizing
md(r"""
---
## 8. Sizing on the map

This section follows paper §7.4–§7.5 and §8. The evaluation map is exact given its inputs; the
sizing question is what those inputs, and the realised data, can support. Four steps: the
overshoot law and bad debt; the expected log of wealth (Kelly); the closed-form **break-even
drift** above which leverage pays; and a **model-free realised backtest** of whether it did.

### 8.1 The overshoot law and the probability of bad debt

At liquidation the log health factor lands at $h_\tau=-u\le0$. The overshoot $u$ is a creeping
atom plus two exponential phases (paper Proposition 7.1):

$$\mathbb E\bigl[f(u)\mathbf 1_{\{\tau\le T\}}\bigr]=P_0f(0)+\sum_{i=1}^2P_i\int_0^\infty f(u)\,r_ie^{-r_iu}\,du,
\qquad r_1=\tfrac1{\eta_{1,-}},\ r_2=\tfrac1{\eta_{2,+}},$$

with weights from three hitting moments, $e^{jh_0}\mathrm{Hit}(j,-j)=P_0+P_1\frac{r_1}{r_1+j}+P_2\frac{r_2}{r_2+j}$,
$j=0,1,2$. Recovery is non-positive once $u\ge u^\star=-\log(b(1+\ell))$, so the bad-debt
probability is $P_1e^{-r_1u^\star}+P_2e^{-r_2u^\star}$ in closed form. The cell checks the
mixture against the hitting moments at $j=5,10$, which it was not fitted to.
""")

code(r"""
from optimal_long_short.kelly import overshoot_law, expected_log_wealth

u_star = -math.log(b * (1 + lam))
rows = []
for H in (1.07, 1.16, 1.30):
    law = overshoot_law(PARAMS, math.log(H), T)
    rows.append({"H0": H, "p_liq": law.p_liq, "P0 creep": law.p_creep,
                 "P1 (X1 down-jump)": law.p_jump[0], "P2 (X2 up-jump)": law.p_jump[1],
                 "P(bad debt)": law.tail(u_star), "check error j=5,10": law.max_check_error})
print(f"u* = -log(b(1+l)) = {u_star:.3f}")
pd.DataFrame(rows).set_index("H0")
""")

md(r"""
### 8.2 Expected log wealth

Per unit equity, wealth is $e^{X_{2,T}}(e^{h_T}-b)/(H_0-b)$ on survival and
$e^{X_{2,\tau}}(e^{-u}-c)/(H_0-b)$, $c=b(1+\ell)$, on liquidation. With
$\log(e^h-b)=h-\sum_{n\ge1}(b^n/n)e^{-nh}$ for $h\ge0$, $\mathbb E[\log W_T]$ is a convergent
series of killed joint moments $S(-n,n)$ plus a one-dimensional integral against the overshoot law
(paper eq. 40). Strict Kelly is $-\infty$ whenever bad debt has positive probability, so bad-debt
paths are floored at $10^{-4}$ (paper §7.5). As $h_0\to\infty$ the value must tend to
$\mathbb E[X_{1,T}]$.
""")

code(r"""
import time
rows = []
for H in (1.07, 1.16, 1.30, 2.0, 1e3):
    t0 = time.time()
    r = expected_log_wealth(PARAMS, math.log(H), b=b, liq_bonus=lam, T=T)
    rows.append({"H0": H, "L0": H / (H - b), "E[log W_T]": r["E_log_W"],
                 "seconds": time.time() - t0})
EX1 = (PARAMS.mu1 - 0.5 * PARAMS.sigma1**2 - PARAMS.lam1 * (
    PARAMS.p1 / (1 - PARAMS.eta1_pos) + (1 - PARAMS.p1) / (1 + PARAMS.eta1_neg) - 1)
    + PARAMS.lam1 * (PARAMS.p1 * PARAMS.eta1_pos - (1 - PARAMS.p1) * PARAMS.eta1_neg)) * T
print(f"unlevered limit E[X_1,T] = {EX1:.6f}")
pd.DataFrame(rows).set_index("H0").round(6)
""")

md(r"""
### 8.3 The break-even drift

Write $x=n_2S_{2,0}$ for debt per unit of equity ($L_0=1+x$). As $x\to0$ liquidation vanishes and
$W_T=(1+x)e^{X_{1,T}}-xe^{X_{2,T}}$ to first order, so (paper eq. 42)

$$\frac{d\Phi}{dx}\Big|_{0}=\bigl(\mathbb Ee^{X_{1,T}}-\mathbb Ee^{X_{2,T}}\bigr)-\alpha\,\mathrm{Cov}\bigl(e^{X_{1,T}},e^{X_{1,T}}-e^{X_{2,T}}\bigr),
\qquad
\frac{d\,\mathbb E\log W_T}{dx}\Big|_{0}=1-\mathbb Ee^{X_{2,T}-X_{1,T}}.$$

The **break-even drift view** $\nu^\star$ is the spread drift $\nu=\mu_1-\mu_2$ (price growth plus
net carry) at which the derivative vanishes. For Kelly it is $\Psi_{\mathrm M}(-1,1)=0$, which does
not depend on $T$ and, with independent jumps, is closed-form (paper eq. 43):

$$\nu^\star_{\mathrm{Kelly}}=\underbrace{\sigma_1^2-\rho\sigma_1\sigma_2}_{\text{diffusion}}
+\underbrace{\lambda_1\bigl[M_1(1)+M_1(-1)-2\bigr]}_{\text{WBTC jumps}}.$$

The diffusion part is negative because WBTC is the less volatile leg ($\sigma_1<\rho\sigma_2$); the
break-even is positive only through WBTC's own jumps. The cell recomputes the headline values and
reads all three calibrations from `results/breakeven_drift.json` (written by
`jobs/breakeven_drift.py`, which also checks the switch on a 30-point grid over the full buffer
range).
""")

code(r"""
from jobs.breakeven_drift import kelly_closed_form, breakeven

BE = json.load(open(ROOT / "results/breakeven_drift.json"))
mu2_head = BE["drift_estimates"]["headline_4h"]["mu2"]
live = {str(o): breakeven(PARAMS, o, mu2_head) for o in (0.5, 1.0, 2.0, "kelly")}
print("headline, recomputed now:", {k: round(v, 4) for k, v in live.items()})
print("headline, closed form   :", {k: round(v, 4) for k, v in kelly_closed_form(PARAMS).items()}, "\n")
tab = pd.DataFrame({name: {**{f"nu* {k}": v for k, v in c["nu_star"].items()},
                           "Kelly diffusion part": c["kelly_closed_form"]["diffusion_part"],
                           "Kelly jump part": c["kelly_closed_form"]["jump_part"],
                           "reversed book Kelly nu*": c["kelly_closed_form_reversed_book"]["nu_star"],
                           "grid confirms switch": all(c["grid_check"].values())}
                    for name, c in BE["calibrations"].items()})
tab
""")

md(r"""
### 8.4 What the data say about the drift

The ECF $\mu_i$ of the calibration are **normalizers fitted to de-meaned returns**, not drift
estimates (their difference, $-0.86$, even has the wrong sign); this notebook's benchmark therefore
already uses the window's sample drift (Section 4). The drift has to come from sample
means: $\nu=\nu_{\mathrm{price}}+c$, with the net carry $c=r_s-r_b$ observed on-chain. The bootstrap
below shows how loosely the ECF pins the normalizers; the sample drift itself has a Newey--West
standard error of $0.34$ (headline window) and $0.19$ (3.7-year daily history).
""")

code(r"""
BS = json.load(open(ROOT / "results/bootstrap_se.json"))
print(f"block bootstrap: B={BS['n_boot_kept']}, n_starts={BS['n_starts']}, "
      f"lambda bound={BS['lam_bound']:.0f}, resamples at the bound: {BS['frac_railed']:.0%}\n")
bs = pd.DataFrame({"point": BS["point"], "bootstrap SE": BS["se"]})
bs["95% CI"] = [f"[{lo:+.4g}, {hi:+.4g}]" for lo, hi in (BS["ci"][k] for k in bs.index)]
bs.loc["mu1 - mu2 (ECF normalizer)"] = [BS["spread"]["point"], BS["spread"]["se"],
                       f"[{BS['spread']['ci'][0]:+.3f}, {BS['spread']['ci'][1]:+.3f}]"]
bs.round(4)
""")

code(r"""
D = pd.DataFrame(BE["drift_estimates"]).T[["nu_price", "carry", "nu", "se_nw"]]
print("sample spread drift nu = price growth + carry, per year:")
display(D.round(4))
rows = []
for name, c in BE["calibrations"].items():
    rule = c["bayes_kelly_closed_rule"]
    rows.append({"calibration": name, "nu_hat": c["nu_hat_used"],
                 **{f"L0* at nu_hat ({k})": v for k, v in c["optimal_leverage_at_nu_hat"].items()},
                 "P(lever pays) carry prior, Kelly": c["P_leverage_pays"]["kelly"],
                 "P(lever pays) flat prior, Kelly": c["P_leverage_pays_flat_prior"]["kelly"],
                 "posterior mean m": rule["posterior_mean"],
                 "Bayes-Kelly threshold nu*+Ts^2/2": rule["threshold_on_posterior_mean"],
                 "Bayes-Kelly levers": rule["levers"]})
pd.DataFrame(rows).set_index("calibration").round(3).T
""")

md(r"""
The sample drift ($\approx+0.10$ in both windows) lies above every headline break-even and most
full-history ones, so at the point estimate moderate leverage is optimal. But its standard error is
several times the gap: separating $+0.10$ from the full-history Kelly break-even $+0.05$ at two
standard errors needs a standard error of $0.025$, about two centuries of data. The posterior
probability that leverage pays is about one half under a carry-anchored prior
$\nu\sim\mathcal N(c,0.20^2)$ and about $0.6$ under a flat one. Bayes--Kelly has an exact rule at the
margin, lever iff $m>\nu^\star_{\mathrm{Kelly}}+\tfrac12Ts^2$; the variance term is about $0.001$, so
the unlevered choice under the carry prior reflects the prior's **shrinkage toward zero relative
drift**, not estimation risk. Under the flat prior it levers.

### 8.5 Did leverage pay? A realised backtest

`jobs/realised_backtest.py` replays all 37 cross-section pairs hour by hour on the full Aave v3
history (`data/AAVE_full`, January 2023 to October 2026): oracle-implied prices, the collateral's
supply rate, the debt's borrow rate, and Aave's liquidation rules (50% of the debt repaid, 100% at
$HF\le0.95$, the reserve's bonus). Books are reset to $L_0$ every 1–365 days or held; the benchmark
holds the collateral unlevered. No model is involved.
""")

code(r"""
RB = pd.read_csv(ROOT / "results/realised_backtest_pairs.csv")
RB["horizon"] = RB.horizon.astype(str); RB["L0"] = RB.L0.astype(str)
base = RB[(RB["mode"] == "aave") & (RB.cost_bps == 0)]
cols = ["1.25", "1.5", "2.0", "3.0"]
med = base.pivot_table(index="horizon", columns="L0", values="excess", aggfunc="median")
pos = base.assign(w=base.excess > 0).pivot_table(index="horizon", columns="L0", values="w", aggfunc="sum")
order = ["1", "7", "30", "90", "365", "hold"]
print("median excess log growth over unlevered, per year, across 37 pairs (paper Table 8):")
display(med.loc[order, cols].round(3))
print("number of pairs in which leverage paid:")
display(pos.loc[order, cols].astype(int))
print(f"combinations with t > 2: {(base.t > 2).sum()},  t < -2: {(base.t < -2).sum()}  of {len(base)}")
""")

code(r"""
RBJ = json.load(open(ROOT / "results/realised_backtest.json"))
years = pd.DataFrame({f"{q['collateral']}/{q['debt']}": {y: v["excess"] for y, v in q["by_year_L2_30d"].items()}
                      for q in RBJ["pairs"] if q["group"] == "cross-asset"}).T
print("cross-asset books, L0=2, 30-day rolls: excess growth by calendar year")
print("positive pair-years:", {y: f"{(years[y] > 0).sum()}/{years[y].notna().sum()}" for y in years})
rbe = pd.DataFrame([{"pair": f"{q['collateral']}/{q['debt']}", **q["realised_breakeven"]} for q in RBJ["pairs"]]).set_index("pair")
paid = base[(base.L0 == "1.25") & (base.horizon == "1")].set_index(base[(base.L0 == "1.25") & (base.horizon == "1")].collateral + "/" + base[(base.L0 == "1.25") & (base.horizon == "1")].debt).excess
print(f"\nrealised break-even rule (nu_hat > nu*_hat) vs whether 1.25x daily leverage paid: "
      f"{(np.sign(rbe.margin) == np.sign(paid.reindex(rbe.index))).sum()}/37 agree "
      "(a first-order identity, not independent evidence)")
rbe[["nu_hat", "nu_star_hat", "margin", "se_margin_nw"]].sort_values("margin", ascending=False).head(8).round(3)
""")

md(r"""
Leverage rarely paid: at $L_0=1.25$ it beat the unlevered book in 4 of 37 pairs (all long BTC, in a
window where BTC rose 3.6-fold and more than doubled against ETH), never significantly, and at
$2\times$ no cross-asset pair-year in 2025 was positive. The window is one cycle --- a bull market to
the 2025 peaks, then $-53\%$ (BTC) and $-68\%$ (ETH) to mid-2026 --- so this describes a history,
not a law.

**The realised break-even exposes the model.** For WBTC/WETH the realised
$\hat\nu^\star=\mathrm{Var}(X_1)-\mathrm{Cov}(X_1,X_2)$ is negative, against the model's $+0.048$.
The model's jumps are independent, so it cannot reproduce the legs' common jumps:
""")

code(r"""
from jobs.realised_backtest import load_pair

HZ = json.load(open(ROOT / "results/horizon_analysis.json"))["calibrations"]["daily_full"]
def model_moments(q):
    j2 = lambda pp, e_up, e_dn: pp * 2 * e_up**2 + (1 - pp) * 2 * e_dn**2
    v1 = q["sigma1"]**2 + q["lam1"] * j2(q["p1"], q["eta1_pos"], q["eta1_neg"])
    v2 = q["sigma2"]**2 + q["lam2"] * j2(q["p2"], q["eta2_pos"], q["eta2_neg"])
    cv = q["rho"] * q["sigma1"] * q["sigma2"]
    return {"var WBTC": v1, "var WETH": v2, "cov": cv, "corr": cv / math.sqrt(v1 * v2),
            "spread var": v1 + v2 - 2 * cv, "var1 - cov (Kelly nu*)": v1 - cv}
d = np.log(load_pair("WBTC", "WETH")[["S1", "S2"]].resample("1D").last()).diff().dropna()
v1, v2, cv = d.S1.var() * 365, d.S2.var() * 365, d.cov().iloc[0, 1] * 365
real = {"var WBTC": v1, "var WETH": v2, "cov": cv, "corr": cv / math.sqrt(v1 * v2),
        "spread var": v1 + v2 - 2 * cv, "var1 - cov (Kelly nu*)": v1 - cv}
big = d[(d.abs() > 3 * d.std()).any(axis=1)]
co = ((np.sign(big.S1) == np.sign(big.S2)) & (big.abs() > 1.5 * d.std()).all(axis=1)).mean()
print(f"days with a 3-sd move in either leg: {len(big)}; share where both legs move >1.5 sd the same way: {co:.0%}")
pd.DataFrame({"model (full-history daily fit)": model_moments(HZ), "realised (daily)": real}).round(3)
""")

md(r"""
With only Brownian correlation to carry co-movement, the fit understates both legs' variances and,
by much more, their covariance, so the spread variance is overstated by about 60% and the
break-even turns positive. The same overstatement pushes the model's liquidation probabilities
above realised frequencies out of sample:
""")

code(r"""
RO = json.load(open(ROOT / "results/rolling_backtest.json"))["summary"]
BI = json.load(open(ROOT / "results/rolling_backtest_bias.json"))
rows = []
for key, v in RO.items():
    for h in ("0.05", "0.1", "0.15"):
        rows.append({"predictor": key, "h0": h, "mean predicted": v[h]["mean_pred"],
                     "realised": v[h]["realised_freq"], "Brier": v[h]["brier"],
                     "Brier base rate": v[h]["brier_climatology"],
                     "bias 95% CI (block bootstrap)": "[{:+.3f}, {:+.3f}]".format(*BI[key][h]["block_bootstrap_ci95"])})
print("33 out-of-sample months, WBTC/WETH (paper Table 11)")
pd.DataFrame(rows).set_index(["predictor", "h0"]).round(3)
""")

md(r"""
**What survives.** The evaluation map is exact and fast given its inputs, so it is the right tool
for evaluating a buffer under a stated drift view or stress scenario. It is not a tool for
calibrating one: the drift that decides whether to lever cannot be estimated, leverage rarely paid
ex post, and the model's independent-jump restriction biases its break-evens and liquidation
probabilities. A common jump component is the most consequential extension (paper §8.4); a
decision-theoretic treatment of sizing is outlined in paper §8.5.
""")


# ---------------------------------------------------------------- sources
md(r"""
---
## 9. Source index

### Package modules (`optimal_long_short/`)

| File | Contents |
|---|---|
| `aave_hourly.py` | Subgraph fetcher, `RateCurve`, `published_curve`, reserve configs |
| `screen.py` | `ScreenConfig`, `apply_screen`, `pair_cross_section` |
| `rate_curve.py` | `implied_utilization`, `fit_rate_curve`, `JacobiParams`, `fit_jacobi`, `regime_summary` |
| `model_params.py` | `KouParams` and the drift normalisation $\mathbb E[e^{X_i(t)}]=e^{\mu_i t}$ |
| `kou_model.py` | `BivariateKouModel`, `KouZTiltedDynamics` |
| `root_finder.py` | `CharacteristicRootFinder` (degree-6), `SixRoots` |
| `laplace_resolvent.py` | `ParticularSolution`, `HomogeneousSolution`, `GeneralSolution` ($3\times3$) |
| `moments.py` | `ConditionalMoments`, killed and conditional moments |
| `inversion.py` | `TalbotInverter`, `StehfestInverter`, `DeHoogInverter` |
| `recovery.py` | `hitting_moment_transform` (FP), `wealth_moments` (continuous crossing), `recovery_coefficient` |
| `overshoot_recovery.py` | Bivariate tilt, `joint_hitting_moment`, `exact_wealth_moments` (exact overshoot) |
| `kelly.py` | `killed_joint_moment`, `overshoot_law`, `expected_log_wealth` |
| `drift_bayes.py` | `_BufferGrid` (exact recovery by default), `optimal_buffer`, `sizing_distribution` |
| `full_history.py` | `hourly_leg`, `pair_panel` (stale-aware), `variance_ratios` |
| `regime_switching.py` | `RegimeSpec`, `RegimeSwitchingDynamics`, `RegimeRootFinder` (degree-12), `RegimeResolvent` ($6\times6$) |
| `barbell.py` | `supplier_leg`, `keeper_leg`, `SupplierLeg`, `BarbellResult` |
| `carry.py` | Deterministic carry as a drift view; depeg scenario overlay |
| `drift.py` | `with_price_drift_view`, `with_muX_drift_view` |
| `monte_carlo.py`, `simulation.py` | Simulation benchmarks |
| `calibration/` | Penalised ECF estimator, standardised frequency grid, threshold initialiser, diagnostics |

### Entry points (`jobs/`)

`fetch_universe.py` · `build_merged_4h.py` · `calibrate_btc_eth.py --merged` ·
`numerical_comparison.py` · `mc_convergence.py` · `sensitivity_analysis.py` ·
`overshoot_table.py` · `regime_calibration.py` · `breakeven_drift.py` · `kelly_sizing.py` ·
`bootstrap_se.py` · `model_comparison.py` · `horizon_analysis.py` · `rolling_backtest.py` ·
`backtest_bias.py` · `realised_backtest.py` · `fig_realised_backtest.py` · `validate_recovery.py` ·
`validate_regime_switching.py` · `drift_bayes.py` (legacy capped grid) · `recalibrate_extended.py`

### Data (`data/AAVE/`)

* `hourly_<token>.parquet` — 12 reserves × 14,321 hours, 2025-01-01 to 2026-08-20
* `merged_wbtc_weth_4h.parquet` — the headline 840-return panel, rebuilt by `jobs/build_merged_4h.py`
* `reserve_configs_v3_eth.csv` — LTV, liquidation threshold and bonus, reserve factor, and the
  published rate-curve parameters $(U_{\mathrm{opt}},r_0,s_1,s_2)$
* `realized_rates_aug2026.json` — trailing-mean APRs

`data/AAVE_full/` holds the full Aave v3 history (2023-01-27 to 2026-10-13) for the eight
screened reserves, with an `n_snapshots` staleness column; fetched by
`jobs/fetch_universe.py --start 2023-01-27 --out-dir data/AAVE_full`.

Subgraphs: hourly history `JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk` (Messari schema);
reserve configuration `Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g` (official Aave v3).

### Results (`results/`)

* `pair_cross_section.csv` — the 37 admissible ordered pairs
* `params_WBTC_WETH_headline.json` — canonical calibration (penalised ECF, 840 four-hour returns)
* `bootstrap_se.json` — block bootstrap under the headline bounds (`bootstrap_se_lam500.json`: the
  superseded run under the tighter $\lambda\le500$ default)
* `params_WBTC_WETH_benchmark.json` — the merged paper's benchmark (ECF shape, sample drifts)
* `decision_rules.json` — liquidation-budget selections (Table 6)
* `merged/` — Sections 6–7 outputs at the benchmark (`jobs/run_merged_benchmark.sh`)
* `breakeven_drift.json` — break-even drifts, sample drifts, posteriors, Bayes--Kelly rule (Table 7, Figure 2)
* `realised_backtest.json`, `realised_backtest_pairs.csv` — the 37-pair realised backtest (Table 8, Figure 3)
* `horizon_analysis.json`, `rolling_backtest.json`, `rolling_backtest_bias.json` — full-history
  calibrations and the out-of-sample liquidation backtest (Tables 11--12)
* `regime_calibrated.json`, `model_comparison.json` — regime table and Appendix B.2 (the
  `merged/` copy of the regime file is at the benchmark)
* `task4_numbers.json` — **superseded** capped-grid thresholds (kept for reference only)
* `cir_usdc_weth_aave_v3.csv`, `jcir_usdc_weth_aave_v3.csv` — **superseded**; retained only as
  the documented failure case motivating the Jacobi model (Section 2.3)

### Paper (`latex/optimal_long_short_merged.tex`, 37pp; slides `optimal_long_short_merged_slides.tex`)

The merged paper combines the coauthor's version (Sections 2–5, Appendix A) with this
repository's recovery, sizing and empirical work.

| Section | Contents |
|---|---|
| 2 | Position, health factor, feasible set; the evaluation map, eq. (7) |
| 3–5 | Kou dynamics and tilting, moment reduction, $3\times3$ resolvent |
| 6 | Benchmark (drift kept outside the fit, Table 2); validation vs Monte Carlo (Tables 3–4); sensitivity (Figure 1) |
| 7 | **Liquidation recovery**: 7.3 exact jump overshoot (Table 5), 7.4 overshoot law and bad debt, 7.5 expected log wealth |
| 8.1 | Decision rules on the map, incl. liquidation budget and fractional Kelly (Table 6) |
| 8.2–8.3 | **Break-even drift** (Table 7, Figure 2), sample drifts, posteriors, Bayes--Kelly |
| 8.4 | **Realised 37-pair backtest** and the independent-jump finding (Table 8, Figure 3) |
| 8.5 | Decision-theoretic extension |
| App. A | Killing at the barrier, downward-phase conditions, coincident rates ($2\times2$) |
| App. B | Penalised ECF estimator; bootstrap (Table 9) |
| App. C | Full history and out-of-sample liquidation backtest (Tables 10–11) |
| App. D | Carry as a drift (Prop. D.1), cross-section (Table 12), regime switching (Table 13, Prop. D.2), depeg |
| App. E | Proofs |

### Related notebooks

`01_health_factor_and_leverage` · `02_survival_and_integer_moments` ·
`03_sizing_frontier_and_drift_views` · `04_kou_calibration_diagnostics` ·
`06_tutorial_from_scratch` (first-principles derivations of everything here)

### Open items

* Figures for the cross-section and the regime results are not yet generated;
  `fig_carry_frontier.pdf` is unreferenced, and `fig_carry_depeg.pdf` is used only by the slides.
* The benchmark spec now lives at `strategies/BENCHMARK_BRIEF.md`, with the strategy notebooks.
* Common jumps: the model's independent jumps understate the legs' covariance (Section 8.5);
  a common Poisson component is the most consequential model extension.
* Regime is exogenous; endogenising utilization is left open.
* A market-impact channel that makes the liquidation bonus endogenous in the overshoot regime.
""")


def main() -> None:
    cells = []
    for kind, text in CELLS:
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "id": f"cell-{len(cells):03d}", "metadata": {},
                          "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "id": f"cell-{len(cells):03d}", "execution_count": None,
                          "metadata": {}, "outputs": [],
                          "source": text.splitlines(keepends=True)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / "05_integrated_framework.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
