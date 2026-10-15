# Revision Report — Round 2

**Manuscript:** `latex/optimal_long_short.tex`.
**Compiles:** `pdflatex` twice, clean, **no undefined references** (Glasserman2004 resolved from
`finance.bib`; the three round-1 additions resolve from `additions.bib`).
**`\TODO` count: 0** — every [CODE] task was run on the real Aave data.
**Length:** **45 pp** (≈11.8k words of body text by `texcount`), up from 43 pp at the start of the
round: the regime demotion and repetition trims (C6) were outweighed by the two new tables the
referee requested (out-of-sample model comparison and the barrier-crossing diagnostic
`tab:barrier`) plus the spread-drift bootstrap row. The length target and the add-evidence tasks
pull in opposite directions; flagged for the author.

All round-1 improvements were preserved: unconditional-wealth objective, bootstrap SEs,
bridge-corrected MC benchmark, variance-matched overshoot design, calibrated regime model,
extended-sample recalibration, full cross-section appendix, replication section, barbell removal.

---

## Group A — Correctness

**A1 Reconcile the two survival probabilities at h0=0.07 — DONE [CODE].**
Root cause: `tab:results` labelled its first row `h0=0.07` but the value `0.3093` had been computed
at the true feasible minimum `h0_min≈0.066`. At *exactly* `h0=0.07` the Laplace value is `0.33029`,
which is what `tab:mcconv` already used. The `tab:results` first row was regenerated at `h0=0.07`
(Laplace `0.3303`, naive MC `0.3421`, mean `1.1035`, var `0.0322`, skew `0.6535`, kurt `0.4790`) and
`tab:diff` updated to match (Δp_surv `-0.0118`, ΔMean `+0.0043`, ΔVar `-0.0002`, ΔSkew `-0.0112`,
ΔKurt `-0.0509`). **Step convention:** both simulators use steps *over the one-month horizon*
(`dt=T/n_steps`); the captions of `tab:results`, `tab:diff` and `tab:mcconv` now say so, and the
erroneous "252 steps yr⁻¹ / ≈21 steps" wording in the MC-benchmark section was removed. A single
Laplace value `p_surv(0.07)=0.3303` now appears everywhere; the `≈20 ms`, `≈440×`, `≈3000×` and
`+0.011` claims were re-checked against `results/mc_convergence.json` and are unchanged. Script:
`jobs/mc_convergence.py`; regeneration in `results/a1_results_row.json`.

**A2 Remove double-counted carry in the regime table — DONE [CODE].**
The benchmark drifts already fold in the APRs, yet the old `tab:regime` labelled that APR-adjusted
value the "no-carry single-regime" column and then added regime carry on top. Default convention
applied: all regime runs now start from **price-only** drifts (APRs removed) and add the net carry
back. Columns relabelled **price-only / deterministic mean carry / open normal / open stressed**.
Reran `jobs/regime_calibration.py`. Results (h0=0.10): price-only `0.4957`, det. mean carry
`0.4879` (this reproduces the old "single-regime" number exactly, confirming the diagnosis),
open normal `0.4882`, open stressed `0.4870`. So the **deterministic mean carry costs −0.78 pp**,
the **opening regime only −0.12 pp**, and the convexity correction **< 0.05 pp**. A sentence now
notes that this −0.78 pp survival cost, invisible to the carry-to-barrier ratio CB, shows CB
understates carry's influence near the barrier. The Bayesian prior's no-double-counting argument
was checked and remains consistent (it concerns the APR-adjusted *sizing* drift, a separate object).

**A3 Rerun the bootstrap with the headline estimator — DONE [CODE].**
`jobs/bootstrap_se.py` now recalibrates each resample with the **same multi-start ECF optimiser**
as the headline fit (n_starts=8), not a single warm start. Results (B=200): **41% of resamples rail
to the λ=500 bound**; bootstrap **SE(μ̂1−μ̂2) = 1.06** (point −0.858), CI [−2.67, 1.38], with
**drift correlation +0.45**. `tab:bootstrap` gained a `μ1−μ2` row, real λ SEs (86, 78) with
bound-limited upper CIs, and a rewritten caption.

**A4 Use the bootstrap spread SE in the Bayesian layer — DONE [CODE].**
The bootstrap spread SE (1.06) is *twice* the realised-variance SE (0.54), so they are **not**
close; the Bayesian likelihood now uses the bootstrap SE. Consequences: the drift is only
**≈0.8 estimator standard errors from zero** (was "1.6" under the realised-variance SE), and the
posterior is more prior-dominated. `tab:bayes` interior probabilities recomputed (see below); the
abstract and discussion "1.6 standard errors" wording updated. Script: `jobs/drift_bayes.py`.

**A5 Reconcile WETH stressed-hour fractions — DONE (explained) [CODE-checked].**
The `19.9%` (regime calibration) thresholds the **raw** utilization series; the `13.6%`
(cross-section) uses the **curve-implied** utilization, which corrects the known upward bias in the
raw borrowed/supplied ratio. Both places now state this explicitly, and note that using the
curve-implied measure would make the (already negligible) regime channel smaller still. Confirmed
numerically: raw util >0.90 over the window = 0.1992.

## Group B — Restructuring

**B1 Make the extended sample central — DONE [CODE, default].**
Recomputed the survival/moment table and the interior thresholds ν*(α) at the extended-sample
parameters (`jobs/recalibrate_extended.py`). The extended spread is more volatile
(σ_Z 0.175 → 0.232), so ν*(α) **rises** to **+0.12, +0.24, +0.48** (from +0.06/+0.10/+0.20). The
flipped drift (+0.01 raw, −0.01 APR-adjusted) stays far below all of them, so the **unlevered
corner solution survives** the window change. This is now stated in the abstract, in the
`sec:discussion` drift paragraph (with the ν* numbers), and in the (reframed) appendix paragraph,
whose heading is now "the drift flips sign out of window, yet the corner solution survives."

**B2 Demote the regime-switching machinery — DONE [TEXT, default: appendix].**
The 6×6 construction (Prop. `prop:regime`, Lemma `lem:regime-invertible`, `eq:psi-regime`,
`eq:regime-system`, the implementation notes and verification) moved to a new
**Appendix `app:regime`, "Regime-switching carry."** Main text keeps one paragraph (the kinked
curve, calibrated states, `tab:regime`, the negligible-at-monthly-horizons finding) plus Remark
`rem:jensen`. The introduction's fourth contribution was rewritten (it no longer calls the
convexity "the economically material feature"; it says the channel is incorporated and shown
negligible). **Both "earlier draft" references deleted.** Abstract, roadmap and conclusion updated.

**B3 Strengthen the model comparison — DONE [CODE].**
Added (i) a parameter-count-penalised (BIC-style) in-sample criterion — Kou still wins; (ii) an
**out-of-sample** ECF test (fit 60%, score 40%): Kou's edge over Merton nearly vanishes
(`1.13e-3` vs `1.14e-3`, both beating GBM `1.60e-3`), so the double-exponential advantage is largely
in-sample; (iii) a **barrier-crossing diagnostic** (`tab:barrier`): empirical one-month crossing
frequency vs each fitted model's simulated frequency under discrete monitoring. The fits disagree by
orders of magnitude and none matches the data, because the weakly-identified drift dominates the
one-month horizon — the liquidation-relevant point the referee asked for. Script:
`jobs/model_comparison.py`.

## Group C — Text-only consistency

- **C1** `rem:overshoot` "grows without bound" replaced (grows with jump size, bad debt only when
  jump *variance* also grows). `tab:overshoot` caption's false "insensitive to both" corrected:
  the per-crossing recovery is design-independent but its expected contribution responds through
  p_liq; the continuous-crossing mean and p_liq values for both designs are now given.
- **C2** Introduction consolidated from "three contributions"/six items to **four contributions**;
  roadmap updated to the final section+appendix order.
- **C3** Double-penalty note added after the objective: δ is a deliberate extra charge for the
  *non-pecuniary* costs of forced exit; δ=0 recovers pure mean–variance.
- **C4** Oracle remark softened: TVL-implied prices are "intended to approximate" the Chainlink
  feed (unsupported "track" claim removed).
- **C5** Brownian-bridge correction now cited to **Glasserman (2004)** (already in `finance.bib`),
  replacing `Sepp2004` at that point.
- **C6** Removed the repeated "3×3 for every k" statement outside the abstract/intro/`sec:resolvent`;
  removed the inline module name from the carry section; trimmed repeated headline numbers from the
  conclusion.

## Group D — Restored round-1 corrections (all verified by search)

D1 `V_{1,τ}=V_{2,τ}/b`; D2 `w_i→n_i`; D3 liquidation bonus `λ→ℓ` throughout; D4 `α_k→Ψ_k` with
q>Ψ_k stated; D5 `Hit(a,c)→Hit(a₁,a₂)` and tilt superscripts; D6 spread drift `δ→ν` (incl. `ν*`);
D7 `CB(h0)→CB` and volatility renamed `σ̄_Z` (realised total, incl. jumps) with a defining sentence;
D8 restricted the "negative net carry" claim to stablecoin/ETH-debt pairs and noted the wrapped-BTC
exception; D9 "second-order stochastic"→"mean–variance", claim restricted to mean-increasing/
variance-decreasing criteria; D10 "toward zero"→"toward the unlevered variance ≈0.007"; D11
`e^{μ1T}=1.02`→`≈1.026`; D12 added the ≈0.7-σ figure with jumps (σ̄_Z√T≈0.103); D13 "Collateral
factor"→"Liquidation threshold"; D14 `χ_i+½σ_i²`→`λ_iχ_i+½σ_i²`; D15 step convention (folded into
A1); D16 "row e–f"→"row e"; D17 "fast and exact"→"fast, noise-free"; D18 title changed to
"Leveraged DeFi Long–Short Positions under Liquidation and Jump Risk: A Semi-Analytic Evaluation
Framework". Verification searches for all listed strings return only intended uses.

## Numbers to sanity-check against the code

| Claim | Value | Source |
|---|---|---|
| Laplace p_surv(h0=0.07) | 0.33029 | `mc_convergence.py`, `moments.py` |
| naive MC p_surv(h0=0.07), 252 steps/horizon | 0.3421 | `a1_results_row.json` |
| regime: det. mean carry / opening / convexity | −0.78 / −0.12 / <0.05 pp | `regime_calibration.py` |
| bootstrap SE(μ1−μ2), drift corr, frac railed | 1.06, 0.45, 0.41 | `bootstrap_se.py` |
| model comparison OOS (GBM/Merton/Kou) | 1.60e-3 / 1.14e-3 / 1.13e-3 | `model_comparison.py` |
| extended-sample ν*(α) | +0.12 / +0.24 / +0.48 | `recalibrate_extended.py` |
| extended spread drift (raw / APR-adj) | +0.01 / −0.01 | `recalibrate_extended.py` |

## Open items for the author

1. `tab:params` (round-1 T4) remains grouped, not split — unchanged this round.
2. The Merton barrier-crossing entry is `0.000` because its fitted positive mean jump is
   poorly identified on the asymmetric spread (already the paper's stated caveat); it is reported
   as-is to illustrate the divergence, not as a credible crossing rate.
3. Verify the Glasserman (2004) citation resolves to the intended `finance.bib` entry in your build.
