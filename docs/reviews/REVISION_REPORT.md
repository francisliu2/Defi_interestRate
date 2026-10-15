# Revision Report

**Manuscript:** `latex/optimal_long_short.tex` (the brief named `optimal_long_short-rewritten.tex`;
no such file exists in the repo, so the live manuscript was edited).
**Compiles:** `latexmk -pdf` runs clean, 43 pages, **no undefined references or citations**.
**`\TODO` count: 0** — the "all CODE, take the time" option was chosen, so every [CODE] task was
executed on the real Aave data rather than stubbed.

**New deliverables**

- `latex/optimal_long_short.tex` — revised manuscript.
- `latex/additions.bib` — new references (Cai 2009; Cai & Kou 2011; Asmussen, Avram & Pistorius 2004),
  wired via `\bibliography{finance,additions}`.
- `latex/barbell_spinoff.tex` — the cut barbell section, preserved verbatim (T2).
- `latex/fig_buffer_drift.pdf` — new figure (T7).
- `jobs/{bootstrap_se,mc_convergence,regime_calibration,overshoot_table,model_comparison,recalibrate_extended,drift_bayes}.py`
  — reproduction scripts (referenced in the new *Replication* section).
- `results/{bootstrap_se,mc_convergence,regime_calibrated,model_comparison,recalib_extended,revision_moderate}.json`
  — machine-readable outputs.

---

## Priority 1 — Framing and scope

**T1 Reposition novelty — DONE.** "first exact, semi-analytic solution" → "develops a semi-analytic
evaluation map"; a sentence now states plainly what is borrowed (first-passage machinery for
exponential-type Lévy models) and what is new (application to the health-factor payoff + the
evaluation map). The literature paragraph adds Cai (2009, hyperexponential), Cai & Kou (2011,
mixed-exponential) and Asmussen–Avram–Pistorius (2004, phase-type), noting the finite linear
system is a general feature of that class. The k-independence of the 3×3 system is now stated once
as "a direct consequence of the tilt … not a separate result."

**T2 Remove barbell — DONE (default: cut).** Section 8 (`sec:barbell`) deleted and moved verbatim to
`latex/barbell_spinoff.tex`. The Jensen-gap observation is retained as Remark `rem:jensen` in the
regime-switching subsection. The one dangling reference in the conclusion was removed. *Author
override:* to reinstate, `\input{barbell_spinoff.tex}` and apply the "keep + repair" edits (name the
three comparison strategies, define keeper income, re-base weights, strip evaluative language).

**T3 Abstract / headline claims — DONE.** "pre-registered" removed in all three places
("screened by a rule fixed before examining carry" / "fixed-rule"). "up to ten percentage points"
removed from the abstract and conclusion (replaced by the calibrated T9 result). Abstract rewritten
to ≈250 words (was ≈370).

**T4 Repetition and length — PARTIAL.**
- 3×3-for-every-k claim: kept in abstract, once in the intro (as a consequence of the tilt), once in
  §5; removed from the intro contribution list and the conclusion.
- `k=88` removed; replaced by "the admissibility bound binding only for large upward jumps in the
  debt leg."
- A single *Replication* section now consolidates package modules, `jobs/` scripts and data paths;
  the most prominent inline code mentions were removed, though a few `\texttt{...}` script references
  remain in captions by design (each table/figure names its generator).
- **`tab:params` was not split.** It is already organised into four labelled blocks (Kou inputs /
  derived model / portfolio inputs / derived portfolio); splitting it further was judged low value
  and left for the author. *Not done.*
- **Net length rose (40→43 pp), not fell.** The mandated new empirical content (bootstrap table,
  MC-convergence table + subsection, model comparison, extended-sample recalibration, the full
  37-pair appendix, the buffer figure) outweighs the barbell cut and the repetition trims. The main
  text was tightened, but the page target and the new-results tasks pull in opposite directions;
  flagged for the author.

## Priority 2 — Internal consistency

**T5 Objective consistency — DONE.** `eq:Phi-general`, `eq:Phi` and `eq:objective` are now stated on
**unconditional** terminal wealth `W_T`. The survival-conditional moments are kept in the evaluation
map as diagnostics, with an explicit sentence that they must not be maximised directly (conditioning
on survival inflates the conditional mean as the buffer thins — a survivorship artefact). Downstream
references to Φ were made consistent, including the frontier caption.

**T6 Bayesian prior / likelihood — DONE.** Now states that the calibrated μ_i already include the
supply/borrow APRs, so anchoring the prior mean at the net carry (−0.025) encodes a *zero
price-spread-drift* view with no double counting; that the three convention-dependent point
estimates (−0.858, −0.118, +0.188) are model uncertainty a single Gaussian likelihood does not
capture (posterior probabilities read as indicative magnitudes); and that the δ\* thresholds hold
all other parameters at their calibrated values. The optional mixed-likelihood robustness was not
run (the dispersion is reported and discussed instead).

**T7 Optimal buffer vs. drift — DONE [CODE].** New Figure `fig:buffer-drift` plots H₀\*(ν) for
ν∈[−0.3,+0.5], α∈{0.5,1,2}, with the unlevered corner shown. The Bayesian subsection is reframed as
"one way to integrate this curve against a posterior for ν."

## Priority 3 — Numerical validation

**T8 Straw-man MC — DONE [CODE].** New subsection `sec:mc-benchmark` + Table `tab:mcconv`: a
Brownian-bridge-corrected simulator (exact intra-step crossing probability) agrees with the Laplace
value 0.33029 within MC error at **every** step count (z≈0 even at 125 steps), while the naive
252-step-style grid check is biased +0.011 at 125 steps and still +0.003 at 4,000 — proving the
1.1-pt gap is discretisation, not an analytic-method flaw. Talbot node sensitivity reported (varying
M from 16 to 48 moves p_surv by <1e-8). The 440× claim is qualified everywhere as being against the
naive simulator and shown to *grow* (~3,000×) against the accuracy-matched one.

**T9 Regime calibration — DONE [CODE].** Regime states estimated by thresholding the WETH
utilization series at U_opt=0.90: frac_stressed 19.9%, q₁=16.3, q₂=65.5, c₁=−2.1%, c₂=−4.0%.
Table `tab:regime` replaced; at benchmark params and **T=1/12** the opening-regime effect is **≤0.1
pp**, versus the illustrative 10.1 pp of the old table (extreme params, T=1/4). The horizon
inconsistency the brief flagged is resolved (everything now at T=1/12), and the abstract/intro/
conclusion were corrected to "small once calibrated."

**T10 Overshoot like-for-like — DONE [CODE].** Table `tab:overshoot` now shows two designs: A
(size×s, intensity fixed → variance ∝ s²) and B (size×s, intensity×1/s² → variance fixed). The
bad-debt flip occurs only under Design A (a far more volatile market); under variance-preserving
Design B the correction is milder (−36% at s=20) and recovery stays positive. The text and the
abstract/intro now distinguish "bigger jumps" from "more volatile market."

## Priority 4 — Calibration and data

**T11 Parameter uncertainty + model comparison — DONE [CODE].** Block-bootstrap SEs (Table
`tab:bootstrap`, B=200): σ and ρ sharply identified (~6% relative SE); drifts have SEs larger than
the estimates; jump intensities rail to the optimiser bound — λ is not separately identified from
the η's along the variance ridge. Nested comparison on the spread Z (GBM/Merton/Kou): Kou wins on
the ECF objective (1.7e-6 vs 1.3e-5 vs 3.4e-4) and matches empirical tail quantiles almost exactly;
the discussion states plainly that the fitted kurtosis comes from many small jumps, not rare large
ones, so the flash-crash motivation is carried only by the scaled experiments of §6.3.

**T12 Rare-large-jump recalibration — DONE [CODE].** Recalibrated on the extended 1.6-year sample
(excess kurtosis 9.3 vs 3.4; realised spread moves to ±6–9%): jumps roughly double (to 1–2%, still
far from flash-crash) and **the spread drift flips −0.86 → +0.01/yr**. Reported as direct evidence
that the §8 driver is sample-specific and weakly identified — the qualitative conclusions
(evaluation map robust, drift not) survive and are reinforced.

**T13 Oracle-based liquidation prices — DONE (text); optional CODE not done.** New Remark
`rem:oracle` explains the discrete Chainlink heartbeat/deviation mechanism and both bias directions
(continuous monitoring overstates crossing frequency; stale-then-gap oracles understate overshoot),
and notes the calibration uses reserve oracle-implied (TVL/balance) prices, not the block-by-block
feed. Recalibration on raw oracle feeds was not run (no separate feed series in the data set).

## Priority 5 — Model mechanics and theory

**T14 Partial liquidation — DONE (text); optional CODE not done.** Remark `rem:liquidation`
rewritten: it no longer claims the absorbing-zero convention is conservative for the
*conditional* moments (it is not — conditioning discards the liquidated paths); it states what the
convention does bound (unconditional wealth on the total-loss event) and that Aave's 50% close
factor typically restores H>1 and the position continues, so full absorption/closure is a
stylisation. The close-factor Monte Carlo was not run.

**T15 Theory tightening — DONE.** Assumption `ass:six-root`: a paragraph now derives the
three-and-three root split from convexity of ψ_Z between its poles and cites the standard count
(Kou–Wang; Sepp; Cai–Kou), reducing the assumption to the measure-zero exclusions verified
numerically. Lemma `lem:regime-invertible` is explicitly treated as a numerically verified condition
for the calibrated (q₁,q₂), not a theorem. The phase-operator boundary conditions in `prop:Ubar`
are now justified as the zero-extension conditions of the phase-type barrier problem, with citations.

**T16 Static horizon — DONE.** Remark `rem:dynamic` expanded: addresses the collateral-topping-up
trader, argues the absorbing model is the relevant description of jump-driven crashes (liquidation
within a block or two, before a remedial tx confirms), and states when the static model is a good
approximation (short stress episodes) versus pessimistic (well-monitored calm markets).

## Priority 6 — Minor

**T17 Cross-section — DONE [CODE].** Full 37-pair table moved to a new Appendix
(`app:cross-section`, `tab:carry-xs-full`), generated from `results/pair_cross_section.csv`; the
main-text table keeps representative rows and points to the appendix. The screening rule is stated
precisely once (§7.2), and a sentence explains why one month is the relevant carry horizon.

**T18 Assorted — DONE.** The frontier caption's "panel (a)" reference is removed (replaced by a note
that the frontier is a survival-conditional diagnostic). An out-of-scope sentence on E-mode, supply/
borrow caps and isolation mode was added to the intro. Evaluative language was neutralised
("uncomfortable" → "a genuine limitation"; "least comfortably" cut; "indictment/Taleb/disciplined"
went with the barbell). "depeg-fragile" was kept as a precise technical descriptor.

---

## Numbers a reader should sanity-check against the code

| Claim | Value | Source |
|---|---|---|
| Overshoot correction, fitted jumps, H₀=1.16 | −3.1% (E[R] 0.162→0.157) | `overshoot_table.py` |
| Overshoot, variance-preserving s=20 | −36%, recovery stays +0.041 | `overshoot_table.py` |
| Bridge-MC vs Laplace p_surv(0.07) | 0.3303 vs 0.33029 (z≈0) | `mc_convergence.py` |
| Regime opening effect, calibrated, T=1/12 | ≤0.1 pp | `regime_calibration.py` |
| Bootstrap SE(σ₁), SE(μ₁) | 0.016, 0.71 | `bootstrap_se.py` |
| Model comparison ECF obj (GBM/Merton/Kou) | 3.4e-4 / 1.3e-5 / 1.7e-6 | `model_comparison.py` |
| Extended-sample spread drift | −0.86 → +0.01 | `recalibrate_extended.py` |
| Posterior P(worthwhile), α=1 | 11% | `drift_bayes.py` |

## Open items for the author

1. `tab:params` split (T4) — left grouped, not split.
2. Net page count rose; decide whether to move more of §5–§6 to an appendix to hit a length target.
3. Optional CODE not run: mixed-convention likelihood (T6), oracle-feed recalibration (T13),
   close-factor Monte Carlo (T14).
4. Verify the three new references in `additions.bib` against your citation database.
