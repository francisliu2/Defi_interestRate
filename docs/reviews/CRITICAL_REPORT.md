# Critical Report

**Paper:** *A Semi-Analytic Framework for Long-Short Position Sizing in DeFi under Liquidation and Jump Risk*
**Reviewer role:** Adversarial referee (quantitative finance / applied probability)
**Scope:** Methodology, calibration, empirical claims, and the paper's own headline "negative result."

---

## 1. One-paragraph verdict

This is a genuinely good paper with an unusually honest spine, and it is undermined in exactly two places — one technical, one epistemic — that the author has been careful enough to flag but not careful enough to resolve. The mathematical core (change of measure → 1-D killed expectation → a `3×3` Laplace resolvent that is *the same size for every moment order*) is elegant, correct-looking, and a real contribution: it lifts the Kou–Wang / Sepp barrier-option machinery to a barrier-killed **long–short ratio** via one clean tilting step. The empirical discipline (pre-registered cross-section, internal consistency checks to `1e-11`, refusal to report an "optimal" buffer) is better than the field's norm. But the paper markets fat tails and ruin while (a) modelling the one quantity where jump-overshoot actually bites — liquidation **recovery / bad debt** — with an admittedly optimistic *continuous-crossing* approximation, and (b) resting every headline number on a **single asset pair, a single 4.5-month window, and an ECF objective whose frequency weighting actively down-weights the tail**. The negative result is the most valuable thing in the paper and is also the least stress-tested. Accept in principle; the criticisms below are what stand between "elegant" and "trustworthy."

---

## 2. What the paper gets right (and should keep)

1. **The reduction is the real contribution.** The exponential tilt `dP⁽ᵏ⁾/dP ∝ e^{kX₂}` absorbing the short-leg factor, leaving a univariate first-exit problem for `Z = X₁ − X₂`, is the correct and non-obvious move. Because exponentials are eigenfunctions of the Kou generator, the resolvent stays `3×3` for *every* admissible integer `k`; higher moments enlarge only the finite forcing sum, not the linear system (Prop. 5.5, §4.2). This is genuinely useful and cleanly proven in Appendix B.

2. **Separating the evaluation map from the objective (§2.3, eq. `eval-map`).** Computing `(p_liq, {Mₖ}, {Mₖ/p_surv})` as a preference-free object, then applying utility afterward, is good epistemic hygiene. It is also what licenses the honest conclusion in §8.

3. **Correctly diagnosing the Monte Carlo bias.** The persistent `Δp_surv < 0` near the barrier is correctly attributed to the discrete-time Euler–Poisson scheme *missing intra-period liquidations*, not to path scarcity (Table 6). This is exactly right and is a point most simulation-based DeFi papers get wrong.

4. **Convexity, not level, of the rate curve.** The finding that deterministic carry is inert for volatile pairs (`|CB| < 0.03`) but the *kink* in the utilization curve moves survival by ~10pp (§7.3) is a sharp, correct, and non-obvious result. Recovering utilization by inverting the published rate curve rather than trusting the borrowed/supplied ratio (which misclassifies 25.4% vs 10.4% of stressed hours) is a nice methodological catch.

5. **The negative result is stated, not buried (§8).** Refusing to report an optimal buffer because its existence hinges on the sign of a weakly identified drift is the intellectually correct call and the paper's best feature.

---

## 3. Major criticisms

### 3.1 The recovery term is modelled where jumps *don't* matter and punted where they *do* (§6, Remark 6.2)

The survival probabilities and killed moments handle jump-crossings of the barrier *correctly* — that is the whole point of the Kou–Wang first-passage transform. But the **recovery** formula `R_τ = V₂,τ (1 − b(1+λ))/b` assumes the health factor equals **exactly one** at `τ` (continuous crossing). Under a jump-diffusion the barrier is overwhelmingly crossed *by a jump*, which overshoots: `V₂ > b·V₁`, recovery is strictly less than the formula returns, and can be **negative (bad debt)**.

This is not a minor caveat, for three compounding reasons:
- Recovery is quantitatively **large**: it raises mean terminal wealth from 0.910 to 1.014 and *halves* variance at `H₀ = 1.16` (§6.2). A term that big cannot be modelled with the loosest available approximation.
- It sits in the **left tail of wealth**, i.e. exactly the region the paper insists (correctly) that mean-variance ignores and higher moments capture. The paper sells tail-aware objectives, then models the wealth left tail with an *optimistic upper bound*. That is a self-contradiction at the level of the paper's own thesis.
- The exact treatment is available and the author knows it: under the double-exponential law the overshoot is itself exponentially distributed by memorylessness (Kou 2002). Deferring "the most valuable open problem here" (the author's own words) is defensible for a first paper, but the recovery *numbers currently in the paper* should be labelled as an upper bound **everywhere they appear**, and any claim built on them (e.g. "recovery does not change the sizing conclusion") should be re-derived, or at least bounded, under the pessimistic (bad-debt) case. As written, "recovery is large but harmless to the conclusion" is asserted from the optimistic side only.

**Recommendation:** either implement the exponential-overshoot recovery (feasible with the existing `(τ, Z_τ)` transform), or demote all recovery figures to explicit upper bounds and show the conclusion survives the lower bound too.

### 3.2 Everything rests on one pair, one window, and a tail-suppressing calibration

The headline table (WBTC/WETH), the sensitivity fan, the frontier, and the entire §8 negative result are calibrated to **840 four-hour observations over a single 4.5-month window** (2026-03-17 → 08-04). Three problems stack:

- **Sample-specific sign.** The negative result is driven by `μ₁ − μ₂ = −0.858/yr` — i.e. ETH out-grew BTC *in this particular window*. The paper is admirably clear that the drift is weakly identified, but then does not show the negative result is stable across windows or pairs. A referee cannot tell whether "no interior optimum" is a structural feature of health-factor sizing or an artifact of one bull-ETH quarter. **At minimum, repeat the §8 monotonicity finding across rolling windows and across several of the 37 pairs.**

- **The ECF weighting down-weights the tail.** The objective uses Gaussian frequency damping `w(u,v) = exp(−a(u²+v²))` with `a = 1` (§A.2), explicitly to "down-weight high frequencies sensitive to microstructure noise." But high frequencies of the characteristic function *are* the tail. A paper whose entire motivation is fat tails and flash crashes is fitting its measure with a weighting that systematically discounts tail information. This deserves, at least, a sensitivity check on `a`, and ideally a tail-focused diagnostic (the paper proposes `Z`-tail quantile and barrier-crossing-frequency checks in §A.4 — **run them and report them**, don't just recommend them).

- **The fitted jumps are tiny.** Calibrated mean jump sizes are `η ≈ 0.004–0.011` (0.4–1.1%) at intensities `λ ≈ 358–451/yr`. That is *many small jumps* — a near-diffusion that manufactures excess kurtosis, not the 10–20% daily moves and flash crashes the introduction invokes. Combined with the known identification ridge that preserves only jump *variance* (Ramezani–Zeng 2007, cited), the "Kou jump" interpretation is doing far less tail work than the framing implies. The genuinely dangerous large-jump regime may simply be absent from the fitted measure. This should be confronted head-on, not left for the reader to compute from the parameter table.

### 3.3 The regime model switches the wrong state variable

§7.3's Markov-modulated carry lets **only the drift** switch (`ψⁱ(s) = ψ_Z⁽ᵏ⁾(s) + cᵢs`), holding diffusion, jump law, and correlation fixed across regimes. This keeps the algebra cheap (the `6×6` system) — a nice property — but it captures the *rate* response to stress while ignoring the price response that dominates crypto crashes: **volatility spikes and correlations collapse toward 1** (which raises `σ_Z`, the barrier scale that §5 sensitivity shows is the single strongest driver of survival). The paper's own sensitivity analysis says volatility and correlation dominate; the regime extension then freezes exactly those. The `−10.1pp` opening-regime effect is therefore a *lower bound* on regime importance, and the more consequential regime channel is omitted. This should be stated, and ideally the extension generalized to a vol/correlation switch (which breaks the "only drift changes" shortcut and is genuinely harder — worth acknowledging as such).

### 3.4 The chosen illustrative objective is the one the paper argues against

The numerical objective (eq. `Phi`) is penalized conditional **mean-variance**, which the paper (in the Taleb rewrite, explicitly; in the original, implicitly) concedes is ill-suited to a fat-tailed, ruin-bearing setting. The "no interior optimum" result is then partly a property of mean-variance under monotone first two moments. The setup — a leveraged bet with an absorbing ruin barrier and a survival probability computed *explicitly* — is a textbook case for a **growth-optimal / Kelly objective with a ruin constraint**, or the leverage-efficiency criterion of Peters & Adamou (time-average vs ensemble-average growth). That the paper never evaluates a growth-optimal objective, despite having every ingredient in hand and `p_surv(h₀)` on a plate, is a conspicuous omission. It is plausible that a Kelly-type criterion *does* produce an interior optimum where mean-variance does not — which would materially change the §8 verdict. **This is the single most valuable additional experiment the paper could run.**

---

## 4. Minor / technical issues

- **Feasible region is economically marginal.** `h₀_min ≈ 0.066` gives `H₀_min ≈ 1.068`, at which `p_surv ≈ 0.31` over one month (Table 5) — a 69% liquidation probability. The high-leverage region where the model's outputs are most "interesting" is one no risk-managed desk would occupy. Worth a sentence situating who the feasible outputs are actually *for*.

- **Inconsistent horizons.** Main results use `T = 1/12` (one month); the regime table (Table 8) uses `T = 1/4`. Cross-comparing the headline survival numbers with the regime survival numbers is therefore apples-to-oranges. Standardize, or flag prominently.

- **Assumption 5.1 (generic six-root regime) is assumed, not established** for the calibrated parameters — simple roots, clean `3/3` sign split across the imaginary axis. It is verified numerically but not argued from the structure of the Lévy exponent (convexity of `ψ_Z` on the real axis gives most of it). A short lemma would close the gap; as written, a structural assumption is doing load-bearing work behind a numerical check.

- **Regime invertibility is only "generic" (Lemma 7.5).** Honestly stated as measure-zero exclusion rather than a pointwise theorem, with condition-number evidence (`81 → 844`). Acceptable, but the actual calibrated `(q₁, q₂)` is never *proven* to avoid the null set.

- **Symbol collision on `τ`.** `τ` is the liquidation time throughout, then reused in §A.3 for the unconstrained calibration parameter vector (`τ = (μ₁, a₁, …, f)`). Rename the latter.

- **Speedup is measured against a knowingly biased benchmark.** The `≈440×` figure compares against a 252-step MC that the paper *itself* shows is biased near the barrier. A fair accuracy-matched comparison would use a Brownian-bridge barrier correction or finer stepping (slower MC), which only *strengthens* the speed argument — so state it that way, and don't let a reader infer the analytic method was benchmarked against an accurate MC.

- **"First exact semi-analytic solution" claim.** The novelty is real but the phrasing invites a priority fight. The barrier-option resolvent (Sepp 2004) is prior art; the genuine novelty is the *change-of-measure reduction of a long–short ratio* to that machinery. Frame the claim around the reduction, not around "first."

---

## 5. Robustness experiments a revision should include

1. **Overshoot recovery** (exponential overshoot via the `(τ, Z_τ)` transform), or explicit upper/lower recovery bounds and a re-derived §8 conclusion under the lower bound.
2. **Rolling-window and cross-pair replication of the §8 monotonicity result** — is "no interior optimum" structural or sample-specific?
3. **A growth-optimal / Kelly-with-ruin objective** on the existing evaluation map, checked for an interior optimum.
4. **ECF weighting sensitivity** (`a`) and the tail diagnostics the paper already proposes (`Z`-tail quantiles, empirical vs model barrier-crossing frequency).
5. **A volatility/correlation regime switch**, or at minimum a scenario showing how much the `−10.1pp` drift-only figure grows once `σ_Z` is allowed to spike in the stressed state.

---

## 6. Summary scorecard

| Dimension | Assessment |
|---|---|
| Mathematical correctness | Strong — reduction and resolvent are clean and consistency-checked |
| Novelty | Real, but over-claimed in phrasing ("first exact"); frame around the tilt reduction |
| Calibration rigour | Weak spot — one pair, one window, tail-suppressing weighting, tiny fitted jumps |
| Recovery / bad-debt treatment | Weakest link — optimistic approximation in the tail the paper claims to care about |
| Objective choice | Coherent framework, but the illustrative objective contradicts the thesis; Kelly untested |
| Intellectual honesty | Excellent — the negative result and the identification limit are the paper's strongest suit |
| Overall | Accept with major revisions; the honesty is rare and the core is sound, but the empirical and recovery claims are not yet load-bearing |

---

*Prepared alongside `latex/main_taleb.tex`, a full stylistic rewrite of the source in the voice of N. N. Taleb. The rewrite embraces the paper's genuine strengths — the absorbing barrier as ruin, fat tails, the negative result — and sharpens (rather than hides) the same weak points catalogued above, especially §3.1 (recovery) and §3.4 (mean-variance vs growth-optimal).*
