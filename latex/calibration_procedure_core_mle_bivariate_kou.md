# Calibration of the Bivariate Kou Model under the Physical Measure

## Core calibration principle

The calibration is based on the maximum-likelihood treatment of the double-exponential jump-diffusion (DEJD) model in Ramezani and Zeng (2007). The jump-filtering step used below is not treated as an independent estimator and is not presented as a separate nonparametric jump-detection methodology. It is used only to construct stable starting values for the likelihood optimizer.

This choice keeps the calibration deliberately conservative. The final reported parameters are likelihood estimates of the parametric Kou model, while the preliminary classification of extreme returns is only a numerical device used to reduce sensitivity to poor initial guesses and local optima.

The model is calibrated under the physical measure $P$. This is appropriate because the liquidation-sizing problem requires physical survival probabilities and physical killed payoff moments, not risk-neutral option prices.

---

## Model to be calibrated

Let the synchronized log returns over sampling interval $\Delta$ be

$$
r_{i,t}
=
\log S_{i,t+\Delta}-\log S_{i,t},
\qquad i=1,2.
$$

The bivariate Kou specification is

$$
dX_{i,t}
=
\mu_i^{\mathrm{eff}}dt
+
\sigma_i dB_{i,t}
+
d\left(\sum_{n=1}^{N_{i,t}}J_i^{(n)}\right),
\qquad i=1,2,
$$

with

$$
d\langle B_1,B_2\rangle_t=\rho\,dt,
$$

and independent Poisson jump counters $N_i$ with intensities $\lambda_i$. The jump-size density for asset $i$ is

$$
f_i(j)
=
\frac{p_i}{\eta_{i,+}}e^{-j/\eta_{i,+}}\mathbf 1_{\{j>0\}}
+
\frac{1-p_i}{\eta_{i,-}}e^{j/\eta_{i,-}}\mathbf 1_{\{j<0\}}.
$$

The calibrated parameter vector is

$$
\theta^P
=
\left(
\mu_1^{\mathrm{eff}},\mu_2^{\mathrm{eff}},
\sigma_1,\sigma_2,\rho,
\lambda_1,\lambda_2,
p_1,p_2,
\eta_{1,+},\eta_{1,-},
\eta_{2,+},\eta_{2,-}
\right).
$$

The expected price-growth drift can be recovered after estimation through

$$
\mu_i
=
\mu_i^{\mathrm{eff}}
+
\frac12\sigma_i^2
+
\lambda_i\chi_i,
$$

where

$$
\chi_i
=
\frac{p_i}{1-\eta_{i,+}}
+
\frac{1-p_i}{1+\eta_{i,-}}
-1.
$$

The constraints imposed during estimation are

$$
\sigma_i>0,\qquad
\lambda_i>0,\qquad
p_i\in(0,1),
$$

$$
0<\eta_{i,+}<1,\qquad
\eta_{i,-}>0,\qquad
|\rho|<1.
$$

The restriction $\eta_{i,+}<1$ is required so that the price-jump compensator is finite.

---

## Sampling frequency

The baseline calibration should use a fixed sampling interval $\Delta$, for example hourly or four-hour returns.

The sampling frequency should be chosen before calibration and kept fixed for the reported parameter estimates. Very high-frequency data can introduce microstructure noise, stale quotes, exchange-specific frictions, and artificial outliers. Daily data often contains too few jump observations for stable tail estimation. Hourly or four-hour data is usually a reasonable compromise for liquid crypto pairs.

The final estimates should not be obtained by pooling several sampling frequencies into one likelihood. Different $\Delta$'s correspond to different transition densities. Repeating the calibration at alternative $\Delta$'s is useful as a robustness check, but the reported baseline should correspond to a single transition horizon.

---

## Step 1: preliminary starting values

Following the practical logic of likelihood calibration in Ramezani and Zeng (2007), the optimizer is initialized from simple return-based estimates. This step is not the final estimator.

First compute sample means and robust scale estimates for each return series. Define a central-return set

$$
\mathcal C_i
=
\left\{
t:
|r_{i,t}-\bar r_i|
\le
c\,\widehat\sigma_i\sqrt{\Delta}
\right\},
$$

where $c$ is a fixed cutoff, for example $c=4$. Returns outside the central set are treated as tail observations for initialization only.

Estimate preliminary diffusion parameters from the central returns:

$$
\widehat\sigma_i^{(0)}
=
\frac{\operatorname{sd}(r_{i,t}:t\in\mathcal C_i)}{\sqrt{\Delta}},
$$

and estimate the Brownian correlation from jointly central observations:

$$
\widehat\rho^{(0)}
=
\operatorname{corr}
\left(
r_{1,t},r_{2,t}:
t\in\mathcal C_1\cap\mathcal C_2
\right).
$$

Let the preliminary tail set be

$$
\mathcal J_i
=
\{t:t\notin\mathcal C_i\}.
$$

Then initialize the jump intensity as

$$
\widehat\lambda_i^{(0)}
=
\frac{|\mathcal J_i|}{n\Delta},
$$

where $n\Delta$ is the total sample length in model time units.

Initialize the upward-jump probability by

$$
\widehat p_i^{(0)}
=
\frac{
|\{t\in\mathcal J_i:r_{i,t}>0\}|
}{
|\mathcal J_i|
}.
$$

Initialize the positive and negative jump-size means by

$$
\widehat\eta_{i,+}^{(0)}
=
\operatorname{mean}
\left(
r_{i,t}:t\in\mathcal J_i,\ r_{i,t}>0
\right),
$$

$$
\widehat\eta_{i,-}^{(0)}
=
\operatorname{mean}
\left(
-r_{i,t}:t\in\mathcal J_i,\ r_{i,t}<0
\right).
$$

If too few tail observations are available for one side of the distribution, replace the corresponding initial value by a conservative small positive value and rely on the likelihood optimization to refine it. The final estimate is still determined by MLE, not by the threshold classification.

Initialize $\mu_i^{\mathrm{eff}}$ from the sample mean after subtracting the preliminary jump contribution, or simply initialize it from the average return divided by $\Delta$. Drift is weakly identified at short horizons, so the likelihood should not be forced to fit the data primarily through $\mu_i^{\mathrm{eff}}$.

---

## Step 2: constrained maximum likelihood

The final parameters are obtained by constrained maximum likelihood.

For one observation $r_t=(r_{1,t},r_{2,t})$, let $f_\theta(r_t;\Delta)$ denote the bivariate transition density of the Kou model over interval $\Delta$. The log-likelihood is

$$
\ell(\theta)
=
\sum_{t=1}^n
\log f_\theta(r_t;\Delta).
$$

The MLE is

$$
\widehat\theta^P
=
\arg\max_{\theta\in\Theta}
\ell(\theta),
$$

where $\Theta$ imposes the admissibility constraints listed above.

The transition density can be evaluated using the joint characteristic exponent

$$
\Psi_\theta(u,v),
$$

through Fourier inversion:

$$
f_\theta(r_1,r_2;\Delta)
=
\frac{1}{(2\pi)^2}
\int_{\mathbb R^2}
\exp\{-i(ur_1+vr_2)\}
\exp\{\Delta\Psi_\theta(u,v)\}
\,du\,dv.
$$

Equivalently, the same density may be computed by truncating the Poisson mixture over jump counts. The Fourier representation is usually cleaner for a bivariate implementation because the Brownian correlation is handled directly in $\Psi_\theta(u,v)$.

The numerical maximization should use transformed unconstrained parameters, for example

$$
\sigma_i=\exp(a_i),
\qquad
\lambda_i=\exp(b_i),
\qquad
p_i=\frac{1}{1+\exp(-c_i)},
$$

$$
\eta_{i,+}=\frac{1}{1+\exp(-d_i)},
\qquad
\eta_{i,-}=\exp(e_i),
\qquad
\rho=\tanh(g).
$$

This avoids invalid parameter proposals and keeps the optimizer inside the economically meaningful region.

The likelihood should be optimized from the preliminary starting point and from several perturbed starting points around it. The final estimate is the admissible local maximum with the largest likelihood value.

---

## Step 3: compensation and consistency check

After estimating

$$
\widehat\theta^P,
$$

compute

$$
\widehat\chi_i
=
\frac{\widehat p_i}{1-\widehat\eta_{i,+}}
+
\frac{1-\widehat p_i}{1+\widehat\eta_{i,-}}
-1.
$$

If the model is reported in terms of expected price-growth drifts, recover

$$
\widehat\mu_i
=
\widehat\mu_i^{\mathrm{eff}}
+
\frac12\widehat\sigma_i^2
+
\widehat\lambda_i\widehat\chi_i.
$$

The liquidation engine should use $\widehat\mu_i^{\mathrm{eff}}$, not the uncompensated expected-growth drift.

---

## Step 4: validation on the log-ratio process

The calibration must be validated on the object that drives liquidation:

$$
Z_t=X_{1,t}-X_{2,t}.
$$

The fitted model should be checked against empirical properties of

$$
\Delta Z_t=r_{1,t}-r_{2,t}.
$$

At minimum, compare empirical and model-implied values of

$$
\mathbb E[\Delta Z],
\qquad
\operatorname{Var}(\Delta Z),
\qquad
\operatorname{Skew}(\Delta Z),
\qquad
\operatorname{Kurt}(\Delta Z),
$$

and the lower-tail quantiles

$$
q_{1\%}^{Z},
\qquad
q_{5\%}^{Z}.
$$

Since the application is a first-passage liquidation problem, the most important validation is barrier crossing. For a grid of health buffers $h_0$ and horizons $T$, compare the empirical frequency of

$$
\min_{0\le s\le T}Z_s<-h_0
$$

with the model-implied liquidation probability

$$
p_{\mathrm{liq}}(h_0,T).
$$

This validation is not an additional calibration criterion in the baseline procedure. It is an out-of-sample diagnostic that checks whether the fitted physical-measure model is suitable for liquidation-risk evaluation.

---

## Step 5: uncertainty assessment

The baseline method reports the constrained MLE. Parameter uncertainty can be assessed by either:

1. the inverse Hessian of the negative log-likelihood;
2. a parametric bootstrap from the fitted bivariate Kou model;
3. a rolling-window recalibration exercise.

The bootstrap is the most useful for the present application because it propagates parameter uncertainty into

$$
p_{\mathrm{liq}}(h_0,T),
\qquad
E[\Pi_T\mid \tau>T],
\qquad
h_0^\star.
$$

Bayesian calibration is not used as the baseline. It is a possible extension if the goal is to report posterior credible intervals for liquidation probabilities or optimal health buffers. It is not necessary for the main calibration because the target model is already parametric and likelihood-estimable.

---

## Why this procedure is defensible

The procedure is deliberately centered on one calibration literature: maximum-likelihood estimation of the double-exponential jump-diffusion process. The preliminary tail classification is not introduced as a separate estimator and is not used to claim nonparametric jump identification. It only supplies numerically sensible starting values.

This avoids two common weaknesses:

1. **Filter-only calibration.** A pure threshold-based calibration is sensitive to the cutoff and sampling frequency. It is therefore not used for the final reported parameters.

2. **Cold-start MLE.** A pure likelihood search from arbitrary starting values can be unstable because jump intensity, jump size, and diffusion volatility can partially substitute for one another. The initialization step reduces this numerical instability without changing the estimator.

The final estimates are therefore standard parametric MLE estimates of the bivariate Kou model under $P$, initialized in a transparent and reproducible way.

---

## Suggested paper wording

> We estimate the physical-measure parameters of the bivariate Kou model by constrained maximum likelihood, following the likelihood-based calibration of the double-exponential jump-diffusion in Ramezani and Zeng (2007). A preliminary tail-classification step is used only to construct stable starting values for the optimizer; the reported parameters are not threshold estimates. The likelihood is evaluated from the bivariate transition density implied by the joint characteristic exponent. Since liquidation depends on the log-ratio process $Z=X_1-X_2$, the fitted model is validated on the empirical left tail and barrier-crossing behavior of $Z$, in addition to the marginal return distributions of the two assets.

---

## References

Kou, S. G. (2002). A jump-diffusion model for option pricing. *Management Science*, 48(8), 1086–1101.

Ramezani, C. A., & Zeng, Y. (2007). Maximum likelihood estimation of the double exponential jump-diffusion process. *Annals of Finance*, 3, 487–507. https://doi.org/10.1007/s10436-006-0062-y
