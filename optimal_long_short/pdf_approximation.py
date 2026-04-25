"""
GB2 approximation to the conditional survival payoff distribution.

Fits a Generalised Beta of the Second Kind (GB2) distribution to the
four conditional moments

    mu_k^+ = E[Pi_T^k | tau > T],   k = 1, 2, 3, 4,

obtained from the Laplace-resolvent machinery via ConditionalMoments.

GB2 density
-----------
    f_Y(y) = a y^{ap-1} / [beta^{ap} B(p,q) (1 + (y/beta)^a)^{p+q}],   y > 0.

Moments
-------
    E[Y^k] = beta^k * Gamma(p + k/a) * Gamma(q - k/a) / [Gamma(p) * Gamma(q)],

valid for k < a*q.  The first four moments exist iff q > 4/a.

Calibration (Appendix A of the paper)
--------------------------------------
Step 1 – shape parameters (a, p, q) from scale-free ratios
    R_k = mu_k^+ / (mu_1^+)^k,   k = 2, 3, 4.
Step 2 – scale beta from the first moment.

Full payoff law
---------------
    Pi_T  ~  p_liq * delta_0  +  p_surv * GB2(a, beta, p, q).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

from optimal_long_short.inversion import TalbotInverter
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.model_params import KouParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy


# ---------------------------------------------------------------------------
# GB2 distribution — stateless functions
# ---------------------------------------------------------------------------

def gb2_log_moment(k: float, a: float, beta: float, p: float, q: float) -> float:
    """
    log E[Y^k] for Y ~ GB2(a, beta, p, q).

    Requires k < a*q for the moment to exist.
    Uses log-gamma arithmetic to avoid overflow for large k.
    """
    return (
        k * np.log(beta)
        + gammaln(p + k / a)
        + gammaln(q - k / a)
        - gammaln(p)
        - gammaln(q)
    )


def gb2_moment(k: float, a: float, beta: float, p: float, q: float) -> float:
    """E[Y^k] for Y ~ GB2(a, beta, p, q)."""
    return float(np.exp(gb2_log_moment(k, a, beta, p, q)))


def gb2_pdf(y: np.ndarray, a: float, beta: float, p: float, q: float) -> np.ndarray:
    """
    GB2 probability density function.

    Parameters
    ----------
    y : array-like, strictly positive.
    a, beta, p, q : GB2 parameters.

    Returns
    -------
    np.ndarray of the same shape as y.
    """
    y = np.asarray(y, dtype=float)
    u = (y / beta) ** a
    log_f = (
        np.log(a)
        + (a * p - 1) * np.log(y)
        - a * p * np.log(beta)
        - betaln(p, q)
        - (p + q) * np.log1p(u)
    )
    return np.exp(log_f)


def gb2_cdf(
    y: np.ndarray,
    a: float,
    beta: float,
    p: float,
    q: float,
    quad_limit: int = 200,
) -> np.ndarray:
    """
    GB2 cumulative distribution function via numerical quadrature.

    Parameters
    ----------
    y : array-like, strictly positive.
    quad_limit : maximum number of subintervals passed to scipy.integrate.quad.

    Returns
    -------
    np.ndarray of the same shape as y.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    cdf = np.empty_like(y)
    for i, yi in enumerate(y):
        cdf[i], _ = quad(
            gb2_pdf, 0.0, yi, args=(a, beta, p, q), limit=quad_limit
        )
    return cdf


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_shape(log_targets: np.ndarray) -> tuple[float, float, float, float]:
    """
    Internal: fit (a, p, q) from the four log-moment targets, then recover beta.

    Parameterises q = 4/a + exp(xi) so that q > 4/a is always satisfied.

    Returns (a, beta, p, q).
    """
    def objective(x: np.ndarray) -> float:
        a = np.exp(x[0])
        p = np.exp(x[1])
        q = 4.0 / a + np.exp(x[2])

        # beta pinned to the first moment
        log_beta = log_targets[0] - (
            gammaln(p + 1.0 / a) + gammaln(q - 1.0 / a) - gammaln(p) - gammaln(q)
        )
        beta = np.exp(log_beta)

        resid = 0.0
        for k in range(2, 5):
            pred = gb2_log_moment(k, a, beta, p, q)
            resid += (pred - log_targets[k - 1]) ** 2
        return resid

    x0 = np.array([np.log(2.0), np.log(1.0), np.log(1.0)])
    res = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 30_000},
    )

    a = np.exp(res.x[0])
    p = np.exp(res.x[1])
    q = 4.0 / a + np.exp(res.x[2])
    log_beta = log_targets[0] - (
        gammaln(p + 1.0 / a) + gammaln(q - 1.0 / a) - gammaln(p) - gammaln(q)
    )
    beta = np.exp(log_beta)
    return a, beta, p, q


def fit_gb2(
    mu_plus: list[float],
) -> tuple[float, float, float, float]:
    """
    Fit GB2(a, beta, p, q) to four conditional survival moments.

    Parameters
    ----------
    mu_plus : list of four floats
        [mu_1^+, mu_2^+, mu_3^+, mu_4^+]  where mu_k^+ = E[Pi_T^k | tau > T].

    Returns
    -------
    (a, beta, p, q) : fitted GB2 parameters.

    Raises
    ------
    ValueError
        If any moment is non-positive or q <= 4/a after fitting.
    """
    if len(mu_plus) != 4:
        raise ValueError(f"Expected 4 moments, got {len(mu_plus)}.")
    if any(m <= 0 for m in mu_plus):
        raise ValueError("All moments must be strictly positive.")

    log_targets = np.log(np.asarray(mu_plus, dtype=float))
    a, beta, p, q = _fit_shape(log_targets)

    if q <= 4.0 / a:
        raise ValueError(
            f"Fitted parameters violate the existence condition q > 4/a: "
            f"q={q:.4f}, 4/a={4/a:.4f}."
        )
    return a, beta, p, q


# ---------------------------------------------------------------------------
# GB2Approximation dataclass — full workflow
# ---------------------------------------------------------------------------

@dataclass
class GB2Approximation:
    """
    GB2 approximation to the conditional survival payoff Pi_T | tau > T.

    Wraps the complete workflow:
      1. Compute four conditional moments via Laplace-resolvent inversion.
      2. Fit GB2(a, beta, p, q) by moment matching (Appendix A of the paper).
      3. Expose the fitted density, CDF, and moment diagnostics.

    Parameters
    ----------
    params : KouParams
        Kou model parameters.
    strategy : UnitExposureLongShortStrategy
        Portfolio strategy (supplies h0, T, market params).
    inverter : TalbotInverter, optional
        Numerical Laplace inverter. Defaults to TalbotInverter(M=36).

    Attributes (set after calling fit())
    ------
    a, beta, p, q : float
        Fitted GB2 parameters.
    mu_plus : list[float]
        Four conditional moments [mu_1^+, ..., mu_4^+].
    p_surv : float
        Survival probability P(tau > T).
    """

    params: KouParams
    strategy: UnitExposureLongShortStrategy
    inverter: TalbotInverter = field(default_factory=lambda: TalbotInverter(M=36))

    # fitted quantities (populated by fit())
    a: float = field(init=False, default=float("nan"))
    beta: float = field(init=False, default=float("nan"))
    p: float = field(init=False, default=float("nan"))
    q: float = field(init=False, default=float("nan"))
    mu_plus: list[float] = field(init=False, default_factory=list)
    p_surv: float = field(init=False, default=float("nan"))

    def fit(self) -> "GB2Approximation":
        """
        Compute moments and fit the GB2 distribution.  Returns self for chaining.
        """
        cm = ConditionalMoments(
            params=self.params,
            strategy=self.strategy,
            inverter=self.inverter,
        )
        self.p_surv = cm.p_surv()
        self.mu_plus = [cm.conditional_moment(k) for k in range(1, 5)]
        self.a, self.beta, self.p, self.q = fit_gb2(self.mu_plus)
        return self

    def _check_fitted(self) -> None:
        if np.isnan(self.a):
            raise RuntimeError("Call .fit() before using this method.")

    def pdf(self, y: np.ndarray) -> np.ndarray:
        """GB2 density of Y = Pi_T | tau > T evaluated at y."""
        self._check_fitted()
        return gb2_pdf(y, self.a, self.beta, self.p, self.q)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        """GB2 CDF of Y = Pi_T | tau > T evaluated at y."""
        self._check_fitted()
        return gb2_cdf(y, self.a, self.beta, self.p, self.q)

    def moment_check(self) -> dict[int, dict[str, float]]:
        """
        Compare fitted GB2 moments against the Laplace targets.

        Returns
        -------
        dict mapping k -> {"target": mu_k^+, "gb2": E[Y^k], "rel_err": |gb2/target - 1|}
        """
        self._check_fitted()
        result = {}
        for k in range(1, 5):
            target = self.mu_plus[k - 1]
            fitted = gb2_moment(k, self.a, self.beta, self.p, self.q)
            result[k] = {
                "target": target,
                "gb2": fitted,
                "rel_err": abs(fitted / target - 1.0),
            }
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the fit."""
        self._check_fitted()
        lines = [
            f"GB2 fit   a={self.a:.4f}  beta={self.beta:.6f}  "
            f"p={self.p:.4f}  q={self.q:.4f}",
            f"Existence q > 4/a:  {self.q:.4f} > {4/self.a:.4f}  "
            f"-> {self.q > 4/self.a}",
            f"p_surv = {self.p_surv:.6f}",
            "",
            f"  {'k':>2}  {'target':>12}  {'GB2':>12}  {'rel_err':>10}",
            "  " + "-" * 40,
        ]
        for k, d in self.moment_check().items():
            lines.append(
                f"  {k:>2}  {d['target']:>12.6f}  {d['gb2']:>12.6f}"
                f"  {d['rel_err']:>10.2e}"
            )
        return "\n".join(lines)
