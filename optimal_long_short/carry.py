"""Carry-adjusted position sizing for leveraged DeFi long-short books.

This module folds AAVE-style interest accrual (supply APR on the collateral
leg, borrow APR on the debt leg) into the semi-analytic long-short framework of
``optimal_long_short.tex`` **without** changing any of the core Kou /
Laplace-resolvent machinery.

The key observation is that, under a *deterministic* rate over the horizon,
carry enters as a per-leg price-growth drift shift.  With collateral value
compounding at the supply rate and the outstanding liability compounding at the
borrow rate,

    V_{1,t} = n_1 S_{1,0} e^{X_{1,t}} e^{r_s t},
    V_{2,t} = n_2 S_{2,0} e^{X_{2,t}} e^{r_b t},

the health factor H_t = b V_1 / V_2 picks up the *net* carry (r_s - r_b) in its
drift, while the terminal payoff Pi_T = (V_{1,T} - V_{2,T}) 1_{tau>T} picks up
the per-leg accrual e^{r_s T}, e^{r_b T}.  Both effects are captured *exactly
and consistently* by the single substitution

    mu_1 -> mu_1 + r_s   (collateral / long leg = asset 1),
    mu_2 -> mu_2 + r_b   (debt / short leg      = asset 2),

because ``KouParams.mu_i`` is defined so that E[exp(X_{i,t})] = exp(mu_i t).
Every downstream object (characteristic function, killed moments, survival
probability, the Phi objective) therefore applies verbatim to the
carry-adjusted parameters.

The mapping is *generalized to any pair*: the caller supplies which asset is the
collateral leg and which is the debt leg together with their annualized rates,
so the same code serves WBTC/WETH-with-funding, USDC/WETH carry, or any other
AAVE market.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from optimal_long_short.model.drift_service import with_price_drift_view
from optimal_long_short.model.model_params import KouParams
from optimal_long_short.model.risk_report import h0_liquidation_moment_report


class RateSource(str, Enum):
    """Which deterministic rate to use for the horizon carry drift.

    ``CIR_THETA``
        The calibrated CIR / CIR+Jump long-run mean (``theta``); the
        model-implied expected rate.  See the caveat in
        :func:`resolve_rate`.
    ``SPOT``
        The current supply/borrow APR at ``t = 0`` held constant to ``T``.
    ``REALIZED``
        The realized historical mean APR over a trailing window.
    """

    CIR_THETA = "cir_theta"
    SPOT = "spot"
    REALIZED = "realized"


def resolve_rate(
    source: RateSource | str,
    *,
    cir_theta: float | None = None,
    spot: float | None = None,
    realized: float | None = None,
) -> float:
    """Return the annualized deterministic rate for the requested source.

    Exactly the field matching ``source`` must be supplied.

    Notes
    -----
    ``CIR_THETA`` is exposed for completeness, but the CIR fits shipped in
    ``results/cir_usdc_weth_aave_v3.csv`` are numerically unstable (very large
    ``kappa`` and a ``theta`` that hits its optimiser bound).  Prefer ``SPOT``
    or ``REALIZED`` for the demonstration until the rate calibration is
    revisited.
    """
    source = RateSource(source)
    value = {
        RateSource.CIR_THETA: cir_theta,
        RateSource.SPOT: spot,
        RateSource.REALIZED: realized,
    }[source]
    if value is None:
        raise ValueError(
            f"Rate source {source.value!r} selected but its value was not provided."
        )
    return float(value)


@dataclass(frozen=True)
class CarryLeg:
    """One leg of the carry book.

    Attributes
    ----------
    asset : str
        Human-readable asset label (e.g. ``"USDC"``), for reporting only.
    rate : float
        Annualized rate as a decimal.  For the collateral leg this is the
        supply APR (earned); for the debt leg it is the borrow APR (paid).
        Both are entered as positive numbers; the sign convention is handled
        by :func:`apply_carry`.
    """

    asset: str
    rate: float


@dataclass(frozen=True)
class CarrySpec:
    """A carry trade: long the collateral leg, short (borrow) the debt leg.

    The collateral leg maps to asset 1 (``S_1``, the long leg of the paper) and
    the debt leg maps to asset 2 (``S_2``, the short leg).  This matches the
    long-short convention of ``optimal_long_short.tex`` and generalizes to any
    AAVE market pair.
    """

    collateral: CarryLeg
    debt: CarryLeg

    @property
    def net_carry(self) -> float:
        """Net annualized carry earned on the health-factor drift, r_s - r_b."""
        return self.collateral.rate - self.debt.rate


def apply_carry(params: KouParams, spec: CarrySpec) -> KouParams:
    """Fold deterministic carry into the per-leg price-growth drifts.

    Returns a :class:`KouParams` copy with

        mu_1 -> mu_1 + spec.collateral.rate   (supply APR, earned),
        mu_2 -> mu_2 + spec.debt.rate         (borrow APR, paid),

    so that the health-factor drift gains the net carry ``spec.net_carry`` and
    the terminal payoff gains the per-leg accrual, with the semi-analytic form
    preserved.  ``muX`` drifts and jump compensators are recomputed
    automatically by ``KouParams.__post_init__``.
    """
    return with_price_drift_view(
        params,
        delta_mu1=spec.collateral.rate,
        delta_mu2=spec.debt.rate,
    )


def objective_phi(row: dict[str, float], *, alpha: float, delta: float) -> float:
    """Penalised conditional mean-variance objective of eq. (Phi).

    Phi = E[Pi_T | tau>T] - (alpha/2) Var(Pi_T | tau>T) - delta * p_liq,

    read off a single row of :func:`h0_liquidation_moment_report`.
    """
    mean = row["conditional_mean"]
    var = row.get("conditional_variance", 0.0)
    return mean - 0.5 * alpha * var - delta * row["p_liq"]


def conditional_sharpe(row: dict[str, float]) -> float:
    """Conditional Sharpe ratio E[Pi_T|tau>T] / sd(Pi_T|tau>T) for a row.

    Returns ``nan`` when the conditional variance is non-positive.
    """
    var = row.get("conditional_variance", 0.0)
    if var <= 0.0:
        return float("nan")
    return row["conditional_mean"] / math.sqrt(var)


@dataclass(frozen=True)
class CarryOptimizationResult:
    """Outcome of maximising Phi over an h0 grid under carry.

    Attributes
    ----------
    h0_star, H0_star, L0_star : float
        Optimal log-health, health ratio, and initial gross leverage.
    phi_star, sharpe_star : float
        Objective value and conditional Sharpe at the optimum.
    p_liq_star : float
        Liquidation probability at the optimum.
    carried_params : KouParams
        The carry-adjusted parameters used for the report.
    rows : list[dict[str, float]]
        The full per-h0 report, each row augmented with ``phi`` and
        ``conditional_sharpe``.
    net_carry : float
        The net carry (r_s - r_b) that was applied.
    """

    h0_star: float
    H0_star: float
    L0_star: float
    phi_star: float
    sharpe_star: float
    p_liq_star: float
    carried_params: KouParams
    rows: list[dict[str, float]]
    net_carry: float


def optimal_carry_h0(
    params: KouParams,
    spec: CarrySpec,
    h0_grid: Iterable[float],
    *,
    b: float,
    T: float,
    alpha: float,
    delta: float,
    S10: float = 1.0,
    S20: float = 1.0,
    max_moment_order: int = 4,
) -> CarryOptimizationResult:
    """Maximise the penalised mean-variance objective under carry.

    Applies :func:`apply_carry`, evaluates
    :func:`h0_liquidation_moment_report` on ``h0_grid``, augments each row with
    ``phi`` and ``conditional_sharpe``, and returns the argmax over Phi.
    """
    carried = apply_carry(params, spec)
    rows = h0_liquidation_moment_report(
        carried,
        h0_grid,
        b=b,
        T=T,
        S10=S10,
        S20=S20,
        max_moment_order=max_moment_order,
    )
    if not rows:
        raise ValueError("h0_grid produced no admissible rows.")
    for row in rows:
        row["phi"] = objective_phi(row, alpha=alpha, delta=delta)
        row["conditional_sharpe"] = conditional_sharpe(row)

    best = max(rows, key=lambda r: r["phi"])
    return CarryOptimizationResult(
        h0_star=best["h0"],
        H0_star=best["H0"],
        L0_star=best["initial_leverage"],
        phi_star=best["phi"],
        sharpe_star=best["conditional_sharpe"],
        p_liq_star=best["p_liq"],
        carried_params=carried,
        rows=rows,
        net_carry=spec.net_carry,
    )


def size_for_target_pliq(
    params: KouParams,
    spec: CarrySpec,
    target_pliq: float,
    *,
    b: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    h0_lo: float | None = None,
    h0_hi: float = 3.0,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> dict[str, float]:
    """Size the carry book to a target liquidation probability.

    A practitioner sizing rule that sidesteps the corner/branch structure of the
    penalised objective: choose the cushion ``h0`` such that
    ``p_liq(h0) = target_pliq``.  Because ``p_liq`` is strictly decreasing in
    ``h0`` (more cushion -> safer), the root is unique and found by bisection on
    the carry-adjusted dynamics.

    Returns the report row at the solved ``h0`` (leverage, moments, realized
    ``p_liq``), or the ``h0_hi`` endpoint row if the target is not attainable on
    ``[h0_lo, h0_hi]`` (flagged via ``target_attained``).
    """
    if not (0.0 < target_pliq < 1.0):
        raise ValueError(f"target_pliq must be in (0, 1), got {target_pliq}")
    carried = apply_carry(params, spec)
    # Never search the sub-barrier region: H0 = e^{h0} < 1 means the position
    # starts already liquidated (p_liq == 1), where the Laplace inversion is
    # ill-conditioned.  The smallest meaningful cushion is h0 = 0 (H0 = 1).
    lo = max(math.log(b), 0.0) + 1e-4 if h0_lo is None else float(h0_lo)
    hi = float(h0_hi)

    def pliq_at(h0: float) -> float:
        row = h0_liquidation_moment_report(
            carried, [h0], b=b, T=T, S10=S10, S20=S20, max_moment_order=1
        )[0]
        return row["p_liq"]

    p_lo, p_hi = pliq_at(lo), pliq_at(hi)
    attained = p_hi <= target_pliq <= p_lo
    if not attained:
        h0_star = hi if target_pliq < p_hi else lo
    else:
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if pliq_at(mid) > target_pliq:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        h0_star = 0.5 * (lo + hi)

    row = h0_liquidation_moment_report(
        carried, [h0_star], b=b, T=T, S10=S10, S20=S20, max_moment_order=4
    )[0]
    row["conditional_sharpe"] = conditional_sharpe(row)
    row["target_pliq"] = target_pliq
    row["target_attained"] = 1.0 if attained else 0.0
    row["net_carry"] = spec.net_carry
    return row


def depeg_scenario(
    params: KouParams,
    spec: CarrySpec,
    h0: float,
    depeg_fractions: Iterable[float],
    *,
    b: float,
    T: float,
    S10: float = 1.0,
    S20: float = 1.0,
    max_moment_order: int = 4,
) -> list[dict[str, float]]:
    """Reprice a fixed position under discrete collateral-depeg shocks.

    A separate scenario overlay (not part of the base Kou dynamics): a
    stablecoin-collateral depeg of fraction ``d`` scales the collateral value,
    hence the health factor, by ``(1 - d)``.  This is equivalent to entering the
    *same* holdings at a reduced initial cushion

        h0_eff = h0 + log(1 - d),

    so we reprice the carry-adjusted position at each shocked ``h0_eff``.  Rows
    whose shocked cushion breaches ``log(b)`` (instant liquidation) are reported
    with ``p_liq = 1``.

    Parameters
    ----------
    h0 : float
        The (unshocked) log-health of the position being stressed, e.g. the
        optimum ``h0_star`` from :func:`optimal_carry_h0`.
    depeg_fractions : iterable of float
        Collateral haircuts in [0, 1); ``0.0`` is the no-shock baseline.
    """
    carried = apply_carry(params, spec)
    log_b = math.log(b)
    out: list[dict[str, float]] = []
    for d in depeg_fractions:
        d = float(d)
        if not (0.0 <= d < 1.0):
            raise ValueError(f"depeg fraction must be in [0, 1), got {d}")
        h0_eff = h0 + math.log1p(-d)
        # H_0 = e^{h0_eff} <= 1: the shocked position is liquidatable at inception
        # (and for h0_eff <= log b the holdings themselves are undefined).
        if h0_eff <= 0.0:
            out.append(
                {
                    "depeg_fraction": d,
                    "h0_eff": h0_eff,
                    "H0_eff": math.exp(h0_eff),
                    "p_liq": 1.0,
                    "p_surv": 0.0,
                    "conditional_mean": 0.0,
                    "instant_liquidation": 1.0,
                }
            )
            continue
        row = h0_liquidation_moment_report(
            carried,
            [h0_eff],
            b=b,
            T=T,
            S10=S10,
            S20=S20,
            max_moment_order=max_moment_order,
        )[0]
        row["depeg_fraction"] = d
        row["h0_eff"] = h0_eff
        row["H0_eff"] = row["H0"]
        row["instant_liquidation"] = 0.0
        out.append(row)
    return out
