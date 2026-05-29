"""Shared helpers for package-backed job entry points."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from optimal_long_short.kou_model import BivariateKouModel
from optimal_long_short.market_params import MarketParams
from optimal_long_short.model_params import KouParams
from optimal_long_short.moments import ConditionalMoments
from optimal_long_short.strategy import UnitExposureLongShortStrategy
from optimal_long_short.drift import apply_price_drift_view

REPO_ROOT = Path(__file__).resolve().parents[2]
LATEX_DIR = REPO_ROOT / "latex"
RESULTS_DIR = REPO_ROOT / "results"


class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return bool(obj)
        return super().default(obj)


def kou_params_to_dict(params: KouParams) -> dict[str, float]:
    """Convert KouParams to a plain JSON-serializable dict."""
    return {k: float(v) for k, v in dataclasses.asdict(params).items()}


def load_calibrated_params(
    path: Path = RESULTS_DIR / "params_WBTC_WETH.json",
    price_drift_view: dict[str, float] | None = None,
) -> tuple[KouParams, dict[str, float]]:
    """Load saved rate-adjusted KouParams and AAVE constraints.

    ``price_drift_view`` may contain ``mu1``, ``mu2``, ``delta_mu1``, and/or
    ``delta_mu2``.  Views are annualized expected price-growth drifts, matching
    the saved JSON convention.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run  python jobs/calibrate_btc_eth.py  first."
        )
    payload = json.loads(path.read_text())
    p = payload["params"]
    params = KouParams(
        mu1=p["mu1"],
        sigma1=p["sigma1"],
        lam1=p["lam1"],
        p1=p["p1"],
        eta1_pos=p["eta1_pos"],
        eta1_neg=p["eta1_neg"],
        mu2=p["mu2"],
        sigma2=p["sigma2"],
        lam2=p["lam2"],
        p2=p["p2"],
        eta2_pos=p["eta2_pos"],
        eta2_neg=p["eta2_neg"],
        rho=p["rho"],
    )
    constraint = dict(payload["aave_constraint"])
    initial_prices = payload.get("meta", {}).get("initial_prices", {})
    if initial_prices:
        constraint["S10"] = float(initial_prices["WBTC"])
        constraint["S20"] = float(initial_prices["WETH"])
        constraint["initial_price_datetime"] = initial_prices.get("datetime")
        constraint["initial_price_block"] = initial_prices.get("block")
    else:
        constraint.setdefault("S10", 1.0)
        constraint.setdefault("S20", 1.0)
    return apply_price_drift_view(params, price_drift_view), constraint


def dt_from_freq(freq: str) -> float:
    """Convert common return frequencies into year fractions."""
    seconds_by_freq = {"15min": 15 * 60, "1h": 3_600, "4h": 4 * 3_600, "1d": 86_400}
    seconds = seconds_by_freq.get(freq)
    if seconds is None:
        raise ValueError(f"Unknown frequency {freq!r}. Choose from {list(seconds_by_freq)}.")
    return seconds / (365.0 * 24.0 * 3_600.0)


def aave_constraint(b: float, ltv_max: float, liq_bonus: float) -> dict[str, float]:
    """Return the AAVE health-buffer constraint dictionary used by jobs."""
    h0_min = math.log(b / ltv_max)
    return {
        "b": b,
        "ltv_max": ltv_max,
        "liq_bonus": liq_bonus,
        "h0_min": h0_min,
        "H0_min": math.exp(h0_min),
    }


def rate_adjusted_params(raw: KouParams, mu1_adjust: float, mu2_adjust: float) -> KouParams:
    """Return a copy of KouParams with only the two drift terms adjusted."""
    return KouParams(
        mu1=raw.mu1 + mu1_adjust,
        sigma1=raw.sigma1,
        lam1=raw.lam1,
        p1=raw.p1,
        eta1_pos=raw.eta1_pos,
        eta1_neg=raw.eta1_neg,
        mu2=raw.mu2 + mu2_adjust,
        sigma2=raw.sigma2,
        lam2=raw.lam2,
        p2=raw.p2,
        eta2_pos=raw.eta2_pos,
        eta2_neg=raw.eta2_neg,
        rho=raw.rho,
    )


def jump_cumulants(lam: float, p: float, ep: float, en: float) -> tuple[float, float, float]:
    """Annualized jump cumulants lambda E[J], lambda E[J^2], lambda E[J^4]."""
    k1 = lam * (p * ep - (1 - p) * en)
    k2 = lam * (p * 2 * ep ** 2 + (1 - p) * 2 * en ** 2)
    k4 = lam * (p * math.factorial(4) * ep ** 4 + (1 - p) * math.factorial(4) * en ** 4)
    return k1, k2, k4


def standardised_moments(r1: float, r2: float, r3: float, r4: float) -> tuple[float, float, float, float]:
    """Convert first four raw moments into mean, variance, skewness, excess kurtosis."""
    mean = r1
    var = r2 - r1 ** 2
    std = var ** 0.5
    skew = (r3 - 3 * r1 * r2 + 2 * r1 ** 3) / std ** 3
    kurt = (r4 - 4 * r1 * r3 + 6 * r1 ** 2 * r2 - 3 * r1 ** 4) / var ** 2 - 3
    return mean, var, skew, kurt


def frontier_moments(
    kou_kw: dict[str, float],
    h0_grid: np.ndarray,
    b: float,
    T: float,
    include_variance: bool,
    S10: float = 1.0,
    S20: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Compute survival, conditional mean, optional variance, and leverage over h0."""
    params = KouParams(**kou_kw)
    market = MarketParams(b=b, S10=S10, S20=S20)
    ps_list: list[float] = []
    mu_list: list[float] = []
    var_list: list[float] = []

    for h0 in h0_grid:
        try:
            strategy = UnitExposureLongShortStrategy(h0=h0, market=market, T=T)
            moments = ConditionalMoments(params=params, strategy=strategy)
            ps_list.append(float(np.clip(moments.p_surv(), 0.0, 1.0)))
            mu_list.append(float(moments.conditional_mean()))
            if include_variance:
                var_list.append(max(float(moments.conditional_variance()), 0.0))
        except Exception:
            ps_list.append(np.nan)
            mu_list.append(np.nan)
            if include_variance:
                var_list.append(np.nan)

    ps = np.array(ps_list)
    mu = np.array(mu_list)
    leverage = np.exp(h0_grid) / (np.exp(h0_grid) - b)
    if include_variance:
        return ps, mu, np.array(var_list), leverage
    return ps, mu


def plot_ecf_spread_fit(
    r1: np.ndarray,
    r2: np.ndarray,
    dt: float,
    out: Path,
    title: str,
    estimated_params: KouParams,
    estimated_label: str = "ECF estimate",
    true_params: KouParams | None = None,
    true_label: str = "True model",
    fontsize: float = 10.0,
) -> None:
    """Plot empirical and model CFs along the spread direction (s, -s)."""
    from optimal_long_short.calibration import empirical_cf
    import matplotlib.pyplot as plt

    sz = max(float(np.std(r1 - r2, ddof=1)), 1e-8)
    s_max = 5.0 / sz
    s_grid = np.linspace(0.01 / sz, s_max, 200)
    freqs = np.column_stack([s_grid, -s_grid])

    phi_hat = empirical_cf(r1, r2, freqs)
    est_model = BivariateKouModel(estimated_params)
    phi_est = np.array([np.exp(dt * est_model.levy_khintchine(s, -s)) for s in s_grid])

    phi_true = None
    if true_params is not None:
        true_model = BivariateKouModel(true_params)
        phi_true = np.array([np.exp(dt * true_model.levy_khintchine(s, -s)) for s in s_grid])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, qty, label in [
        (axes[0], np.abs, r"Modulus $|\phi(s,-s)|$"),
        (axes[1], np.angle, r"Phase $\arg\phi(s,-s)$"),
    ]:
        ax.plot(s_grid * sz, qty(phi_hat), "k.", ms=2, label="Empirical", alpha=0.6)
        if phi_true is not None:
            ax.plot(s_grid * sz, qty(phi_true), "b--", lw=1.5, label=true_label)
        ax.plot(s_grid * sz, qty(phi_est), "r-", lw=1.5, label=estimated_label)
        ax.set_xlabel(r"$s / \hat{\sigma}_Z$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
