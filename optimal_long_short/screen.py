"""Pre-registered asset screen and long-short pair cross-section.

The screen is fixed *before* any model output is examined and selects on
liquidity and protocol admissibility only, never on realized performance.  Every
admissible ordered pair is then reported; no pair is chosen as a headline on the
basis of its results.  This is what keeps the empirical section free of
selection bias, and it is why the screening quantities below are all measured at
the *start* of the estimation window.

Screen
------
1. Universe fixed in advance (:data:`optimal_long_short.aave_hourly.RESERVE_TOKENS`):
   blue-chip majors and their canonical wrappers.  No governance tokens, no
   principal-token or yield-bearing wrappers.
2. Not frozen and not paused.
3. Listed for the full estimation window.
4. Ranked by USD TVL at the window start; the top ``n_assets`` are kept.
5. Collateral admissibility: ``baseLTVasCollateral > 0`` and
   ``usageAsCollateralEnabled``.  Debt admissibility: ``borrowingEnabled``.

The resulting cross-section reports, for every ordered (collateral, debt) pair,
the contract terms, the realized carry, the debt leg's utilization regime, and
the diagnostic that decides whether carry matters at all: the ratio of the carry
drift ``c*T`` to the barrier's diffusive scale ``sigma_Z*sqrt(T)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_long_short.aave_hourly import DATA_DIR, published_curve
from optimal_long_short.rate_curve import implied_utilization

#: Maps the fetcher's token keys to reserve symbols in the config table.
TOKEN_SYMBOL = {
    "weth": "WETH",
    "wsteth": "wstETH",
    "weeth": "weETH",
    "cbbtc": "cbBTC",
    "tbtc": "tBTC",
    "wbtc": "WBTC",
    "usdc": "USDC",
    "usdt": "USDT",
    "dai": "DAI",
    "gho": "GHO",
    "usde": "USDe",
    "link": "LINK",
}

WINDOW_START = pd.Timestamp("2026-03-17", tz="UTC")
WINDOW_END = pd.Timestamp("2026-08-04", tz="UTC")
HOURS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class ScreenConfig:
    """Screening parameters, fixed before results are examined."""

    n_assets: int = 8
    window_start: pd.Timestamp = WINDOW_START
    window_end: pd.Timestamp = WINDOW_END
    horizon_years: float = 1.0 / 12.0
    # Sampling interval (hours) for the realised spread volatility sigma_Z.
    # Hourly returns carry oracle microstructure noise (negative lag-1
    # autocorrelation) that inflates sigma_Z; 24 gives a daily estimate.
    vol_sampling_hours: int = 1


def load_panel(
    data_dir: Path | None = None,
    *,
    tokens: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the fetched hourly parquets keyed by reserve symbol."""
    data_dir = data_dir or DATA_DIR
    tokens = tokens or TOKEN_SYMBOL
    panel: dict[str, pd.DataFrame] = {}
    for token, symbol in tokens.items():
        path = data_dir / f"hourly_{token}.parquet"
        if path.is_file():
            panel[symbol] = pd.read_parquet(path)
    return panel


def apply_screen(
    panel: dict[str, pd.DataFrame],
    configs: pd.DataFrame,
    cfg: ScreenConfig | None = None,
) -> pd.DataFrame:
    """Apply the pre-registered screen and return the surviving reserves.

    Ranking uses TVL in USD at ``cfg.window_start`` -- information available at
    the moment the window opens, so the selection is genuinely ex ante.
    """
    cfg = cfg or ScreenConfig()
    rows = []
    for symbol, frame in panel.items():
        if symbol not in configs.index:
            continue
        cfgrow = configs.loc[symbol]
        if bool(cfgrow["isFrozen"]) or bool(cfgrow["isPaused"]):
            continue
        if frame.index[0] > cfg.window_start or frame.index[-1] < cfg.window_end:
            continue  # not listed for the full window
        at_start = frame.loc[: cfg.window_start].iloc[-1]
        rows.append(
            {
                "symbol": symbol,
                "tvl_usd_at_start": float(at_start["TVL_USD"]),
                "ltv": float(cfgrow["baseLTVasCollateral"]),
                "liq_threshold": float(cfgrow["reserveLiquidationThreshold"]),
                "liq_bonus": float(cfgrow["liquidationBonus"]),
                "reserve_factor": float(cfgrow["reserveFactor"]),
                "borrowing_enabled": bool(cfgrow["borrowingEnabled"]),
                "collateral_enabled": bool(cfgrow["usageAsCollateralEnabled"]),
                "u_opt": float(cfgrow["optimalUtilisationRate"]),
            }
        )

    screen = pd.DataFrame(rows).set_index("symbol")
    screen = screen.sort_values("tvl_usd_at_start", ascending=False)
    screen["rank"] = np.arange(1, len(screen) + 1)
    screen["in_universe"] = screen["rank"] <= cfg.n_assets
    screen["admissible_collateral"] = (
        screen["in_universe"] & screen["collateral_enabled"] & (screen["ltv"] > 0.0)
    )
    screen["admissible_debt"] = screen["in_universe"] & screen["borrowing_enabled"]
    return screen


def _window(frame: pd.DataFrame, cfg: ScreenConfig) -> pd.DataFrame:
    return frame.loc[cfg.window_start : cfg.window_end]


def pair_cross_section(
    panel: dict[str, pd.DataFrame],
    configs: pd.DataFrame,
    screen: pd.DataFrame,
    cfg: ScreenConfig | None = None,
) -> pd.DataFrame:
    """Report every admissible ordered (collateral, debt) pair.

    The whole cross-section is returned deliberately: no pair is promoted to a
    headline on the basis of its numbers.

    Key columns
    -----------
    ``b``, ``ltv_max``, ``h0_min``
        Contract terms from the collateral leg.
    ``net_carry``
        Realized ``r_s - r_b`` over the window, annualized decimal.
    ``sigma_Z``
        Annualized volatility of the log price ratio, the barrier's diffusive
        scale.
    ``carry_over_barrier``
        ``c*T / (sigma_Z*sqrt(T))`` -- how many barrier standard deviations the
        carry drift is worth over the horizon.  Small values mean deterministic
        carry cannot matter for sizing, whatever its sign.
    ``debt_frac_stressed``, ``debt_convexity``
        Utilization regime of the debt leg, from inverting its published rate
        curve.  A debt leg that never reaches its kink has no rate channel.
    """
    cfg = cfg or ScreenConfig()
    T = cfg.horizon_years

    collateral = screen.index[screen["admissible_collateral"]]
    debt = screen.index[screen["admissible_debt"]]

    # Per-leg window statistics, computed once.
    stats: dict[str, dict] = {}
    for symbol in set(collateral) | set(debt):
        frame = _window(panel[symbol], cfg)
        curve = published_curve(configs, symbol)
        u = implied_utilization(curve, frame["borrower_variable_apr"].values / 100.0)
        rf = float(configs.loc[symbol, "reserve_factor"]) if "reserve_factor" in configs.columns else float(configs.loc[symbol, "reserveFactor"])
        stressed = u > curve.u_opt
        rb = np.array([curve.borrow_rate(x) for x in u])
        rs = rb * u * (1.0 - rf)
        price = frame["price_usd"].astype(float)
        stats[symbol] = {
            "log_price": np.log(price.values),
            "index": frame.index,
            "r_s": float(np.mean(frame["lender_variable_apr"].values / 100.0)),
            "r_b": float(np.mean(frame["borrower_variable_apr"].values / 100.0)),
            "frac_stressed": float(np.mean(stressed)),
            "convexity": (
                float(np.mean(rs[stressed]) / np.mean(rs[~stressed]))
                if stressed.any() and (~stressed).any()
                else float("nan")
            ),
            "u_mean": float(np.mean(u)),
            "u_max": float(np.max(u)),
        }

    rows = []
    for c_sym in collateral:
        for d_sym in debt:
            if c_sym == d_sym:
                continue
            cs, ds = stats[c_sym], stats[d_sym]
            n = min(len(cs["log_price"]), len(ds["log_price"]))
            ratio = cs["log_price"][-n:] - ds["log_price"][-n:]
            k = max(int(cfg.vol_sampling_hours), 1)
            dz = np.diff(ratio[::k])
            dz = dz[np.isfinite(dz)]
            sigma_z = (
                float(np.std(dz, ddof=1) * np.sqrt(HOURS_PER_YEAR / k))
                if dz.size > 2
                else float("nan")
            )

            b = float(screen.loc[c_sym, "liq_threshold"])
            ltv_max = float(screen.loc[c_sym, "ltv"])
            carry = cs["r_s"] - ds["r_b"]
            barrier = sigma_z * np.sqrt(T)
            rows.append(
                {
                    "collateral": c_sym,
                    "debt": d_sym,
                    "b": b,
                    "ltv_max": ltv_max,
                    "h0_min": float(np.log(b / ltv_max)) if ltv_max > 0 else np.nan,
                    "liq_bonus": float(screen.loc[c_sym, "liq_bonus"]),
                    "r_s": cs["r_s"],
                    "r_b": ds["r_b"],
                    "net_carry": carry,
                    "sigma_Z": sigma_z,
                    "carry_over_barrier": carry * T / barrier if barrier > 0 else np.nan,
                    "debt_u_mean": ds["u_mean"],
                    "debt_u_max": ds["u_max"],
                    "debt_frac_stressed": ds["frac_stressed"],
                    "debt_convexity": ds["convexity"],
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["collateral", "debt"]
    ).reset_index(drop=True)
