"""Aave v3 Ethereum hourly reserve history and configuration fetcher.

Replaces the ad-hoc ``data/OLD_AAVE/get_aave_hourly.py`` script.  Three fixes
matter relative to that version:

1. **API key loading.**  ``data/OLD_AAVE/.env`` declares the variable name with a
   trailing space (``"THE_GRAPH_API_KEY "``), so ``os.environ.get`` returned
   ``None`` and every fetch silently ran unauthenticated.  :func:`graph_api_key`
   strips whitespace from both names and values and searches several locations.
2. **Universe.**  The old ``get_data_coin_aave`` hardcoded a five-coin dict, so
   WBTC/wstETH/cbBTC could never be fetched through the hourly path even though
   they were listed in ``RESERVE_TOKENS``.  Markets are now resolved by
   underlying-asset address over the whole universe.
3. **Utilization.**  The quantity the interest-rate model actually needs is
   ``U = borrowed_USD / supplied_USD``, the argument of Aave's piecewise-linear
   rate curve.  It is derived here rather than rediscovered downstream.

The hourly snapshots come from the Messari-schema subgraph; reserve
configuration (LTV, liquidation threshold, liquidation bonus, reserve factor)
comes from the official Aave v3 subgraph.  Rates in the Messari schema are
already annualized percentages, so no RAY conversion is applied.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "AAVE"

# Messari-schema subgraph: marketHourlySnapshots (history).
SUBGRAPH_HOURLY_V3_ETH = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"
# Official Aave v3 subgraph: reserve configuration snapshot.
SUBGRAPH_CONFIG_V3_ETH = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"

#: Blue-chip Ethereum-mainnet underlying assets.  Membership here is a
#: *universe* definition fixed in advance; the liquidity and admissibility
#: screens in :mod:`optimal_long_short.screen` do the actual selection.
RESERVE_TOKENS: dict[str, str] = {
    "weth":   "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "wsteth": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
    "weeth":  "0xCd5fE23C85820F7B72D0926FC9b05b43E359b7ee",
    "cbbtc":  "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf",
    "tbtc":   "0x18084fbA666a33d37592fA2633fD49a74DD93a88",
    "wbtc":   "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "usdc":   "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "usdt":   "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "dai":    "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "gho":    "0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f",
    "usde":   "0x4c9EDD5852cd905f086C759E8383e09bff1E68B3",
    "link":   "0x514910771AF9Ca656af840dff83E8264EcF986CA",
}

_ENV_CANDIDATES = (
    REPO_ROOT / ".env",
    REPO_ROOT / "data" / "OLD_AAVE" / ".env",
)
_KEY_NAMES = ("THE_GRAPH_API_KEY", "GRAPH_API_KEY")
# The Graph gateway keys are 32 hex characters; the repo root .env holds a
# 48-character key for a different service, so length disambiguates them.
_GATEWAY_KEY_LEN = 32


def _parse_env(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file, stripping whitespace from names and values."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        out[name.strip()] = value.strip().strip("'\"")
    return out


def graph_api_key() -> str:
    """Return The Graph gateway API key.

    Searches the process environment then the repo's ``.env`` files, stripping
    whitespace from variable names (see the module docstring for why).  When
    several candidates are found, one of gateway-key length is preferred.
    """
    found: list[str] = []
    for name in _KEY_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            found.append(value.strip())
    for path in _ENV_CANDIDATES:
        env = _parse_env(path)
        for name in _KEY_NAMES:
            if env.get(name):
                found.append(env[name])
    if not found:
        raise RuntimeError(
            "No Graph API key found.  Set THE_GRAPH_API_KEY in the environment "
            f"or in one of: {', '.join(str(p) for p in _ENV_CANDIDATES)}"
        )
    for value in found:
        if len(value) == _GATEWAY_KEY_LEN:
            return value
    return found[0]


def endpoint(subgraph_id: str, api_key: str | None = None) -> str:
    """Gateway URL for a decentralized-network subgraph."""
    return (
        "https://gateway.thegraph.com/api/"
        f"{api_key or graph_api_key()}/subgraphs/id/{subgraph_id}"
    )


def gql(
    query: str,
    url: str,
    variables: dict | None = None,
    *,
    max_retries: int = 5,
    timeout: float = 90.0,
) -> dict:
    """POST a GraphQL query, retrying on rate limits and transient failures."""
    payload = {"query": query, "variables": variables or {}}
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code == 429:
            time.sleep(float(response.headers.get("Retry-After", delay)))
            delay *= 2
            continue
        response.raise_for_status()
        body = response.json()
        if "errors" in body:
            raise RuntimeError(json.dumps(body["errors"], indent=2))
        return body["data"]
    raise RuntimeError(f"Query failed after {max_retries} attempts: {url}")


# ---------------------------------------------------------------------------
# Hourly history
# ---------------------------------------------------------------------------

_MARKET_QUERY = """
query($token: String!) {
  markets(where: {inputToken: $token}) {
    id
    name
    inputToken { symbol decimals }
  }
}
"""

_SNAPSHOT_QUERY = """
query($mkt: String!, $first: Int!, $skip: Int!, $ts: Int!) {
  marketHourlySnapshots(
    where: {market: $mkt, timestamp_gt: $ts}
    orderBy: timestamp
    orderDirection: asc
    first: $first
    skip: $skip
  ) {
    timestamp
    blockNumber
    totalValueLockedUSD
    totalDepositBalanceUSD
    totalBorrowBalanceUSD
    hourlyLiquidateUSD
    inputTokenBalance
    rates { side type rate }
  }
}
"""

# The subgraph rejects skip beyond this, so paging switches to timestamp
# advancement once the window is exhausted.
_MAX_SKIP = 5000


def resolve_market(token: str, url: str) -> tuple[str, str, int]:
    """Return ``(market_id, symbol, decimals)`` for an underlying-asset address."""
    address = RESERVE_TOKENS.get(token.lower(), token).lower()
    markets = gql(_MARKET_QUERY, url, {"token": address})["markets"]
    if not markets:
        raise ValueError(f"No Aave market found for {token!r} ({address}).")
    tok = markets[0]["inputToken"]
    return markets[0]["id"], tok["symbol"], int(tok["decimals"])


def _snapshot_row(snap: dict) -> dict:
    row = {
        "timestamp": int(snap["timestamp"]),
        "block": int(snap["blockNumber"]),
        "TVL_USD": float(snap["totalValueLockedUSD"]),
        "supplied_USD": float(snap["totalDepositBalanceUSD"]),
        "borrowed_USD": float(snap["totalBorrowBalanceUSD"]),
        "liquidations": float(snap["hourlyLiquidateUSD"]),
        "token_balance": float(snap["inputTokenBalance"]),
    }
    for side in ("lender", "borrower"):
        for kind in ("variable", "stable"):
            row[f"{side}_{kind}_apr"] = None
    # Messari rates are already annualized percentages.
    for rate in snap["rates"]:
        row[f"{rate['side'].lower()}_{rate['type'].lower()}_apr"] = float(rate["rate"])
    return row


def fetch_hourly(
    token: str,
    *,
    url: str | None = None,
    start_ts: int = 0,
    end_ts: int | None = None,
    batch: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch hourly market snapshots for one reserve.

    Parameters
    ----------
    token : str
        Key of :data:`RESERVE_TOKENS`, or a raw underlying-asset address.
    start_ts, end_ts : int
        Unix-second bounds.  ``start_ts`` is exclusive (the subgraph filters on
        ``timestamp_gt``); ``end_ts`` truncates after fetching.

    Returns
    -------
    DataFrame indexed by hourly UTC timestamp, forward-filled onto a complete
    hourly grid, with a derived ``utilization`` column.
    """
    url = url or endpoint(SUBGRAPH_HOURLY_V3_ETH)
    market_id, symbol, decimals = resolve_market(token, url)
    if verbose:
        print(f"  {token:>7} -> market {market_id} ({symbol})")

    raw: list[dict] = []
    cursor, skip = int(start_ts), 0
    while True:
        chunk = gql(
            _SNAPSHOT_QUERY,
            url,
            {"mkt": market_id, "first": batch, "skip": skip, "ts": cursor},
        )["marketHourlySnapshots"]
        if not chunk:
            break
        raw.extend(chunk)
        if end_ts is not None and int(chunk[-1]["timestamp"]) >= end_ts:
            break
        skip += len(chunk)
        if skip >= _MAX_SKIP:
            # Advance the timestamp cursor instead of paging deeper.
            cursor, skip = int(chunk[-1]["timestamp"]), 0

    if not raw:
        raise ValueError(f"No snapshots returned for {token!r}.")

    frame = pd.DataFrame(map(_snapshot_row, raw))
    frame = frame.drop_duplicates(subset="timestamp")
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    frame = frame.set_index("datetime").sort_index()
    if end_ts is not None:
        frame = frame[frame["timestamp"] <= end_ts]

    # Snapshots can be missing or duplicated within an hour; collapse onto a
    # regular hourly grid and forward-fill gaps.
    frame.index = frame.index.ceil("h")
    counts = frame.groupby(level=0).size()
    frame = frame.groupby(level=0).mean()
    grid = pd.date_range(frame.index[0], frame.index[-1], freq="1h", tz="UTC")
    frame = frame.reindex(grid, method="ffill")
    # Number of real snapshots in each hour; 0 marks a forward-filled (stale) hour.
    frame["n_snapshots"] = counts.reindex(grid).fillna(0).astype(int).values
    frame.index.name = "datetime"

    frame["utilization"] = (frame["borrowed_USD"] / frame["supplied_USD"]).clip(0.0, 1.0)
    # Oracle-implied USD price per whole token.  ``inputTokenBalance`` is in the
    # token's smallest unit, so it must be scaled by 10**decimals; files written
    # before this fix hold TVL / raw balance, off by exactly that constant (log
    # returns are unaffected).
    frame["price_usd"] = frame["TVL_USD"] / (
        frame["token_balance"].replace(0.0, pd.NA) / 10.0**decimals
    )
    frame["symbol"] = symbol
    if verbose:
        print(
            f"  {token:>7}    {len(frame):5d} hours "
            f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}  "
            f"U in [{frame.utilization.min():.3f}, {frame.utilization.max():.3f}]"
        )
    return frame


def fetch_universe(
    tokens: Iterable[str] = tuple(RESERVE_TOKENS),
    *,
    start_ts: int = 0,
    end_ts: int | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch hourly history for several reserves, caching one parquet each.

    Reserves that fail (delisted, no market, subgraph error) are reported and
    skipped rather than aborting the run.
    """
    out_dir = out_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    url = endpoint(SUBGRAPH_HOURLY_V3_ETH)

    frames: dict[str, pd.DataFrame] = {}
    for token in tokens:
        try:
            frame = fetch_hourly(
                token, url=url, start_ts=start_ts, end_ts=end_ts, verbose=verbose
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [SKIP] {token}: {exc}")
            continue
        frame.to_parquet(out_dir / f"hourly_{token}.parquet")
        frames[token] = frame
    return frames


# ---------------------------------------------------------------------------
# Reserve configuration
# ---------------------------------------------------------------------------

_RESERVE_QUERY = """
query {
  reserves(first: 200) {
    symbol
    underlyingAsset
    decimals
    isFrozen
    isPaused
    reserveFactor
    usageAsCollateralEnabled
    borrowingEnabled
    baseLTVasCollateral
    reserveLiquidationThreshold
    reserveLiquidationBonus
    availableLiquidity
    totalLiquidity
    liquidityRate
    variableBorrowRate
    optimalUtilisationRate
    baseVariableBorrowRate
    variableRateSlope1
    variableRateSlope2
    price { priceInEth }
  }
}
"""


@dataclass(frozen=True)
class RateCurve:
    """Aave's deterministic piecewise-linear borrow-rate curve.

    ``r_b(U) = base + slope1 * U / U_opt``                      for ``U <= U_opt``
    ``r_b(U) = base + slope1 + slope2 * (U - U_opt)/(1 - U_opt)`` otherwise.

    All rates are annualized decimals.  The break at ``u_opt`` is the source of
    the convexity that a supplier is long and a borrower is short.
    """

    u_opt: float
    base: float
    slope1: float
    slope2: float

    def borrow_rate(self, u: float) -> float:
        if u <= self.u_opt:
            return self.base + self.slope1 * (u / self.u_opt if self.u_opt else 0.0)
        excess = (u - self.u_opt) / (1.0 - self.u_opt)
        return self.base + self.slope1 + self.slope2 * excess

    def supply_rate(self, u: float, reserve_factor: float) -> float:
        """Supplier's rate: ``U * r_b(U) * (1 - reserve_factor)``."""
        return u * self.borrow_rate(u) * (1.0 - reserve_factor)


def _ray(value) -> float:
    return int(value) / 1e27


def _bps(value) -> float:
    return int(value) / 1e4


def _parse_reserve(raw: dict) -> dict:
    decimals = int(raw["decimals"])
    bonus = int(raw["reserveLiquidationBonus"])
    total = int(raw["totalLiquidity"]) / 10**decimals
    available = int(raw["availableLiquidity"]) / 10**decimals
    return {
        "symbol": raw["symbol"],
        "underlyingAsset": raw["underlyingAsset"],
        "isFrozen": raw["isFrozen"],
        "isPaused": raw["isPaused"],
        "reserveFactor": _bps(raw["reserveFactor"]),
        # Stored as 1e4 + bonus, so 10500 means a 5% bonus.
        "liquidationBonus": (bonus - 1e4) / 1e4 if bonus else 0.0,
        "baseLTVasCollateral": _bps(raw["baseLTVasCollateral"]),
        "reserveLiquidationThreshold": _bps(raw["reserveLiquidationThreshold"]),
        "usageAsCollateralEnabled": raw["usageAsCollateralEnabled"],
        "borrowingEnabled": raw["borrowingEnabled"],
        "totalLiquidity": total,
        "availableLiquidity": available,
        "utilization": 1.0 - available / total if total > 0 else float("nan"),
        "liquidityRate": _ray(raw["liquidityRate"]),
        "variableBorrowRate": _ray(raw["variableBorrowRate"]),
        "optimalUtilisationRate": _ray(raw["optimalUtilisationRate"]),
        "baseVariableBorrowRate": _ray(raw["baseVariableBorrowRate"]),
        "variableRateSlope1": _ray(raw["variableRateSlope1"]),
        "variableRateSlope2": _ray(raw["variableRateSlope2"]),
        "price_eth": int(raw["price"]["priceInEth"]) / 1e18,
    }


def fetch_reserve_configs(url: str | None = None) -> pd.DataFrame:
    """Fetch current reserve configuration for every Aave v3 Ethereum reserve.

    Includes the published rate-curve parameters (``optimalUtilisationRate``,
    ``baseVariableBorrowRate``, ``variableRateSlope1``, ``variableRateSlope2``)
    needed to build :class:`RateCurve` without estimating slopes from data.
    """
    url = url or endpoint(SUBGRAPH_CONFIG_V3_ETH)
    reserves = gql(_RESERVE_QUERY, url)["reserves"]
    rows = []
    for raw in reserves:
        try:
            rows.append(_parse_reserve(raw))
        except (KeyError, TypeError, ValueError):
            continue
    frame = pd.DataFrame(rows).set_index("symbol").sort_values(
        "totalLiquidity", ascending=False
    )
    return frame


def published_curve(configs: pd.DataFrame, symbol: str) -> RateCurve:
    """Build a :class:`RateCurve` from the protocol's published parameters."""
    row = configs.loc[symbol]
    return RateCurve(
        u_opt=float(row["optimalUtilisationRate"]),
        base=float(row["baseVariableBorrowRate"]),
        slope1=float(row["variableRateSlope1"]),
        slope2=float(row["variableRateSlope2"]),
    )
