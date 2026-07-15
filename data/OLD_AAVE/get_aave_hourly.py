import pandas as pd 
from dotenv import load_dotenv
import os
import time
import json
import requests
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

API_KEY = os.environ.get("THE_GRAPH_API_KEY") or os.environ.get("GRAPH_API_KEY")

# token ETH addresses
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT_ADDRESS = "0xdac17f958d2ee523a2206206994597c13d831ec7"
DAI_ADDRESS = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
WETH_ADDRESS = '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
USDE_ADDRESS = "0x4c9EDD5852cd905f086C759E8383e09bff1E68B3"

# underlying-asset addresses (Ethereum mainnet) for the reserve-config snapshot
RESERVE_TOKENS = {
    "usdc":   "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "usdt":   "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "dai":    "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "weth":   "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "wbtc":   "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "wsteth": "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
    "reth":   "0xae78736Cd615f374D3085123A210448E74Fc6393",
    "link":   "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "gho":    "0x40D16FC0246aD3160Ccc09B8D0D3A2cD28aE6C2f",
}


def run_query(query: str, endpoint: str, variables: dict | None = None):
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(endpoint, json=payload)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def gql(query: str, endpoint: str, variables: dict | None = None) -> dict:
    while True:
        r = requests.post(endpoint, json={"query": query, "variables": variables or {}})
        if r.status_code == 429:                           # hit the rate limit → wait & retry
            time.sleep(int(r.headers.get("Retry-After", "2")))
            continue
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(json.dumps(data["errors"], indent=2))
        return data["data"]

def to_row(snap: dict) -> dict:
        row = {
            "timestamp"      : int(snap["timestamp"]),
            "block"          : int(snap["blockNumber"]),
            "TVL_USD"        : float(snap["totalValueLockedUSD"]),
            "supplied_USD"   : float(snap["totalDepositBalanceUSD"]),
            "borrowed_USD"   : float(snap["totalBorrowBalanceUSD"]),
            "liquidations"   : float(snap['hourlyLiquidateUSD']),
            "token_balance"  : float(snap["inputTokenBalance"]),
        }

        # initialise APR columns with NaN
        for side in ("lender", "borrower"):
            for typ in ("variable", "stable"):
                row[f"{side}_{typ}_apr"] = None

        SECONDS_PER_YEAR = 60 * 60 * 24 * 365
        for r in snap["rates"]:
            col = f"{r['side'].lower()}_{r['type'].lower()}_apr"
            row[col] = float(r["rate"])
            # row[col] = float(r["rate"]) / 1e27 * SECONDS_PER_YEAR * 100  # % APR
        return row
    
def fetch_daily_snapshots(query, mkt_id: str, endpoint: str, batch: int = 1000, start_ts = 0) -> list[dict]:
      out, skip = [], 0
      while True:
          chunk = gql(query, endpoint, {"mkt": mkt_id, "first": batch, "skip": skip, 'ts': start_ts})\
                      ["marketHourlySnapshots"]
          if not chunk:
              break
          out.extend(chunk)
          skip += len(chunk)
      return out

def get_data_coin_aave(name, start_ts, endpoint):
    coins = {
        "usdt": USDT_ADDRESS,
        "usdc": USDC_ADDRESS,
        "dai": DAI_ADDRESS,
        "weth": WETH_ADDRESS,
        "usde" : USDE_ADDRESS
    }

    coin_address = coins[name]

    MKT_QUERY = """
    query($token: String!) {
        markets(where: {inputToken: $token}) {
        id
        name
        inputToken { symbol }
        }
    }
    """
    markets = gql(MKT_QUERY, endpoint = endpoint, variables = {"token": coin_address})["markets"]
    if not markets:
        raise ValueError(f"{name.upper()} market not found in this subgraph.")
    MARKET_ID = markets[0]["id"]
    print("Found market:", markets[0]["name"], "→", MARKET_ID)


    SNAP_QUERY = """
    query($mkt: String!, $first: Int!, $skip: Int!, $ts: Int!) {
        marketHourlySnapshots(
        where: {market: $mkt, timestamp_gt: $ts}
        orderBy: timestamp
        orderDirection: asc
        first: $first
        skip: $skip
        ) {
        id
        timestamp
        blockNumber
        totalValueLockedUSD
        totalDepositBalanceUSD
        totalBorrowBalanceUSD
        hourlyLiquidateUSD
        inputTokenBalance
        rates {
            side          # LENDER / BORROWER
            type          # VARIABLE / STABLE
            rate          # per-second ray (1e27)
        }
        }
    }
    """

    raw_snaps = fetch_daily_snapshots(SNAP_QUERY, MARKET_ID, endpoint, batch = 1000, start_ts= start_ts)
    print("Fetched", len(raw_snaps), "hourly snapshots")

    
    df = pd.DataFrame(map(to_row, raw_snaps))
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    df.index = df.index.ceil('h')
    df = df.groupby(level = 0).mean()
    full_idx = pd.date_range(df.index[0].ceil("h"),
                            df.index[-1].ceil("h"),
                            freq="1H")
    df = df.reindex(full_idx,method = 'ffill')
    df = df.iloc[:-1,:]
    return df 

def update_aave_parquet(coin: str, version: str, url: str, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        prev = pd.read_parquet(out_path)
        start_ts = int(prev.index[-1].timestamp())
        new = get_data_coin_aave(coin, start_ts=start_ts, endpoint=url)

        full = pd.concat([prev, new], axis=0)
        full = full[~full.index.duplicated(keep="last")]
        full.to_parquet(out_path)
        print(f"----- Updated Historical AAVE {version} ETH Data ({coin}) -----")

    except FileNotFoundError:
        full = get_data_coin_aave(coin, start_ts=0, endpoint=url)
        full.to_parquet(out_path)
        print(f"----- Collected Full Historical AAVE {version} ETH Data ({coin}) -----")



  
# ---------------------------------------------------------------------------
# Current reserve-configuration snapshot (Aave V3 official subgraph)
# ---------------------------------------------------------------------------

# Aave V3 Ethereum mainnet protocol subgraph (decentralized network).
# Confirm the id at https://thegraph.com/explorer if the query returns nothing.
AAVE_V3_ETH_SUBGRAPH = "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g"

RESERVE_QUERY = """
query($assets: [Bytes!]!) {
  reserves(where: {underlyingAsset_in: $assets}) {
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
    price { priceInEth }
  }
}
"""


def parse_reserve(r: dict) -> dict:
    """Convert raw subgraph integers into human-readable units.

    Rates are RAY (1e27); LTV/threshold/factor/bonus are basis points (1e4);
    liquidity amounts are in token units (10**decimals); priceInEth is 1e18.
    """
    dec = int(r["decimals"])
    bonus_bps = int(r["reserveLiquidationBonus"])
    return {
        "symbol":                      r["symbol"],
        "underlyingAsset":             r["underlyingAsset"],
        "isFrozen":                    r["isFrozen"],
        "isPaused":                    r["isPaused"],
        "reserveFactor":               int(r["reserveFactor"]) / 1e4,
        "liquidationBonus":            (bonus_bps - 1e4) / 1e4 if bonus_bps else 0.0,
        "baseLTVasCollateral":         int(r["baseLTVasCollateral"]) / 1e4,
        "reserveLiquidationThreshold": int(r["reserveLiquidationThreshold"]) / 1e4,
        "usageAsCollateralEnabled":    r["usageAsCollateralEnabled"],
        "borrowingEnabled":            r["borrowingEnabled"],
        "availableLiquidity":          int(r["availableLiquidity"]) / 10 ** dec,
        "totalLiquidity":              int(r["totalLiquidity"]) / 10 ** dec,
        "liquidityRate":               int(r["liquidityRate"]) / 1e27,
        "variableBorrowRate":          int(r["variableBorrowRate"]) / 1e27,
        "price_eth":                   int(r["price"]["priceInEth"]) / 1e18,
    }


def get_reserve_configs(tokens: dict, endpoint: str) -> pd.DataFrame:
    """Fetch the current reserve config/state for the given tokens.

    `tokens` maps a short name (e.g. "usdc") to its underlying-asset address.
    Returns one row per token, indexed by short name.
    """
    addr_to_name = {addr.lower(): name for name, addr in tokens.items()}
    assets = list(addr_to_name.keys())

    reserves = gql(RESERVE_QUERY, endpoint=endpoint, variables={"assets": assets})["reserves"]
    if not reserves:
        raise ValueError("No reserves returned — check the subgraph id / addresses.")

    rows = {}
    for r in reserves:
        name = addr_to_name.get(r["underlyingAsset"].lower())
        if name is None:
            continue
        # keep the first match per token (avoids stale/duplicate reserves)
        rows.setdefault(name, parse_reserve(r))

    missing = [n for n in tokens if n not in rows]
    if missing:
        print(f"[WARN] no reserve found for: {', '.join(missing)}")

    df = pd.DataFrame([rows[n] for n in tokens if n in rows],
                      index=[n for n in tokens if n in rows])
    df.index.name = "token"
    return df


if __name__ == "__main__":
    url_v3_eth = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{AAVE_V3_ETH_SUBGRAPH}"
    configs = get_reserve_configs(RESERVE_TOKENS, url_v3_eth)
    pd.set_option("display.max_columns", None, "display.width", 200)
    print(configs)

    out_path = Path("./data/AAVE/aave_v3_reserve_configs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    configs.to_csv(out_path)
    print(f"----- Saved reserve configs for {len(configs)} tokens -> {out_path} -----")


def _update_hourly_snapshots():
   for coin in ['usdt', 'usdc', 'dai', 'usde', 'weth']:
        # v3
        try:
            id_v3_eth = "JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk"
            url_v3_eth = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{id_v3_eth}"
            update_aave_parquet(coin, "v3", url_v3_eth, f'./data/AAVE/aave_v3_{coin}_eth.parquet')
        except Exception as e:
            print(f"[ERROR] coin={coin} version=v3 -> {e}")

        # v2
        try:
            id_v2_eth = "C2zniPn45RnLDGzVeGZCx2Sw3GXrbc9gL4ZfL8B8Em2j"
            url_v2_eth = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{id_v2_eth}"
            update_aave_parquet(coin, "v2", url_v2_eth, f'./data/AAVE/aave_v2_{coin}_eth.parquet')
        except Exception as e:
            print(f"[ERROR] coin={coin} version=v2 -> {e}")