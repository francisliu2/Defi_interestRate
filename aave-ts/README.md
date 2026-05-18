# aave-ts

TypeScript tool for fetching Aave V3 Ethereum market data via on-chain contract calls. Supports live market snapshots, historical rate/liquidity series, or both — configurable through a typed parameter class or CLI flags.

## Setup

```bash
cp .env.example .env   # add your RPC URL
npm install
```

`.env`:

```env
RPC_URL=https://mainnet.infura.io/v3/<YOUR_KEY>
```

Free RPC options: [Infura](https://infura.io), [Alchemy](https://alchemy.com), or `https://rpc.ankr.com/eth` (no key, rate-limited).

## Usage

All data fetching goes through `src/run.ts` via `npm run run`:

```bash
npm run run -- --mode=<markets|history|both> --assets=<SYMBOL,...> --days=<N> --frequency=<hourly|6h|12h|daily> --output=<table|json>
```

| Parameter | Default | Description |
| --- | --- | --- |
| `--mode` | `both` | What to fetch: `markets`, `history`, or `both` |
| `--assets` | `USDC` | Comma-separated asset symbols (see below for full list) |
| `--days` | `30` | History window in days (1–365); ignored in `markets` mode |
| `--frequency` | `daily` | Sampling interval: `hourly`, `6h`, `12h`, or `daily` |
| `--concurrency` | `1` | Max simultaneous RPC requests (1–50); raise if fetching many assets |
| `--no-persist` | *(off)* | Disable saving results to Parquet (persistence is on by default) |
| `--out-dir` | `./data/AAVE` | Directory for Parquet files |
| `--chain` | `ethereum` | Chain name embedded in the filename |
| `--output` | `table` | Print a formatted table or raw `json` to stdout |

`--frequency` controls how many on-chain block queries are made per day:

| Frequency | Blocks between samples | Samples per day | 30-day total |
| --- | --- | --- | --- |
| `daily` | 7 200 | 1 | 31 |
| `12h` | 3 600 | 2 | 61 |
| `6h` | 1 800 | 4 | 121 |
| `hourly` | 300 | 24 | 721 |

Each sample makes 3 parallel RPC calls (reserve data, oracle price, block). Higher frequency multiplies that cost. Infura's free tier (100k requests/day) comfortably covers daily and 6h; hourly over many assets or long windows may approach the limit.

### Examples

```bash
# Live snapshot of all ~70 Aave V3 assets
npm run fetch

# 30-day daily USDC borrow rate history (default)
npm run fetch:history

# 7-day hourly history for USDC
npm run run -- --mode=history --assets=USDC --days=7 --frequency=hourly

# Both market snapshot + 6h history, filtered to USDC and WETH
npm run run -- --mode=both --assets=USDC,WETH --days=30 --frequency=6h

# 60-day daily history for three stablecoins, JSON output
npm run run -- --mode=history --assets=USDC,USDT,DAI --days=60 --output=json

# Fetch without saving to Parquet
npm run run -- --mode=history --assets=USDC --days=7 --no-persist

# Save to a custom directory
npm run run -- --mode=history --assets=WETH --days=30 --out-dir=../data
```

### Sample output — table mode

```text
Markets — 2026-05-18T19:35:11.244Z

Asset       Supply APY%  Borrow APY%    Liquidity (USD)
────────────────────────────────────────────────────────
WETH              1.43%        1.93%       $393,447,871
USDC              3.36%        4.07%       $168,442,812

History — USDC (9 samples)

Datetime                  Price     Supplied USD     Borrowed USD  Supply APR%  VarBorrow APR%
───────────────────────────────────────────────────────────────────────────────────────────────
2026-05-16T19:39         $1.0001    $1,842,301,024     $912,456,788      3.2999%         3.9927%
2026-05-17T01:40         $1.0001    $1,843,100,512     $913,201,344      3.3075%         3.9973%
...
2026-05-18T19:48         $1.0001    $1,845,200,000     $914,800,000      3.3000%         3.9928%

Saved USDC: ./data/AAVE/USDC_AAVEv3_ethereum_6h_rates.parquet (+9 new, 9 total)
```

### Sample output — JSON mode

```json
{
  "fetchedAt": "2026-05-18T19:35:20.786Z",
  "markets": [
    { "symbol": "USDC", "supplyApy": 3.36, "borrowApy": 4.07, "liquidityUSD": 168442812 }
  ],
  "history": {
    "USDC": [
      {
        "datetime": "2026-05-16T19:39:00.000Z",
        "block": 22500000,
        "close": 1.0001,
        "token_balance": 1842301024.5,
        "supplied_usd": 1842301024.5,
        "borrowed_usd": 912456788.2,
        "tvl_usd": 1842301024.5,
        "supply_apr": 3.2999,
        "variable_borrow_apr": 3.9927,
        "stable_borrow_apr": 0.0
      }
    ]
  }
}
```

## Parquet schema

Each history row persisted to `./data/AAVE/<SYMBOL>_AAVEv3_<chain>_<frequency>_rates.parquet` contains:

| Column | Type | Description |
| --- | --- | --- |
| `datetime` | string | ISO 8601 timestamp of the sampled block |
| `block` | int64 | Block number |
| `close` | double | Asset price in USD (Aave oracle) |
| `token_balance` | double | Total aToken supply in token units |
| `supplied_usd` | double | Total deposited value in USD |
| `borrowed_usd` | double | Total borrowed value in USD (stable + variable) |
| `tvl_usd` | double | Total value locked in USD (= `supplied_usd`) |
| `supply_apr` | double | Annualised supply APR % (what lenders earn) |
| `variable_borrow_apr` | double | Annualised variable borrow APR % |
| `stable_borrow_apr` | double | Annualised stable borrow APR % (deprecated in Aave v3, usually 0) |

Rows are deduplicated by `datetime` and sorted chronologically on each write. **Existing files must be deleted if you ran a previous version** — the old schema (`supplyApy`, `borrowApy`) is incompatible.

## Available assets

Any symbol from Aave V3 Ethereum is valid: `USDC`, `USDT`, `DAI`, `WETH`, `WBTC`, `AAVE`, `LINK`, `UNI`, `GHO`, `cbBTC`, `sUSDe`, `USDe`, and ~60 more. Passing an unknown symbol exits with an error listing all valid options.

## Architecture

```text
src/
  params.ts       OrchestratorParams class — typed fields, validation, CLI parser
  orchestrator.ts Orchestrator class — dispatches fetchers and persistence in parallel
  fetchMarkets.ts fetchMarkets() — reads live data from UiPoolDataProvider contract
  fetchHistory.ts fetchHistory() — reads historical blocks via PoolDataProvider
  persist.ts      persistHistory() — merges and writes HistoryRows to Parquet
  run.ts          CLI entry point — parses args, runs orchestrator, prints output
data/AAVE/
  USDC_AAVEv3_ethereum_daily_rates.parquet
  WETH_AAVEv3_ethereum_6h_rates.parquet
  ...
```

`markets` and `history` fetches run concurrently. When multiple assets are requested in `history` mode, all assets are also fetched in parallel.

### Using the orchestrator programmatically

```ts
import { Orchestrator } from "./orchestrator";
import { OrchestratorParams } from "./params";

const params = new OrchestratorParams({
  mode: "history",
  assets: ["USDC", "WETH"],
  days: 90,
  frequency: "6h",
  concurrency: 1,
  persist: true,
  outDir: "./data/AAVE",
  chain: "ethereum",
});

const orchestrator = new Orchestrator(process.env.RPC_URL!);
const result = await orchestrator.run(params);
// result.history["USDC"]   → HistoryRow[]
// result.persisted["USDC"] → { filepath, addedRows, totalRows }
```

## Key packages

| Package | Purpose |
| --- | --- |
| `@aave/contract-helpers` | ABI + typed wrappers for Aave V3 contracts |
| `@aave/math-utils` | Converts raw ray values to human-readable APY/APR |
| `@bgd-labs/aave-address-book` | Canonical contract addresses (tracks deployments) |
| `ethers` v5 | Ethereum RPC provider and historical block queries |
| `@dsnp/parquetjs` | Read/write Parquet files in Node.js |
