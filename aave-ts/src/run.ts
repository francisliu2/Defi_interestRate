import "dotenv/config";
import { OrchestratorParams } from "./params";
import { Orchestrator } from "./orchestrator";
import { MarketRow } from "./fetchMarkets";
import { HistoryRow } from "./fetchHistory";
import { PersistResult } from "./persist";

const PROVIDER_URL = process.env.RPC_URL ?? "https://eth.llamarpc.com";

function printMarketsTable(rows: MarketRow[]): void {
  console.log(`\nMarkets — ${new Date().toISOString()}\n`);
  console.log(
    `${"Asset".padEnd(10)} ${"Supply APY%".padStart(12)} ${"Borrow APY%".padStart(12)} ${"Liquidity (USD)".padStart(18)}`
  );
  console.log("─".repeat(56));
  for (const row of rows) {
    const liq = "$" + row.liquidityUSD.toLocaleString("en-US", { maximumFractionDigits: 0 });
    console.log(
      `${row.symbol.padEnd(10)} ${(row.supplyApy.toFixed(2) + "%").padStart(12)} ${(row.borrowApy.toFixed(2) + "%").padStart(12)} ${liq.padStart(18)}`
    );
  }
}

function printHistoryTable(symbol: string, rows: HistoryRow[]): void {
  console.log(`\nHistory — ${symbol} (${rows.length} samples)\n`);
  console.log(
    `${"Datetime".padEnd(20)} ${"Price".padStart(10)} ${"Supplied USD".padStart(16)} ${"Borrowed USD".padStart(16)} ${"Supply APR%".padStart(12)} ${"VarBorrow APR%".padStart(15)}`
  );
  console.log("─".repeat(91));
  for (const row of rows) {
    const dt = row.datetime.slice(0, 16);
    const price = "$" + row.close.toFixed(4);
    const sup = "$" + row.supplied_usd.toLocaleString("en-US", { maximumFractionDigits: 0 });
    const bor = "$" + row.borrowed_usd.toLocaleString("en-US", { maximumFractionDigits: 0 });
    console.log(
      `${dt.padEnd(20)} ${price.padStart(10)} ${sup.padStart(16)} ${bor.padStart(16)} ${(row.supply_apr.toFixed(4) + "%").padStart(12)} ${(row.variable_borrow_apr.toFixed(4) + "%").padStart(15)}`
    );
  }
}

async function main(): Promise<void> {
  let params: OrchestratorParams;
  try {
    params = OrchestratorParams.fromCLI();
  } catch (err) {
    console.error(`Error: ${(err as Error).message}`);
    console.error(`Usage: tsx src/run.ts --mode=<markets|history|both> --assets=USDC,WETH --days=30 --frequency=<hourly|6h|12h|daily> --concurrency=5 --output=<table|json>`);
    process.exit(1);
  }

  const orchestrator = new Orchestrator(PROVIDER_URL);
  const result = await orchestrator.run(params);

  if (params.output === "json") {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  if (result.markets) printMarketsTable(result.markets);

  if (result.history) {
    for (const [symbol, rows] of Object.entries(result.history)) {
      printHistoryTable(symbol, rows);
    }
  }

  if (result.persisted) {
    console.log();
    for (const [symbol, r] of Object.entries(result.persisted) as [string, PersistResult][]) {
      console.log(`Saved ${symbol}: ${r.filepath} (+${r.addedRows} new, ${r.totalRows} total)`);
    }
  }
}

main().catch(console.error);
