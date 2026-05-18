import { ethers } from "ethers";
import { AaveV3Ethereum } from "@bgd-labs/aave-address-book";
import { OrchestratorParams, FREQUENCY_BLOCKS } from "./params";
import { fetchMarkets, MarketRow } from "./fetchMarkets";
import { fetchHistory, HistoryRow } from "./fetchHistory";
import { persistHistory, PersistResult } from "./persist";

export interface OrchestratorResult {
  fetchedAt: string;
  markets?: MarketRow[];
  history?: Record<string, HistoryRow[]>;
  persisted?: Record<string, PersistResult>;
}

export class Orchestrator {
  private provider: ethers.providers.JsonRpcProvider;

  constructor(rpcUrl: string) {
    this.provider = new ethers.providers.JsonRpcProvider(rpcUrl);
  }

  async run(params: OrchestratorParams): Promise<OrchestratorResult> {
    const result: OrchestratorResult = { fetchedAt: new Date().toISOString() };
    const tasks: Promise<void>[] = [];

    if (params.mode === "markets" || params.mode === "both") {
      const filter = params.mode === "both" ? params.assets.map(String) : undefined;
      tasks.push(
        fetchMarkets(this.provider, filter).then((rows) => {
          result.markets = rows;
        })
      );
    }

    if (params.mode === "history" || params.mode === "both") {
      tasks.push(
        Promise.all(
          params.assets.map(async (symbol) => {
            const address = AaveV3Ethereum.ASSETS[symbol].UNDERLYING;
            const rows = await fetchHistory(
              this.provider,
              address,
              params.days,
              FREQUENCY_BLOCKS[params.frequency],
              params.concurrency
            );

            let persistResult: PersistResult | undefined;
            if (params.persist) {
              persistResult = await persistHistory(
                rows,
                String(symbol),
                params.chain,
                params.frequency,
                params.outDir
              );
            }

            return { symbol: String(symbol), rows, persistResult };
          })
        ).then((entries) => {
          result.history = Object.fromEntries(entries.map((e) => [e.symbol, e.rows]));
          if (params.persist) {
            result.persisted = Object.fromEntries(
              entries
                .filter((e) => e.persistResult)
                .map((e) => [e.symbol, e.persistResult!])
            );
          }
        })
      );
    }

    await Promise.all(tasks);
    return result;
  }
}
