import { ethers } from "ethers";
import { AaveV3Ethereum } from "@bgd-labs/aave-address-book";
import { OrchestratorParams, FREQUENCY_BLOCKS } from "./params";
import { fetchMarkets, MarketRow } from "./fetchMarkets";
import { buildHistoryBlockSchedule, fetchHistory, HistoryRow } from "./fetchHistory";
import { persistHistory, PersistResult } from "./persist";

export interface OrchestratorResult {
  fetchedAt: string;
  markets?: MarketRow[];
  history?: Record<string, HistoryRow[]>;
  persisted?: Record<string, PersistResult>;
}

export class Orchestrator {
  private provider: ethers.providers.JsonRpcProvider;
  private rpcUrl: string;

  constructor(rpcUrl: string) {
    this.rpcUrl = rpcUrl;
    this.provider = new ethers.providers.JsonRpcProvider(rpcUrl);
  }

  async run(params: OrchestratorParams): Promise<OrchestratorResult> {
    const fetchedAt = new Date().toISOString();
    const result: OrchestratorResult = { fetchedAt };
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
        (async () => {
          const entries: { symbol: string; address: string; rows: HistoryRow[]; persistResult?: PersistResult }[] = [];
          const latestBlock = await this.provider.getBlockNumber();
          const blockSchedule = buildHistoryBlockSchedule(
            latestBlock,
            params.days,
            FREQUENCY_BLOCKS[params.frequency]
          );

          for (const symbol of params.assets) {
            const address = AaveV3Ethereum.ASSETS[symbol].UNDERLYING;
            const rows = await fetchHistory(
              this.provider,
              address,
              params.days,
              FREQUENCY_BLOCKS[params.frequency],
              params.concurrency,
              String(symbol),
              blockSchedule
            );

            entries.push({ symbol: String(symbol), address, rows });
          }

          if (entries.length > 1) {
            const commonBlocks = entries
              .map((entry) => new Set(entry.rows.map((row) => row.block)))
              .reduce((acc, blocks) => new Set([...acc].filter((block) => blocks.has(block))));

            for (const entry of entries) {
              const before = entry.rows.length;
              entry.rows = entry.rows.filter((row) => commonBlocks.has(row.block));
              const dropped = before - entry.rows.length;
              if (dropped > 0) {
                process.stderr.write(
                  `\n  aligned ${entry.symbol}: dropped ${dropped} samples not present for every requested asset\n`
                );
              }
            }

            if (commonBlocks.size === 0) {
              throw new Error(
                `No common successful history samples across requested assets: ${entries.map((entry) => entry.symbol).join(", ")}. ` +
                `Try a shorter lookback, latest-only --days=0, or an archive-capable RPC.`
              );
            }
          }

          if (params.persist) {
            for (const entry of entries) {
              entry.persistResult = await persistHistory(
                entry.rows,
                {
                  symbol: entry.symbol,
                  chain: params.chain,
                  frequency: params.frequency,
                  blocksPerSample: FREQUENCY_BLOCKS[params.frequency],
                  requestedDays: params.days,
                  scheduledLatestBlock: blockSchedule.latestBlock,
                  scheduledSampleCount: blockSchedule.blockTags.length,
                  assetAddress: entry.address,
                  outDir: params.outDir,
                  fetchedAt,
                  providerUrl: this.rpcUrl,
                }
              );
            }
          }

          result.history = Object.fromEntries(entries.map((e) => [e.symbol, e.rows]));
          if (params.persist) {
            result.persisted = Object.fromEntries(
              entries
                .filter((e) => e.persistResult)
                .map((e) => [e.symbol, e.persistResult!])
            );
          }
        })()
      );
    }

    await Promise.all(tasks);
    return result;
  }
}
