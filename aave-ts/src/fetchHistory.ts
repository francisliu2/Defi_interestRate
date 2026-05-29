import "dotenv/config";
import { ethers } from "ethers";
import { AaveV3Ethereum } from "@bgd-labs/aave-address-book";

// Returns (unbacked, accruedToTreasury, totalAToken, totalStableDebt,
//          totalVariableDebt, liquidityRate, variableBorrowRate, stableBorrowRate, ...)
const RESERVE_DATA_ABI = [
  "function getReserveData(address asset) view returns (uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint256, uint40)",
];

const ORACLE_ABI = ["function getAssetPrice(address asset) view returns (uint256)"];
const ERC20_ABI = ["function decimals() view returns (uint8)"];

const BLOCKS_PER_DAY = 7_200;

export interface HistoryRow {
  datetime: string;       // ISO 8601
  block: number;
  close: number;          // oracle price in USD
  token_balance: number;  // total aToken supply in token units
  supplied_usd: number;
  borrowed_usd: number;
  tvl_usd: number;        // same as supplied_usd
  supply_apr: number;     // liquidityRate — what lenders earn
  variable_borrow_apr: number;
  stable_borrow_apr: number;
}

export interface HistoryBlockSchedule {
  latestBlock: number;
  blockTags: number[];
}

function rayToPercent(ray: ethers.BigNumber): number {
  return parseFloat(ethers.utils.formatUnits(ray, 27)) * 100;
}

export function buildHistoryBlockSchedule(
  latestBlock: number,
  days: number,
  blocksPerSample: number = BLOCKS_PER_DAY
): HistoryBlockSchedule {
  const numSamples = Math.floor((days * BLOCKS_PER_DAY) / blocksPerSample) + 1;
  const blockTags = Array.from({ length: numSamples }, (_, i) =>
    latestBlock - (numSamples - 1 - i) * blocksPerSample
  );
  return { latestBlock, blockTags };
}

function rpcErrorSummary(err: unknown): string {
  const e = err as { code?: string; reason?: string; error?: { code?: string; error?: { message?: string } } };
  const parts = [
    e.code,
    e.reason,
    e.error?.code ? `provider ${e.error.code}` : undefined,
    e.error?.error?.message,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" / ") : "unknown RPC error";
}

function isNonRetryableHistoricalCall(err: unknown): boolean {
  const e = err as { code?: string; message?: string };
  return e.code === "CALL_EXCEPTION" && /missing revert data|Transaction reverted/i.test(e.message ?? "");
}

// Runs tasks with at most `limit` in-flight at once, preserving order.
async function withConcurrencyLimit<T>(
  tasks: (() => Promise<T>)[],
  limit: number
): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let next = 0;

  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      results[i] = await tasks[i]();
    }
  }

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

export async function fetchHistory(
  provider: ethers.providers.JsonRpcProvider,
  assetAddress: string,
  days: number,
  blocksPerSample: number = BLOCKS_PER_DAY,
  concurrency: number = 1,
  assetSymbol: string = assetAddress,
  schedule?: HistoryBlockSchedule
): Promise<HistoryRow[]> {
  const dataProvider = new ethers.Contract(
    AaveV3Ethereum.AAVE_PROTOCOL_DATA_PROVIDER,
    RESERVE_DATA_ABI,
    provider
  );
  const oracle = new ethers.Contract(AaveV3Ethereum.ORACLE, ORACLE_ABI, provider);
  const token = new ethers.Contract(assetAddress, ERC20_ABI, provider);

  const [resolvedLatestBlock, decimals] = await Promise.all([
    schedule ? Promise.resolve(schedule.latestBlock) : provider.getBlockNumber(),
    token.decimals() as Promise<number>,
  ]);

  const blockTags = schedule?.blockTags ?? buildHistoryBlockSchedule(
    resolvedLatestBlock,
    days,
    blocksPerSample
  ).blockTags;
  const numSamples = blockTags.length;

  async function fetchWithRetry<T>(
    label: string,
    blockTag: number,
    fn: () => Promise<T>,
    maxRetries = 4
  ): Promise<T> {
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await fn();
      } catch (err) {
        if (isNonRetryableHistoricalCall(err)) {
          throw err;
        }
        if (attempt === maxRetries) {
          throw new Error(
            `${label} failed for ${assetSymbol} at block ${blockTag} after ${maxRetries + 1} attempts: ${(err as Error).message}`
          );
        }
        await new Promise((r) => setTimeout(r, 2_000 * (attempt + 1)));
      }
    }
    throw new Error("unreachable");
  }

  let completed = 0;
  let skipped = 0;
  const tasks = blockTags.map((blockTag) => async (): Promise<HistoryRow | null> => {
    try {
      const [block, data, priceRaw] = await Promise.all([
        fetchWithRetry("getBlock", blockTag, () => provider.getBlock(blockTag)),
        fetchWithRetry("getReserveData", blockTag, () =>
          dataProvider.getReserveData(assetAddress, { blockTag }) as Promise<ethers.BigNumber[]>
        ),
        fetchWithRetry("getAssetPrice", blockTag, () =>
          oracle.getAssetPrice(assetAddress, { blockTag }) as Promise<ethers.BigNumber>
        ),
      ]);

      if (!block) {
        throw new Error(`getBlock returned null for ${assetSymbol} at block ${blockTag}`);
      }

      const close = parseFloat(ethers.utils.formatUnits(priceRaw, 8));
      const tokenBal = parseFloat(ethers.utils.formatUnits(data[2], decimals));
      const stableDebt = parseFloat(ethers.utils.formatUnits(data[3], decimals));
      const variableDebt = parseFloat(ethers.utils.formatUnits(data[4], decimals));
      const supplied_usd = tokenBal * close;
      const borrowed_usd = (stableDebt + variableDebt) * close;

      return {
        datetime: new Date(block.timestamp * 1000).toISOString(),
        block: blockTag,
        close,
        token_balance: tokenBal,
        supplied_usd,
        borrowed_usd,
        tvl_usd: supplied_usd,
        supply_apr: rayToPercent(data[5]),
        variable_borrow_apr: rayToPercent(data[6]),
        stable_borrow_apr: rayToPercent(data[7]),
      };
    } catch (err) {
      skipped++;
      process.stderr.write(
        `\n  warning: skipped ${assetSymbol} block ${blockTag}: ${rpcErrorSummary(err)}\n`
      );
      return null;
    } finally {
      completed++;
      process.stderr.write(`\r  fetching ${assetSymbol} ${completed}/${numSamples} samples...`);
    }
  });

  const rows = (await withConcurrencyLimit(tasks, concurrency)).filter(
    (row): row is HistoryRow => row !== null
  );
  process.stderr.write(`\r  fetched ${rows.length}/${numSamples} ${assetSymbol} samples`);
  if (skipped > 0) {
    process.stderr.write(` (${skipped} skipped)`);
  }
  process.stderr.write(".   \n");

  if (rows.length === 0) {
    throw new Error(
      `All ${numSamples} history samples failed for ${assetSymbol}. ` +
      `Your RPC may not support historical eth_call for the requested block range; try a shorter lookback or an archive-capable RPC.`
    );
  }

  return rows;
}
