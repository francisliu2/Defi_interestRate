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

function rayToPercent(ray: ethers.BigNumber): number {
  return parseFloat(ethers.utils.formatUnits(ray, 27)) * 100;
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
  concurrency: number = 1
): Promise<HistoryRow[]> {
  const dataProvider = new ethers.Contract(
    AaveV3Ethereum.AAVE_PROTOCOL_DATA_PROVIDER,
    RESERVE_DATA_ABI,
    provider
  );
  const oracle = new ethers.Contract(AaveV3Ethereum.ORACLE, ORACLE_ABI, provider);
  const token = new ethers.Contract(assetAddress, ERC20_ABI, provider);

  const [latestBlock, decimals] = await Promise.all([
    provider.getBlockNumber(),
    token.decimals() as Promise<number>,
  ]);

  const numSamples = Math.floor((days * BLOCKS_PER_DAY) / blocksPerSample) + 1;
  const blockTags = Array.from({ length: numSamples }, (_, i) =>
    latestBlock - (numSamples - 1 - i) * blocksPerSample
  );

  let completed = 0;
  const tasks = blockTags.map((blockTag) => async (): Promise<HistoryRow> => {
    const [block, data, priceRaw] = await Promise.all([
      provider.getBlock(blockTag),
      dataProvider.getReserveData(assetAddress, { blockTag }),
      oracle.getAssetPrice(assetAddress, { blockTag }),
    ]);
    completed++;
    process.stderr.write(`\r  fetching ${completed}/${numSamples} samples...`);

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
  });

  const rows = await withConcurrencyLimit(tasks, concurrency);
  process.stderr.write(`\r  fetched ${numSamples}/${numSamples} samples.   \n`);
  return rows;
}
