import "dotenv/config";
import { ethers } from "ethers";
import { UiPoolDataProvider, ChainId } from "@aave/contract-helpers";
import { formatReserves } from "@aave/math-utils";
import { AaveV3Ethereum } from "@bgd-labs/aave-address-book";
import dayjs from "dayjs";

export interface MarketRow {
  symbol: string;
  supplyApy: number;
  borrowApy: number;
  liquidityUSD: number;
}

export async function fetchMarkets(
  provider: ethers.providers.JsonRpcProvider,
  assetFilter?: string[]
): Promise<MarketRow[]> {
  const { UI_POOL_DATA_PROVIDER, POOL_ADDRESSES_PROVIDER } = AaveV3Ethereum;

  const poolDataProvider = new UiPoolDataProvider({
    uiPoolDataProviderAddress: UI_POOL_DATA_PROVIDER,
    provider,
    chainId: ChainId.mainnet,
  });

  const { reservesData, baseCurrencyData } =
    await poolDataProvider.getReservesHumanized({
      lendingPoolAddressProvider: POOL_ADDRESSES_PROVIDER,
    });

  const formatted = formatReserves({
    reserves: reservesData,
    currentTimestamp: dayjs().unix(),
    marketReferenceCurrencyDecimals:
      baseCurrencyData.marketReferenceCurrencyDecimals,
    marketReferencePriceInUsd:
      baseCurrencyData.marketReferenceCurrencyPriceInUsd,
  });

  const rows: MarketRow[] = formatted.map((r) => ({
    symbol: r.symbol,
    supplyApy: parseFloat(r.supplyAPY) * 100,
    borrowApy: parseFloat(r.variableBorrowAPY) * 100,
    liquidityUSD: parseFloat(r.availableLiquidityUSD),
  }));

  return assetFilter?.length ? rows.filter((r) => assetFilter.includes(r.symbol)) : rows;
}
