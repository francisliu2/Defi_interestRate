import { AaveV3Ethereum } from "@bgd-labs/aave-address-book";

export type AssetSymbol = keyof typeof AaveV3Ethereum.ASSETS;
export type FetchMode = "markets" | "history" | "both";
export type OutputFormat = "table" | "json";
export type Frequency = "hourly" | "6h" | "12h" | "daily";

// Ethereum averages ~12s per block → blocks per sampling interval
export const FREQUENCY_BLOCKS: Record<Frequency, number> = {
  hourly: 300,   // 3_600s / 12s
  "6h":   1_800, // 21_600s / 12s
  "12h":  3_600, // 43_200s / 12s
  daily:  7_200, // 86_400s / 12s
};

export interface ParamsInput {
  mode?: FetchMode;
  assets?: string[];
  days?: number;
  frequency?: Frequency;
  concurrency?: number;
  persist?: boolean;
  outDir?: string;
  chain?: string;
  output?: OutputFormat;
}

export class OrchestratorParams {
  readonly mode: FetchMode;
  readonly assets: AssetSymbol[];
  readonly days: number;
  readonly frequency: Frequency;
  readonly concurrency: number;
  readonly persist: boolean;
  readonly outDir: string;
  readonly chain: string;
  readonly output: OutputFormat;

  constructor(opts: ParamsInput = {}) {
    this.mode = opts.mode ?? "both";
    this.assets = (opts.assets ?? ["USDC"]) as AssetSymbol[];
    this.days = opts.days ?? 30;
    this.frequency = opts.frequency ?? "daily";
    this.concurrency = opts.concurrency ?? 1;
    this.persist = opts.persist ?? true;
    this.outDir = opts.outDir ?? "./data/AAVE";
    this.chain = opts.chain ?? "ethereum";
    this.output = opts.output ?? "table";
    this.validate();
  }

  private validate(): void {
    const validModes: FetchMode[] = ["markets", "history", "both"];
    if (!validModes.includes(this.mode)) {
      throw new Error(`Invalid mode "${this.mode}". Must be: ${validModes.join(" | ")}`);
    }

    if (this.days < 1 || this.days > 365) {
      throw new Error(`days must be 1–365, got ${this.days}`);
    }

    if (!(this.frequency in FREQUENCY_BLOCKS)) {
      throw new Error(
        `Invalid frequency "${this.frequency}". Must be: ${Object.keys(FREQUENCY_BLOCKS).join(" | ")}`
      );
    }

    if (!Number.isInteger(this.concurrency) || this.concurrency < 1 || this.concurrency > 50) {
      throw new Error(`concurrency must be an integer between 1 and 50, got ${this.concurrency}`);
    }

    const knownAssets = Object.keys(AaveV3Ethereum.ASSETS);
    for (const a of this.assets) {
      if (!knownAssets.includes(a)) {
        throw new Error(
          `Unknown asset "${a}". Available: ${knownAssets.join(", ")}`
        );
      }
    }

    if (this.mode !== "markets" && this.assets.length === 0) {
      throw new Error(`At least one asset is required for mode "${this.mode}"`);
    }
  }

  // Parses: --mode=history --assets=USDC,WETH --days=60 --frequency=6h --concurrency=5 --no-persist --out-dir=./data --chain=ethereum --output=json
  static fromCLI(argv: string[] = process.argv.slice(2)): OrchestratorParams {
    const parsed: Record<string, string> = {};
    const flags = new Set<string>();
    for (const arg of argv) {
      const kvMatch = arg.match(/^--([a-zA-Z-]+)=(.+)$/);
      if (kvMatch) { parsed[kvMatch[1]] = kvMatch[2]; continue; }
      const flagMatch = arg.match(/^--([a-zA-Z-]+)$/);
      if (flagMatch) flags.add(flagMatch[1]);
    }
    return new OrchestratorParams({
      mode: parsed.mode as FetchMode | undefined,
      assets: parsed.assets ? parsed.assets.split(",").map((s) => s.trim()) : undefined,
      days: parsed.days ? parseInt(parsed.days, 10) : undefined,
      frequency: parsed.frequency as Frequency | undefined,
      concurrency: parsed.concurrency ? parseInt(parsed.concurrency, 10) : undefined,
      persist: flags.has("no-persist") ? false : undefined,
      outDir: parsed["out-dir"],
      chain: parsed.chain,
      output: parsed.output as OutputFormat | undefined,
    });
  }
}
