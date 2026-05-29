import * as parquet from "@dsnp/parquetjs";
import { createHash } from "crypto";
import * as fs from "fs";
import * as path from "path";
import { HistoryRow } from "./fetchHistory";

const SCHEMA = new parquet.ParquetSchema({
  datetime:           { type: "UTF8" },
  block:              { type: "INT64" },
  close:              { type: "DOUBLE" },
  token_balance:      { type: "DOUBLE" },
  supplied_usd:       { type: "DOUBLE" },
  borrowed_usd:       { type: "DOUBLE" },
  tvl_usd:            { type: "DOUBLE" },
  supply_apr:         { type: "DOUBLE" },
  variable_borrow_apr: { type: "DOUBLE" },
  stable_borrow_apr:  { type: "DOUBLE" },
});

export interface PersistResult {
  id: string;
  filepath: string;
  manifestPath: string;
  addedRows: number;
  totalRows: number;
}

export interface PersistHistoryParams {
  symbol: string;
  chain: string;
  frequency: string;
  blocksPerSample: number;
  requestedDays: number;
  scheduledLatestBlock?: number;
  scheduledSampleCount?: number;
  assetAddress: string;
  outDir: string;
  fetchedAt: string;
  providerUrl?: string;
}

interface ManifestRow {
  id: string;
  parquet_file: string;
  fetched_at: string;
  symbol: string;
  chain: string;
  asset_address: string;
  frequency: string;
  blocks_per_sample: number;
  requested_days: number;
  scheduled_latest_block: number | "";
  scheduled_sample_count: number | "";
  sample_count: number;
  start_datetime: string;
  end_datetime: string;
  start_block: number;
  end_block: number;
  min_block: number;
  max_block: number;
  rpc_host: string;
  rpc_url_hash: string;
}

const MANIFEST_FILENAME = "manifest.csv";
const MANIFEST_COLUMNS: (keyof ManifestRow)[] = [
  "id",
  "parquet_file",
  "fetched_at",
  "symbol",
  "chain",
  "asset_address",
  "frequency",
  "blocks_per_sample",
  "requested_days",
  "scheduled_latest_block",
  "scheduled_sample_count",
  "sample_count",
  "start_datetime",
  "end_datetime",
  "start_block",
  "end_block",
  "min_block",
  "max_block",
  "rpc_host",
  "rpc_url_hash",
];

function csvEscape(value: string | number): string {
  const s = String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function manifestLine(row: ManifestRow): string {
  return MANIFEST_COLUMNS.map((column) => csvEscape(row[column])).join(",") + "\n";
}

function manifestHeader(): string {
  return MANIFEST_COLUMNS.join(",") + "\n";
}

function shortHash(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 12);
}

function rpcHost(value: string | undefined): string {
  if (!value) return "";
  try {
    return new URL(value).host;
  } catch {
    return value;
  }
}

function buildPersistId(params: PersistHistoryParams, rows: HistoryRow[]): string {
  const first = rows[0];
  const last = rows[rows.length - 1];
  const fingerprint = shortHash(JSON.stringify({
    symbol: params.symbol,
    chain: params.chain,
    assetAddress: params.assetAddress.toLowerCase(),
    frequency: params.frequency,
    blocksPerSample: params.blocksPerSample,
    requestedDays: params.requestedDays,
    scheduledLatestBlock: params.scheduledLatestBlock,
    scheduledSampleCount: params.scheduledSampleCount,
    fetchedAt: params.fetchedAt,
    startBlock: first?.block,
    endBlock: last?.block,
    sampleCount: rows.length,
  }));
  return `hist_${fingerprint}`;
}

function buildFilepath(outDir: string, id: string): string {
  return path.resolve(outDir, `${id}.parquet`);
}

function appendManifest(manifestPath: string, row: ManifestRow): void {
  const tmpPath = manifestPath + ".tmp";
  const existing = fs.existsSync(manifestPath) ? fs.readFileSync(manifestPath, "utf8") : "";
  const prefix = existing.length > 0
    ? existing.endsWith("\n") ? existing : existing + "\n"
    : manifestHeader();
  fs.writeFileSync(tmpPath, prefix + manifestLine(row));
  fs.renameSync(tmpPath, manifestPath);
}

// Writes rows to a new ID-addressed parquet file and records the fetch in manifest.csv.
export async function persistHistory(
  rows: HistoryRow[],
  params: PersistHistoryParams
): Promise<PersistResult> {
  const sorted = rows.slice().sort((a, b) => a.datetime.localeCompare(b.datetime));
  if (sorted.length === 0) {
    throw new Error(`No history rows to persist for ${params.symbol}`);
  }

  fs.mkdirSync(params.outDir, { recursive: true });
  const id = buildPersistId(params, sorted);
  const filepath = buildFilepath(params.outDir, id);
  const manifestPath = path.resolve(params.outDir, MANIFEST_FILENAME);

  if (fs.existsSync(filepath)) {
    throw new Error(`Refusing to overwrite existing parquet file: ${filepath}`);
  }

  const tmpPath = filepath + ".tmp";
  const writer = await parquet.ParquetWriter.openFile(SCHEMA, tmpPath);
  for (const row of sorted) {
    await writer.appendRow(row as unknown as Record<string, unknown>);
  }
  await writer.close();
  fs.renameSync(tmpPath, filepath);

  const blocks = sorted.map((row) => row.block);
  appendManifest(manifestPath, {
    id,
    parquet_file: path.basename(filepath),
    fetched_at: params.fetchedAt,
    symbol: params.symbol,
    chain: params.chain,
    asset_address: params.assetAddress,
    frequency: params.frequency,
    blocks_per_sample: params.blocksPerSample,
    requested_days: params.requestedDays,
    scheduled_latest_block: params.scheduledLatestBlock ?? "",
    scheduled_sample_count: params.scheduledSampleCount ?? "",
    sample_count: sorted.length,
    start_datetime: sorted[0].datetime,
    end_datetime: sorted[sorted.length - 1].datetime,
    start_block: sorted[0].block,
    end_block: sorted[sorted.length - 1].block,
    min_block: Math.min(...blocks),
    max_block: Math.max(...blocks),
    rpc_host: rpcHost(params.providerUrl),
    rpc_url_hash: params.providerUrl ? shortHash(params.providerUrl) : "",
  });

  return {
    id,
    filepath,
    manifestPath,
    addedRows: sorted.length,
    totalRows: sorted.length,
  };
}
