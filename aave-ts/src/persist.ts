import * as parquet from "@dsnp/parquetjs";
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
  filepath: string;
  addedRows: number;
  totalRows: number;
}

export function buildFilepath(
  outDir: string,
  symbol: string,
  chain: string,
  frequency: string
): string {
  return path.resolve(outDir, `${symbol}_AAVEv3_${chain}_${frequency}_rates.parquet`);
}

async function readExisting(filepath: string): Promise<Map<string, HistoryRow>> {
  const map = new Map<string, HistoryRow>();
  const reader = await parquet.ParquetReader.openFile(filepath);
  const cursor = reader.getCursor();
  let record;
  while ((record = await cursor.next())) {
    const row = record as HistoryRow;
    map.set(row.datetime, row);
  }
  await reader.close();
  return map;
}

// Merges new rows into the existing parquet file (deduplicates by datetime).
// Writes to a temp file first then renames to avoid partial writes.
export async function persistHistory(
  rows: HistoryRow[],
  symbol: string,
  chain: string,
  frequency: string,
  outDir: string
): Promise<PersistResult> {
  fs.mkdirSync(outDir, { recursive: true });
  const filepath = buildFilepath(outDir, symbol, chain, frequency);

  const existing = fs.existsSync(filepath)
    ? await readExisting(filepath)
    : new Map<string, HistoryRow>();

  const beforeSize = existing.size;

  for (const row of rows) {
    existing.set(row.datetime, row);
  }

  const merged = Array.from(existing.values()).sort((a, b) =>
    a.datetime.localeCompare(b.datetime)
  );

  const tmpPath = filepath + ".tmp";
  const writer = await parquet.ParquetWriter.openFile(SCHEMA, tmpPath);
  for (const row of merged) {
    await writer.appendRow(row as unknown as Record<string, unknown>);
  }
  await writer.close();
  fs.renameSync(tmpPath, filepath);

  return { filepath, addedRows: merged.length - beforeSize, totalRows: merged.length };
}
