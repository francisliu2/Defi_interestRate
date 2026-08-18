#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def summarize(df, col):
    s = df[col].dropna().astype(float)
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def process_file(fp, out_plot_dir):
    name = os.path.splitext(os.path.basename(fp))[0]
    df = pd.read_csv(fp)
    # try to parse timestamp
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        except Exception:
            pass

    cols = []
    for candidate in ['supply_rate_comp', 'variable_borrow_rate_comp']:
        if candidate in df.columns:
            cols.append(candidate)

    results = []
    if not cols:
        print(f"No target columns found in {fp}")
        return results

    # Time-series plot overlay
    plt.figure(figsize=(10, 4))
    for c in cols:
        plt.plot(df['timestamp'] if 'timestamp' in df.columns else df.index, df[c], label=c)
    plt.legend()
    plt.title(f"{name} rates time series")
    ts_path = os.path.join(out_plot_dir, f"{name}_timeseries.png")
    plt.tight_layout()
    plt.savefig(ts_path)
    plt.close()

    # Histograms and summaries
    for c in cols:
        plt.figure(figsize=(6, 4))
        series = pd.to_numeric(df[c], errors='coerce').dropna()
        plt.hist(series, bins=60)
        plt.title(f"{name} - {c} distribution")
        plt.xlabel('rate')
        plt.ylabel('count')
        hist_path = os.path.join(out_plot_dir, f"{name}_{c}_hist.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        stats = summarize(df, c)
        stats.update({"asset": name, "column": c})
        results.append(stats)

    return results


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, 'data', 'raw')
    out_dir = os.path.join(repo_root, 'results')
    plots_dir = os.path.join(out_dir, 'plots')
    ensure_dir(plots_dir)

    files = [
        os.path.join(data_dir, 'DAI.csv'),
        os.path.join(data_dir, 'WETH.csv'),
    ]

    all_stats = []
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
        stats = process_file(f, plots_dir)
        all_stats.extend(stats)

    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_fp = os.path.join(out_dir, 'rate_statistics.csv')
        summary_df.to_csv(summary_fp, index=False)
        print(f"Wrote summary to {summary_fp}")
        print(f"Wrote plots to {plots_dir}")
    else:
        print("No statistics generated.")


if __name__ == '__main__':
    main()
