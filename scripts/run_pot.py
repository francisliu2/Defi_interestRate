"""Run POT analysis on DAI/WETH 'max' CSVs under data/raw/coins_prices.

Saves `results/pot_results.csv` and diagnostic plots under `results/plots/`.
"""
from pathlib import Path
import sys
import pandas as pd

repo = Path.cwd()
sys.path.insert(0, str(repo / 'src'))
import pot

coins_dir = repo / 'data' / 'raw' / 'coins_prices'
patterns = ['*dai*max*.csv', '*weth*max*.csv', '*dai*max*.CSV', '*weth*max*.CSV']
files = []
for p in patterns:
    files.extend(list(coins_dir.glob(p)))

if not files:
    print('No dai/weth max CSVs found under', coins_dir)
    raise SystemExit(1)

out_dir = repo / 'results' / 'plots'
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for fp in files:
    print('Analyzing', fp.name)
    try:
        r = pot.analyze_file(fp, out_dir=out_dir)
        for s in r['summaries']:
            row = {**{'file': r['file'], 'price_col': r['price_col']}, **s}
            rows.append(row)
    except Exception as e:
        print('Error analyzing', fp, e)

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(repo / 'results' / 'pot_results.csv', index=False)
    print('Wrote results to', repo / 'results' / 'pot_results.csv')
else:
    print('No results')
