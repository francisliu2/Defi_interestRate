"""Peak-Over-Threshold (POT) utilities for jump detection and sizing.

Functions:
- compute_log_returns(prices): log returns
- infer_dt_from_index(index): infer sampling dt in years
- select_threshold(returns, quantiles=...): choose threshold minimizing skewness+excess kurtosis of excesses
- fit_shifted_exponential(excesses): fit expo (MLE)
- estimate_intensity(count, total_time_years)
- analyze_returns(returns, dt, side='positive', q_grid=...): run selection + fit + diagnostics

Also plotting helpers to save diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return np.diff(np.log(prices))


def infer_dt_from_index(index: pd.Index) -> float:
    """Infer median sampling interval in years from a datetime index."""
    median_sec = index.to_series().diff().dt.total_seconds().median()
    if pd.isna(median_sec) or median_sec <= 0:
        return 1.0 / 365.0
    return float(median_sec) / (3600 * 24 * 365)


def _skew_kurtosis(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return float('nan'), float('nan')
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0, -3.0
    skew = np.mean(((x - m) / s) ** 3)
    kurt = np.mean(((x - m) / s) ** 4)
    excess_kurtosis = kurt - 3.0
    return float(skew), float(excess_kurtosis)


def select_threshold(returns: np.ndarray,
                     quantiles: np.ndarray = None,
                     min_excess: int = 30,
                     side: str = 'positive') -> Dict:
    """Select threshold by scanning quantiles and minimising |skew| + |excess_kurtosis| of excesses.

    side: 'positive' (right tail) or 'negative' (left tail, operate on -returns)
    Returns diagnostics dict with chosen threshold, quantile, and arrays of diagnostics.
    """
    if quantiles is None:
        quantiles = np.linspace(0.90, 0.995, 30)
    r = np.asarray(returns, dtype=float)
    if side == 'negative':
        r = -r
    diagnostics = []
    for q in quantiles:
        thr = np.quantile(r, q)
        excess = r[r > thr] - thr
        if excess.size < min_excess:
            diagnostics.append({'quantile': float(q), 'threshold': float(thr), 'n_excess': int(excess.size), 'skew': float('nan'), 'excess_kurtosis': float('nan')})
            continue
        skew, exk = _skew_kurtosis(excess)
        diagnostics.append({'quantile': float(q), 'threshold': float(thr), 'n_excess': int(excess.size), 'skew': skew, 'excess_kurtosis': exk})

    diag_df = pd.DataFrame(diagnostics)
    # drop rows with nan skew
    cand = diag_df.dropna(subset=['skew'])
    if cand.empty:
        # fallback to highest quantile available
        best = diag_df.iloc[-1]
    else:
        # objective minimize |skew| + |excess_kurtosis|
        obj = (cand['skew'].abs() + cand['excess_kurtosis'].abs())
        best = cand.loc[obj.idxmin()]

    thr = float(best['threshold'])
    return {'threshold': thr, 'quantile': float(best['quantile']), 'diag': diag_df}


def fit_shifted_exponential(excesses: np.ndarray) -> Dict[str, float]:
    """Fit shifted exponential (support x>=0) to excesses: MLE lambda = 1/mean(excesses).

    Returns {'lambda':..., 'mean_excess':..., 'n':...}
    """
    x = np.asarray(excesses, dtype=float)
    x = x[x >= 0]
    n = x.size
    if n == 0:
        return {'lambda': float('nan'), 'mean_excess': float('nan'), 'n': 0}
    mean_ex = float(x.mean())
    if mean_ex <= 0:
        return {'lambda': float('nan'), 'mean_excess': mean_ex, 'n': n}
    lam = 1.0 / mean_ex
    return {'lambda': float(lam), 'mean_excess': mean_ex, 'n': int(n)}


def estimate_intensity(n_exceedances: int, total_time_years: float) -> float:
    if total_time_years <= 0:
        return float('nan')
    return float(n_exceedances) / float(total_time_years)


def analyze_returns(returns: np.ndarray, index: Optional[pd.Index] = None, side: str = 'positive',
                    quantiles: np.ndarray = None, min_excess: int = 30, out_dir: Optional[Path] = None,
                    asset_name: str = 'asset', col_name: str = 'returns') -> Dict:
    """Run full POT analysis on an array of returns.

    Returns summary dict and writes diagnostic plots if out_dir provided.
    """
    r = np.asarray(returns, dtype=float)
    if index is None:
        total_time_years = len(r) * (1.0 / 365.0)
    else:
        total_time_years = infer_dt_from_index(index) * len(r)

    sel = select_threshold(r, quantiles=quantiles, min_excess=min_excess, side=side)
    thr = sel['threshold']
    if side == 'negative':
        r_proc = -r
    else:
        r_proc = r
    excess = r_proc[r_proc > thr] - thr
    fit = fit_shifted_exponential(excess)
    n_exc = int(fit['n'])
    intensity = estimate_intensity(n_exc, total_time_years)

    summary = {
        'asset': asset_name,
        'column': col_name,
        'side': side,
        'threshold': thr,
        'quantile': sel.get('quantile'),
        'n_exceedances': n_exc,
        'intensity_per_year': intensity,
        'lambda': fit.get('lambda'),
        'mean_excess': fit.get('mean_excess'),
        'total_observations': len(r),
        'total_time_years': total_time_years
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # plot diagnostic table of selection
        diag = sel['diag']
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            ax.axis('off')
            tbl = ax.table(cellText=diag.round(6).values.tolist(), colLabels=diag.columns.tolist(), loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            plt.tight_layout()
            fig.savefig(out_dir / f'pot_{asset_name}_{col_name}_{side}_selection_table.png', bbox_inches='tight', transparent=True)
            plt.close(fig)
        except Exception:
            pass

        # histogram of excesses with fitted expo pdf
        try:
            if excess.size > 0 and not np.isnan(fit.get('lambda', np.nan)):
                lam = fit['lambda']
                fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
                ax2.hist(excess, bins=40, density=True, alpha=0.6)
                xs = np.linspace(0, excess.max(), 200)
                ax2.plot(xs, lam * np.exp(-lam * xs), color='C1', lw=1.2, label=f'exp(λ={lam:.3g})')
                ax2.set_title(f'{asset_name} {col_name} excesses ({side})')
                ax2.grid(False)
                ax2.legend()
                plt.tight_layout()
                fig2.savefig(out_dir / f'pot_{asset_name}_{col_name}_{side}_excess_hist.png', bbox_inches='tight', transparent=True)
                plt.close(fig2)
        except Exception:
            pass

    return summary


def analyze_file(fp: Path, price_col_candidates=('close', 'max', 'price'), sides=('positive', 'negative'),
                 out_dir: Optional[Path] = None) -> Dict:
    df = pd.read_csv(fp, parse_dates=['timestamp']) if 'timestamp' in pd.read_csv(fp, nrows=0).columns else pd.read_csv(fp)
    # pick price column
    cols = [c for c in price_col_candidates if c in df.columns]
    if not cols:
        # try lowercase variants
        lc = [c.lower() for c in df.columns]
        found = None
        for cand in price_col_candidates:
            if cand in lc:
                found = df.columns[lc.index(cand)]
                cols = [found]
                break
    if not cols:
        raise ValueError(f'No price column found in {fp}')
    price_col = cols[0]
    prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
    returns = compute_log_returns(prices.values)
    # build index for returns (use timestamp index shifted)
    if 'timestamp' in df.columns:
        ret_index = pd.to_datetime(df['timestamp']).dropna().iloc[1:len(returns)+1] if len(returns)>0 else None
    else:
        ret_index = None

    summaries = []
    for side in sides:
        summ = analyze_returns(returns, index=ret_index, side=side, out_dir=out_dir, asset_name=fp.stem, col_name=price_col)
        summaries.append(summ)
    return {'file': str(fp), 'price_col': price_col, 'summaries': summaries}
