"""
Data preparation: prices -> synchronised log-returns with inferred dt_years.

All model parameters are annualised using crypto calendar time:
    1 year = 365 * 24 * 3600 seconds.

Supported target frequencies and their dt_years:
    "15min"  → 15 * 60 / (365 * 24 * 3600)
    "1h"     → 3600 / (365 * 24 * 3600)  = 1 / 8760
    "4h"     → 4 * 3600 / (...)           = 4 / 8760
    "1d"     → 86400 / (...)              = 1 / 365
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_YEAR: float = 365.0 * 24.0 * 3600.0

_FREQ_SECONDS: dict[str, float] = {
    "15min": 15 * 60,
    "1h":    3_600,
    "4h":    4 * 3_600,
    "1d":    86_400,
}


@dataclass
class PreparedReturns:
    """
    Clean, synchronised bivariate log-returns ready for ECF calibration.

    Attributes
    ----------
    r1, r2 : (N,) arrays of log-returns.
    dt_years : Observation interval in annualised years.
    n_obs : Number of return observations N.
    freq_label : Human-readable frequency string (e.g. "1h", "1d", "inferred").
    timestamps : (N+1,) array of epoch seconds or None.
    """
    r1: np.ndarray
    r2: np.ndarray
    dt_years: float
    n_obs: int
    freq_label: str
    timestamps: np.ndarray | None = None


def prepare_returns(
    prices1: np.ndarray,
    prices2: np.ndarray,
    timestamps: np.ndarray | None = None,
    target_freq: str | None = None,
) -> PreparedReturns:
    """
    Clean price series and compute synchronised log-returns.

    Parameters
    ----------
    prices1, prices2 : 1-D arrays of positive asset prices, equal length.
    timestamps : 1-D array of POSIX timestamps (seconds) aligned with prices,
                 or None.  If None, equally spaced observations are assumed
                 and ``target_freq`` must be given.
    target_freq : One of "15min", "1h", "4h", "1d".  Used both to resample
                  irregular price data (requires timestamps) and to set dt_years
                  when timestamps are absent.

    Returns
    -------
    PreparedReturns

    Notes
    -----
    For irregular oracle data (e.g. Chainlink), pass the raw tick-level prices
    and timestamps together with a ``target_freq`` to grid-resample using the
    latest available price at each grid point (forward fill), then compute
    log-returns on the resampled series.
    """
    prices1 = np.asarray(prices1, dtype=float).ravel()
    prices2 = np.asarray(prices2, dtype=float).ravel()

    if len(prices1) != len(prices2):
        raise ValueError("prices1 and prices2 must have the same length.")

    if timestamps is not None:
        ts = np.asarray(timestamps, dtype=float).ravel()
        if len(ts) != len(prices1):
            raise ValueError("timestamps must have the same length as prices.")

        if target_freq is not None:
            prices1, prices2, ts = _resample(prices1, prices2, ts, target_freq)

        dt_years = float(np.median(np.diff(ts)) / SECONDS_PER_YEAR)
        freq_label = target_freq if target_freq else "inferred"
        ts_out = ts
    else:
        if target_freq is None:
            raise ValueError(
                "Provide either timestamps or target_freq "
                "(or both for irregular-tick resampling)."
            )
        if target_freq not in _FREQ_SECONDS:
            raise ValueError(
                f"Unknown target_freq {target_freq!r}. "
                f"Choose from {list(_FREQ_SECONDS)}."
            )
        dt_years = _FREQ_SECONDS[target_freq] / SECONDS_PER_YEAR
        freq_label = target_freq
        ts_out = None

    # Drop non-positive / non-finite prices
    valid = np.isfinite(prices1) & (prices1 > 0) & np.isfinite(prices2) & (prices2 > 0)
    if not valid.all():
        prices1, prices2 = prices1[valid], prices2[valid]
        if ts_out is not None:
            ts_out = ts_out[valid]

    if len(prices1) < 2:
        raise ValueError("Fewer than 2 valid price observations after cleaning.")

    r1 = np.diff(np.log(prices1))
    r2 = np.diff(np.log(prices2))

    # Drop any return pairs containing non-finite values
    ok = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[ok], r2[ok]
    if ts_out is not None:
        ts_out = ts_out[1:][ok]

    return PreparedReturns(
        r1=r1,
        r2=r2,
        dt_years=dt_years,
        n_obs=len(r1),
        freq_label=freq_label,
        timestamps=ts_out,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(
    prices1: np.ndarray,
    prices2: np.ndarray,
    ts: np.ndarray,
    target_freq: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-fill prices to a regular grid with spacing given by target_freq.
    """
    step = _FREQ_SECONDS[target_freq]
    t0 = float(np.ceil(ts[0] / step) * step)
    t1 = float(np.floor(ts[-1] / step) * step)
    grid = np.arange(t0, t1 + 0.5 * step, step)

    # forward-fill: for each grid point, use the last observed price at or before it
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ts) - 1)

    return prices1[idx], prices2[idx], grid
