"""Mechanics of the hourly replay in jobs/realised_backtest.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jobs"))
from realised_backtest import roll_mask, simulate  # noqa: E402

B, BONUS = 0.8, 0.05


def run(S1, S2, L, full=False, s1=0.0, r2=0.0, cost=0.0):
    n = len(S1)
    return simulate(np.asarray(S1, float), np.asarray(S2, float), np.full(n, s1), np.full(n, r2),
                    roll_mask(n, 0, 0), L, B, BONUS, full, cost)


def test_flat_prices_keep_equity():
    eq, nliq, ruined = run([1.0] * 50, [1.0] * 50, 3.0)
    assert np.allclose(eq, 1.0) and nliq == 0 and not ruined


def test_interest_accrual():
    n = 8761
    eq, _, _ = run([1.0] * n, [1.0] * n, 2.0, s1=0.03, r2=0.05)
    # collateral 2 at 3%, debt 1 at 5%, one year
    assert abs(eq[-1] - (2 * np.exp(0.03) - np.exp(0.05))) < 1e-9


def test_partial_liquidation_close_factor():
    # L=3: n1=3, n2=2, HF = 0.8*3/2 = 1.2; S1 0.8 -> HF 0.96 (> 0.95): 50% close
    eq, nliq, ruined = run([1.0, 0.8], [1.0, 1.0], 3.0)
    repay = 1.0
    n1 = 3 - repay * (1 + BONUS) / 0.8
    assert nliq == 1 and not ruined
    assert abs(eq[-1] - (n1 * 0.8 - 1.0)) < 1e-12
    assert B * n1 * 0.8 / 1.0 >= 1.0          # health restored


def test_full_close_below_095_and_full_mode():
    # S1 0.78 -> HF 0.936 <= 0.95: whole debt repaid
    eq, nliq, _ = run([1.0, 0.78], [1.0, 1.0], 3.0)
    assert nliq == 1 and abs(eq[-1] - (3 * 0.78 - 2 * (1 + BONUS))) < 1e-12
    eq_f, nliq_f, _ = run([1.0, 0.8], [1.0, 1.0], 3.0, full=True)
    assert nliq_f == 1 and abs(eq_f[-1] - (3 * 0.8 - 2 * (1 + BONUS))) < 1e-12


def test_bad_debt_ruins():
    eq, _, ruined = run([1.0, 0.5, 2.0], [1.0, 1.0, 1.0], 3.0)
    assert ruined and eq[-1] == 0.0
