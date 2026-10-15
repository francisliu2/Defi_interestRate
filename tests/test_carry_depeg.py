"""Depeg overlay: a shock that leaves H_0 <= 1 is an instant liquidation."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimal_long_short.carry import CarryLeg, CarrySpec, depeg_scenario  # noqa: E402
from optimal_long_short.model.model_params import KouParams  # noqa: E402

P = KouParams(mu1=0.05, sigma1=0.3, lam1=5.0, p1=0.4, eta1_pos=0.02, eta1_neg=0.03,
              mu2=0.04, sigma2=0.35, lam2=5.0, p2=0.4, eta2_pos=0.02, eta2_neg=0.03, rho=0.8)
SPEC = CarrySpec(collateral=CarryLeg("A", 0.01), debt=CarryLeg("B", 0.02))
B, T = 0.78, 1 / 12


def test_shock_below_solvency_is_instant_liquidation():
    h0 = 0.05
    d = 0.10                       # h0 + log(0.9) = -0.055: H_0 < 1 but above b
    assert math.log(B) < h0 + math.log1p(-d) <= 0.0
    (row,) = depeg_scenario(P, SPEC, h0, [d], b=B, T=T, max_moment_order=2)
    assert row["instant_liquidation"] == 1.0 and row["p_liq"] == 1.0


def test_small_shock_is_priced():
    (row,) = depeg_scenario(P, SPEC, 0.30, [0.05], b=B, T=T, max_moment_order=2)
    assert row["instant_liquidation"] == 0.0 and 0.0 <= row["p_liq"] < 1.0
