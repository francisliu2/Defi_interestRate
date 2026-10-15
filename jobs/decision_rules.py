"""Decision rules applied to the evaluation map (merged paper, Section 8.1).

Liquidation-budget rule: the largest initial leverage whose one-month liquidation
probability does not exceed a budget p_bar, found by bisection in H0 on the
Laplace survival probability.  The mean--variance and Kelly selections at the
benchmark drift are read from results/breakeven_drift.json (headline calibration,
sample drift), which searches the full buffer range including the unlevered
position.

Usage: python jobs/decision_rules.py [params.json]
Writes results/decision_rules.json.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from optimal_long_short.inversion import TalbotInverter  # noqa: E402
from optimal_long_short.market_params import MarketParams  # noqa: E402
from optimal_long_short.model_params import KouParams  # noqa: E402
from optimal_long_short.moments import ConditionalMoments  # noqa: E402
from optimal_long_short.strategy import UnitExposureLongShortStrategy  # noqa: E402

KEYS = ["mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
        "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho"]
T = 1.0 / 12.0


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/params_WBTC_WETH_benchmark.json"
    blob = json.load(open(path))
    p = KouParams(**{k: blob["params"][k] for k in KEYS})
    c = blob["aave_constraint"]
    b, H_min = c["b"], c["H0_min"]
    inv = TalbotInverter(M=32)

    def p_liq(H):
        s = UnitExposureLongShortStrategy(h0=math.log(H), market=MarketParams(b=b, S10=1, S20=1), T=T)
        return 1.0 - ConditionalMoments(params=p, strategy=s, inverter=inv).p_surv()

    rows = []
    for budget in (0.01, 0.05, 0.10, 0.20):
        lo, hi = H_min, 5.0
        if p_liq(lo) <= budget:
            H = lo
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if p_liq(mid) <= budget else (mid, hi)
            H = hi
        rows.append({"p_bar": budget, "H0": H, "L0": H / (H - b), "p_liq": p_liq(H)})
        print(f"p_liq <= {budget:.2f}: H0 = {H:.4f}, L0 = {H / (H - b):.3f}, p_liq = {rows[-1]['p_liq']:.4f}")

    be = json.load(open(ROOT / "results/breakeven_drift.json"))["calibrations"]["headline_4h"]
    out = {"params_file": str(path.name), "b": b, "H0_min": H_min, "T": T,
           "liquidation_budget": rows,
           "optimal_leverage_at_sample_drift": be["optimal_leverage_at_nu_hat"],
           "nu_star": be["nu_star"], "nu_hat": be["nu_hat_used"]}
    json.dump(out, open(ROOT / "results/decision_rules.json", "w"), indent=2, default=float)
    print("SAVED results/decision_rules.json")


if __name__ == "__main__":
    main()
