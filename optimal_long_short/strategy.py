import math
from dataclasses import dataclass
from typing import Optional

from optimal_long_short.market_params import MarketParams


def minimum_feasible_h0(b: float, ltv_max: Optional[float] = None) -> float:
    """Return the initial log-health lower bound.

    Without an origination-LTV constraint, initial solvency requires ``h0 > 0``.
    If ``ltv_max`` is supplied, the protocol constraint
    ``V2,0 / V1,0 <= ltv_max`` is equivalent to
    ``h0 >= log(b / ltv_max)``.
    """
    if not (0.0 < b < 1.0):
        raise ValueError(f"b must be in (0, 1), got {b}")
    if ltv_max is None:
        return 0.0
    if not (0.0 < ltv_max < b):
        raise ValueError(
            f"ltv_max must be in (0, b) with b = {b}, got {ltv_max}"
        )
    return math.log(b / ltv_max)


@dataclass
class UnitExposureLongShortStrategy:
    """
    Long-short strategy with unit net initial exposure.

    The user supplies h0, the initial log-health
        h0 = log(b * w1 * S10 / (w2 * S20)),
    and the portfolio weights are recovered from the normalisation
        w1 * S10 - w2 * S20 = 1.

    Attributes
    ----------
    h0 : float
        Initial log-health. Must satisfy h0 > 0 when no origination-LTV
        constraint is supplied, or h0 >= log(b / ltv_max) otherwise.
    market : MarketParams
        Market and contract parameters (b, S10, S20).
    T : float
        Investment horizon. Must be strictly positive.
    ltv_max : float, optional
        Maximum borrow-to-collateral ratio permitted at origination. When
        supplied, it must lie in (0, b).
    """
    h0: float
    market: MarketParams
    T: float
    ltv_max: Optional[float] = None

    def __post_init__(self) -> None:
        h0_min = minimum_feasible_h0(self.market.b, self.ltv_max)
        if self.ltv_max is None and self.h0 <= h0_min:
            raise ValueError(
                "h0 must be greater than 0 so that the position is solvent "
                f"at inception, got {self.h0}"
            )
        if (
            self.ltv_max is not None
            and self.h0 < h0_min
            and not math.isclose(self.h0, h0_min, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "h0 violates the origination-LTV constraint: "
                f"h0 must be at least log(b / ltv_max) = {h0_min:.6f}, "
                f"got {self.h0}"
            )
        if self.T <= 0.0:
            raise ValueError(f"T must be strictly positive, got {self.T}")

    @property
    def H(self) -> float:
        """Initial health ratio H0 = exp(h0)."""
        return math.exp(self.h0)

    @property
    def initial_ltv(self) -> float:
        """Initial borrow-to-collateral ratio V2,0 / V1,0 = b / H0."""
        return self.market.b / self.H

    @property
    def w2(self) -> float:
        """Short-leg weight: w2 = b / (S20 * (exp(h0) - b))."""
        return self.market.b / (self.market.S20 * (self.H - self.market.b))

    @property
    def w1(self) -> float:
        """Long-leg weight: w1 = exp(h0) / (S10 * (exp(h0) - b))."""
        return self.H / (self.market.S10 * (self.H - self.market.b))
