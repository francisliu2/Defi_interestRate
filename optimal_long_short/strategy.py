import math
from dataclasses import dataclass

from optimal_long_short.market_params import MarketParams


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
        Initial log-health. Must satisfy h0 > log(b).
    market : MarketParams
        Market and contract parameters (b, S10, S20).
    T : float
        Investment horizon. Must be strictly positive.
    """
    h0: float
    market: MarketParams
    T: float

    def __post_init__(self) -> None:
        if self.h0 <= math.log(self.market.b):
            raise ValueError(
                f"h0 must be greater than log(b) = {math.log(self.market.b):.6f}, "
                f"got {self.h0}"
            )
        if self.T <= 0.0:
            raise ValueError(f"T must be strictly positive, got {self.T}")

    @property
    def H(self) -> float:
        """Initial health ratio H0 = exp(h0)."""
        return math.exp(self.h0)

    @property
    def w2(self) -> float:
        """Short-leg weight: w2 = b / (S20 * (exp(h0) - b))."""
        return self.market.b / (self.market.S20 * (self.H - self.market.b))

    @property
    def w1(self) -> float:
        """Long-leg weight: w1 = exp(h0) / (S10 * (exp(h0) - b))."""
        return self.H / (self.market.S10 * (self.H - self.market.b))
