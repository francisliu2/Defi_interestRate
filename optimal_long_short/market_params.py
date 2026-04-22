from dataclasses import dataclass


@dataclass
class MarketParams:
    """
    Market/contract parameters for the DeFi long-short position.

    b : float
        Collateral factor in (0, 1). Defines the liquidation threshold via
        the health ratio H_t = b * w1 * S1_t / (w2 * S2_t); the position is
        liquidated the first time H_t falls below 1.
    """
    b: float
    S10: float
    S20: float

    def __post_init__(self) -> None:
        if not (0.0 < self.b < 1.0):
            raise ValueError(f"b must be in (0, 1), got {self.b}")
        if self.S10 <= 0.0:
            raise ValueError(f"S10 must be positive, got {self.S10}")
        if self.S20 <= 0.0:
            raise ValueError(f"S20 must be positive, got {self.S20}")
