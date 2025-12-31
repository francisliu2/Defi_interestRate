from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple
import numpy as np
from params import JumpDiffusionParams
from log_health import HealthProcessParameters, LogHealthProcess  

@dataclass
class StablecoinShortParams:
    """Parameters for stablecoin-short strategy."""
    W0: float = 10_000.0
    T: float = 30 / 365.0  # horizon in years
    r_X: float = 0.04
    r_Y: float = 0.08
    b_X: float = 0.8
    X0: float = 1.0
    Y0: float = 1.0
    fht_func: Optional[Callable[..., float]] = None

    # Jump-diffusion parameters
    X_params: JumpDiffusionParams = None
    Y_params: JumpDiffusionParams = None

    def __post_init__(self):
        if self.W0 <= 0:
            raise ValueError("W0 must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if not (0 < self.b_X <= 1):
            raise ValueError("b_X must be in (0,1]")
        if self.X0 <= 0 or self.Y0 <= 0:
            raise ValueError("Initial prices must be positive")
        if self.X_params is None or self.Y_params is None:
            raise ValueError("X_params and Y_params must be provided")


class StablecoinShortStrategy:
    """Stablecoin short strategy using log-health from JumpDiffusion processes."""

    def __init__(self, params: StablecoinShortParams):
        self.p = params
        # Build log-health process (h = log(b_X * w_X * X / (w_Y * Y)))
        self.health_process = LogHealthProcess(
            HealthProcessParameters(
                # small positive placeholders so parameter validation passes;
                # actual values are updated from `weights_from_r` before use
                w_X=1.00000001,
                w_Y=0.00000001,
                b_X=self.p.b_X,
                mu_X=self.p.X_params.mu,
                mu_Y=self.p.Y_params.mu,
                sigma_X=self.p.X_params.sigma,
                sigma_Y=self.p.Y_params.sigma,
                rho=self.p.X_params.rho,
                lambda_X=self.p.X_params.lam,
                lambda_Y=self.p.Y_params.lam,
                delta_X=self.p.X_params.delta,
                delta_Y=self.p.Y_params.delta,
                eta_X=self.p.X_params.eta,
                eta_Y=self.p.Y_params.eta,
                X0=self.p.X0,
                Y0=self.p.Y0
            )
        )

    def psi_h(self, theta: float) -> float:
        """Convenience wrapper: forward psi_h calls to the internal health process."""
        return self.health_process.psi_h(theta)

    # ---------------------------
    # Weight mapping: r -> (w_X, w_Y)
    # ---------------------------
    def weights_from_r(self, r: float) -> Tuple[float, float]:
        if r <= 0:
            raise ValueError("Allocation r must be positive")
        w_X = float(r)
        w_Y = r - 1.0
        return w_X, w_Y

    # ---------------------------
    # Expected wealth & variance
    # ---------------------------
    def expected_wealth(self, r: float) -> float:
        w_X, w_Y = self.weights_from_r(r)
        # update health process weights
        self.health_process.p.w_X = w_X
        self.health_process.p.w_Y = w_Y

        # carry component
        carry_ann = w_X * self.p.r_X - w_Y * self.p.r_Y
        carry_return = carry_ann * self.p.T

        # price component via log-health Lévy exponents
        E_X = np.exp(self.health_process.psi_h(1)) - 1
        E_Y = np.exp(self.health_process.psi_h(-1)) - 1
        price_return = w_X * E_X - w_Y * E_Y

        return self.p.W0 * (1 + carry_return + price_return)

    def wealth_variance(self, r: float) -> float:
        w_X, w_Y = self.weights_from_r(r)
        self.health_process.p.w_X = w_X
        self.health_process.p.w_Y = w_Y

        var_h = self.health_process.variance_diffusion  # approximate diffusion variance
        return (w_X ** 2 + w_Y ** 2) * self.p.W0 ** 2 * var_h

    # ---------------------------
    # Log-health & liquidation
    # ---------------------------
    def h0(self, r: float) -> float:
        w_X, w_Y = self.weights_from_r(r)
        self.health_process.p.w_X = w_X
        self.health_process.p.w_Y = w_Y
        return self.health_process.h0()

    def liquidation_probability(self, r: float) -> Optional[float]:
        if self.p.fht_func is None:
            return None
        h0_val = self.h0(r)
        return self.p.fht_func(h0_val, self.health_process.psi_h, self.p.T)

    # ---------------------------
    # Utility function
    # ---------------------------
    def utility(self, r: float, rho1: float = 1.0, rho2: float = 1.0) -> Dict[str, Any]:
        meanW = self.expected_wealth(r)
        varW = self.wealth_variance(r)
        p_liq = self.liquidation_probability(r)

        if p_liq is None:
            U = meanW - rho1 * varW
        else:
            U = meanW - rho1 * varW - rho2 * p_liq

        return {
            "r": r,
            "U": U,
            "meanW": meanW,
            "varW": varW,
            "p_liq": p_liq
        }


# ---------------------------
# Convenience factory
# ---------------------------
def create_stablecoin_short_strategy(
    W0: float = 10_000.0,
    days: int = 30,
    r_X: float = 0.04,
    r_Y: float = 0.08,
    b_X: float = 0.8,
    X0: float = 1.0,
    Y0: float = 1.0,
    X_params: JumpDiffusionParams = None,
    Y_params: JumpDiffusionParams = None,
    fht_func: Optional[Callable[..., float]] = None
) -> StablecoinShortStrategy:
    params = StablecoinShortParams(
        W0=W0,
        T=days / 365.0,
        r_X=r_X,
        r_Y=r_Y,
        b_X=b_X,
        X0=X0,
        Y0=Y0,
        X_params=X_params,
        Y_params=Y_params,
        fht_func=fht_func
    )
    return StablecoinShortStrategy(params)