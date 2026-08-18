import math
from dataclasses import dataclass, field


@dataclass
class KouParams:
    """
    Parameters for the bivariate Kou double-exponential jump-diffusion model,
    following Sepp (2004) notation.

    Each asset i=1,2 has log-price

        X_i(t) = mu_i^X * t + sigma_i * B_i(t) + sum_{n=1}^{N_i(t)} J_i^n,

    where mu_i is the continuously compounded expected asset-price growth rate,
    i.e. E_P[S_{i,t}/S_{i,0}] = E_P[exp(X_{i,t})] = exp(mu_i * t).

    eta_i_pos, eta_i_neg are the **means** of positive/negative jump sizes
    (requires eta_i_pos < 1).

    The price-jump compensator is

        chi_i = E[exp(J_i) - 1] = M_i(1) - 1
              = p_i / (1 - eta_i_pos) + (1 - p_i) / (1 + eta_i_neg) - 1,

    and the log-price drift is

        mu_i^X = mu_i - 0.5 * sigma_i^2 - lambda_i * chi_i.

    This ensures E_P[exp(X_{i,t})] = exp(mu_i * t) exactly (no-shorting
    limit: as h0 -> inf, E[Pi_T] -> exp(mu_1 * T)).

    The muX drift is computed automatically in __post_init__ and stored
    as muX1 and muX2. All downstream objects (Laplace
    resolvent, Monte Carlo) use the muX drift.
    """
    # --- Asset 1 ---
    mu1: float       # continuously compounded expected price growth rate of X_1
    sigma1: float    # volatility of X_1
    lam1: float      # jump intensity of X_1
    p1: float        # probability of an upward jump in X_1
    eta1_pos: float  # mean upward jump size (also requires < 1/k for moment order k)
    eta1_neg: float  # mean of downward jump size in X_1 (eta_{1,-})

    # --- Asset 2 ---
    mu2: float       # continuously compounded expected price growth rate of X_2
    sigma2: float    # volatility of X_2
    lam2: float      # jump intensity of X_2
    p2: float        # probability of an upward jump in X_2
    eta2_pos: float  # mean upward jump size (also requires < 1/k for moment order k)
    eta2_neg: float  # mean of downward jump size in X_2 (eta_{2,-})

    # --- Correlation ---
    rho: float       # correlation between the two Brownian motions

    # --- Derived (set in __post_init__) ---
    jump_compensator1: float = field(init=False, repr=False)
    jump_compensator2: float = field(init=False, repr=False)
    muX1: float = field(init=False, repr=False)
    muX2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = {
            name: getattr(self, name)
            for name in (
                "mu1", "sigma1", "lam1", "p1", "eta1_pos", "eta1_neg",
                "mu2", "sigma2", "lam2", "p2", "eta2_pos", "eta2_neg", "rho",
            )
        }
        nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
        if nonfinite:
            raise ValueError(f"Kou parameters must be finite; invalid: {', '.join(nonfinite)}")
        for index in (1, 2):
            sigma = getattr(self, f"sigma{index}")
            intensity = getattr(self, f"lam{index}")
            probability = getattr(self, f"p{index}")
            eta_pos = getattr(self, f"eta{index}_pos")
            eta_neg = getattr(self, f"eta{index}_neg")
            if sigma <= 0.0:
                raise ValueError(f"sigma{index} must be positive, got {sigma}")
            if intensity <= 0.0:
                raise ValueError(f"lam{index} must be positive, got {intensity}")
            if not 0.0 < probability < 1.0:
                raise ValueError(f"p{index} must lie in (0, 1), got {probability}")
            if not 0.0 < eta_pos < 1.0:
                raise ValueError(f"eta{index}_pos must lie in (0, 1), got {eta_pos}")
            if eta_neg <= 0.0:
                raise ValueError(f"eta{index}_neg must be positive, got {eta_neg}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho}")

        # chi_i = E[exp(J_i) - 1] = M_i(1) - 1
        chi1 = self.p1 / (1.0 - self.eta1_pos) + (1.0 - self.p1) / (1.0 + self.eta1_neg) - 1.0
        chi2 = self.p2 / (1.0 - self.eta2_pos) + (1.0 - self.p2) / (1.0 + self.eta2_neg) - 1.0
        # Full log-to-price correction: despite the historical attribute name,
        # this contains both the diffusion Ito term and the jump-price
        # compensator.  mu_i^X = mu_i - 0.5*sigma_i^2 - lambda_i*chi_i
        # ensures E[exp(X_{i,t})] = exp(mu_i * t).
        self.jump_compensator1 = 0.5 * self.sigma1 ** 2 + self.lam1 * chi1
        self.jump_compensator2 = 0.5 * self.sigma2 ** 2 + self.lam2 * chi2
        self.muX1 = self.mu1 - self.jump_compensator1
        self.muX2 = self.mu2 - self.jump_compensator2

    @property
    def diffusion_ito_correction1(self) -> float:
        return 0.5 * self.sigma1**2

    @property
    def diffusion_ito_correction2(self) -> float:
        return 0.5 * self.sigma2**2

    @property
    def jump_price_compensator1(self) -> float:
        """Return lambda_1 E[exp(J_1)-1], excluding the diffusion term."""
        return self.jump_compensator1 - self.diffusion_ito_correction1

    @property
    def jump_price_compensator2(self) -> float:
        """Return lambda_2 E[exp(J_2)-1], excluding the diffusion term."""
        return self.jump_compensator2 - self.diffusion_ito_correction2

    @property
    def mean_jump_rate1(self) -> float:
        """Return lambda_1 E[J_1], the jump contribution to E[X_1(t)]/t."""
        mean_jump = self.p1 * self.eta1_pos - (1.0 - self.p1) * self.eta1_neg
        return self.lam1 * mean_jump

    @property
    def mean_jump_rate2(self) -> float:
        """Return lambda_2 E[J_2], the jump contribution to E[X_2(t)]/t."""
        mean_jump = self.p2 * self.eta2_pos - (1.0 - self.p2) * self.eta2_neg
        return self.lam2 * mean_jump

    @property
    def log_to_price_correction1(self) -> float:
        """Return 0.5 sigma_1^2 + lambda_1 E[exp(J_1)-1]."""
        return self.jump_compensator1

    @property
    def log_to_price_correction2(self) -> float:
        """Return 0.5 sigma_2^2 + lambda_2 E[exp(J_2)-1]."""
        return self.jump_compensator2
