from dataclasses import dataclass, field


@dataclass
class KouParams:
    """
    Parameters for the bivariate Kou double-exponential jump-diffusion model,
    following Sepp (2004) notation.

    Each asset i=1,2 has log-price

        X_i(t) = mu_i^eff * t + sigma_i * B_i(t) + sum_{n=1}^{N_i(t)} J_i^n,

    where mu_i is the continuously compounded expected asset-price growth rate,
    i.e. E_P[S_{i,t}/S_{i,0}] = E_P[exp(X_{i,t})] = exp(mu_i * t).

    eta_i_pos, eta_i_neg are the **means** of positive/negative jump sizes
    (requires eta_i_pos < 1).

    The price-jump compensator is

        chi_i = E[exp(J_i) - 1] = M_i(1) - 1
              = p_i / (1 - eta_i_pos) + (1 - p_i) / (1 + eta_i_neg) - 1,

    and the effective log-price drift is

        mu_i^eff = mu_i - 0.5 * sigma_i^2 - lambda_i * chi_i.

    This ensures E_P[exp(X_{i,t})] = exp(mu_i * t) exactly (no-shorting
    limit: as h0 -> inf, E[Pi_T] -> exp(mu_1 * T)).

    The effective drift is computed automatically in __post_init__ and stored
    as effective_mu1 and effective_mu2. All downstream objects (Laplace
    resolvent, Monte Carlo) use the effective drift.
    """
    # --- Asset 1 ---
    mu1: float       # continuously compounded expected price growth rate of X_1
    sigma1: float    # volatility of X_1
    lam1: float      # jump intensity of X_1
    p1: float        # probability of an upward jump in X_1
    eta1_pos: float  # mean of upward jump size in X_1  (eta_{1,+}; requires < 1)
    eta1_neg: float  # mean of downward jump size in X_1 (eta_{1,-})

    # --- Asset 2 ---
    mu2: float       # continuously compounded expected price growth rate of X_2
    sigma2: float    # volatility of X_2
    lam2: float      # jump intensity of X_2
    p2: float        # probability of an upward jump in X_2
    eta2_pos: float  # mean of upward jump size in X_2  (eta_{2,+}; requires < 1/k for tilt order k)
    eta2_neg: float  # mean of downward jump size in X_2 (eta_{2,-})

    # --- Correlation ---
    rho: float       # correlation between the two Brownian motions

    # --- Derived (set in __post_init__) ---
    jump_compensator1: float = field(init=False, repr=False)
    jump_compensator2: float = field(init=False, repr=False)
    effective_mu1: float = field(init=False, repr=False)
    effective_mu2: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # chi_i = E[exp(J_i) - 1] = M_i(1) - 1
        chi1 = self.p1 / (1.0 - self.eta1_pos) + (1.0 - self.p1) / (1.0 + self.eta1_neg) - 1.0
        chi2 = self.p2 / (1.0 - self.eta2_pos) + (1.0 - self.p2) / (1.0 + self.eta2_neg) - 1.0
        # mu_i^eff = mu_i - 0.5*sigma_i^2 - lambda_i*chi_i
        # ensures E[exp(X_{i,t})] = exp(mu_i * t)
        self.jump_compensator1 = 0.5 * self.sigma1 ** 2 + self.lam1 * chi1
        self.jump_compensator2 = 0.5 * self.sigma2 ** 2 + self.lam2 * chi2
        self.effective_mu1 = self.mu1 - self.jump_compensator1
        self.effective_mu2 = self.mu2 - self.jump_compensator2
