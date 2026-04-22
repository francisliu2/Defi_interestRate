from dataclasses import dataclass


@dataclass
class KouParams:
    """
    Parameters for the bivariate Kou double-exponential jump-diffusion model.

    Each asset i=1,2 has log-price X_i with drift mu_i, volatility sigma_i,
    jump intensity lambda_i, and double-exponential jumps with upward rate
    eta_i_pos, downward rate eta_i_neg, and upward jump probability p_i.
    The two Brownian parts have correlation rho.
    """
    # --- Asset 1 ---
    mu1: float       # drift of X_1
    sigma1: float    # volatility of X_1
    lam1: float      # jump intensity of X_1
    p1: float        # probability of an upward jump in X_1
    eta1_pos: float  # rate of upward jump size in X_1  (eta_{1,+})
    eta1_neg: float  # rate of downward jump size in X_1 (eta_{1,-})

    # --- Asset 2 ---
    mu2: float       # drift of X_2
    sigma2: float    # volatility of X_2
    lam2: float      # jump intensity of X_2
    p2: float        # probability of an upward jump in X_2
    eta2_pos: float  # rate of upward jump size in X_2  (eta_{2,+})
    eta2_neg: float  # rate of downward jump size in X_2 (eta_{2,-})

    # --- Correlation ---
    rho: float       # correlation between the two Brownian motions
