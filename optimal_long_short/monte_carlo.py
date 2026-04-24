from dataclasses import dataclass

import numpy as np

from optimal_long_short.model_params import KouParams
from optimal_long_short.strategy import UnitExposureLongShortStrategy


@dataclass
class MonteCarloResult:
    """
    Results from a Monte Carlo simulation of the long-short DeFi strategy.

    Attributes
    ----------
    S1 : np.ndarray, shape (n_paths,)
        Terminal asset 1 prices S1_T.
    S2 : np.ndarray, shape (n_paths,)
        Terminal asset 2 prices S2_T.
    survived : np.ndarray of bool, shape (n_paths,)
        True if the path was not liquidated before T (tau > T).
    payoff : np.ndarray, shape (n_paths,)
        Killed payoff Pi_T = (w1*S1_T - w2*S2_T)^+ * 1_{tau > T}.
    p_surv : float
        Estimated survival probability P(tau > T).
    conditional_mean : float
        E[Pi_T | tau > T].
    conditional_variance : float
        Var(Pi_T | tau > T).
    """
    S1: np.ndarray
    S2: np.ndarray
    survived: np.ndarray
    payoff: np.ndarray
    p_surv: float
    conditional_mean: float
    conditional_variance: float


@dataclass
class MonteCarlo:
    """
    Monte Carlo simulator for the bivariate Kou long-short strategy.

    Simulates paths of (X1_t, X2_t) over [0, T] using n_steps Euler steps,
    applies the Kou double-exponential jumps exactly at each step via
    compound Poisson thinning, tracks the health ratio H_t = b*w1*S1_t/(w2*S2_t),
    and records the terminal payoff Pi_T = (w1*S1_T - w2*S2_T)^+ * 1_{tau > T}.

    Parameters
    ----------
    params : KouParams
        Kou model parameters.
    strategy : UnitExposureLongShortStrategy
        Portfolio strategy, supplying h0, T, w1, w2, and market params.
    n_paths : int
        Number of simulated paths.
    n_steps : int
        Number of time steps per path.
    seed : int or None
        Random seed for reproducibility.
    """
    params: KouParams
    strategy: UnitExposureLongShortStrategy
    n_paths: int = 10_000
    n_steps: int = 252
    seed: int = None

    def _sample_kou_jumps(
        self,
        rng: np.random.Generator,
        lam: float,
        p: float,
        eta_pos: float,
        eta_neg: float,
        dt: float,
        n: int,
    ) -> np.ndarray:
        """
        Sample the total Kou jump in one time step for n paths.

        Number of jumps ~ Poisson(lam * dt); each jump is
        +Exp(mean=eta_pos) with probability p, -Exp(mean=eta_neg) with prob 1-p.
        eta_pos and eta_neg are the **means** of the exponential distributions
        (Sepp 2004 convention).
        """
        n_jumps = rng.poisson(lam * dt, size=n)
        total = np.zeros(n)
        for i in np.where(n_jumps > 0)[0]:
            nj = n_jumps[i]
            signs = rng.choice([1, -1], size=nj, p=[p, 1 - p])
            sizes = np.where(
                signs == 1,
                rng.exponential(eta_pos, size=nj),
                -rng.exponential(eta_neg, size=nj),
            )
            total[i] = sizes.sum()
        return total

    def run(self) -> MonteCarloResult:
        """
        Run the Monte Carlo simulation and return a MonteCarloResult.

        Returns
        -------
        MonteCarloResult
        """
        p = self.params
        s = self.strategy
        m = s.market
        T, n_paths, n_steps = s.T, self.n_paths, self.n_steps
        dt = T / n_steps

        rng = np.random.default_rng(self.seed)

        # --- initialise log-prices and log-health ---
        X1 = np.zeros(n_paths)
        X2 = np.zeros(n_paths)
        alive = np.ones(n_paths, dtype=bool)   # 1 = not yet liquidated

        # Cholesky for correlated Brownians: dB1 = dW1, dB2 = rho*dW1 + sqrt(1-rho^2)*dW2
        sqrt_dt = np.sqrt(dt)
        rho = p.rho
        sqrt_1mrho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

        # Log-price drift per step: X_i is the log-price, so no Ito correction needed
        drift1 = p.effective_mu1 * dt
        drift2 = p.effective_mu2 * dt

        for _ in range(n_steps):
            # correlated Brownian increments
            W1 = rng.standard_normal(n_paths)
            W2 = rng.standard_normal(n_paths)
            dB1 = sqrt_dt * W1
            dB2 = sqrt_dt * (rho * W1 + sqrt_1mrho2 * W2)

            # Kou jumps
            J1 = self._sample_kou_jumps(rng, p.lam1, p.p1, p.eta1_pos, p.eta1_neg, dt, n_paths)
            J2 = self._sample_kou_jumps(rng, p.lam2, p.p2, p.eta2_pos, p.eta2_neg, dt, n_paths)

            X1 += drift1 + p.sigma1 * dB1 + J1
            X2 += drift2 + p.sigma2 * dB2 + J2

            # check liquidation: H_t = b * w1 * S10 * exp(X1) / (w2 * S20 * exp(X2))
            #                       = exp(h0 + X1 - X2)  < 1  iff  X1 - X2 < -h0
            alive &= (X1 - X2) >= -s.h0

        # --- terminal values ---
        S1_T = m.S10 * np.exp(X1)
        S2_T = m.S20 * np.exp(X2)

        payoff_gross = np.maximum(s.w1 * S1_T - s.w2 * S2_T, 0.0)
        payoff = payoff_gross * alive   # killed payoff Pi_T

        # --- conditional moments ---
        p_surv = alive.mean()
        survived_payoffs = payoff[alive]
        cond_mean = survived_payoffs.mean() if alive.any() else 0.0
        cond_var  = survived_payoffs.var()  if alive.any() else 0.0

        return MonteCarloResult(
            S1=S1_T,
            S2=S2_T,
            survived=alive,
            payoff=payoff,
            p_surv=float(p_surv),
            conditional_mean=float(cond_mean),
            conditional_variance=float(cond_var),
        )
