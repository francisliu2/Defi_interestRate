import cmath
from dataclasses import dataclass, field

from optimal_long_short.model_params import KouParams


@dataclass
class BivariateKouModel:
    """
    Bivariate Kou double-exponential jump-diffusion model.

    The joint log-price process (X1, X2) has Lévy-Khintchine exponent

        Psi(u, v) = i*(mu1*u + mu2*v)
                    - 0.5*(sigma1^2*u^2 + sigma2^2*v^2 + 2*rho*sigma1*sigma2*u*v)
                    + lam1*(phi_J1(u) - 1)
                    + lam2*(phi_J2(v) - 1),

    where the Kou characteristic function of asset i's jump is

        phi_Ji(u) = p_i * eta_i_pos / (eta_i_pos - i*u)
                  + (1 - p_i) * eta_i_neg / (eta_i_neg + i*u).
    """
    params: KouParams

    def phi_J(self, u: complex, i: int) -> complex:
        """
        Characteristic function of the jump distribution for asset i (i=1 or 2).

            phi_Ji(u) = p_i * eta_i_pos / (eta_i_pos - i*u)
                      + (1 - p_i) * eta_i_neg / (eta_i_neg + i*u)
        """
        p = self.params
        if i == 1:
            pi, ep, en = p.p1, p.eta1_pos, p.eta1_neg
        elif i == 2:
            pi, ep, en = p.p2, p.eta2_pos, p.eta2_neg
        else:
            raise ValueError(f"i must be 1 or 2, got {i}")
        return pi * ep / (ep - 1j * u) + (1 - pi) * en / (en + 1j * u)

    def levy_khintchine(self, u: complex, v: complex) -> complex:
        """
        Bivariate Lévy-Khintchine exponent Psi(u, v) such that

            E[exp(i*(u*X1_t + v*X2_t))] = exp(t * Psi(u, v)).

        Parameters
        ----------
        u, v : complex
            Fourier arguments for X1 and X2 respectively.

        Returns
        -------
        complex
            The value of Psi(u, v).
        """
        p = self.params
        diffusion = (
            1j * (p.mu1 * u + p.mu2 * v)
            - 0.5 * (
                p.sigma1**2 * u**2
                + p.sigma2**2 * v**2
                + 2 * p.rho * p.sigma1 * p.sigma2 * u * v
            )
        )
        jumps = (
            p.lam1 * (self.phi_J(u, 1) - 1)
            + p.lam2 * (self.phi_J(v, 2) - 1)
        )
        return diffusion + jumps


@dataclass
class KouZTiltedDynamics:
    """
    Lévy-Khintchine exponent and Kou phase rates of Z = X1 - X2
    under the k-tilted measure P^(2,k).

    Under P^(2,k), defined by dP^(2,k)/dP|_{F_t} = exp(k*X2_t - t*Psi(0,-ik)),
    the log-ratio process Z is Lévy with exponent (eq. psiZk in the paper):

        psi_Z^(k)(s) = mu_Z^(k) * s
                       + 0.5 * sigma_Z^2 * s^2
                       + lam1 * (M1(s) - 1)
                       + lam2 * (M2(k - s) - M2(k)),

    where
        sigma_Z^2  = sigma1^2 + sigma2^2 - 2*rho*sigma1*sigma2,
        mu_Z^(k)   = (mu1 - mu2) - k*(sigma2^2 - rho*sigma1*sigma2),

    and M_i(s) = E[e^{s*J_i}] is the MGF of the Kou jump for asset i:

        M_i(s) = p_i * eta_i_pos / (eta_i_pos - s)
               + (1 - p_i) * eta_i_neg / (eta_i_neg + s).

    The k-dependent Kou phase rates (eq. r in the paper) are:

        r1_pos = eta1_pos           r1_neg = eta1_neg
        r2_pos = eta2_neg + k       r2_neg = eta2_pos - k

    These are the exponential rates of the upward (+) and downward (-)
    jump components of Z under P^(2,k), and appear in the phase operators
    K_{j,±}^(k) used to form the barrier linear system.

    Parameters
    ----------
    params : KouParams
        Model parameters.
    k : int
        Tilting order (k=0 for untilted, k=1 or k=2 for moment computations).
        Requires eta2_pos > k for the tilt to be well-defined.
    """
    params: KouParams
    k: int

    def __post_init__(self) -> None:
        if self.params.eta2_pos <= self.k:
            raise ValueError(
                f"eta2_pos must be greater than k for the tilt to be well-defined; "
                f"got eta2_pos={self.params.eta2_pos}, k={self.k}"
            )

    @staticmethod
    def _mgf_kou(s: complex, pi: float, ep: float, en: float) -> complex:
        """MGF of a Kou double-exponential jump: M(s) = p*ep/(ep-s) + (1-p)*en/(en+s)."""
        return pi * ep / (ep - s) + (1 - pi) * en / (en + s)

    def mgf_J1(self, s: complex) -> complex:
        """MGF of the jump distribution of X1: M1(s) = E[e^{s*J1}]."""
        p = self.params
        return self._mgf_kou(s, p.p1, p.eta1_pos, p.eta1_neg)

    def mgf_J2(self, s: complex) -> complex:
        """MGF of the jump distribution of X2: M2(s) = E[e^{s*J2}]."""
        p = self.params
        return self._mgf_kou(s, p.p2, p.eta2_pos, p.eta2_neg)

    @property
    def sigma_Z_sq(self) -> float:
        """Variance of the Brownian part of Z: sigma_Z^2 = sigma1^2 + sigma2^2 - 2*rho*sigma1*sigma2."""
        p = self.params
        return p.sigma1**2 + p.sigma2**2 - 2 * p.rho * p.sigma1 * p.sigma2

    @property
    def mu_Z(self) -> float:
        """Drift of Z under P^(2,k): mu_Z^(k) = (mu1 - mu2) - k*(sigma2^2 - rho*sigma1*sigma2)."""
        p = self.params
        return (p.mu1 - p.mu2) - self.k * (p.sigma2**2 - p.rho * p.sigma1 * p.sigma2)

    # ------------------------------------------------------------------
    # Phase rates  r_{j,±}^(k)  (Kou jump rates of Z under P^(2,k))
    # ------------------------------------------------------------------

    @property
    def r1_pos(self) -> float:
        """r_{1,+}^(k) = eta1_pos  (upward rate from X1 jumps; k-invariant)."""
        return self.params.eta1_pos

    @property
    def r1_neg(self) -> float:
        """r_{1,-}^(k) = eta1_neg  (downward rate from X1 jumps; k-invariant)."""
        return self.params.eta1_neg

    @property
    def r2_pos(self) -> float:
        """r_{2,+}^(k) = eta2_neg + k  (upward rate from X2 jumps under tilt)."""
        return self.params.eta2_neg + self.k

    @property
    def r2_neg(self) -> float:
        """r_{2,-}^(k) = eta2_pos - k  (downward rate from X2 jumps under tilt)."""
        return self.params.eta2_pos - self.k

    def __call__(self, s: complex) -> complex:
        """
        Evaluate psi_Z^(k)(s).

        Parameters
        ----------
        s : complex
            Argument in the moment-generating domain.

        Returns
        -------
        complex
            psi_Z^(k)(s).
        """
        return (
            self.mu_Z * s
            + 0.5 * self.sigma_Z_sq * s**2
            + self.params.lam1 * (self.mgf_J1(s) - 1)
            + self.params.lam2 * (self.mgf_J2(self.k - s) - self.mgf_J2(self.k))
        )
