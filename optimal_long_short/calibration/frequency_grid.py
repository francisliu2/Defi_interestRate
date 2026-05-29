"""
Standardized frequency grid for bivariate Kou ECF calibration.

Raw frequencies are scaled by the per-period robust standard deviation of
the returns, so the same dimensionless base grid works for any sampling
frequency (hourly, 4-hourly, daily, etc.).

For a return series with scale s_i, a dimensionless base value a maps to
raw frequency a / s_i.  The ECF argument u * r_i ≈ a * (r_i / s_i) is
then dimensionless and comparable across frequencies.

Groups produced by build_from_returns:
    "marginal_1"  (±a/s1,  0      )  identifies asset-1 marginal Kou law
    "marginal_2"  (0,     ±a/s2   )  identifies asset-2 marginal Kou law
    "joint_pp"    (±a/s1, ±b/s2   )  identifies Brownian correlation rho
    "joint_pm"    (±a/s1, ∓b/s2   )  mixed-sign joint direction
    "spread"      (±a/sz, ∓a/sz   )  identifies Z = X1 - X2 (DeFi liquidation)
    "jump_tail"   high-base marginal+spread points (jump-sensitive regime)

Each primary point (u, v) is paired with its conjugate (-u, -v) (same
label and base weight).  This doubles the grid size but ensures that the
ECF objective treats the real and imaginary parts of phi symmetrically,
improving optimizer numerical balance.

Weights are attached to the standardized base frequency, not the raw
frequency, so they are identical across sampling rates:
    w(a, b) = exp(-weight_decay * (a^2 + b^2))

Using Gaussian decay with weight_decay=0.02 (default) gives useful weight
at medium-high base frequencies (w(5)≈0.61, w(10)≈0.14) which are critical
for identifying jump-size parameters.  The old rational form 1/(1+a^2+b^2)
kills these frequencies (w(5)≈0.04, w(10)≈0.01) and hides jump-tail shape.

The spread group receives an additional ``spread_weight`` multiplier
(default 2) to emphasise the Z = X1 - X2 direction that governs DeFi
liquidation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MARGINAL_BASE: np.ndarray  = np.array([0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0])
_JOINT_BASE: np.ndarray     = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
_SPREAD_BASE: np.ndarray    = np.array([0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
_JUMP_TAIL_BASE: np.ndarray = np.array([8.0, 12.0, 16.0, 20.0, 25.0])


def _robust_scale(r: np.ndarray, min_scale: float = 1e-8) -> float:
    """IQR-based robust scale: (q75 - q25) / 1.349."""
    q25, q75 = np.percentile(r, [25, 75])
    return float(max((q75 - q25) / 1.349, min_scale))


@dataclass
class StandardizedCalibrationGrid:
    """
    Three-group frequency grid whose raw frequencies adapt to the return scale.

    The grid is built by calling build_from_returns(r1, r2), which computes
    per-asset robust scales (s1, s2, sz) and scales each base frequency
    a -> a / s_i.  Weights use the dimensionless base value a, not the raw
    frequency a/s_i, so they are identical across sampling rates.

    weight_decay : float, default 0.02
        Gaussian decay parameter: w(a, b) = exp(-weight_decay*(a^2+b^2)).
        Lower values give more weight to high-frequency points (jump tail),
        higher values concentrate weight on low-frequency (mean, variance).
    spread_weight : float, default 2.0
        Multiplier applied to spread-group point weights, emphasising the
        Z = X1 - X2 direction that governs DeFi liquidation.
    """
    marginal_base: np.ndarray = field(default_factory=lambda: _MARGINAL_BASE.copy())
    joint_base: np.ndarray    = field(default_factory=lambda: _JOINT_BASE.copy())
    spread_base: np.ndarray   = field(default_factory=lambda: _SPREAD_BASE.copy())
    jump_tail_base: np.ndarray = field(default_factory=lambda: _JUMP_TAIL_BASE.copy())
    weight_decay: float = 0.02
    spread_weight: float = 2.0
    jump_tail_weight: float = 1.0
    min_scale: float = 1e-8
    max_raw_freq: float = 5_000.0

    def build_from_returns(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Build the frequency grid adapted to the empirical return distribution.

        Parameters
        ----------
        r1, r2 : (N,) log-return arrays.

        Returns
        -------
        freqs      : (M, 2)  raw frequency pairs.
        weights    : (M,)    Gaussian decay weights based on standardized base frequency.
        groups     : (M,)    string labels ("marginal_1", ..., "spread").
        scale_info : dict    with keys "s1", "s2", "sz".
        """
        s1 = _robust_scale(r1, self.min_scale)
        s2 = _robust_scale(r2, self.min_scale)
        sz = _robust_scale(r1 - r2, self.min_scale)

        pairs:   list[tuple[float, float]] = []
        base_ab: list[tuple[float, float]] = []
        labels:  list[str] = []
        w_mults: list[float] = []

        def _add(u: float, v: float, a: float, b: float, lbl: str,
                 w_mult: float = 1.0) -> None:
            """Add (u, v) and its conjugate (-u, -v) with given weight multiplier."""
            if abs(u) > self.max_raw_freq or abs(v) > self.max_raw_freq:
                return
            for su, sv in [(u, v), (-u, -v)]:
                pairs.append((su, sv))
                base_ab.append((a, b))
                labels.append(lbl)
                w_mults.append(w_mult)

        for a in self.marginal_base:
            _add(float(a) / s1, 0.0,  float(a), 0.0,  "marginal_1")
            _add(0.0, float(a) / s2,  0.0,  float(a), "marginal_2")

        for a in self.joint_base:
            for b in self.joint_base:
                u  = float(a) / s1
                vp = float(b) / s2
                _add(u,  vp, float(a), float(b), "joint_pp")
                _add(u, -vp, float(a), float(b), "joint_pm")

        for a in self.spread_base:
            s = float(a) / sz
            _add(s, -s, float(a), float(a), "spread", w_mult=self.spread_weight)

        for a in self.jump_tail_base:
            _add(float(a) / s1, 0.0,         float(a), 0.0,         "jump_tail", w_mult=self.jump_tail_weight)
            _add(0.0, float(a) / s2,         0.0,      float(a),    "jump_tail", w_mult=self.jump_tail_weight)
            _add(float(a) / sz, -float(a) / sz, float(a), float(a), "jump_tail", w_mult=self.jump_tail_weight)

        freqs    = np.array(pairs)    # (M, 2)
        base_arr = np.array(base_ab)  # (M, 2)
        groups   = np.array(labels)
        weights  = np.exp(-self.weight_decay * (base_arr[:, 0] ** 2 + base_arr[:, 1] ** 2)) * np.array(w_mults)

        return freqs, weights, groups, {"s1": s1, "s2": s2, "sz": sz}
