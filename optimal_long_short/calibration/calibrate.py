"""
Public API for bivariate Kou ECF calibration.

Two calling conventions are supported:

    # Already-prepared returns
    result = calibrate_ecf(r1, r2, dt_years=1/(365*24))

    # Raw prices + timestamps (resampling and dt inference happen automatically)
    result = calibrate_ecf(
        prices1=p1, prices2=p2, timestamps=ts, target_freq="1h"
    )

In both cases every fitted parameter is annualized.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from optimal_long_short.model_params import KouParams
from .transforms import (
    ParameterBounds, _DEFAULT_BOUNDS,
    unc_to_params, theta_to_params, params_to_theta, nat_to_unc,
)
from .frequency_grid import StandardizedCalibrationGrid
from .initializer import initialize, build_multistart_cloud, pot_anchors, _lambda_from_moments
from .ecf_objective import (
    empirical_cf, model_cf, objective_unc, objective_unc_pot_anchored, objective_by_group,
)
from .time_grid import prepare_returns


@dataclass
class ECFCalibrationResult:
    """
    Full result of bivariate Kou ECF calibration.

    Core outputs
    ------------
    params           : Fitted KouParams (all annualized).
    objective        : Best normalized ECF objective Q_N.
    objective_by_group : Normalized sub-objective per frequency group.
    success          : Whether the best optimizer run reported convergence.
    message          : Optimizer convergence message.
    n_iter           : Iterations used by the best optimizer run.

    Data provenance
    ---------------
    n_obs            : Number of return observations N.
    dt_years         : Annualized observation interval Delta.
    frequency_label  : Human-readable frequency ("1h", "1d", "provided", …).

    Initialization
    --------------
    theta0           : KouParams at the initializer point (before optimization).
    best_start_index : Which element of the multi-start cloud gave the best result.
    starts_objectives : Objective value from each starting point.

    Grid
    ----
    freqs  : (M, 2) raw frequency pairs used.
    weights: (M,)  weights.
    groups : (M,)  group labels.
    scale_info : dict with keys "s1", "s2", "sz" (per-period robust scales).

    Post-calibration
    ----------------
    diagnostics : dict from diagnose_calibration (empty if run_diagnostics=False).
    """
    params: KouParams
    objective: float
    objective_by_group: dict
    success: bool
    message: str
    n_iter: int
    n_obs: int
    dt_years: float
    frequency_label: str
    theta0: KouParams
    best_start_index: int
    starts_objectives: list
    freqs: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    scale_info: dict
    diagnostics: dict = field(default_factory=dict)


def calibrate_ecf(
    r1: np.ndarray | None = None,
    r2: np.ndarray | None = None,
    dt_years: float | None = None,
    *,
    prices1: np.ndarray | None = None,
    prices2: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
    target_freq: str = "1h",
    grid: StandardizedCalibrationGrid | None = None,
    bounds: ParameterBounds | None = None,
    theta0: np.ndarray | None = None,
    n_starts: int = 20,
    threshold: float | None = None,
    max_moment_order: int = 4,
    moment_anchor_weight: float = 0.05,
    eta_anchor_weight: float = 0.02,
    seed: int | None = None,
    run_diagnostics: bool = True,
) -> ECFCalibrationResult:
    """
    Estimate bivariate Kou parameters by minimising the ECF distance.

    The estimator matches the empirical characteristic function of (r1, r2)
    to exp(dt * Psi_theta(u, v)) over a standardized three-group frequency
    grid using a bounded L-BFGS-B optimizer run from a structured multi-start
    cloud seeded by a robust MAD threshold initializer.

    Parameters (returns path)
    -------------------------
    r1, r2    : 1-D arrays of observed bivariate log-returns (positional OK).
    dt_years  : Annualized time step, e.g. 1/(365*24) for hourly crypto.

    Parameters (prices path)
    ------------------------
    prices1, prices2 : Raw price series (any positive values).
    timestamps       : POSIX timestamps in seconds aligned with prices, or None.
    target_freq      : Resampling target: "15min", "1h", "4h", "1d".

    Common parameters
    -----------------
    grid             : StandardizedCalibrationGrid; auto-built if None.
    bounds           : ParameterBounds; defaults with max_moment_order.
    theta0           : 13-element natural-space starting vector; auto if None.
    n_starts         : Multi-start cloud size.
    threshold          : MAD multiplier for diffusion-subset classification.
    max_moment_order   : K such that K*eta2_pos < 1 is enforced.
    moment_anchor_weight : Weight of the log-lambda POT soft penalty.
    eta_anchor_weight    : Weight of the log-eta POT soft penalty (lambda-eta
                           degeneracy guard; default 0.02).
    seed               : RNG seed for the multi-start perturbation cloud.
    run_diagnostics    : Whether to call diagnose_calibration after fitting.

    Returns
    -------
    ECFCalibrationResult
    """
    # ------------------------------------------------------------------
    # 1. Resolve inputs into (r1, r2, dt_years, freq_label)
    # ------------------------------------------------------------------
    freq_label = "provided"
    if r1 is None or r2 is None or dt_years is None:
        if prices1 is None or prices2 is None:
            raise ValueError(
                "Provide either (r1, r2, dt_years) or (prices1, prices2)."
            )
        prep = prepare_returns(
            prices1, prices2, timestamps=timestamps, target_freq=target_freq
        )
        r1, r2      = prep.r1, prep.r2
        dt_years    = prep.dt_years
        freq_label  = prep.freq_label

    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if r1.ndim != 1 or r1.shape != r2.shape:
        raise ValueError("r1 and r2 must be 1-D arrays of equal length.")
    if dt_years is None or dt_years <= 0:
        raise ValueError(f"dt_years must be positive, got {dt_years!r}.")

    # ------------------------------------------------------------------
    # 2. Build grid, empirical CF, and initial parameter cloud
    # ------------------------------------------------------------------
    if bounds is None:
        bounds = ParameterBounds(max_moment_order=max_moment_order)
    if grid is None:
        grid = StandardizedCalibrationGrid()

    freqs, weights, groups, scale_info = grid.build_from_returns(r1, r2)
    phi_hat = empirical_cf(r1, r2, freqs)

    theta_nat0 = theta0 if theta0 is not None else initialize(
        r1, r2, dt_years, bounds, threshold
    )
    rng = np.random.default_rng(seed)
    starts_unc = build_multistart_cloud(theta_nat0, n_starts, rng, bounds)
    unc_bds = bounds.unc_bounds()

    # ------------------------------------------------------------------
    # 3. Multi-start L-BFGS-B optimization
    # ------------------------------------------------------------------
    if moment_anchor_weight > 0 or eta_anchor_weight > 0:
        # Lambda anchor: two-moment estimator (more reliable than POT extrapolation)
        sigma1_init = float(theta_nat0[1])
        sigma2_init = float(theta_nat0[7])
        lam1_anchor = _lambda_from_moments(r1, dt_years, sigma1_init, bounds.lambda_min, bounds.lambda_max)
        lam2_anchor = _lambda_from_moments(r2, dt_years, sigma2_init, bounds.lambda_min, bounds.lambda_max)
        log_lam1_anch = float(np.log(max(lam1_anchor, 1e-12)))
        log_lam2_anch = float(np.log(max(lam2_anchor, 1e-12)))
        # Eta anchor: POT excess means (unbiased for exponential, prevents eta collapse)
        eta_anch = pot_anchors(r1, r2, dt_years, bounds=bounds)
        obj_fn = objective_unc_pot_anchored
        opt_args = (phi_hat, dt_years, freqs, weights,
                    log_lam1_anch, log_lam2_anch, moment_anchor_weight,
                    eta_anch["log_ep1"], eta_anch["log_en1"],
                    eta_anch["log_ep2"], eta_anch["log_en2"],
                    eta_anchor_weight, bounds)
    else:
        obj_fn = objective_unc
        opt_args = (phi_hat, dt_years, freqs, weights, bounds)

    best_fun  = np.inf
    best_res  = None
    best_idx  = 0
    starts_objs: list[float] = []

    for i, tau0 in enumerate(starts_unc):
        res = minimize(
            obj_fn,
            x0=tau0,
            args=opt_args,
            method="L-BFGS-B",
            bounds=unc_bds,
            options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-9},
        )
        fun = float(res.fun) if np.isfinite(res.fun) else 1e100
        starts_objs.append(fun)
        if fun < best_fun:
            best_fun = fun
            best_res = res
            best_idx = i

    if best_res is None:
        raise RuntimeError("All optimizer starts failed to produce a finite objective.")

    # ------------------------------------------------------------------
    # 4. Assemble result
    # ------------------------------------------------------------------
    best_params = unc_to_params(best_res.x, bounds)
    grp_obj     = objective_by_group(
        best_res.x, phi_hat, dt_years, freqs, weights, groups, bounds
    )
    theta0_params = theta_to_params(theta_nat0)

    result = ECFCalibrationResult(
        params             = best_params,
        objective          = best_fun,
        objective_by_group = grp_obj,
        success            = bool(best_res.success),
        message            = str(best_res.message),
        n_iter             = int(best_res.nit),
        n_obs              = len(r1),
        dt_years           = float(dt_years),
        frequency_label    = freq_label,
        theta0             = theta0_params,
        best_start_index   = best_idx,
        starts_objectives  = starts_objs,
        freqs              = freqs,
        weights            = weights,
        groups             = groups,
        scale_info         = scale_info,
    )

    if run_diagnostics:
        from .diagnostics import diagnose_calibration
        result.diagnostics = diagnose_calibration(result, r1, r2)

    return result
