"""
Laplace-Resolvent Semi-Analytics for a Ratio Barrier Payoff
under Kou (double-exponential jump-diffusion) Dynamics.
"""
from optimal_long_short.calibration import (  # noqa: F401
    CalibrationGrid,
    ECFCalibrationResult,
    StandardizedCalibrationGrid,
    ParameterBounds,
    PreparedReturns,
    calibrate_ecf,
    empirical_cf,
    params_to_theta,
    prepare_returns,
)
from optimal_long_short.model.drift_service import (  # noqa: F401
    apply_price_drift_view,
    drift_summary,
    with_muX_drift_view,
    with_price_drift_view,
)
from optimal_long_short.carry import (  # noqa: F401
    CarryLeg,
    CarryOptimizationResult,
    CarrySpec,
    RateSource,
    apply_carry,
    conditional_sharpe,
    depeg_scenario,
    objective_phi,
    optimal_carry_h0,
    resolve_rate,
    size_for_target_pliq,
)
from optimal_long_short.drift_bayes import (  # noqa: F401
    gaussian_posterior,
    interior_threshold,
    optimal_buffer,
    sizing_distribution,
    spread_drift_standard_error,
)
from optimal_long_short.overshoot_recovery import (  # noqa: F401
    BivariateTiltedZDynamics,
    exact_recovery_moments,
    exact_wealth_moments,
    joint_hitting_moment,
)
from optimal_long_short.model.risk_report import h0_liquidation_moment_report  # noqa: F401
from optimal_long_short.model.strategy import minimum_feasible_h0  # noqa: F401
