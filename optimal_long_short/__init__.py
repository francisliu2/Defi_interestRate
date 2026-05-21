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
