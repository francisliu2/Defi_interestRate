"""
Bivariate Kou ECF calibration package.

Public API
----------
calibrate_ecf               main estimation function
ECFCalibrationResult        expanded result dataclass
empirical_cf                empirical characteristic function
StandardizedCalibrationGrid data-adaptive frequency grid
ParameterBounds             parameter bounds with moment-admissibility
PreparedReturns             output of prepare_returns
prepare_returns             price series -> synchronised log-returns
params_to_theta             KouParams -> natural-space flat vector
ewm_smooth                  exponential weighted mean smoother for returns

CalibrationGrid is kept as an alias for StandardizedCalibrationGrid for
backward compatibility with code written against the old flat calibration.py.
"""
from .calibrate import ECFCalibrationResult, calibrate_ecf
from .ecf_objective import empirical_cf
from .frequency_grid import StandardizedCalibrationGrid
from .transforms import ParameterBounds, params_to_theta
from .time_grid import PreparedReturns, prepare_returns
from .preprocess import ewm_smooth

CalibrationGrid = StandardizedCalibrationGrid

__all__ = [
    "ECFCalibrationResult",
    "calibrate_ecf",
    "empirical_cf",
    "StandardizedCalibrationGrid",
    "CalibrationGrid",
    "ParameterBounds",
    "params_to_theta",
    "PreparedReturns",
    "prepare_returns",
    "ewm_smooth",
]
