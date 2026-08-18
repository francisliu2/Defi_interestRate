"""Entry point for paired moving-block calibration uncertainty."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimal_long_short.calibration.jobs.calibration_uncertainty import main


if __name__ == "__main__":
    main()
