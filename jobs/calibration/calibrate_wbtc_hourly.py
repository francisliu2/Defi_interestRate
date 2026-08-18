"""Entry point for WBTC hourly empirical calibration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optimal_long_short.calibration.jobs.calibrate_wbtc_hourly import main

if __name__ == "__main__":
    main()
