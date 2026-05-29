"""Entry point for generic empirical ECF calibration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimal_long_short.job_runners.calibrate_empirical import main

if __name__ == "__main__":
    main()

