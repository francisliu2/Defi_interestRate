"""Entry point for expected-log-return-mean spread sensitivity analysis."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimal_long_short.job_runners.mu_spread_sensitivity import main


if __name__ == "__main__":
    main()
