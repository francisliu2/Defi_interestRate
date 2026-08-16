"""Entry point for the empirical semi-analytical/Monte Carlo comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimal_long_short.job_runners.empirical_method_comparison import main


if __name__ == "__main__":
    main()
