"""Entry point for numerical robustness diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimal_long_short.laplace.jobs.numerical_diagnostics import main

if __name__ == "__main__":
    main()
