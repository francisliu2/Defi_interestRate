"""Entry point for numerical comparison tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optimal_long_short.monte_carlo.jobs.numerical_comparison import main

if __name__ == "__main__":
    main()
