"""Entry point for h0 liquidation and moment CSV reports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optimal_long_short.job_runners.h0_liquidation_report import main

if __name__ == "__main__":
    main()
