"""Entry point for the data-selected WETH/WBTC empirical showcase."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimal_long_short.job_runners.calibrate_eth_btc import main

if __name__ == "__main__":
    main()
