"""Entry point for health-buffer evaluation-map generation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optimal_long_short.job_runners.health_buffer_evaluation_map import main

if __name__ == "__main__":
    main()
