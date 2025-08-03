# Hawkes First Passage Time Analysis for DeFi Lending

This project implements Hawkes jump-diffusion models for analyzing first passage times in DeFi lending platforms, focusing on optimal long-short positioning and liquidation risk management.

## Project Structure

```
INTEREST_RATE/
├── latex/                  # LaTeX documents and academic paper
│   ├── notes.tex          # Main paper
│   ├── definitions.tex    # Mathematical definitions
│   └── finance.bib        # Bibliography
├── src/                   # Python source code
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Data files
│   ├── raw/              # Raw market data
│   ├── processed/        # Processed/calibrated data
│   └── results/          # Analysis results
├── scripts/              # Standalone scripts
├── tests/                # Unit tests
├── docs/                 # Documentation
└── config/               # Configuration files
```

## Installation

```bash
pip install -e .
```

## Usage

See notebooks/ for example usage and analysis workflows.

## Academic Paper

The theoretical framework is detailed in `latex/notes.tex`, which covers:
- Hawkes jump-diffusion processes with cross-excitation
- Health factor dynamics in DeFi lending
- Wrong-way risk management
- First passage time analysis
- Peak-over-threshold calibration methods

## License

MIT License - see LICENSE file for details.