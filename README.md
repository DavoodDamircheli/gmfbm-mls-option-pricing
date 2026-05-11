# GMFBM MLS/FPM Option Pricing Codes

This repository contains the Python codes used to reproduce the numerical examples, temporal convergence study, and stability diagnostics for the paper:

> **A Semismooth Newton-Based Non-Cardinal MLS Collocation Method for Option Pricing and Sensitivity Analysis under Generalized \emph{Mixed} Fractional Brownian Motion**

The implementation focuses on option pricing under the generalized mixed fractional Brownian motion (GMFBM) model using a finite-point / moving least-squares (MLS/FPM) meshfree collocation method in log-price space.

## Repository structure

```text
gmfbm-mls-option-pricing/
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
│
├── codes/
│   ├── gmfbm_example1_mls.py
│   ├── gmfbm_temporal_convergence_fixed_Nx400.py
│   └── stability_diagnostics_reliable.py
│
├── results/
│   ├── example1/
│   ├── temporal_convergence/
│   └── stability_diagnostics/
│
├── figures/
│
└── docs/
    ├── index.md
    └── usage.md
```

## Code description

### `codes/gmfbm_example1_mls.py`

Main reproducibility script for the first numerical example. It solves the GMFBM option-pricing problem in log-price coordinates using an MLS/FPM derivative reconstruction and Crank--Nicolson time stepping. It includes:

- European put pricing,
- American put pricing using a projected Crank--Nicolson approximation, with optional PSOR for the LCP,
- price curve and surface plots,
- exploratory Crank--Nicolson amplification diagnostics.

Typical run:

```bash
python codes/gmfbm_example1_mls.py
```

Expected output folder:

```text
gmfbm_example1_output/
```

### `codes/gmfbm_temporal_convergence_fixed_Nx400.py`

Temporal convergence experiment for Example 1 with fixed spatial resolution. The script fixes the spatial grid, varies the number of time steps, and compares the computed solutions against a fine same-grid reference solution. It reports normalized price errors over a financial window.

Typical run:

```bash
python codes/gmfbm_temporal_convergence_fixed_Nx400.py
```

Expected output folder:

```text
temporal_convergence_results/
```

Expected outputs include:

```text
ex1_temporal_convergence_fixed_Nx400.csv
ex1_temporal_convergence_fixed_Nx400.png
ex1_temporal_convergence_fixed_Nx400.pdf
```

### `codes/stability_diagnostics_reliable.py`

Stability diagnostic script for the semi-discrete GMFBM log-price operator. It generates a CSV file and a LaTeX table for the reported stability diagnostic.

Typical run:

```bash
python codes/stability_diagnostics_reliable.py
```

Expected output folder:

```text
gmfbm_stability_output/
```

Expected outputs include:

```text
stability_diagnostics_reliable.csv
stability_diagnostics_reliable_table.tex
```

**Important note.** This diagnostic checks the energy-stable finite-point/SBP realization of the semi-discrete operator. If the exploratory row-wise MLS derivative matrices are used instead, the stability diagnostic should be rerun using the corresponding exact mass matrix and assembled spatial operator.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running all scripts

From the repository root:

```bash
python codes/gmfbm_example1_mls.py
python codes/gmfbm_temporal_convergence_fixed_Nx400.py
python codes/stability_diagnostics_reliable.py
```

## Suggested GitHub Pages setup

This repository includes a `docs/` folder that can be used as the GitHub Pages source.

After pushing the repository to GitHub:

1. Go to **Settings**.
2. Open **Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose the main branch and select `/docs` as the publishing folder.
5. Save.

Then edit:

```text
docs/index.md
```

to describe the paper, method, and code usage.

## Suggested citation text for the paper

In the numerical results section, the repository can be cited as:

```latex
All numerical experiments reported in this section were implemented in Python.
The source codes for the MLS/FPM option-pricing solver, temporal convergence
study, and stability diagnostics are available at
\url{https://github.com/YOUR-GITHUB-USERNAME/gmfbm-mls-option-pricing}.
```

A separate code availability statement can be added near the end of the paper:

```latex
\section*{Code Availability}
The Python codes used to reproduce the numerical experiments in this paper are
available at
\url{https://github.com/YOUR-GITHUB-USERNAME/gmfbm-mls-option-pricing}.
The repository includes the implementation of the GMFBM MLS/FPM option-pricing
solver, temporal convergence tests, and stability diagnostic scripts.
```

## License

This repository is currently distributed under the MIT License. Replace the license if the journal or collaborators require a different one.
