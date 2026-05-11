# GMFBM MLS/FPM Option Pricing Codes

This page accompanies the paper:

> **A Semismooth Newton-Based Non-Cardinal MLS Collocation Method for Option Pricing and Sensitivity Analysis under Generalized \emph{Mixed} Fractional Brownian Motion**

The repository provides Python scripts for reproducing the main numerical example, temporal convergence study, and stability diagnostics for GMFBM option pricing using an MLS/FPM meshfree collocation method.

## Quick start

```bash
git clone https://github.com/DavoodDamircheli/gmfbm-mls-option-pricing.git
cd gmfbm-mls-option-pricing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the scripts:

```bash
python codes/gmfbm_example1_mls.py
python codes/gmfbm_temporal_convergence_fixed_Nx400.py
python codes/stability_diagnostics_reliable.py
```

## Contents

- Main Example 1 solver: `codes/gmfbm_example1_mls.py`
- Temporal convergence experiment: `codes/gmfbm_temporal_convergence_fixed_Nx400.py`
- Stability diagnostics: `codes/stability_diagnostics_reliable.py`
