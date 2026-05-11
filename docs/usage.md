# Usage Notes

## Main example

```bash
python codes/gmfbm_example1_mls.py
```

This script generates option value plots and exploratory amplification diagnostics.

## Temporal convergence

```bash
python codes/gmfbm_temporal_convergence_fixed_Nx400.py
```

This script fixes the spatial grid and varies the time step to estimate temporal convergence rates.

## Stability diagnostics

```bash
python codes/stability_diagnostics_reliable.py
```

This script generates a CSV file and a LaTeX table for the semi-discrete stability diagnostic.

## Important stability note

The stability diagnostic script checks the energy-stable finite-point/SBP realization of the semi-discrete operator. If a different MLS derivative matrix is used, the diagnostic should be recomputed for that exact operator.
