# gmfbm-mls — non-cardinal MLS collocation for GMFBM option pricing (manuscript v38)

This package reproduces every table and figure of Section 9 of the manuscript
"A Semismooth Newton-Based Non-Cardinal MLS Collocation Method for Option Pricing and
Sensitivity Analysis under Generalized Mixed Fractional Brownian Motion" (v38).

## Contents

| path | content |
|---|---|
| `gmfbm_mls/mls.py` | 1D MLS shape functions with exact derivatives (moment-matrix recursion), weight (1-z^2)^5 |
| `gmfbm_mls/model.py` | GMFBM coefficient a(tau), closed-form European prices and Greeks, time meshes |
| `gmfbm_mls/solvers.py` | averaged Crank-Nicolson coefficient scheme, coefficient Dirichlet data, Fischer-Burmeister semismooth Newton for American puts, tangent sensitivities, Greeks |
| `gmfbm_mls/fd.py` | independent finite-difference reference (centred FD, Rannacher start, Brennan-Schwartz) |
| `experiments/exp_stability.py` | Tables 3-5: certificates, mass-matrix bounds, radius/degree scan, node sets, propagator; `extra`: certificates for every further American configuration and the quartic-basis runs |
| `experiments/exp_european.py` | Tables 6-9: European pointwise, spatial, temporal, 1+beta, T = 2 |
| `experiments/exp_american.py` | Tables 10-11: reference study, American pointwise, semismooth Newton, boundary mismatch |
| `experiments/exp_greeks.py` | Tables 12-13: Greeks and tangent sensitivities |
| `experiments/exp_hurst.py` | Table 14: effect of the Hurst exponents |
| `experiments/exp_efficiency.py` | Table 15: accuracy versus run time, MLS versus FD |
| `experiments/make_tables.py` | writes the LaTeX table bodies to `../tables/` |
| `experiments/make_figures.py` | writes all figures to `../figures/` |
| `results/*.json` | raw numbers behind every table |

## Requirements

Python >= 3.10, `numpy`, `scipy`, `matplotlib`, `numba` (see `requirements.txt`).

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh        # about 1 hour on a laptop; efficiency timings depend on the machine
```

`run_all.sh` runs every experiment, then `make_tables.py` and `make_figures.py`. The
manuscript includes the generated tables with `\input{tables/...}`.

## Conventions

* log-price `x = ln(S/K)` on `[-6, 6]`; MLS degree 3, support radius 3.5 h (node-dependent
  radius 3.5 x local mean spacing on non-uniform point sets);
* Dirichlet data on the first and last MLS coefficients; American obstacle on reconstructed
  prices `q = M_h U`;
* errors are `max |u - u_h| / K` over 401 log-uniform points in `[0.2K, 2.5K]` unless stated.
