# Non-cardinal MLS collocation for GMFBM option pricing

Code for the paper *A Semismooth Newton-Based Non-Cardinal MLS Collocation Method for Option Pricing and Sensitivity Analysis under Generalized Mixed Fractional Brownian Motion* (D. Damircheli, S.-M.-M. Kazemi).

The package implements exactly the discretization analyzed in the paper:

| Paper | Code |
|---|---|
| Sec. 3: non-cardinal MLS shape functions, mass matrix `M_h`, derivative matrices `D_{k,h}` with exact derivatives (eq. (mls_derivs)) | `gmfbm_mls/mls.py` |
| Sec. 2, Prop. (closed form): GMFBM coefficient `a(tau)`, its parameter derivatives, exact European prices, Greeks and parameter sensitivities | `gmfbm_mls/model.py` |
| Sec. 4: averaged Crank–Nicolson in coefficient form `(M_h - s_n L_{n+1}) U^{n+1} = (M_h + s_n L_n) U^n`, coefficient Dirichlet rows, uniform / graded / two-sided time meshes | `gmfbm_mls/solvers.py: solve_european, time_mesh` |
| Sec. 5, Alg. 1: American LCP on reconstructed prices `q = M_h U`, Fischer–Burmeister semismooth Newton with Armijo line search | `gmfbm_mls/solvers.py: solve_american_put, fb_semismooth_newton` |
| Sec. 8: Greeks via chain rule, Theta from the PDE, tangent equations reusing the price LU factors | `gmfbm_mls/solvers.py: greeks, solve_tangent` |
| Sec. 9: independent reference for the American put (centred FD + Brennan–Schwartz) | `gmfbm_mls/benchmark.py` |

Production settings (Sec. 9): degree `m = 3`, weight `(1 - r^2)^5`, support radius `3.5 h`, `N_x` uniform subintervals on `[-6, 6]`, two-sided graded time mesh.  The dissipativity condition (Assumption 13) is checked by `experiments/run_experiments.py dissipativity`. It holds for this configuration and fails for larger supports and for `m = 4`, so re-run it if you change the MLS parameters.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the paper

```bash
python experiments/run_experiments.py all     # results/*.json  (about 1-1.5 h on 2 cores)
python experiments/make_tables.py             # tables/*.tex
python experiments/make_figures.py            # figures/*.pdf
```

Individual experiments: `dissipativity`, `propagator`, `euro_space`, `euro_time`, `benchmarks`, `american` (needs `benchmarks`), `sensitivity`.
`python experiments/make_tables.py main_with_markers.tex main.tex` inserts the tables into a manuscript at the `%%TABLE:<name>%%` markers, so the numbers in the paper are always the numbers in `results/`.

## Quick check

```bash
python -m pytest -q tests
```

## Minimal example

```python
import numpy as np
from gmfbm_mls.model import EXAMPLE1 as m
from gmfbm_mls.solvers import Discretization, time_mesh, solve_european, solve_american_put

d = Discretization(nx=400)                       # m=3, (1-r^2)^5, support 3.5h
taus = time_mesh(400, m.T, "two_sided")
x = np.log(np.array([0.75, 1.0, 1.25]))          # x = ln(S/K)
E = d.eval_matrices(x, kmax=0)[0]                # MLS reconstruction at x
print(E @ solve_european(m, d, taus, "put", "financial").U, m.european(m.K * np.exp(x), m.T))
print(E @ solve_american_put(m, d, taus).U)
```
