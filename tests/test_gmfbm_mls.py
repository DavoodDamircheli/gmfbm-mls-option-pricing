import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmfbm_mls.mls import uniform_space  # noqa: E402
from gmfbm_mls.model import EXAMPLE1 as m  # noqa: E402
from gmfbm_mls.solvers import (Discretization, solve_american_put, solve_european,  # noqa: E402
                               solve_tangent, time_mesh)


def test_polynomial_reproduction_and_noncardinality():
    S = uniform_space(60, degree=3, support=3.5)
    E = S.matrices(kmax=3)
    x = S.nodes
    for q in range(4):
        for k, Ek in enumerate(E):
            exact = np.zeros_like(x) if k > q else np.prod(range(q - k + 1, q + 1)) * x ** (q - k)
            assert np.max(np.abs(Ek @ x**q - exact)) < 1e-6 * max(1.0, np.abs(exact).max())
    assert abs(E[0] - np.eye(x.size)).max() > 0.3          # M_h is not the identity


def test_dissipativity():
    d = Discretization(100)
    M, D1, D2 = (d.D[k].toarray() for k in range(3))
    Mi = M[1:-1, 1:-1]
    for tau in (0.0, m.T):
        A = np.linalg.solve(Mi, (m.a(tau) * (D2 - D1) + (m.r - m.delta) * D1 - m.r * M)[1:-1, 1:-1])
        assert np.linalg.eigvalsh(0.5 * (A + A.T)).max() < 0


def test_european_against_closed_form():
    d = Discretization(200)
    U = solve_european(m, d, time_mesh(400, m.T, "two_sided"), "put", "exact").U
    x = np.log(np.linspace(0.5, 1.5, 11))
    err = np.abs(d.eval_matrices(x, 0)[0] @ U - m.european(m.K * np.exp(x), m.T)).max()
    assert err < 1e-2


def test_american_and_tangent():
    d = Discretization(100)
    sol = solve_american_put(m, d, time_mesh(100, m.T, "two_sided"))
    assert sol.info["max_final_residual"] < 1e-10 and sol.info["complementarity"] < 1e-10
    taus = time_mesh(100, 0.5, "two_sided")
    s = solve_european(m, d, taus, "put", "exact", keep_history=True, keep_factors=True)
    Z = solve_tangent(m, d, s, "sigma")
    eps = 1e-6
    fd = (solve_european(m.with_(sigma=m.sigma + eps), d, taus).U
          - solve_european(m.with_(sigma=m.sigma - eps), d, taus).U) / (2 * eps)
    assert np.max(np.abs(Z - fd)) < 1e-6
