"""Accuracy versus run time: non-cardinal MLS (uniform and strike-clustered nodes) versus the
three-point finite-difference scheme with the same time stepping (Table 13)."""
import time

import numpy as np

from common import (GMFBM, EXAMPLE1, MLSPricer, eval_points, fd_solve, interp_log_cubic, ref_on,
                    save, time_mesh)
from exp_stability import sinh_nodes


def best_time(fn, reps=2):
    best, out = np.inf, None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def run():
    m = GMFBM(**EXAMPLE1)
    Se = eval_points(m.K)
    exE = m.european(Se, m.T, "put")
    refA = ref_on(m, Se)
    fd_solve(m, 50, time_mesh(m.T, 10, "twosided"), "put", bc="exact")  # numba warm-up
    fd_solve(m, 50, time_mesh(m.T, 10, "twosided"), "american")
    rows = []
    for Nx in (100, 200, 400, 800, 1600):
        tau = time_mesh(m.T, Nx, "twosided")
        for method in ("MLS uniform", "MLS clustered", "FD uniform"):
            for kind in ("european", "american"):
                if method.startswith("MLS"):
                    nodes = None if method == "MLS uniform" else sinh_nodes(Nx, 0.75)

                    def job():
                        p = MLSPricer(m, Nx, nodes=nodes)
                        if kind == "european":
                            U = p.european(tau, "put", "exact")["U"]
                        else:
                            U = p.american(tau)["U"]
                        return p.eval_matrices(Se)[0] @ U
                else:
                    def job():
                        S, u = fd_solve(m, Nx, tau, "put" if kind == "european" else "american",
                                        bc="exact" if kind == "european" else "fin")
                        return interp_log_cubic(S, u, Se, m.S_ref)
                t, v = best_time(job)
                ref = exE if kind == "european" else refA
                rows.append(dict(Nx=Nx, Nt=Nx, method=method, kind=kind, time=t,
                                 err=float(np.abs(v - ref).max() / m.K)))
                print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    save("efficiency", run())
