import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

from gmfbm_mls.model import GMFBM, EXAMPLE1, EXAMPLE2, time_mesh, eval_points  # noqa: E402
from gmfbm_mls.solvers import MLSPricer  # noqa: E402
from gmfbm_mls.fd import fd_solve, interp_log_cubic  # noqa: E402


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def orders(errs, ns):
    """Observed orders log(E1/E2)/log(n2/n1)."""
    out = [None]
    for i in range(1, len(errs)):
        out.append(float(np.log(errs[i - 1] / errs[i]) / np.log(ns[i] / ns[i - 1])))
    return out


def american_reference(model, Nx=12000, Nt=4000, rannacher=2):
    S, u = fd_solve(model, Nx, time_mesh(model.T, Nt, "twosided"), "american", rannacher=rannacher)
    return S, u


def ref_on(model, S_eval, Nx=12000, Nt=4000):
    S, u = american_reference(model, Nx, Nt)
    return interp_log_cubic(S, u, S_eval, model.S_ref)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t
