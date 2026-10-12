"""
Independent reference solver for the American put (and European sanity check).

Second-order centred finite differences in x = ln(S/K) on a fine uniform grid,
averaged Crank--Nicolson in tau with a Rannacher start (two steps of two implicit
Euler half steps) and a time mesh graded at both tau = 0 and tau = T.  At each step the
tridiagonal LCP is solved exactly by the Brennan--Schwartz algorithm (valid for the
put, whose exercise region is x < x_f(tau)).  This benchmark shares no code with
the MLS solver.
"""
from __future__ import annotations

import numpy as np

from .model import GMFBM
from .solvers import time_mesh


def _tridiag_ops(model: GMFBM, h: float, a: float):
    """Coefficients (sub, diag, sup) of L = a(D2 - D1) + (r - delta) D1 - r I."""
    mu = model.r - model.delta - a
    lo = a / h**2 - mu / (2 * h)
    di = -2 * a / h**2 - model.r
    up = a / h**2 + mu / (2 * h)
    return lo, di, up


def _brennan_schwartz(lo, di, up, rhs, phi):
    """Solve the tridiagonal LCP A u >= rhs, u >= phi, complementarity, where
    rows 0 and n-1 are Dirichlet rows.  Backward elimination, then forward
    substitution with projection."""
    n = di.size
    d = di.copy()
    b = rhs.copy()
    for i in range(n - 2, -1, -1):
        f = up[i] / d[i + 1]
        d[i] -= f * lo[i + 1]
        b[i] -= f * b[i + 1]
    u = np.empty(n)
    u[0] = max(phi[0], b[0] / d[0])
    for i in range(1, n):
        u[i] = max(phi[i], (b[i] - lo[i] * u[i - 1]) / d[i])
    return u


def reference_put(model: GMFBM, nx: int = 12000, nt: int = 4000, american: bool = True,
                  xmin: float = -6.0, xmax: float = 6.0):
    x = np.linspace(xmin, xmax, nx + 1)
    h = x[1] - x[0]
    S = model.K * np.exp(x)
    phi = np.maximum(model.K - S, 0.0) if american else np.full_like(S, -np.inf)
    u = np.maximum(model.K - S, 0.0)
    taus = time_mesh(nt, model.T, "two_sided")
    n = x.size

    def step(u, t0, t1, theta):
        dt = t1 - t0
        lo1, di1, up1 = _tridiag_ops(model, h, model.a(t1))
        lo0, di0, up0 = _tridiag_ops(model, h, model.a(t0))
        # right-hand side (I + (1-theta) dt L0) u
        rhs = u.copy()
        if theta < 1.0:
            c = (1 - theta) * dt
            rhs[1:-1] += c * (lo0 * u[:-2] + di0 * u[1:-1] + up0 * u[2:])
        lo = np.full(n, -theta * dt * lo1)
        di = np.full(n, 1.0 - theta * dt * di1)
        up = np.full(n, -theta * dt * up1)
        lo[0] = up[0] = 0.0
        di[0] = 1.0
        lo[-1] = up[-1] = 0.0
        di[-1] = 1.0
        rhs[0] = model.K if american else model.K * np.exp(-model.r * t1)
        rhs[-1] = 0.0
        if american:
            return _brennan_schwartz(lo, di, up, rhs, phi)
        return _brennan_schwartz(lo, di, up, rhs, np.full(n, -np.inf))

    for k in range(nt):
        t0, t1 = taus[k], taus[k + 1]
        if k < 2:
            tm = 0.5 * (t0 + t1)
            u = step(u, t0, tm, 1.0)
            u = step(u, tm, t1, 1.0)
        else:
            u = step(u, t0, t1, 0.5)
    return x, S, u
