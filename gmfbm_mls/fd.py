"""Independent second-order finite-difference solver (reference and efficiency comparison).

Centred differences in x = ln(S/S_*), averaged Crank-Nicolson in tau (optional Rannacher start:
the first `rannacher` steps are replaced by two implicit-Euler half steps each), and the
Brennan-Schwartz algorithm for the tridiagonal American-put complementarity problem.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _tri_solve(lo, di, up, b):
    n = b.size
    c = np.empty(n)
    d = np.empty(n)
    c[0] = up[0] / di[0]
    d[0] = b[0] / di[0]
    for i in range(1, n):
        den = di[i] - lo[i] * c[i - 1]
        c[i] = up[i] / den
        d[i] = (b[i] - lo[i] * d[i - 1]) / den
    x = np.empty(n)
    x[n - 1] = d[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


@njit(cache=True)
def _brennan_schwartz(lo, di, up, b, obst):
    # put: exercise region on the left -> eliminate the superdiagonal from the right,
    # then substitute forward with projection onto the obstacle.
    n = b.size
    d = di.copy()
    bb = b.copy()
    for i in range(n - 2, -1, -1):
        mlt = up[i] / d[i + 1]
        d[i] -= mlt * lo[i + 1]
        bb[i] -= mlt * bb[i + 1]
    x = np.empty(n)
    x[0] = max(bb[0] / d[0], obst[0])
    for i in range(1, n):
        x[i] = max((bb[i] - lo[i] * x[i - 1]) / d[i], obst[i])
    return x


def fd_solve(model, Nx, tau, kind="american", xmin=-6.0, xmax=6.0, rannacher=0, bc="fin"):
    """Return (S nodes, values at tau[-1])."""
    x = np.linspace(xmin, xmax, Nx + 1)
    dx = x[1] - x[0]
    S = model.S_ref * np.exp(x)
    K, r, q = model.K, model.r, model.delta
    put = kind in ("put", "american")
    u = np.maximum(K - S, 0.0) if put else np.maximum(S - K, 0.0)
    obst = u[1:-1].copy()
    n = Nx - 1
    ones = np.ones(n)

    def coeffs(a):
        lo = a / dx ** 2 - (r - q - a) / (2 * dx)
        di = -2 * a / dx ** 2 - r
        up = a / dx ** 2 + (r - q - a) / (2 * dx)
        return lo, di, up

    def bvals(t):
        if bc == "exact":
            return model.european(S[0], t, "put" if put else "call"), model.european(S[-1], t, "put" if put else "call")
        if kind == "american":
            return K - S[0], 0.0
        if kind == "put":
            return K * np.exp(-r * t) - S[0] * np.exp(-q * t), 0.0
        return 0.0, S[-1] * np.exp(-q * t) - K * np.exp(-r * t)

    def substep(u, t0, t1, th):
        dt = t1 - t0
        lo0, di0, up0 = coeffs(model.a(t0))
        lo1, di1, up1 = coeffs(model.a(t1))
        ui = u[1:-1]
        rhs = ui + (1 - th) * dt * (lo0 * u[:-2] + di0 * ui + up0 * u[2:])
        bl, br = bvals(t1)
        rhs[0] += th * dt * lo1 * bl
        rhs[-1] += th * dt * up1 * br
        L = -th * dt * lo1 * ones
        D = (1 - th * dt * di1) * ones
        Uu = -th * dt * up1 * ones
        if kind == "american":
            v = _brennan_schwartz(L, D, Uu, rhs, obst)
        else:
            v = _tri_solve(L, D, Uu, rhs)
        out = np.empty_like(u)
        out[0], out[-1] = bl, br
        out[1:-1] = v
        return out

    for k in range(len(tau) - 1):
        if k < rannacher:
            tm = 0.5 * (tau[k] + tau[k + 1])
            u = substep(u, tau[k], tm, 1.0)
            u = substep(u, tm, tau[k + 1], 1.0)
        else:
            u = substep(u, tau[k], tau[k + 1], 0.5)
    return S, u


def interp_log_cubic(S_nodes, values, S_eval, Sref):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(np.log(S_nodes / Sref), values)
    return cs(np.log(np.asarray(S_eval) / Sref))
