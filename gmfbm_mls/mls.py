"""Non-cardinal moving least squares (MLS) shape functions in 1D with exact derivatives.

Shape functions  psi(x, x_j) = w_j(x) * gamma(x)^T p(x_j),  G(x) gamma(x) = p(x),
G(x) = sum_j w_j(x) p(x_j) p(x_j)^T.  Derivatives follow from differentiating
G gamma = p (recursion (3.4) of the paper); no numerical differentiation is used.
At every evaluation point x0 the scaled, shifted basis p_l(x) = ((x - x0)/hloc)^l
is used, so that G(x0) is well conditioned.
"""
from math import comb, factorial

import numpy as np
import scipy.sparse as sp

# weight W(s) = (1 - s^2)^5 on |s| < 1 (C^4), as a polynomial in s
_W = np.polynomial.Polynomial([1.0, 0.0, -1.0]) ** 5
_WD = [_W.deriv(k) if k else _W for k in range(5)]


def weight_derivs(s, kmax):
    """Return array (kmax+1, len(s)) of d^k W / ds^k at s (zero outside |s|<1)."""
    s = np.asarray(s, dtype=float)
    out = np.zeros((kmax + 1, s.size))
    inside = np.abs(s) < 1.0
    for k in range(kmax + 1):
        out[k, inside] = _WD[k](s[inside])
    return out


class MLS1D:
    """MLS approximation space on a 1D node set.

    nodes  : increasing array of N nodes (x_1 = x_min, x_N = x_max)
    degree : polynomial degree m
    radius : support radius of each node (scalar or array of length N)
    """

    def __init__(self, nodes, degree=3, radius=None, radius_factor=3.5):
        self.x = np.asarray(nodes, dtype=float)
        self.N = self.x.size
        self.m = degree
        # local spacing at each node: mean gap over the (at most) two neighbours on each side;
        # on a uniform grid this is exactly the grid spacing h
        loc = np.empty(self.N)
        for j in range(self.N):
            lo, hi = max(0, j - 2), min(self.N - 1, j + 2)
            loc[j] = (self.x[hi] - self.x[lo]) / (hi - lo)
        self.hloc = loc
        if radius is None:
            radius = radius_factor * loc
        self.rho = np.broadcast_to(np.asarray(radius, dtype=float), (self.N,)).copy()
        self.rho_max = self.rho.max()

    def _neighbors(self, x0):
        lo = np.searchsorted(self.x, x0 - self.rho_max, side="left")
        hi = np.searchsorted(self.x, x0 + self.rho_max, side="right")
        idx = np.arange(lo, hi)
        idx = idx[np.abs(x0 - self.x[idx]) < self.rho[idx]]
        return idx

    def shape_derivs(self, x0, kmax):
        """Return (idx, D) with D[k, :] = d^k/dx^k psi(x0, x_idx), k = 0..kmax."""
        idx = self._neighbors(x0)
        xj = self.x[idx]
        hs = np.interp(x0, self.x, self.hloc)
        m = self.m
        P = ((xj - x0) / hs)[:, None] ** np.arange(m + 1)[None, :]  # (n, m+1): p(x_j)
        s = (x0 - xj) / self.rho[idx]
        Wk = weight_derivs(s, kmax) / self.rho[idx][None, :] ** np.arange(kmax + 1)[:, None]
        Gk = [P.T @ (Wk[k][:, None] * P) for k in range(kmax + 1)]
        # p^{(k)}(x0)
        pk = np.zeros((kmax + 1, m + 1))
        for k in range(kmax + 1):
            if k <= m:
                pk[k, k] = factorial(k) / hs ** k
        G0 = Gk[0]
        gam = []
        for k in range(kmax + 1):
            rhs = pk[k].copy()
            for l in range(k):
                rhs -= comb(k, l) * (Gk[k - l] @ gam[l])
            gam.append(np.linalg.solve(G0, rhs))
        D = np.zeros((kmax + 1, idx.size))
        for k in range(kmax + 1):
            acc = np.zeros(idx.size)
            for l in range(k + 1):
                acc += comb(k, l) * Wk[k - l] * (P @ gam[l])
            D[k] = acc
        return idx, D

    def matrices(self, eval_points=None, kmax=2):
        """Sparse matrices [M, D1, ..., Dkmax], rows = evaluation points (default: nodes)."""
        xe = self.x if eval_points is None else np.asarray(eval_points, dtype=float)
        rows, cols, vals = [], [], [[] for _ in range(kmax + 1)]
        for i, x0 in enumerate(xe):
            idx, D = self.shape_derivs(x0, kmax)
            rows.append(np.full(idx.size, i))
            cols.append(idx)
            for k in range(kmax + 1):
                vals[k].append(D[k])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        shape = (xe.size, self.N)
        return [sp.csr_matrix((np.concatenate(vals[k]), (rows, cols)), shape=shape)
                for k in range(kmax + 1)]
