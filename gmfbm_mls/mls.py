"""
Non-cardinal moving least squares (MLS) shape functions in one dimension.

This module implements the MLS construction of Section 3 of the paper exactly:

    psi(x, x_j) = w_j(x) * p(x)^T G(x)^{-1} p(x_j),     G(x) = sum_j w_j(x) p(x_j) p(x_j)^T,

with moving, compactly supported weights w_j(x) = w(|x - x_j| / delta) and a
polynomial basis of degree m.  The shape functions are *not* cardinal,
psi(x_i, x_j) != delta_ij, so the collocation mass matrix

    (M_h)_{ij} = psi(x_i, x_j)

is kept explicitly, together with the derivative matrices
(D_{k,h})_{ij} = d^k/dx^k psi(x_i, x_j), k = 1,...,4.

Derivatives are exact (no finite differences): with gamma(x) = G(x)^{-1} p(x),
    G gamma^{(k)} = p^{(k)} - sum_{l<k} binom(k,l) G^{(k-l)} gamma^{(l)},
    psi_j^{(k)}   = sum_{l<=k} binom(k,l) w_j^{(k-l)} gamma^{(l)} . p(x_j).
At an evaluation point x0 the basis is the fixed polynomial
p_l(x) = ((x - x0)/h)^l, which keeps G well conditioned.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Weight functions.  Each is w(u) for |u| < 1 (zero otherwise), returned
# together with its derivatives d^k w / du^k, k = 0..4.
# ---------------------------------------------------------------------------


def _poly_weight_derivs(coeffs: np.ndarray, u: np.ndarray, kmax: int) -> np.ndarray:
    """Derivatives of an even polynomial weight written in powers of u."""
    out = np.zeros((kmax + 1, u.size))
    c = np.polynomial.Polynomial(coeffs)
    inside = np.abs(u) < 1.0
    for k in range(kmax + 1):
        out[k, inside] = c.deriv(k)(u[inside]) if k > 0 else c(u[inside])
    return out


# w(u) = (1 - u^2)^5 on |u| < 1.  It is C^4 across |u| = 1 and smooth at 0,
# so the MLS shape functions have four continuous derivatives (needed for the
# third and fourth asset-price Greeks).
_WENDLAND_LIKE = np.polynomial.Polynomial([1.0, 0.0, -1.0]) ** 5


def weight_derivs(u: np.ndarray, kmax: int = 4, kind: str = "poly5") -> np.ndarray:
    if kind == "poly5":
        return _poly_weight_derivs(_WENDLAND_LIKE.coef, u, kmax)
    if kind == "cubic_spline":
        # Table 1 cubic spline (C^2), written in r = |u|.
        r = np.abs(u)
        s = np.where(u >= 0.0, 1.0, -1.0)   # sign(0) := 1 (w is even)
        out = np.zeros((kmax + 1, u.size))
        a = r < 0.5
        b = (r >= 0.5) & (r < 1.0)
        pa = np.polynomial.Polynomial([2 / 3, 0.0, -4.0, 4.0])
        pb = np.polynomial.Polynomial([4 / 3, -4.0, 4.0, -4 / 3])
        for k in range(kmax + 1):
            da = pa.deriv(k) if k else pa
            db = pb.deriv(k) if k else pb
            out[k, a] = da(r[a]) * s[a] ** k
            out[k, b] = db(r[b]) * s[b] ** k
        return out
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------


@dataclass
class MLSSpace:
    """MLS approximation space on a 1-D point set."""

    nodes: np.ndarray          # x_1 < ... < x_N
    degree: int = 3            # polynomial degree m
    support: float = 3.5       # delta = support * h (support radius)
    weight: str = "poly5"

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.h = float(np.max(np.diff(self.nodes)))      # fill-distance scale
        self.delta = self.support * self.h

    # -- core evaluation ---------------------------------------------------
    def shape_derivatives(self, x0: float, kmax: int = 2):
        """Return (indices J, array D[k, j]) with D[k] = d^k psi_J / dx^k at x0."""
        m, h, d = self.degree, self.h, self.delta
        u = (x0 - self.nodes) / d
        J = np.nonzero(np.abs(u) < 1.0)[0]
        if J.size < m + 1:
            raise np.linalg.LinAlgError(
                f"only {J.size} nodes in the support at x={x0:.4f}; need {m + 1}")
        wd = weight_derivs(u[J], kmax, self.weight)
        # d/dx = (1/delta) d/du
        wd = wd / d ** np.arange(kmax + 1)[:, None]
        z = (self.nodes[J] - x0) / h
        P = np.vander(z, m + 1, increasing=True)           # rows p(x_j)^T
        # derivatives of the fixed basis p_l(x) = ((x - x0)/h)^l at x = x0
        pk = np.zeros((kmax + 1, m + 1))
        for k in range(kmax + 1):
            if k <= m:
                pk[k, k] = factorial(k) / h ** k
        Gk = [P.T @ (wd[k][:, None] * P) for k in range(kmax + 1)]
        G = Gk[0]
        gam = []
        for k in range(kmax + 1):
            rhs = pk[k].copy()
            for l in range(k):
                rhs -= comb(k, l) * (Gk[k - l] @ gam[l])
            gam.append(np.linalg.solve(G, rhs))
        D = np.zeros((kmax + 1, J.size))
        for k in range(kmax + 1):
            for l in range(k + 1):
                D[k] += comb(k, l) * wd[k - l] * (P @ gam[l])
        return J, D

    def matrices(self, points: np.ndarray | None = None, kmax: int = 2):
        """Sparse matrices E_k with (E_k)_{ij} = psi^{(k)}(points_i, x_j), k=0..kmax.

        With points = nodes this returns [M_h, D_1h, D_2h, ...].
        """
        pts = self.nodes if points is None else np.asarray(points, dtype=float)
        rows = [[] for _ in range(kmax + 1)]
        cols = [[] for _ in range(kmax + 1)]
        vals = [[] for _ in range(kmax + 1)]
        for i, x0 in enumerate(pts):
            J, D = self.shape_derivatives(x0, kmax)
            for k in range(kmax + 1):
                rows[k].append(np.full(J.size, i))
                cols[k].append(J)
                vals[k].append(D[k])
        shape = (pts.size, self.nodes.size)
        return [
            sp.csr_matrix((np.concatenate(vals[k]), (np.concatenate(rows[k]), np.concatenate(cols[k]))), shape=shape)
            for k in range(kmax + 1)
        ]


def uniform_space(nx: int, xmin: float = -6.0, xmax: float = 6.0, **kw) -> MLSSpace:
    """N_x subintervals, N_x + 1 nodes (the convention of the paper)."""
    return MLSSpace(np.linspace(xmin, xmax, nx + 1), **kw)
