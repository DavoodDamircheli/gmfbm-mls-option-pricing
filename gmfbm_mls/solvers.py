"""
Fully discrete non-cardinal MLS schemes (Sections 4, 5, 8 of the paper).

European (Section 4): coefficient equation with the MLS mass matrix,
    (M_h - s_n L_h(tau_{n+1})) U^{n+1} = (M_h + s_n L_h(tau_n)) U^n,   s_n = (tau_{n+1}-tau_n)/2,
    L_h(tau) = a(tau)(D_2h - D_1h) + (r - delta) D_1h - r M_h,
boundary rows replaced by the coefficient conditions U_1 = g_min, U_N = g_max
(the boundary treatment analysed in Sections 6-7).

American put (Section 5): at every step the LCP
    A U - b >= 0,  M_h U - phi >= 0,  (A U - b) o (M_h U - phi) = 0   (interior rows),
    U_1 = K,  U_N = 0                                                   (boundary rows),
with A = M_h - s_n L_h(tau_{n+1}), b = (M_h + s_n L_h(tau_n)) U^n, i.e. the obstacle is
imposed on the reconstructed nodal prices q = M_h U.  It is solved by the
Fischer--Burmeister semismooth Newton method with an Armijo line search.

Sensitivities (Section 8): Greeks from the derivative matrices, Theta from the PDE,
and model-parameter sensitivities from the discrete tangent equation, which reuses
the LU factorisation of the price matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mls import MLSSpace, uniform_space
from .model import GMFBM


# ---------------------------------------------------------------------------
# Time meshes
# ---------------------------------------------------------------------------

def time_mesh(N: int, T: float, kind: str = "uniform", gamma: float = 2.0) -> np.ndarray:
    """tau_0 = 0 < ... < tau_N = T.  'graded' clusters steps at tau = T (valuation
    date), tau_n = T (1 - (1 - n/N)^gamma), as in Corollary of Section 7."""
    s = np.linspace(0.0, 1.0, N + 1)
    if kind == "uniform":
        return T * s
    if kind == "graded":
        return T * (1.0 - (1.0 - s) ** gamma)
    if kind == "two_sided":
        # graded at both ends (smoothstep, local grading exponent 2 at tau=0 and tau=T):
        # resolves the early-exercise layer at maturity and the GMFBM singularity at tau=T
        return T * s * s * (3.0 - 2.0 * s)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Discretisation container
# ---------------------------------------------------------------------------

@dataclass
class Discretization:
    nx: int
    degree: int = 3
    support: float = 3.5
    weight: str = "poly5"
    xmin: float = -6.0
    xmax: float = 6.0
    kmax: int = 2

    def __post_init__(self):
        self.space: MLSSpace = uniform_space(self.nx, self.xmin, self.xmax, degree=self.degree,
                                             support=self.support, weight=self.weight)
        self.x = self.space.nodes
        self.h = self.space.h
        mats = self.space.matrices(kmax=self.kmax)
        self.M = mats[0].tocsr()
        self.D = mats            # D[k] = k-th derivative matrix, D[0] = M
        self.N = self.x.size

    def S(self, model: GMFBM) -> np.ndarray:
        return model.K * np.exp(self.x)

    def L(self, model: GMFBM, tau: float, a_value: float | None = None):
        a = model.a(tau) if a_value is None else a_value
        D1, D2 = self.D[1], self.D[2]
        return a * (D2 - D1) + (model.r - model.delta) * D1 - model.r * self.M

    def eval_matrices(self, x_eval: np.ndarray, kmax: int = 2):
        return self.space.matrices(points=np.asarray(x_eval), kmax=kmax)


def _set_rows(A: sp.spmatrix, rows) -> sp.csr_matrix:
    """Replace the given rows of A by identity rows (coefficient Dirichlet rows)."""
    A = A.tolil()
    for i in rows:
        A.rows[i] = [i]
        A.data[i] = [1.0]
    return A.tocsr()


# ---------------------------------------------------------------------------
# European options
# ---------------------------------------------------------------------------

def boundary_values(model: GMFBM, disc: Discretization, tau: float, kind: str, bc: str):
    S0, S1 = model.K * np.exp(disc.xmin), model.K * np.exp(disc.xmax)
    if bc == "exact":
        return float(model.european(S0, tau, kind)), float(model.european(S1, tau, kind))
    if bc == "financial":
        if kind == "put":
            return model.K * np.exp(-model.r * tau), 0.0
        return 0.0, S1 * np.exp(-model.delta * tau) - model.K * np.exp(-model.r * tau)
    if bc == "american_put":
        return model.K, 0.0
    raise ValueError(bc)


def payoff(model: GMFBM, S: np.ndarray, kind: str) -> np.ndarray:
    return np.maximum(model.K - S, 0.0) if kind == "put" else np.maximum(S - model.K, 0.0)


@dataclass
class Solution:
    taus: np.ndarray
    U: np.ndarray                       # coefficients at the final level
    history: list = field(default_factory=list)   # coefficients at all levels (optional)
    info: dict = field(default_factory=dict)


def solve_european(model: GMFBM, disc: Discretization, taus: np.ndarray, kind: str = "put",
                   bc: str = "exact", rannacher: int = 0, keep_history: bool = False,
                   keep_factors: bool = False) -> Solution:
    """Averaged Crank--Nicolson in coefficient form (eq. (4.x) of the paper)."""
    M = disc.M
    K_op = (disc.D[2] - disc.D[1]).tocsr()
    C_op = ((model.r - model.delta) * disc.D[1] - model.r * M).tocsr()
    S = disc.S(model)
    U = payoff(model, S, kind)              # U^0_j = h(x_j)
    hist = [U.copy()] if keep_history else []
    factors = []
    bidx = (0, disc.N - 1)
    a_prev = model.a(taus[0])
    for n in range(len(taus) - 1):
        t0, t1 = taus[n], taus[n + 1]
        a1 = model.a(t1)
        if n < rannacher:   # optional Rannacher start: two implicit Euler half steps
            tm = 0.5 * (t0 + t1)
            for (s0, s1) in ((t0, tm), (tm, t1)):
                dt = s1 - s0
                A = _set_rows(M - dt * (model.a(s1) * K_op + C_op), bidx)
                b = M @ U
                b[0], b[-1] = boundary_values(model, disc, s1, kind, bc)
                U = spla.spsolve(A.tocsc(), b)
        else:
            s = 0.5 * (t1 - t0)
            A = _set_rows(M - s * (a1 * K_op + C_op), bidx)
            b = (M + s * (a_prev * K_op + C_op)) @ U
            b[0], b[-1] = boundary_values(model, disc, t1, kind, bc)
            lu = spla.splu(A.tocsc())
            U = lu.solve(b)
            if keep_factors:
                factors.append(lu)
        a_prev = a1
        if keep_history:
            hist.append(U.copy())
    return Solution(taus=taus, U=U, history=hist, info={"factors": factors} if keep_factors else {})


# ---------------------------------------------------------------------------
# American put: Fischer--Burmeister semismooth Newton
# ---------------------------------------------------------------------------

def fb(a, b):
    return np.sqrt(a * a + b * b) - a - b


def fb_semismooth_newton(A: sp.csr_matrix, b: np.ndarray, M: sp.csr_matrix, phi: np.ndarray,
                         bvals: tuple, U0: np.ndarray, tol: float = 1e-10, maxit: int = 50,
                         armijo: float = 1e-4):
    """Solve  A U - b >= 0,  M U - phi >= 0,  complementarity  on interior rows,
    U_1 = bvals[0], U_N = bvals[1]  on boundary rows, by semismooth Newton applied to
    Phi(U) = phi_FB(A U - b, M U - phi) with an Armijo line search on 0.5 |Phi|^2."""
    N = U0.size
    inner = np.arange(1, N - 1)

    def residual(U):
        F = np.empty(N)
        al = (A @ U - b)[inner]
        be = (M @ U - phi)[inner]
        F[inner] = fb(al, be)
        F[0] = U[0] - bvals[0]
        F[-1] = U[-1] - bvals[1]
        return F, al, be

    U = U0.copy()
    U[0], U[-1] = bvals
    F, al, be = residual(U)
    psi = 0.5 * F @ F
    it = 0
    n_linesearch = 0
    active_prev = be <= al
    active_changes = 0
    while np.max(np.abs(F)) > tol and it < maxit:
        rho = np.sqrt(al * al + be * be)
        da = np.empty_like(rho)
        db = np.empty_like(rho)
        nz = rho > 1e-14
        da[nz] = al[nz] / rho[nz] - 1.0
        db[nz] = be[nz] / rho[nz] - 1.0
        da[~nz] = 1.0 / np.sqrt(2.0) - 1.0      # an element of the B-subdifferential at (0,0)
        db[~nz] = 1.0 / np.sqrt(2.0) - 1.0
        dA = np.zeros(N)
        dM = np.zeros(N)
        dA[inner], dM[inner] = da, db
        Jac = sp.diags(dA) @ A + sp.diags(dM) @ M
        Jac = _set_rows(Jac, (0, N - 1))
        dU = spla.spsolve(Jac.tocsc(), -F)
        t = 1.0
        while True:
            Fn, aln, ben = residual(U + t * dU)
            psin = 0.5 * Fn @ Fn
            if psin <= (1.0 - 2.0 * armijo * t) * psi or t < 1e-8:
                break
            t *= 0.5
            n_linesearch += 1
        U = U + t * dU
        F, al, be, psi = Fn, aln, ben, psin
        it += 1
        active = be <= al
        active_changes += int(np.sum(active != active_prev))
        active_prev = active
    return U, {"iterations": it, "residual": float(np.max(np.abs(F))),
               "linesearch_reductions": n_linesearch, "active_changes": active_changes}


def solve_american_put(model: GMFBM, disc: Discretization, taus: np.ndarray, tol: float = 1e-10,
                       keep_history: bool = False) -> Solution:
    M = disc.M
    K_op = (disc.D[2] - disc.D[1]).tocsr()
    C_op = ((model.r - model.delta) * disc.D[1] - model.r * M).tocsr()
    S = disc.S(model)
    phi = payoff(model, S, "put")
    U = phi.copy()
    hist = [U.copy()] if keep_history else []
    stats = []
    a_prev = model.a(taus[0])
    for n in range(len(taus) - 1):
        t0, t1 = taus[n], taus[n + 1]
        s = 0.5 * (t1 - t0)
        a1 = model.a(t1)
        A = (M - s * (a1 * K_op + C_op)).tocsr()
        b = (M + s * (a_prev * K_op + C_op)) @ U
        U, st = fb_semismooth_newton(A, b, M, phi, (model.K, 0.0), U, tol=tol)
        stats.append(st)
        a_prev = a1
        if keep_history:
            hist.append(U.copy())
    # final-step complementarity diagnostics (interior rows)
    inner = slice(1, disc.N - 1)
    resid = (A @ U - b)[inner]
    gap = (M @ U - phi)[inner]
    info = {
        "avg_iterations": float(np.mean([s["iterations"] for s in stats])),
        "max_iterations": int(np.max([s["iterations"] for s in stats])),
        "max_final_residual": float(np.max([s["residual"] for s in stats])),
        "active_changes": int(np.sum([s["active_changes"] for s in stats])),
        "linesearch_reductions": int(np.sum([s["linesearch_reductions"] for s in stats])),
        "min_q_minus_phi": float(gap.min()),
        "min_AU_minus_b": float(resid.min()),
        "complementarity": float(np.max(np.abs(gap * resid))),
    }
    return Solution(taus=taus, U=U, history=hist, info=info)


def exercise_boundary(model: GMFBM, disc: Discretization, U: np.ndarray, x_fine: np.ndarray,
                      E0=None, tol_rel: float = 1e-7) -> float:
    """Largest S at which the reconstructed American put touches the payoff."""
    E0 = disc.eval_matrices(x_fine, kmax=0)[0] if E0 is None else E0
    q = E0 @ U
    S = model.K * np.exp(x_fine)
    gap = q - np.maximum(model.K - S, 0.0)
    ex = np.nonzero((gap <= tol_rel * model.K) & (S < model.K))[0]
    return float(S[ex.max()]) if ex.size else float("nan")


# ---------------------------------------------------------------------------
# Greeks and tangent sensitivities (Section 8)
# ---------------------------------------------------------------------------

def greeks(model: GMFBM, disc: Discretization, U: np.ndarray, tau: float, x_eval: np.ndarray):
    """Delta, Gamma, Speed (needs kmax>=3) and Theta at arbitrary log-prices."""
    E = disc.eval_matrices(x_eval, kmax=3)
    u, ux, uxx, uxxx = (E[k] @ U for k in range(4))
    S = model.K * np.exp(x_eval)
    a = model.a(tau)
    return {
        "price": u,
        "Delta": ux / S,
        "Gamma": (uxx - ux) / S**2,
        "Speed": (uxxx - 3 * uxx + 2 * ux) / S**3,
        "Theta": -(a * (uxx - ux) + (model.r - model.delta) * ux - model.r * u),
    }


def solve_tangent(model: GMFBM, disc: Discretization, sol: Solution, p: str, kind: str = "put",
                  bc: str = "exact", keep_history: bool = False):
    """Discrete tangent equation for d u / d p, p entering only through a(tau).
    Boundary rows carry d g / d p (zero for parameter-independent boundary data).
    The tangent system has the same matrix as the price update, so the stored LU
    factors of the price solve are reused (one extra forward/back substitution
    per step).  Returns Z at the final level (and all levels if keep_history)."""
    assert sol.history, "price solve must keep its history"
    factors = sol.info.get("factors")
    if factors and len(factors) != len(sol.taus) - 1:   # Rannacher start: no reuse
        factors = None
    M = disc.M
    K_op = (disc.D[2] - disc.D[1]).tocsr()
    C_op = ((model.r - model.delta) * disc.D[1] - model.r * M).tocsr()
    taus = sol.taus
    Z = np.zeros(disc.N)
    Zs = [Z.copy()]
    S_b = model.K * np.exp(np.array([disc.xmin, disc.xmax]))
    for n in range(len(taus) - 1):
        t0, t1 = taus[n], taus[n + 1]
        s = 0.5 * (t1 - t0)
        rhs = (M + s * (model.a(t0) * K_op + C_op)) @ Z
        rhs += s * (model.a_param(t1, p) * (K_op @ sol.history[n + 1])
                    + model.a_param(t0, p) * (K_op @ sol.history[n]))
        if bc == "exact":
            rhs[0], rhs[-1] = model.european_param_sensitivity(S_b, t1, p, kind)
        else:
            rhs[0] = rhs[-1] = 0.0
        if factors:
            Z = factors[n].solve(rhs)
        else:
            A = _set_rows(M - s * (model.a(t1) * K_op + C_op), (0, disc.N - 1))
            Z = spla.spsolve(A.tocsc(), rhs)
        if keep_history:
            Zs.append(Z.copy())
    return (Z, Zs) if keep_history else Z
