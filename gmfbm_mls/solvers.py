"""Non-cardinal MLS collocation solvers for GMFBM option pricing.

Coefficient equation  M_h U' = L_h(tau) U,  L_h = a(tau)(D2 - D1) + (r - delta) D1 - r M_h,
averaged Crank-Nicolson  (M_h - s_n L_h(tau_{n+1})) U^{n+1} = (M_h + s_n L_h(tau_n)) U^n,
Dirichlet data imposed on the first and last coefficients.  American puts: Fischer-Burmeister
semismooth Newton on the reconstructed prices q = M_h U (Algorithm 1 of the paper).
"""
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mls import MLS1D


class MLSPricer:
    def __init__(self, model, Nx=None, xmin=-6.0, xmax=6.0, degree=3, radius_factor=3.5,
                 nodes=None, kmax=2):
        self.model = model
        if nodes is None:
            nodes = np.linspace(xmin, xmax, Nx + 1)
        self.x = np.asarray(nodes, dtype=float)
        self.N = self.x.size
        self.h = np.max(np.diff(self.x))
        t0 = time.perf_counter()
        self.space = MLS1D(self.x, degree=degree, radius_factor=radius_factor)
        mats = self.space.matrices(kmax=kmax)
        self.assembly_time = time.perf_counter() - t0
        self.M, self.D1, self.D2 = mats[0], mats[1], mats[2]
        self.Dk = mats
        self.S = model.S_ref * np.exp(self.x)
        n = self.N
        mask = np.ones(n)
        mask[0] = mask[-1] = 0.0
        self.Pint = sp.diags(mask)
        e = np.zeros(n)
        e[0] = e[-1] = 1.0
        self.Ebd = sp.diags(e)
        self.Kd = (self.D2 - self.D1).tocsr()

    # ------------------------------------------------------------------ operators
    def L(self, tau):
        m = self.model
        return (m.a(tau) * self.Kd + (m.r - m.delta) * self.D1 - m.r * self.M).tocsr()

    def step_mats(self, tau0, tau1):
        s = 0.5 * (tau1 - tau0)
        A = (self.M - s * self.L(tau1)).tocsr()
        B = (self.M + s * self.L(tau0)).tocsr()
        return A, B

    def bc_matrix(self, A):
        return (self.Pint @ A + self.Ebd).tocsc()

    # ------------------------------------------------------------------ boundary data
    def boundary(self, tau, kind, bc):
        m = self.model
        Smin, Smax = self.S[0], self.S[-1]
        if bc == "exact":
            return m.european(Smin, tau, kind), m.european(Smax, tau, kind)
        if kind == "put":
            return m.K * np.exp(-m.r * tau) - Smin * np.exp(-m.delta * tau), 0.0
        if kind == "call":
            return 0.0, Smax * np.exp(-m.delta * tau) - m.K * np.exp(-m.r * tau)
        if kind == "american":
            return m.K - Smin, 0.0
        raise ValueError(kind)

    def payoff(self, kind):
        K = self.model.K
        return np.maximum(K - self.S, 0.0) if kind in ("put", "american") else np.maximum(self.S - K, 0.0)

    # ------------------------------------------------------------------ European
    def _ie_step(self, U, t0, t1, kind, bc):
        A = (self.M - (t1 - t0) * self.L(t1)).tocsr()
        rhs = self.Pint @ (self.M @ U)
        g0, g1 = self.boundary(t1, kind, bc)
        rhs[0], rhs[-1] = g0, g1
        return spla.splu(self.bc_matrix(A)).solve(rhs)

    def european(self, tau_grid, kind="put", bc="exact", store=False, keep_lu=False, rannacher=0):
        U = self.payoff(kind)
        Us = [U.copy()] if store else None
        lus = [] if keep_lu else None
        for n in range(len(tau_grid) - 1):
            t0, t1 = tau_grid[n], tau_grid[n + 1]
            if n < rannacher:  # Rannacher start: two implicit-Euler half steps
                tm = 0.5 * (t0 + t1)
                U = self._ie_step(self._ie_step(U, t0, tm, kind, bc), tm, t1, kind, bc)
                if store:
                    Us.append(U.copy())
                continue
            A, B = self.step_mats(t0, t1)
            rhs = self.Pint @ (B @ U)
            g0, g1 = self.boundary(t1, kind, bc)
            rhs[0], rhs[-1] = g0, g1
            lu = spla.splu(self.bc_matrix(A))
            U = lu.solve(rhs)
            if store:
                Us.append(U.copy())
            if keep_lu:
                lus.append(lu)
        out = {"U": U}
        if store:
            out["Us"] = Us
        if keep_lu:
            out["lus"] = lus
        return out

    # ------------------------------------------------------------------ American put
    def american(self, tau_grid, tol=1e-10, sigma_A=1e-4, maxit=50, store=False):
        m = self.model
        eta = self.payoff("american")
        U = eta.copy()
        stats = {"its": [], "reductions": 0, "res": []}
        Us = [U.copy()] if store else None
        n = self.N
        interior = np.arange(1, n - 1)
        for k in range(len(tau_grid) - 1):
            t0, t1 = tau_grid[k], tau_grid[k + 1]
            A, B = self.step_mats(t0, t1)
            b = B @ U
            g0, g1 = self.boundary(t1, "american", "fin")
            V = U.copy()
            V[0], V[-1] = g0, g1

            def Phi(V):
                aa = A @ V - b
                bb = self.M @ V - eta
                rho = np.sqrt(aa ** 2 + bb ** 2)
                F = rho - aa - bb
                F[0] = V[0] - g0
                F[-1] = V[-1] - g1
                return F, aa, bb, rho

            F, aa, bb, rho = Phi(V)
            it = 0
            while np.max(np.abs(F)) > tol and it < maxit:
                safe = rho > 0
                ca = np.where(safe, aa / np.where(safe, rho, 1), 1 / np.sqrt(2)) - 1
                cb = np.where(safe, bb / np.where(safe, rho, 1), 1 / np.sqrt(2)) - 1
                ca[[0, -1]] = 0.0
                cb[[0, -1]] = 0.0
                Hm = (sp.diags(ca) @ A + sp.diags(cb) @ self.M + self.Ebd).tocsc()
                dV = spla.spsolve(Hm, -F)
                psi0 = 0.5 * F @ F
                t = 1.0
                while True:
                    Fn, aan, bbn, rhon = Phi(V + t * dV)
                    if 0.5 * Fn @ Fn <= (1 - 2 * sigma_A * t) * psi0 or t < 1e-12:
                        break
                    t *= 0.5
                    stats["reductions"] += 1
                V = V + t * dV
                F, aa, bb, rho = Fn, aan, bbn, rhon
                it += 1
            stats["its"].append(it)
            stats["res"].append(np.max(np.abs(F)))
            U = V
            if store:
                Us.append(U.copy())
            last = (A, b)
        A, b = last
        aa = (A @ U - b)[interior]
        bb = (self.M @ U - eta)[interior]
        stats["compl"] = np.max(np.abs(aa * bb))
        stats["min_gap"] = bb.min()
        stats["min_dual"] = aa.min()
        out = {"U": U, "stats": stats}
        if store:
            out["Us"] = Us
        return out

    # ------------------------------------------------------------------ tangent sensitivities
    def tangent(self, tau_grid, price_Us, lus, da_dp, kind="put"):
        """Discrete tangent equation for a parameter entering only through a(tau)."""
        Z = np.zeros(self.N)
        Zs = [Z.copy()]
        for n in range(len(tau_grid) - 1):
            t0, t1 = tau_grid[n], tau_grid[n + 1]
            s = 0.5 * (t1 - t0)
            A, B = self.step_mats(t0, t1)
            rhs = B @ Z + s * (da_dp(t1) * (self.Kd @ price_Us[n + 1]) + da_dp(t0) * (self.Kd @ price_Us[n]))
            rhs = self.Pint @ rhs
            rhs[0] = rhs[-1] = 0.0
            Z = lus[n].solve(rhs)
            Zs.append(Z.copy())
        return Zs

    # ------------------------------------------------------------------ evaluation
    def eval_matrices(self, S_eval, kmax=0):
        xe = np.log(np.asarray(S_eval) / self.model.S_ref)
        return self.space.matrices(xe, kmax=kmax)

    def greeks(self, U, S_eval, tau):
        """Delta, Gamma, Speed and (calendar) Theta at S_eval from coefficients U."""
        E = self.eval_matrices(S_eval, kmax=3)
        S = np.asarray(S_eval)
        u, ux, uxx, uxxx = (E[k] @ U for k in range(4))
        m = self.model
        delta = ux / S
        gamma = (uxx - ux) / S ** 2
        speed = (uxxx - 3 * uxx + 2 * ux) / S ** 3
        theta = -(m.a(tau) * (uxx - ux) + (m.r - m.delta) * ux - m.r * u)
        return delta, gamma, speed, theta

    # ------------------------------------------------------------------ stability diagnostics
    def reduced(self):
        I = np.arange(1, self.N - 1)
        Md = self.M.toarray()[np.ix_(I, I)]
        D1 = self.D1.toarray()[np.ix_(I, I)]
        Kd = self.Kd.toarray()[np.ix_(I, I)]
        return Md, D1, Kd
