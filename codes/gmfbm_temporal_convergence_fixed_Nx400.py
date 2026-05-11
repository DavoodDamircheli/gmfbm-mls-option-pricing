#!/usr/bin/env python3
"""
Temporal convergence experiment for Example 1 of the GMFBM MLS option-pricing paper.

This script fixes the spatial grid and varies only the number of time steps.
The temporal error is measured against a same-spatial-grid reference solution
computed with Nt_ref = 6400. This isolates the time discretization error from
the spatial discretization error.

The reported error is
    ||u_{dt} - u_ref||_inf / K
over the financial window 0.2 K <= S <= 2.5 K.

Outputs:
    temporal_convergence_results/ex1_temporal_convergence_fixed_Nx400.csv
    temporal_convergence_results/ex1_temporal_convergence_fixed_Nx400.png
    temporal_convergence_results/ex1_temporal_convergence_fixed_Nx400.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numba import njit


@dataclass(frozen=True)
class Params:
    K: float = 25.0
    T: float = 1.0
    r: float = 0.08
    delta: float = 0.0
    sigma: float = 0.35
    H: tuple[float, ...] = (0.5, 0.59, 0.68, 0.77)
    alpha: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    xmin: float = -6.0
    xmax: float = 6.0

    # MLS/FPM settings used in the paper code.
    degree: int = 4
    num_neighbors: int = 13
    support_scale: float = 2.5

    # American put LCP/PSOR settings.
    omega: float = 1.25
    tol: float = 1.0e-10
    maxit: int = 10000


def gamma_tau(tau: float, p: Params) -> float:
    """GMFBM diffusion coefficient in log-price time-forward form."""
    t = max(p.T - tau, 0.0)
    total = 0.0
    for H_i, alpha_i in zip(p.H, p.alpha):
        exponent = 2.0 * H_i - 1.0
        if abs(exponent) < 1.0e-14:
            factor = 1.0
        elif t == 0.0:
            factor = 0.0
        else:
            factor = t**exponent
        total += H_i * alpha_i**2 * factor
    return p.sigma**2 * total


def payoff(x: np.ndarray, kind: str, p: Params) -> np.ndarray:
    S = p.K * np.exp(x)
    if kind == "put":
        return np.maximum(p.K - S, 0.0)
    if kind == "call":
        return np.maximum(S - p.K, 0.0)
    raise ValueError(f"Unknown option kind: {kind}")


def boundary_values(tau: float, kind: str, american: bool, p: Params) -> tuple[float, float]:
    Smax = p.K * math.exp(p.xmax)
    if kind == "put":
        left = p.K if american else p.K * math.exp(-p.r * tau)
        right = 0.0
    elif kind == "call":
        left = 0.0
        right = Smax * math.exp(-p.delta * tau) - p.K * math.exp(-p.r * tau)
    else:
        raise ValueError(f"Unknown option kind: {kind}")
    return left, right


def local_weights(x: np.ndarray, i: int, derivative_order: int, p: Params) -> tuple[np.ndarray, np.ndarray]:
    xi = x[i]
    dist = np.abs(x - xi)
    m = p.degree + 1
    k = min(max(p.num_neighbors, m + 2), len(x))
    nbrs = np.sort(np.argsort(dist)[:k])

    z = x[nbrs] - xi
    radius = max(float(np.max(np.abs(z))), 1.0e-14)
    rho = p.support_scale * radius

    w = np.exp(-(z / rho) ** 2)
    V = np.vstack([z**q for q in range(m)]).T
    WV = w[:, None] * V
    M = V.T @ WV

    rhs = np.zeros(m)
    rhs[derivative_order] = math.factorial(derivative_order)
    coeff = np.linalg.solve(M, rhs)
    return nbrs, WV @ coeff


def build_rows(Nx: int, p: Params):
    x = np.linspace(p.xmin, p.xmax, Nx + 1)
    n = Nx + 1
    idx = np.full((n, p.num_neighbors), -1, dtype=np.int64)
    d1 = np.zeros((n, p.num_neighbors), dtype=np.float64)
    d2 = np.zeros((n, p.num_neighbors), dtype=np.float64)

    for i in range(n):
        nbr1, c1 = local_weights(x, i, 1, p)
        nbr2, c2 = local_weights(x, i, 2, p)
        if not np.array_equal(nbr1, nbr2):
            raise RuntimeError("First- and second-derivative stencils do not match.")
        idx[i, : len(nbr1)] = nbr1
        d1[i, : len(c1)] = c1
        d2[i, : len(c2)] = c2

    return x, idx, d1, d2


def build_stencil_flat(Nx: int, p: Params):
    x, idx, d1, d2 = build_rows(Nx, p)
    n, kmax = idx.shape
    rows, cols, d1s, d2s, diag = [], [], [], [], []

    for i in range(n):
        for kk in range(kmax):
            j = int(idx[i, kk])
            if j < 0:
                continue
            rows.append(i)
            cols.append(j)
            d1s.append(d1[i, kk])
            d2s.append(d2[i, kk])
            diag.append(1.0 if j == i else 0.0)

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    d1s = np.asarray(d1s, dtype=float)
    d2s = np.asarray(d2s, dtype=float)
    diag = np.asarray(diag, dtype=float)

    interior = (rows >= 1) & (rows <= n - 2) & (cols >= 1) & (cols <= n - 2)
    left = (rows >= 1) & (rows <= n - 2) & (cols == 0)
    right = (rows >= 1) & (rows <= n - 2) & (cols == n - 1)
    return x, idx, d1, d2, rows, cols, d1s, d2s, diag, interior, left, right


def solve_european_from_stencil(stencil, Nt: int, kind: str, p: Params) -> tuple[np.ndarray, np.ndarray]:
    x, _idx, _d1, _d2, rows, cols, d1s, d2s, diag, interior, left_mask, right_mask = stencil
    n = len(x)
    u = payoff(x, kind, p)
    dt = p.T / Nt

    for nstep in range(Nt):
        tau_n = nstep * dt
        tau_p = (nstep + 1) * dt

        a_n = gamma_tau(tau_n, p)
        data_n = a_n * d2s + (p.r - p.delta - a_n) * d1s - p.r * diag
        L_n = csr_matrix((data_n, (rows, cols)), shape=(n, n))
        rhs = u + 0.5 * dt * L_n.dot(u)

        a_p = gamma_tau(tau_p, p)
        data_p = a_p * d2s + (p.r - p.delta - a_p) * d1s - p.r * diag

        Adata = -0.5 * dt * data_p[interior]
        Adata = Adata.copy()
        Adata[rows[interior] == cols[interior]] += 1.0
        A = csr_matrix((Adata, (rows[interior] - 1, cols[interior] - 1)), shape=(n - 2, n - 2))

        left, right = boundary_values(tau_p, kind, american=False, p=p)
        rhs_int = rhs[1:-1].copy()

        if np.any(left_mask):
            for rr, val in zip(rows[left_mask], -0.5 * dt * data_p[left_mask]):
                rhs_int[rr - 1] -= val * left
        if np.any(right_mask):
            for rr, val in zip(rows[right_mask], -0.5 * dt * data_p[right_mask]):
                rhs_int[rr - 1] -= val * right

        unew = np.empty_like(u)
        unew[0] = left
        unew[-1] = right
        unew[1:-1] = spsolve(A, rhs_int)
        u = unew

    return x, u


@njit
def american_psor_numba(idx, d1, d2, x, Nt, K, T, r, delta, sigma, H, alpha, omega, tol, maxit):
    n, kmax = idx.shape
    u = np.empty(n)
    obstacle = np.empty(n)

    for i in range(n):
        S = K * math.exp(x[i])
        val = K - S
        if val < 0.0:
            val = 0.0
        obstacle[i] = val
        u[i] = val

    dt = T / Nt
    avg_iter = 0.0

    for nstep in range(Nt):
        tau_n = nstep * dt
        tau_p = (nstep + 1) * dt

        gamma = np.empty(2)
        for qtau, tau in enumerate((tau_n, tau_p)):
            tt = T - tau
            if tt < 0.0:
                tt = 0.0
            s = 0.0
            for q in range(len(H)):
                exponent = 2.0 * H[q] - 1.0
                if abs(exponent) < 1.0e-14:
                    factor = 1.0
                elif tt == 0.0:
                    factor = 0.0
                else:
                    factor = tt**exponent
                s += H[q] * alpha[q] * alpha[q] * factor
            gamma[qtau] = sigma * sigma * s

        # Explicit CN side: rhs = u + dt/2 L_n u.
        rhs = np.empty(n)
        bcoef_n = r - delta - gamma[0]
        for i in range(n):
            acc = 0.0
            for kk in range(kmax):
                j = idx[i, kk]
                if j < 0:
                    continue
                Lij = gamma[0] * d2[i, kk] + bcoef_n * d1[i, kk]
                if j == i:
                    Lij -= r
                acc += Lij * u[j]
            rhs[i] = u[i] + 0.5 * dt * acc

        left = K
        right = 0.0
        rhs[0] = left
        rhs[n - 1] = right
        u[0] = left
        u[n - 1] = right

        bcoef_p = r - delta - gamma[1]

        for it in range(maxit):
            old = u.copy()
            maxdiff = 0.0

            for i in range(1, n - 1):
                diag = 1.0
                accum = 0.0

                for kk in range(kmax):
                    j = idx[i, kk]
                    if j < 0:
                        continue

                    Lij = gamma[1] * d2[i, kk] + bcoef_p * d1[i, kk]
                    if j == i:
                        Lij -= r
                    Aij = -0.5 * dt * Lij

                    if j == i:
                        diag += Aij
                    else:
                        if j < i:
                            accum += Aij * u[j]
                        else:
                            accum += Aij * old[j]

                y = (rhs[i] - accum) / diag
                newval = old[i] + omega * (y - old[i])
                if newval < obstacle[i]:
                    newval = obstacle[i]
                u[i] = newval

                diff = abs(newval - old[i])
                if diff > maxdiff:
                    maxdiff = diff

            u[0] = left
            u[n - 1] = right

            if maxdiff < tol:
                avg_iter += it + 1
                break

            if it == maxit - 1:
                avg_iter += maxit

    return u, avg_iter / Nt


def solve_american_put(x: np.ndarray, idx: np.ndarray, d1: np.ndarray, d2: np.ndarray, Nt: int, p: Params):
    H = np.asarray(p.H, dtype=np.float64)
    alpha = np.asarray(p.alpha, dtype=np.float64)
    return american_psor_numba(idx, d1, d2, x, Nt, p.K, p.T, p.r, p.delta, p.sigma, H, alpha, p.omega, p.tol, p.maxit)


def normalized_error(x: np.ndarray, u: np.ndarray, xref: np.ndarray, uref: np.ndarray, p: Params) -> float:
    ref = np.interp(x, xref, uref)
    S = p.K * np.exp(x)
    mask = (S >= 0.2 * p.K) & (S <= 2.5 * p.K)
    return float(np.max(np.abs(u[mask] - ref[mask])) / p.K)


def regression(xvals, errors):
    lx = np.log(np.asarray(xvals, dtype=float))
    ly = np.log(np.asarray(errors, dtype=float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    yhat = slope * lx + intercept
    r2 = 1.0 - np.sum((ly - yhat) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(slope), float(intercept), float(r2)


def main():
    p = Params()
    Nx = 400
    Nt_ref = 6400
    Nt_values = [400, 600, 800, 1200, 1600]
    outdir = Path("/mnt/data/temporal_convergence_results")
    outdir.mkdir(parents=True, exist_ok=True)

    stencil = build_stencil_flat(Nx, p)
    x, idx, d1, d2 = stencil[0], stencil[1], stencil[2], stencil[3]

    rows = []

    for opt, kind in [("european_put", "put"), ("european_call", "call")]:
        print(f"Reference solve: {opt}, Nt={Nt_ref}")
        xr, ur = solve_european_from_stencil(stencil, Nt_ref, kind, p)
        prev = None
        for Nt in Nt_values:
            t0 = time.time()
            xh, uh = solve_european_from_stencil(stencil, Nt, kind, p)
            cpu = time.time() - t0
            err = normalized_error(xh, uh, xr, ur, p)
            dt = p.T / Nt
            order = "" if prev is None else math.log(err / prev[1]) / math.log(dt / prev[0])
            rows.append([opt, Nx, Nt, (p.xmax - p.xmin) / Nx, dt, err, order, cpu])
            prev = (dt, err)

    print(f"Reference solve: american_put, Nt={Nt_ref}")
    ur, _avg = solve_american_put(x, idx, d1, d2, Nt_ref, p)
    prev = None
    for Nt in Nt_values:
        t0 = time.time()
        uh, avg_iter = solve_american_put(x, idx, d1, d2, Nt, p)
        cpu = time.time() - t0
        err = normalized_error(x, uh, x, ur, p)
        dt = p.T / Nt
        order = "" if prev is None else math.log(err / prev[1]) / math.log(dt / prev[0])
        rows.append(["american_put", Nx, Nt, (p.xmax - p.xmin) / Nx, dt, err, order, cpu])
        prev = (dt, err)

    csv_path = outdir / "ex1_temporal_convergence_fixed_Nx400.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["option", "Nx", "Nt", "h", "dt", "error", "pairwise_order", "cpu_time"])
        writer.writerows(rows)

    # Plot.
    labels = {
        "european_put": "European put",
        "european_call": "European call",
        "american_put": "American put",
    }
    markers = {"european_put": "o", "european_call": "s", "american_put": "^"}

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for opt in ["european_put", "european_call", "american_put"]:
        opt_rows = [r for r in rows if r[0] == opt]
        dt_vals = np.array([r[4] for r in opt_rows], dtype=float)
        err_vals = np.array([r[5] for r in opt_rows], dtype=float)
        m, b, r2 = regression(dt_vals, err_vals)
        ax.loglog(dt_vals, err_vals, marker=markers[opt], label=f"{labels[opt]} (slope {m:.2f})")
        xs = np.array([dt_vals.min(), dt_vals.max()])
        ax.loglog(xs, np.exp(b) * xs**m, linestyle="--")
        print(f"{opt}: slope={m:.6f}, R2={r2:.6f}")

    ax.set_xlabel(r"$\Delta\tau$")
    ax.set_ylabel("normalized price error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title(r"Example 1 temporal convergence, fixed $N_x=400$")
    fig.tight_layout()
    fig.savefig(outdir / "ex1_temporal_convergence_fixed_Nx400.png", dpi=220)
    fig.savefig(outdir / "ex1_temporal_convergence_fixed_Nx400.pdf")

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
