#!/usr/bin/env python3
"""
First numerical example for GMFBM option pricing using a finite-point / MLS-type
meshfree collocation method in log-price space.

The code is written so we can later use the same assembled operators for stability
analysis.

Model in forward time tau = T - t:
    u_tau = L_tau u

Original S-operator:
    L_tau u = sigma^2 S^2 sum_i H_i alpha_i^2 (T-tau)^(2H_i-1) u_SS
              + (r-delta) S u_S - r u

Log-price variable:
    x = log(S/K), S = K exp(x)
    S u_S = u_x,   S^2 u_SS = u_xx - u_x

Therefore:
    L_tau u = a(tau) u_xx + (r - delta - a(tau)) u_x - r u,
    a(tau) = sigma^2 sum_i H_i alpha_i^2 (T-tau)^(2H_i-1).

Boundary conditions for a put:
    European: u(x_min,tau) = K exp(-r tau), u(x_max,tau) = 0
    American: u(x_min,tau) = K,            u(x_max,tau) = 0

Run:
    python gmfbm_example1_mls.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class GMFBMParams:
    K: float = 100.0
    T: float = 1.0
    r: float = 0.10
    delta: float = 0.0
    sigma: float = 0.35

    # First example in the paper
    H: tuple[float, ...] = (0.5, 0.59, 0.68, 0.77)
    alpha: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)

    # Log-price truncated computational domain x = log(S/K)
    xmin: float = -6.0
    xmax: float = 6.0

    # Numerical parameters
    nx: int = 81
    nt: int = 80
    poly_degree: int = 4
    num_neighbors: int = 13
    support_scale: float = 2.5

    # American LCP solver
    psor_omega: float = 1.25
    psor_tol: float = 1.0e-10
    psor_maxit: int = 500


def gmfbm_diffusion_coefficient(tau: float, p: GMFBMParams) -> float:
    """
    a(tau) = sigma^2 sum_i H_i alpha_i^2 (T - tau)^(2H_i - 1).

    At tau = T, terms with H_i > 1/2 vanish; the H_i = 1/2 term remains finite.
    """
    t = max(p.T - tau, 0.0)
    total = 0.0
    for H_i, alpha_i in zip(p.H, p.alpha):
        exponent = 2.0 * H_i - 1.0
        if abs(exponent) < 1.0e-14:
            factor = 1.0
        elif t == 0.0:
            # H_i > 1/2 in our first example, so exponent > 0.
            factor = 0.0
        else:
            factor = t**exponent
        total += H_i * alpha_i**2 * factor
    return p.sigma**2 * total


def put_payoff_from_x(x: np.ndarray, p: GMFBMParams) -> np.ndarray:
    S = p.K * np.exp(x)
    return np.maximum(p.K - S, 0.0)


def boundary_left(tau: float, p: GMFBMParams, american: bool = False) -> float:
    if american:
        return p.K
    return p.K * np.exp(-p.r * tau)


def boundary_right(tau: float, p: GMFBMParams) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# MLS / finite-point differentiation matrices
# ---------------------------------------------------------------------------

def local_stencil_weights(
    x_nodes: np.ndarray,
    center_index: int,
    derivative_order: int,
    degree: int,
    num_neighbors: int,
    support_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute local weighted least-squares finite-point weights for the derivative
    at x_i.

    Polynomial basis is centered:
        q_k(z) = z^k, z = x - x_i, k = 0,...,degree.

    The returned weights c satisfy approximately
        D^m u(x_i) ~= sum_j c_j u(x_j).

    This is the derivative form of MLS/FPM collocation.
    """
    xi = x_nodes[center_index]
    dist = np.abs(x_nodes - xi)

    # Choose local neighbors; ensure enough points for polynomial reproduction.
    m = degree + 1
    k = max(num_neighbors, m + 2)
    k = min(k, len(x_nodes))
    nbrs = np.argsort(dist)[:k]
    nbrs = np.sort(nbrs)

    z = x_nodes[nbrs] - xi
    radius = max(np.max(np.abs(z)), 1.0e-14)
    h = support_scale * radius

    # Gaussian weights.
    w = np.exp(-(z / h) ** 2)

    # Vandermonde matrix V[j,k] = z_j^k.
    V = np.vstack([z**q for q in range(m)]).T
    W = np.diag(w)

    # Moment matrix.
    M = V.T @ W @ V

    # Derivative of basis at z=0:
    # d^m/dx^m z^k at 0 equals factorial(m) if k=m, else 0.
    rhs = np.zeros(m)
    rhs[derivative_order] = math.factorial(derivative_order)

    # c^T = rhs^T M^{-1} V^T W
    coeff = np.linalg.solve(M, rhs)
    c = W @ V @ coeff
    return nbrs, c


def build_derivative_matrix(
    x_nodes: np.ndarray,
    derivative_order: int,
    degree: int,
    num_neighbors: int,
    support_scale: float,
) -> np.ndarray:
    n = len(x_nodes)
    D = np.zeros((n, n))
    for i in range(n):
        nbrs, c = local_stencil_weights(
            x_nodes=x_nodes,
            center_index=i,
            derivative_order=derivative_order,
            degree=degree,
            num_neighbors=num_neighbors,
            support_scale=support_scale,
        )
        D[i, nbrs] = c
    return D


def assemble_L(tau: float, D1: np.ndarray, D2: np.ndarray, p: GMFBMParams) -> np.ndarray:
    a = gmfbm_diffusion_coefficient(tau, p)
    n = D1.shape[0]
    I = np.eye(n)
    return a * D2 + (p.r - p.delta - a) * D1 - p.r * I


def impose_dirichlet_on_system(
    A: np.ndarray,
    b: np.ndarray,
    left_value: float,
    right_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strongly impose Dirichlet boundary conditions at first and last nodes.
    """
    A = A.copy()
    b = b.copy()

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = left_value

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = right_value

    return A, b


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------

def solve_european_put_mls(p: GMFBMParams):
    x = np.linspace(p.xmin, p.xmax, p.nx)
    tau_grid = np.linspace(0.0, p.T, p.nt + 1)
    dt = tau_grid[1] - tau_grid[0]

    D1 = build_derivative_matrix(
        x, 1, p.poly_degree, p.num_neighbors, p.support_scale
    )
    D2 = build_derivative_matrix(
        x, 2, p.poly_degree, p.num_neighbors, p.support_scale
    )

    u = put_payoff_from_x(x, p)
    U = np.zeros((p.nt + 1, p.nx))
    U[0, :] = u

    I = np.eye(p.nx)

    for n in range(p.nt):
        tau_n = tau_grid[n]
        tau_np1 = tau_grid[n + 1]

        L_n = assemble_L(tau_n, D1, D2, p)
        L_np1 = assemble_L(tau_np1, D1, D2, p)

        A = I - 0.5 * dt * L_np1
        b = (I + 0.5 * dt * L_n) @ u

        A, b = impose_dirichlet_on_system(
            A,
            b,
            boundary_left(tau_np1, p, american=False),
            boundary_right(tau_np1, p),
        )

        u = np.linalg.solve(A, b)
        U[n + 1, :] = u

    return x, tau_grid, U, D1, D2


def psor_lcp(
    A: np.ndarray,
    b: np.ndarray,
    payoff: np.ndarray,
    u0: np.ndarray,
    omega: float,
    tol: float,
    maxit: int,
    fixed_boundary: tuple[float, float],
) -> tuple[np.ndarray, int, float]:
    """
    Projected SOR for the LCP:
        A u >= b, u >= payoff, (A u - b)_i (u_i - payoff_i) = 0.

    Boundary entries are held fixed.
    """
    u = u0.copy()
    u[0] = fixed_boundary[0]
    u[-1] = fixed_boundary[1]

    n = len(u)

    for it in range(1, maxit + 1):
        old = u.copy()

        for i in range(1, n - 1):
            sigma = A[i, :i] @ u[:i] + A[i, i + 1:] @ old[i + 1:]
            y = (b[i] - sigma) / A[i, i]
            u[i] = max(payoff[i], old[i] + omega * (y - old[i]))

        u[0] = fixed_boundary[0]
        u[-1] = fixed_boundary[1]

        err = np.linalg.norm(u - old, ord=np.inf)
        if err < tol:
            return u, it, err

    return u, maxit, err


def solve_american_put_mls(p: GMFBMParams, use_psor: bool = False):
    """
    American put solver.

    By default, this uses a fast projected Crank--Nicolson step:
        solve the linear CN system, then project onto the obstacle.
    This is useful for debugging and plotting.

    Set use_psor=True to solve the LCP at each time step with projected SOR:
        A u >= b, u >= payoff, (A u-b)(u-payoff)=0.
    """
    x = np.linspace(p.xmin, p.xmax, p.nx)
    tau_grid = np.linspace(0.0, p.T, p.nt + 1)
    dt = tau_grid[1] - tau_grid[0]

    D1 = build_derivative_matrix(
        x, 1, p.poly_degree, p.num_neighbors, p.support_scale
    )
    D2 = build_derivative_matrix(
        x, 2, p.poly_degree, p.num_neighbors, p.support_scale
    )

    payoff = put_payoff_from_x(x, p)
    u = payoff.copy()
    U = np.zeros((p.nt + 1, p.nx))
    U[0, :] = u

    I = np.eye(p.nx)
    psor_iterations = []

    for n in range(p.nt):
        tau_n = tau_grid[n]
        tau_np1 = tau_grid[n + 1]

        L_n = assemble_L(tau_n, D1, D2, p)
        L_np1 = assemble_L(tau_np1, D1, D2, p)

        A = I - 0.5 * dt * L_np1
        b = (I + 0.5 * dt * L_n) @ u

        left = boundary_left(tau_np1, p, american=True)
        right = boundary_right(tau_np1, p)
        A, b = impose_dirichlet_on_system(A, b, left, right)

        if use_psor:
            u, it, err = psor_lcp(
                A=A,
                b=b,
                payoff=payoff,
                u0=u,
                omega=p.psor_omega,
                tol=p.psor_tol,
                maxit=p.psor_maxit,
                fixed_boundary=(left, right),
            )
            psor_iterations.append(it)
        else:
            # Fast projected CN approximation.
            u = np.linalg.solve(A, b)
            u = np.maximum(u, payoff)
            u[0] = left
            u[-1] = right
            psor_iterations.append(0)

        U[n + 1, :] = u

    return x, tau_grid, U, D1, D2, np.array(psor_iterations)


# ---------------------------------------------------------------------------
# Stability analysis utilities
# ---------------------------------------------------------------------------

def crank_nicolson_amplification_matrix(
    tau_n: float,
    tau_np1: float,
    dt: float,
    D1: np.ndarray,
    D2: np.ndarray,
    p: GMFBMParams,
) -> np.ndarray:
    """
    For the homogeneous interior problem, CN gives
        U^{n+1} = G_n U^n,
        G_n = (I - dt/2 L_{n+1})^{-1} (I + dt/2 L_n).

    We remove boundary rows/columns and analyze only interior unknowns.
    """
    L_n = assemble_L(tau_n, D1, D2, p)[1:-1, 1:-1]
    L_np1 = assemble_L(tau_np1, D1, D2, p)[1:-1, 1:-1]
    n = L_n.shape[0]
    I = np.eye(n)

    A = I - 0.5 * dt * L_np1
    B = I + 0.5 * dt * L_n
    return np.linalg.solve(A, B)


def stability_scan(D1: np.ndarray, D2: np.ndarray, p: GMFBMParams):
    tau_grid = np.linspace(0.0, p.T, p.nt + 1)
    dt = tau_grid[1] - tau_grid[0]

    spectral_radius = np.zeros(p.nt)
    two_norm = np.zeros(p.nt)
    cond_A = np.zeros(p.nt)

    for n in range(p.nt):
        G = crank_nicolson_amplification_matrix(
            tau_grid[n], tau_grid[n + 1], dt, D1, D2, p
        )
        eigs = np.linalg.eigvals(G)
        spectral_radius[n] = np.max(np.abs(eigs))
        two_norm[n] = np.linalg.norm(G, 2)

        L_np1 = assemble_L(tau_grid[n + 1], D1, D2, p)[1:-1, 1:-1]
        A = np.eye(L_np1.shape[0]) - 0.5 * dt * L_np1
        cond_A[n] = np.linalg.cond(A)

    return tau_grid[1:], spectral_radius, two_norm, cond_A


# ---------------------------------------------------------------------------
# Plotting and main routine
# ---------------------------------------------------------------------------

def make_plots(
    x: np.ndarray,
    tau_grid: np.ndarray,
    U_eur: np.ndarray,
    U_amer: np.ndarray,
    stability_data,
    p: GMFBMParams,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    S = p.K * np.exp(x)

    # Plot at maturity tau = T, against S, cropped to useful financial range.
    mask = (S >= 1.0) & (S <= 250.0)

    plt.figure(figsize=(7, 4.5))
    plt.plot(S[mask], put_payoff_from_x(x, p)[mask], "--", label="Payoff")
    plt.plot(S[mask], U_eur[-1, mask], label="European put")
    plt.plot(S[mask], U_amer[-1, mask], label="American put")
    plt.xlabel("S")
    plt.ylabel("Option value")
    plt.title("GMFBM first example: put price at tau = T")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example1_price_curves.png", dpi=200)
    plt.close()

    # European surface.
    Tgrid, Xgrid = np.meshgrid(tau_grid, x, indexing="ij")
    Sgrid = p.K * np.exp(Xgrid)

    # Use a cropped surface view.
    mask_idx = np.where(mask)[0]
    i0, i1 = mask_idx[0], mask_idx[-1] + 1

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        Sgrid[:, i0:i1],
        Tgrid[:, i0:i1],
        U_eur[:, i0:i1],
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    ax.set_xlabel("S")
    ax.set_ylabel("tau")
    ax.set_zlabel("u")
    ax.set_title("European put surface")
    plt.tight_layout()
    plt.savefig(outdir / "example1_european_surface.png", dpi=200)
    plt.close()

    # Stability quantities.
    tau_s, rho, norm2, cond_A = stability_data

    plt.figure(figsize=(7, 4.5))
    plt.plot(tau_s, rho, label="spectral radius rho(G_n)")
    plt.plot(tau_s, norm2, label="2-norm ||G_n||_2")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("tau")
    plt.ylabel("amplification measure")
    plt.title("CN amplification matrix diagnostics")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "example1_stability_amplification.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(tau_s, cond_A)
    plt.xlabel("tau")
    plt.ylabel("cond(I - dt/2 L)")
    plt.title("Condition number of implicit CN matrix")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "example1_stability_condition.png", dpi=200)
    plt.close()


def main():
    p = GMFBMParams()

    print("Running GMFBM first example with parameters:")
    print(p)
    print(f"H = {p.H}")
    print(f"alpha = {p.alpha}")

    print("\nSolving European put...")
    x, tau_grid, U_eur, D1, D2 = solve_european_put_mls(p)

    print("Solving American put using fast projected CN approximation...")
    _, _, U_amer, _, _, psor_iters = solve_american_put_mls(p, use_psor=False)

    print("Scanning linear CN stability/amplification matrices...")
    stability_data = stability_scan(D1, D2, p)
    tau_s, rho, norm2, cond_A = stability_data

    outdir = Path("gmfbm_example1_output")
    make_plots(x, tau_grid, U_eur, U_amer, stability_data, p, outdir)

    S = p.K * np.exp(x)
    idx_atm = np.argmin(np.abs(S - p.K))

    print("\nSummary")
    print("-------")
    print(f"S closest to K: {S[idx_atm]:.6g}")
    print(f"European put u(S=K,t=0): {U_eur[-1, idx_atm]:.8f}")
    print(f"American put u(S=K,t=0): {U_amer[-1, idx_atm]:.8f}")
    print("American solver: projected CN approximation. Set use_psor=True in solve_american_put_mls for LCP/PSOR.")
    print(f"Max spectral radius rho(G_n): {np.max(rho):.8f}")
    print(f"Max ||G_n||_2: {np.max(norm2):.8f}")
    print(f"Max cond(I - dt/2 L): {np.max(cond_A):.8e}")
    print(f"\nPlots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
