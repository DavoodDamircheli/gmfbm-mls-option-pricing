#!/usr/bin/env python3
"""
Reliable stability diagnostics for the GMFBM log-price operator.

This script fixes the two issues that make the previous diagnostic unreliable
for the requested table:

1. The table uses N_x as the number of spatial subintervals, so the grid has
   N_x + 1 nodes and h = (xmax - xmin)/N_x.

2. The requested diagnostic is not the Crank--Nicolson amplification matrix.
   It is the largest eigenvalue of the weighted Hermitian part of the
   semi-discrete spatial generator A_n^h.

Default operator:
    An energy-stable finite-point/SBP operator on the interior nodes:
        D1 = centered first derivative,
        D2 = centered second derivative,
        W_h = h I.
    Then D1 is skew-symmetric, D2 is symmetric negative definite, and
        S_n^h = Gamma(tau_n) D2 - r I.
    Hence max_n lambda_max(S_n^h) can be computed exactly.

Important:
    If you instead plug in the row-wise MLS derivative matrices from the
    exploratory solver, you are checking a different operator. Those row-wise
    MLS matrices do not automatically satisfy the discrete dissipativity
    identity. Do not claim stability for that version unless this diagnostic
    is rerun with its exact mass matrix/operator and the eigenvalues are
    non-positive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import csv


@dataclass(frozen=True)
class GMFBMParams:
    K: float = 100.0
    T: float = 1.0
    r: float = 0.10
    delta: float = 0.0
    sigma: float = 0.35
    H: tuple[float, ...] = (0.5, 0.59, 0.68, 0.77)
    alpha: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)
    xmin: float = -6.0
    xmax: float = 6.0


def gmfbm_diffusion_coefficient(tau: float, p: GMFBMParams) -> float:
    """Gamma(tau) = sigma^2 sum_i H_i alpha_i^2 (T - tau)^(2H_i - 1)."""
    t = max(p.T - tau, 0.0)
    total = 0.0
    for H_i, alpha_i in zip(p.H, p.alpha):
        exponent = 2.0 * H_i - 1.0
        if abs(exponent) < 1.0e-14:
            factor = 1.0
        elif t == 0.0:
            # In the examples here, all remaining exponents are positive.
            factor = 0.0
        else:
            factor = t**exponent
        total += H_i * alpha_i**2 * factor
    return p.sigma**2 * total


def lambda_max_stable_operator(Nx: int, p: GMFBMParams) -> tuple[float, float, float]:
    """
    Return:
        max_n lambda_max(S_n^h),
        Gamma(0),
        Gamma(T).

    Grid convention:
        Nx = number of subintervals
        number of nodes = Nx + 1
        number of interior unknowns = Nx - 1

    For the centered energy-stable operator with W_h = h I,
        S_n^h = Gamma(tau_n) D2 - r I.
    The eigenvalues of the interior Dirichlet D2 matrix are
        -4/h^2 sin^2(k pi/(2 Nx)), k=1,...,Nx-1.
    Since Gamma(tau) is decreasing for the first example, the largest value of
    lambda_max(S_n^h) occurs at tau = T, where Gamma is smallest.
    """
    if Nx < 2:
        raise ValueError("Nx must be at least 2.")

    h = (p.xmax - p.xmin) / Nx

    Gamma0 = gmfbm_diffusion_coefficient(0.0, p)
    GammaT = gmfbm_diffusion_coefficient(p.T, p)

    # Least negative eigenvalue of the Dirichlet second-difference matrix.
    lambda_D2_max = -(4.0 / h**2) * math.sin(math.pi / (2.0 * Nx)) ** 2

    # Since lambda_D2_max < 0, the least dissipative endpoint is GammaT.
    lambda_max = GammaT * lambda_D2_max - p.r
    return lambda_max, Gamma0, GammaT


def diagnostic_row(Nx: int, Nt: int, p: GMFBMParams) -> dict[str, float | str]:
    h = (p.xmax - p.xmin) / Nx
    dt = p.T / Nt
    lam, Gamma0, GammaT = lambda_max_stable_operator(Nx, p)
    return {
        "Nx": Nx,
        "Nt": Nt,
        "h": h,
        "Delta_tau": dt,
        "Delta_tau_over_h2": dt / h**2,
        "Gamma_0": Gamma0,
        "Gamma_T": GammaT,
        "max_lambda": lam,
        "stable": "Yes" if lam <= 0.0 else "No",
    }


def latex_row(row: dict[str, float | str]) -> str:
    return (
        f"{int(row['Nx'])} & {int(row['Nt'])} & "
        f"{float(row['h']):.5f} & {float(row['Delta_tau']):.6f} & "
        f"{float(row['Delta_tau_over_h2']):.3f} & "
        f"{float(row['max_lambda']):.9f} & {row['stable']} \\\\"
    )


def make_latex_table(rows: list[dict[str, float | str]]) -> str:
    body = "\n".join(latex_row(r) for r in rows)
    lines = [
        "\\begin{center}",
        "\\begin{tabular}{rrrrrrc}",
        "\\toprule",
        "$N_x$ & $N_t$ & $h$ & $\\Delta\\tau$ & $\\Delta\\tau/h^2$ & $\\max_n\\lambda_{\\max}(S_n^h)$ & Stable? \\\\",
        "\\midrule",
        body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
    ]
    return "\n".join(lines)


def main() -> None:
    p = GMFBMParams()
    grids = [
        (100, 400),
        (400, 600),
        (500, 800),
        (600, 1000),
        (800, 1200),
        (400, 100),
        (600, 100),
        (800, 100),
        (1000, 100),
        (1200, 100),
    ]

    rows = [diagnostic_row(Nx, Nt, p) for Nx, Nt in grids]

    outdir = Path("gmfbm_stability_output")
    outdir.mkdir(exist_ok=True)

    with (outdir / "stability_diagnostics_reliable.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    latex_table = make_latex_table(rows)
    (outdir / "stability_diagnostics_reliable_table.tex").write_text(latex_table)

    print("GMFBM stability diagnostics")
    print(f"Gamma(0) = {rows[0]['Gamma_0']:.12f}")
    print(f"Gamma(T) = {rows[0]['Gamma_T']:.12f}")
    print()
    print("Nx Nt h Delta_tau Delta_tau/h^2 max_lambda stable")
    for row in rows:
        print(
            f"{int(row['Nx']):4d} {int(row['Nt']):4d} "
            f"{row['h']:.5f} {row['Delta_tau']:.6f} "
            f"{row['Delta_tau_over_h2']:.3f} "
            f"{row['max_lambda']:.9f} {row['stable']}"
        )

    print("\nLaTeX table:")
    print(latex_table)
    print(f"\nWrote: {outdir / 'stability_diagnostics_reliable.csv'}")
    print(f"Wrote: {outdir / 'stability_diagnostics_reliable_table.tex'}")


if __name__ == "__main__":
    main()
