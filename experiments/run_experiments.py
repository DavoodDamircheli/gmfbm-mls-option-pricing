#!/usr/bin/env python3
"""
Reproduce every table of Section 9 of the paper.

    python experiments/run_experiments.py all          # everything (about 1 hour)
    python experiments/run_experiments.py <name> ...   # a single experiment

Each experiment writes results/<name>.json; experiments/make_tables.py turns these
into the LaTeX tables included by main.tex, and experiments/make_figures.py draws
the figures.  Production discretisation: MLS degree m = 3, weight (1-u^2)^5,
support radius 3.5h, N_x subintervals on [-6, 6], two-sided graded time mesh.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmfbm_mls.benchmark import reference_put          # noqa: E402
from gmfbm_mls.model import EXAMPLE1, EXAMPLE2, GMFBM  # noqa: E402
from gmfbm_mls.solvers import (Discretization, greeks, solve_american_put,  # noqa: E402
                               solve_european, solve_tangent, time_mesh)

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
PTS = np.array([0.5, 0.75, 1.0, 1.25, 1.5])            # S/K sample points
WINDOW = np.linspace(np.log(0.2), np.log(2.5), 401)     # financial window 0.2K <= S <= 2.5K
PROD_NX, PROD_NT, MESH = 800, 800, "two_sided"


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"[saved results/{name}.json]")


def orders(errs, hs):
    return [None] + [math.log(errs[i - 1] / errs[i]) / math.log(hs[i - 1] / hs[i]) for i in range(1, len(errs))]


def window_error(disc, U, ref, K, E=None):
    E = disc.eval_matrices(WINDOW, 0)[0] if E is None else E
    return float(np.max(np.abs(E @ U - ref)) / K)


# ---------------------------------------------------------------------------
def exp_dissipativity():
    """Assumption 13 for the assembled operator.  S_n is affine in a(tau_n), so
    lambda_max(S_n) is convex in a and its maximum over all time levels is attained
    at a(0) or a(T): evaluating both gives the exact maximum over the time grid."""
    out = {"production": [], "support_scan": []}
    for ex_name, m in (("Example 1", EXAMPLE1), ("Example 2", EXAMPLE2)):
        for nx in (100, 200, 400, 800):
            d = Discretization(nx)
            M, D1, D2 = (d.D[k].toarray() for k in range(3))
            Mi = M[1:-1, 1:-1]
            J = np.linalg.solve(Mi, D1[1:-1, 1:-1])
            lam = []
            for tau in (0.0, m.T):
                L = (m.a(tau) * (D2 - D1) + (m.r - m.delta) * D1 - m.r * M)[1:-1, 1:-1]
                A = np.linalg.solve(Mi, L)
                lam.append(float(np.linalg.eigvalsh(0.5 * (A + A.T)).max()))
            sv = np.linalg.svd(Mi, compute_uv=False)
            out["production"].append({
                "example": ex_name, "nx": nx, "h": d.h, "lam_max": max(lam),
                "norm_M": float(sv[0]), "norm_Minv": float(1 / sv[-1]), "cond_M": float(sv[0] / sv[-1]),
                "C_D": float(np.linalg.norm(J, 2) * d.h)})
            print(out["production"][-1])
    m = EXAMPLE1
    for kappa in (3.0, 3.5, 4.0, 4.5):
        try:
            d = Discretization(200, support=kappa)
        except np.linalg.LinAlgError:
            out["support_scan"].append({"support": kappa, "nx": 200, "lam_max": None,
                                        "note": "local polynomial space not unisolvent at the boundary"})
            continue
        M, D1, D2 = (d.D[k].toarray() for k in range(3))
        Mi = M[1:-1, 1:-1]
        lam = max(float(np.linalg.eigvalsh(0.5 * (A + A.T)).max()) for A in
                  (np.linalg.solve(Mi, (m.a(t) * (D2 - D1) + (m.r - m.delta) * D1 - m.r * M)[1:-1, 1:-1])
                   for t in (0.0, m.T)))
        out["support_scan"].append({"support": kappa, "nx": 200, "lam_max": lam})
        print(out["support_scan"][-1])
    save("dissipativity", out)


def exp_propagator(grids=((100, 200), (200, 400), (400, 800), (800, 800)), append=False):
    """max_n ||P_n||_h and rho(P_n) over all steps of the production time mesh, and
    the bound 1 + Theta max|rho_n| of Theorem (energy stability).  W_h = h I on the
    uniform grid, so ||.||_h-norms are spectral norms.  For N_x >= 800 the largest
    singular value and the largest-modulus eigenvalue are computed by ARPACK
    (agreement with dense LAPACK checked to 1e-10 at N_x = 400)."""
    import scipy.sparse.linalg as spla
    m = EXAMPLE1
    rows = json.loads((RES / "propagator.json").read_text()) if append else []
    for nx, nt in grids:
        d = Discretization(nx)
        M, D1, D2 = (d.D[k].toarray() for k in range(3))
        Mi = M[1:-1, 1:-1]
        Kh = np.linalg.solve(Mi, (D2 - D1)[1:-1, 1:-1])
        Jh = np.linalg.solve(Mi, D1[1:-1, 1:-1])
        n_int = Mi.shape[0]
        I = np.eye(n_int)
        taus = time_mesh(nt, m.T, MESH)
        A = lambda t: m.a(t) * Kh + (m.r - m.delta) * Jh - m.r * I
        norm_max, rho_max, rho_coef = 0.0, 0.0, 0.0
        for n in range(nt):
            s = 0.5 * (taus[n + 1] - taus[n])
            P = np.linalg.solve(I - s * A(taus[n + 1]), I + s * A(taus[n]))
            if nx >= 800:
                op = spla.aslinearoperator(P)
                nrm = float(spla.svds(op, k=1, tol=1e-12, return_singular_vectors=False)[0])
                rho = float(np.max(np.abs(spla.eigs(op, k=1, which="LM", tol=1e-12,
                                                     return_eigenvectors=False))))
            else:
                nrm = float(sla.svdvals(P)[0])
                rho = float(np.max(np.abs(np.linalg.eigvals(P))))
            norm_max = max(norm_max, nrm)
            rho_max = max(rho_max, rho)
            if nx >= 800 and n % 100 == 0:
                print(f"  step {n}/{nt}: max norm {norm_max:.9f}", flush=True)
            rho_coef = max(rho_coef, abs(m.a(taus[n]) - m.a(taus[n + 1])) / m.a(taus[n + 1]))
        kmax = float(np.max(np.diff(taus)))
        C_D = float(np.linalg.norm(Jh, 2) * d.h)
        kappa = kmax / d.h
        Theta = 1 + 0.5 * m.r * kmax + 0.5 * kappa * abs(m.r - m.delta) * C_D
        rows.append({"nx": nx, "nt": nt, "max_norm": norm_max, "max_rho": rho_max,
                     "kmax_over_h": kappa, "Theta": Theta, "bound": float(1 + Theta * rho_coef),
                     "growth_const": math.exp(Theta * (m.a(0) - m.a(m.T)) / m.a(m.T))})
        print(rows[-1])
    save("propagator", rows)


def exp_euro_space():
    m = EXAMPLE1
    nxs = [50, 100, 200, 400, 800, 1600]
    res = {"nx": nxs, "h": [], "put": [], "call": [], "Delta": [], "Gamma": [], "Speed": [], "Theta": []}
    ref = {k: m.european(m.K * np.exp(WINDOW), m.T, k) for k in ("put", "call")}
    gx = m.european_greeks(m.K * np.exp(WINDOW), m.T, "put")
    for nx in nxs:
        d = Discretization(nx, kmax=3)
        taus = time_mesh(1600, m.T, MESH)
        E = d.eval_matrices(WINDOW, 0)[0]
        res["h"].append(d.h)
        for kind in ("put", "call"):
            sol = solve_european(m, d, taus, kind, "exact")
            res[kind].append(window_error(d, sol.U, ref[kind], m.K, E))
            if kind == "put":
                g = greeks(m, d, sol.U, m.T, WINDOW)
                for k in ("Delta", "Gamma", "Speed", "Theta"):
                    res[k].append(float(np.max(np.abs(g[k] - gx[k]))))
        print(nx, {k: res[k][-1] for k in ("put", "call", "Delta", "Gamma", "Speed", "Theta")})
    for k in ("put", "call", "Delta", "Gamma", "Speed", "Theta"):
        res[k + "_order"] = orders(res[k], res["h"])
    # production grid with the financial (non-exact) boundary data: truncation check
    d = Discretization(PROD_NX)
    taus = time_mesh(PROD_NT, m.T, MESH)
    E = d.eval_matrices(np.log(PTS), 0)[0]
    pw = {}
    for kind in ("put", "call"):
        u_fin = E @ solve_european(m, d, taus, kind, "financial").U
        u_ex = m.european(m.K * PTS, m.T, kind)
        pw[kind] = {"exact": u_ex.tolist(), "mls": u_fin.tolist(), "abs_err": np.abs(u_fin - u_ex).tolist()}
    res["pointwise_production"] = pw
    save("euro_space", res)


def exp_euro_time():
    m = EXAMPLE1
    nx = 400
    d = Discretization(nx)
    E = d.eval_matrices(WINDOW, 0)[0]
    ref = E @ solve_european(m, d, time_mesh(12800, m.T, "graded"), "put", "exact").U
    Ns = [50, 100, 200, 400, 800, 1600]
    out = {"nx": nx, "N": Ns, "beta": m.beta()}
    for label, kind, rann in (("uniform", "uniform", 0), ("uniform_rannacher", "uniform", 2),
                              ("graded", "graded", 0), ("two_sided", "two_sided", 0)):
        errs = [float(np.max(np.abs(E @ solve_european(m, d, time_mesh(N, m.T, kind), "put", "exact",
                                                          rannacher=rann).U - ref)) / m.K) for N in Ns]
        out[label] = errs
        out[label + "_order"] = orders(errs, [1 / N for N in Ns])
        print(label, errs, out[label + "_order"])
    # 1 + beta for several Hurst mixtures (N_x = 200)
    d2 = Discretization(200)
    E2 = d2.eval_matrices(WINDOW, 0)[0]
    out["beta_table"] = []
    for Hs in ((0.5, 0.6), (0.5, 0.75), (0.5, 0.9)):
        mm = GMFBM(K=m.K, r=m.r, sigma=m.sigma, H=Hs, alpha=(0.5, 0.5))
        rf = E2 @ solve_european(mm, d2, time_mesh(12800, mm.T, "graded"), "put", "exact").U
        errs = [float(np.max(np.abs(E2 @ solve_european(mm, d2, time_mesh(N, mm.T, "uniform"), "put",
                                                           "exact").U - rf)) / mm.K) for N in (100, 200, 400, 800, 1600)]
        o = orders(errs, [1 / N for N in (100, 200, 400, 800, 1600)])
        out["beta_table"].append({"H": Hs, "beta": mm.beta(), "errors": errs, "orders": o[1:]})
        print(out["beta_table"][-1])
    save("euro_time", out)


def _benchmark(m, tag):
    t = time.time()
    x, S, u = reference_put(m, 12000, 4000)
    x2, S2, u2 = reference_put(m, 6000, 2000)
    print(f"benchmark {tag}: {time.time() - t:.0f}s")
    return {"x": x.tolist(), "u": u.tolist(), "x_half": x2.tolist(), "u_half": u2.tolist()}


def exp_benchmarks():
    save("benchmarks", {"Example 1": _benchmark(EXAMPLE1, "Ex1"), "Example 2": _benchmark(EXAMPLE2, "Ex2")})


def _load_bench(name):
    b = json.loads((RES / "benchmarks.json").read_text())[name]
    return np.array(b["x"]), np.array(b["u"]), np.array(b["x_half"]), np.array(b["u_half"])


def exp_american():
    out = {}
    for name, m in (("Example 1", EXAMPLE1), ("Example 2", EXAMPLE2)):
        xb, ub, xh, uh = _load_bench(name)
        ref_w = np.interp(WINDOW, xb, ub)
        ref_pts = np.interp(np.log(PTS), xb, ub)
        bench_unc = float(np.max(np.abs(np.interp(WINDOW, xb, ub) - np.interp(WINDOW, xh, uh))) / 3 / m.K)
        r = {"benchmark_uncertainty_over_K": bench_unc}
        # pointwise at production grid + SSN/LCP diagnostics on several grids
        diag = []
        for nx, nt in ((100, 200), (200, 400), (400, 800), (800, 800), (800, 1600)):
            d = Discretization(nx)
            sol = solve_american_put(m, d, time_mesh(nt, m.T, MESH))
            info = {k: v for k, v in sol.info.items()}
            info.update({"nx": nx, "nt": nt, "window_err_over_K": window_error(d, sol.U, ref_w, m.K)})
            diag.append(info)
            if (nx, nt) == (PROD_NX, PROD_NT):
                E = d.eval_matrices(np.log(PTS), 0)[0]
                q = E @ sol.U
                r["pointwise"] = {"benchmark": ref_pts.tolist(), "mls": q.tolist(),
                                  "abs_err": np.abs(q - ref_pts).tolist()}
            print(name, info)
        r["diagnostics"] = diag
        if name == "Example 1":
            # spatial convergence against the benchmark (fine time mesh)
            nxs = [50, 100, 200, 400, 800, 1600]
            errs, hs = [], []
            for nx in nxs:
                d = Discretization(nx)
                errs.append(window_error(d, solve_american_put(m, d, time_mesh(1600, m.T, MESH)).U, ref_w, m.K))
                hs.append(d.h)
                print("amer space", nx, errs[-1])
            r["space"] = {"nx": nxs, "h": hs, "err": errs, "order": orders(errs, hs)}
            # temporal convergence, same spatial grid, uniform vs two-sided mesh
            d = Discretization(400)
            E = d.eval_matrices(WINDOW, 0)[0]
            rf = E @ solve_american_put(m, d, time_mesh(12800, m.T, MESH)).U
            Ns = [50, 100, 200, 400, 800, 1600]
            tt = {"nx": 400, "N": Ns}
            for kind in ("uniform", "two_sided"):
                e = [float(np.max(np.abs(E @ solve_american_put(m, d, time_mesh(N, m.T, kind)).U - rf)) / m.K)
                     for N in Ns]
                tt[kind] = e
                tt[kind + "_order"] = orders(e, [1 / N for N in Ns])
                print("amer time", kind, e)
            r["time"] = tt
        out[name] = r
    save("american", out)


def exp_sensitivity():
    """Parameter sensitivities by the discrete tangent equation, against the exact
    derivative of the closed form and against central differences of the discrete
    solver.  tau = T/2 (time to maturity 0.5): for T = 1 the valuation-date price
    does not depend on H_i (total variance sigma^2 sum alpha_i^2 T^{2H_i}), so the
    H-sensitivity is tested at an intermediate level."""
    m = EXAMPLE1
    tau_end = 0.5 * m.T
    Spts = m.K * PTS
    out = {"tau": tau_end, "params": {}}
    for p in ("sigma", "H1"):
        ex = m.european_param_sensitivity(Spts, tau_end, p)
        exw = m.european_param_sensitivity(m.K * np.exp(WINDOW), tau_end, p)
        errs, hs = [], []
        for nx in (100, 200, 400, 800):
            d = Discretization(nx)
            taus = time_mesh(800, tau_end, MESH)
            sol = solve_european(m, d, taus, "put", "exact", keep_history=True, keep_factors=True)
            Z = solve_tangent(m, d, sol, p)
            E = d.eval_matrices(WINDOW, 0)[0]
            errs.append(float(np.max(np.abs(E @ Z - exw))))
            hs.append(d.h)
            if nx == 400:
                eps = 1e-6
                if p == "sigma":
                    mp, mm_ = m.with_(sigma=m.sigma + eps), m.with_(sigma=m.sigma - eps)
                else:
                    Hp, Hm = list(m.H), list(m.H)
                    Hp[1] += eps
                    Hm[1] -= eps
                    mp, mm_ = m.with_(H=tuple(Hp)), m.with_(H=tuple(Hm))
                fd = (E @ solve_european(mp, d, taus, "put", "exact").U
                      - E @ solve_european(mm_, d, taus, "put", "exact").U) / (2 * eps)
                tan_vs_fd = float(np.max(np.abs(E @ Z - fd)))
                Ep = d.eval_matrices(np.log(PTS), 0)[0]
                pts_tan = (Ep @ Z).tolist()
        out["params"][p] = {"exact_pts": ex.tolist(), "tangent_pts_nx400": pts_tan, "err": errs, "h": hs,
                            "order": orders(errs, hs), "tangent_vs_fd_nx400": tan_vs_fd}
        print(p, out["params"][p])
    save("sensitivity", out)


EXPS = {"dissipativity": exp_dissipativity, "propagator": exp_propagator, "euro_space": exp_euro_space,
        "euro_time": exp_euro_time, "benchmarks": exp_benchmarks, "american": exp_american,
        "sensitivity": exp_sensitivity,
        "propagator800": lambda: exp_propagator(grids=((800, 800),), append=True)}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(EXPS)
    for n in names:
        t = time.time()
        print(f"=== {n} ===")
        EXPS[n]()
        print(f"=== {n}: {time.time() - t:.0f}s ===")
