"""Stability diagnostics: dissipativity certificates, mass-matrix bounds, propagator norms,
support-radius / degree scan, and non-uniform node sets (Tables 2, 3, 4)."""
import numpy as np
import scipy.sparse.linalg as spla

from common import GMFBM, EXAMPLE1, EXAMPLE2, MLSPricer, eval_points, save, time_mesh


def weights(x):
    return 0.5 * (x[2:] - x[:-2])


def reduced_ops(p):
    Md, D1, Kd = p.reduced()
    Minv = np.linalg.inv(Md)
    return Md, Minv, D1, Kd


def lam_sym(A, w):
    Wh = np.sqrt(w)
    As = (Wh[:, None] * A) / Wh[None, :]
    return np.linalg.eigvalsh(0.5 * (As + As.T)).max()


def certificates(model, p):
    """max over the two extreme values of a of
       lambda_max of the weighted Hermitian part of A = M^{-1} L  (coefficient operator) and
       of B = L M^{-1} (reconstructed-price operator)."""
    Md, Minv, D1, Kd = reduced_ops(p)
    w = weights(p.x)
    n = Md.shape[0]
    lamA, lamB, reA = [], [], []
    for a in (model.a(model.T), model.a(0.0)):
        L = a * Kd + (model.r - model.delta) * D1 - model.r * Md
        A = Minv @ L
        B = L @ Minv
        lamA.append(lam_sym(A, w))
        lamB.append(lam_sym(B, w))
        reA.append(np.linalg.eigvals(A).real.max())
    return max(lamA), max(lamB), max(reA)


def mass_bounds(p):
    Md, Minv, D1, Kd = reduced_ops(p)
    # uniform grid: W_h = h I, so ||.||_{h->h} is the spectral norm
    nM = np.linalg.norm(Md, 2)
    nMi = np.linalg.norm(Minv, 2)
    CD = p.h * np.linalg.norm(Minv @ D1, 2)
    return nM, nMi, CD


def table_dissipativity():
    m1, m2 = GMFBM(**EXAMPLE1), GMFBM(**EXAMPLE2)
    rows = []
    for Nx in (100, 200, 400, 800):
        p1, p2 = MLSPricer(m1, Nx), MLSPricer(m2, Nx)
        lA1, lB1, _ = certificates(m1, p1)
        lA2, lB2, _ = certificates(m2, p2)
        nM, nMi, CD = mass_bounds(p1)
        rows.append(dict(Nx=Nx, h=p1.h, lamA1=lA1, lamA2=lA2, lamB1=lB1, lamB2=lB2,
                         normM=nM, normMinv=nMi, kappaM=nM * nMi, sigmin=1 / nMi, CD=CD))
        print(rows[-1], flush=True)
    return rows


def propagator_norms(model, p, tau, dense):
    Md, Minv, D1, Kd = reduced_ops(p)
    n = Md.shape[0]
    I = np.eye(n)
    KA, JA = Minv @ Kd, Minv @ D1

    def A(t):
        return model.a(t) * KA + (model.r - model.delta) * JA - model.r * I

    nmax, rmax = 0.0, 0.0
    for k in range(len(tau) - 1):
        s = 0.5 * (tau[k + 1] - tau[k])
        Lm = I - s * A(tau[k + 1])
        Rm = I + s * A(tau[k])
        if dense:
            P = np.linalg.solve(Lm, Rm)
            nmax = max(nmax, np.linalg.norm(P, 2))
            rmax = max(rmax, np.abs(np.linalg.eigvals(P)).max())
        else:
            import scipy.linalg as sla
            lu = sla.lu_factor(Lm)
            luT = sla.lu_factor(Lm.T)
            op = spla.LinearOperator((n, n), matvec=lambda v: sla.lu_solve(lu, Rm @ v),
                                     rmatvec=lambda v: Rm.T @ sla.lu_solve(luT, v))
            sv = spla.svds(op, k=1, which="LM", return_singular_vectors=False, tol=1e-12)
            nmax = max(nmax, sv[0])
            ev = spla.eigs(spla.LinearOperator((n, n), matvec=lambda v: sla.lu_solve(lu, Rm @ v)),
                           k=1, which="LM", return_eigenvectors=False, tol=1e-12)
            rmax = max(rmax, np.abs(ev).max())
    return nmax, rmax


def table_propagator():
    m = GMFBM(**EXAMPLE1)
    rows = []
    for Nx, Nt in ((100, 200), (200, 400), (400, 800), (800, 800)):
        p = MLSPricer(m, Nx)
        tau = time_mesh(m.T, Nt, "twosided")
        k = np.diff(tau)
        kappa = k.max() / p.h
        _, _, CD = mass_bounds(p)
        a = m.a(tau)
        rho = np.abs((a[:-1] - a[1:]) / a[1:])
        Xi = 1 + 0.5 * m.r * k.max() + 0.5 * kappa * abs(m.r - m.delta) * CD
        nmax, rmax = propagator_norms(m, p, tau, dense=True)
        rows.append(dict(Nx=Nx, Nt=Nt, kappa=kappa, Pnorm=nmax, Prho=rmax, Xi=Xi,
                         bound=1 + Xi * rho.max()))
        print(rows[-1], flush=True)
    return rows


def radius_scan():
    m = GMFBM(**EXAMPLE1)
    out = []
    for deg, radii in ((3, (3.5, 4.0, 4.5)), (4, (4.5, 5.0, 5.5, 6.0, 6.5))):
        for rf in radii:
            p = MLSPricer(m, 200, degree=deg, radius_factor=rf)
            lA, lB, reA = certificates(m, p)
            out.append(dict(degree=deg, radius=rf, lamA=lA, lamB=lB, maxRe=reA))
            print(out[-1], flush=True)
    # does the scheme actually diverge for radius 4h?  European put, exact traces
    Se = eval_points(m.K)
    ex = m.european(Se, m.T, "put")
    blow = []
    for rf in (3.5, 4.0, 4.5):
        p = MLSPricer(m, 200, radius_factor=rf)
        E = p.eval_matrices(Se)[0]
        with np.errstate(all="ignore"):
            U = p.european(time_mesh(m.T, 400, "twosided"), "put", "exact")["U"]
            err = float(np.nanmax(np.abs(E @ U - ex)) / m.K) if np.all(np.isfinite(U)) else float("inf")
        blow.append(dict(radius=rf, err=err))
        print(blow[-1], flush=True)
    return out, blow


def sinh_nodes(Nx, c, xmax=6.0):
    xi = np.linspace(-1, 1, Nx + 1)
    return c * np.sinh(xi * np.arcsinh(xmax / c))


def jitter_nodes(Nx, amp, seed, xmax=6.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-xmax, xmax, Nx + 1)
    h = x[1] - x[0]
    x[1:-1] += rng.uniform(-amp, amp, Nx - 1) * h
    return x


def table_nodes():
    m = GMFBM(**EXAMPLE1)
    Se = eval_points(m.K)
    ex = m.european(Se, m.T, "put")
    tau = time_mesh(m.T, 1600, "twosided")
    rows = []
    for Nx in (100, 200, 400, 800):
        sets = [("uniform", np.linspace(-6, 6, Nx + 1)), ("sinh", sinh_nodes(Nx, 0.75))]
        sets += [("jitter%d" % s, jitter_nodes(Nx, 0.2, s)) for s in range(3)]
        for name, x in sets:
            p = MLSPricer(m, nodes=x)
            lA, lB, reA = certificates(m, p) if Nx <= 400 else (np.nan, np.nan, np.nan)
            E = p.eval_matrices(Se)[0]
            U = p.european(tau, "put", "exact")["U"]
            rows.append(dict(Nx=Nx, nodes=name, lamA=lA, lamB=lB, maxRe=reA,
                             err=float(np.abs(E @ U - ex).max() / m.K),
                             hmin=float(np.diff(x).min()), hmax=float(np.diff(x).max())))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["dissipativity", "radius", "nodes", "propagator"]
    if "dissipativity" in parts:
        save("stability_dissipativity", table_dissipativity())
    if "radius" in parts:
        save("stability_radius", dict(zip(("scan", "blowup"), radius_scan())))
    if "nodes" in parts:
        save("stability_nodes", table_nodes())
    if "propagator" in parts:
        save("stability_propagator", table_propagator())


def extra_certificates():
    """Certificates for every further configuration in which an American problem is solved."""
    from exp_hurst import CONFIGS
    cases = []
    for Nx in (1600,):
        cases.append(("Ex1 Nx=%d" % Nx, dict(EXAMPLE1), Nx, None))
    cases.append(("Ex1 clustered Nx=800", dict(EXAMPLE1), 800, sinh_nodes(800, 0.75)))
    cases.append(("Ex1 clustered Nx=1600", dict(EXAMPLE1), 1600, sinh_nodes(1600, 0.75)))
    for r in (0.04, 0.12):
        p = dict(EXAMPLE1); p["r"] = r
        cases.append(("Ex1 r=%.2f" % r, p, 800, None))
    for sg in (0.25, 0.45):
        p = dict(EXAMPLE1); p["sigma"] = sg
        cases.append(("Ex1 sigma=%.2f" % sg, p, 800, None))
    for sg in (0.4, 0.6):
        p = dict(EXAMPLE2); p["sigma"] = sg
        cases.append(("Ex2 sigma=%.2f" % sg, p, 800, None))
    for T in (0.5, 2.0):
        p = dict(EXAMPLE1); p["T"] = T
        cases.append(("Ex1 T=%g" % T, p, 800, None))
    for T in (1.0, 0.5):
        for name, H, al in CONFIGS:
            cases.append(("%s T=%g" % (name, T), dict(H=H, alpha=al, sigma=0.35, r=0.08, K=25.0, T=T), 800, None))
    out = []
    for name, prm, Nx, nodes in cases:
        m = GMFBM(**prm)
        p = MLSPricer(m, Nx, nodes=nodes)
        lA, lB, reA = certificates(m, p)
        out.append(dict(case=name, Nx=Nx, lamA=lA, lamB=lB, maxRe=reA))
        print(out[-1], flush=True)
    return out


def quartic_run():
    m = GMFBM(**EXAMPLE1)
    Se = eval_points(m.K)
    ex = m.european(Se, m.T, "put")
    out = []
    for rf in (4.5, 5.0, 5.5, 6.0, 6.5):
        p = MLSPricer(m, 200, degree=4, radius_factor=rf)
        E = p.eval_matrices(Se)[0]
        with np.errstate(all="ignore"):
            U = p.european(time_mesh(m.T, 400, "twosided"), "put", "exact")["U"]
            err = float(np.nanmax(np.abs(E @ U - ex)) / m.K) if np.all(np.isfinite(U)) else float("inf")
        out.append(dict(radius=rf, err=err))
        print(out[-1], flush=True)
    return out


if __name__ == "__main__" and "extra" in sys.argv[1:]:
    save("stability_extra", dict(certificates=extra_certificates(), quartic=quartic_run()))
