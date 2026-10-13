"""European options and valuation-date convergence (Tables 5-8, T = 2 check)."""
import numpy as np

from common import (GMFBM, EXAMPLE1, MLSPricer, eval_points, orders, ref_on, save, time_mesh)

PTS = np.array([0.5, 0.75, 1.0, 1.25, 1.5])


def pointwise():
    m = GMFBM(**EXAMPLE1)
    p = MLSPricer(m, 800)
    tau = time_mesh(m.T, 800, "twosided")
    E = p.eval_matrices(PTS * m.K)[0]
    out = {}
    for kind in ("put", "call"):
        U = p.european(tau, kind, "fin")["U"]
        out[kind] = dict(exact=m.european(PTS * m.K, m.T, kind), mls=E @ U)
    return out


def spatial(model, Nxs=(50, 100, 200, 400, 800, 1600), Nt=1600, american=True):
    Se = eval_points(model.K)
    tau = time_mesh(model.T, Nt, "twosided")
    ref = ref_on(model, Se) if american else None
    rows = []
    for Nx in Nxs:
        p = MLSPricer(model, Nx)
        E = p.eval_matrices(Se)[0]
        row = dict(Nx=Nx, h=p.h)
        for kind in ("put", "call"):
            U = p.european(tau, kind, "exact")["U"]
            row[kind] = float(np.abs(E @ U - model.european(Se, model.T, kind)).max() / model.K)
        if american:
            U = p.american(tau)["U"]
            row["am"] = float(np.abs(E @ U - ref).max() / model.K)
        rows.append(row)
        print(row, flush=True)
    return rows


def temporal(model, Nx=400, Nts=(50, 100, 200, 400, 800, 1600), american=True, Nref=12800):
    Se = eval_points(model.K)
    p = MLSPricer(model, Nx)
    E = p.eval_matrices(Se)[0]
    refE = E @ p.european(time_mesh(model.T, Nref, "graded"), "put", "exact")["U"]
    out = dict(Nt=list(Nts), eur_uniform=[], eur_rannacher=[], eur_graded=[], eur_twosided=[])
    for Nt in Nts:
        for key, mesh, ran in (("eur_uniform", "uniform", 0), ("eur_rannacher", "uniform", 2),
                               ("eur_graded", "graded", 0), ("eur_twosided", "twosided", 0)):
            U = p.european(time_mesh(model.T, Nt, mesh), "put", "exact", rannacher=ran)["U"]
            out[key].append(float(np.abs(E @ U - refE).max() / model.K))
        print(Nt, {k: v[-1] for k, v in out.items() if k != "Nt"}, flush=True)
    if american:
        refA = E @ p.american(time_mesh(model.T, Nref, "twosided"))["U"]
        out["am_uniform"], out["am_twosided"] = [], []
        for Nt in Nts:
            for key, mesh in (("am_uniform", "uniform"), ("am_twosided", "twosided")):
                U = p.american(time_mesh(model.T, Nt, mesh))["U"]
                out[key].append(float(np.abs(E @ U - refA).max() / model.K))
            print(Nt, out["am_uniform"][-1], out["am_twosided"][-1], flush=True)
    return out


def beta_table():
    rows = []
    for H2 in (0.6, 0.75, 0.9):
        m = GMFBM(H=(0.5, H2), alpha=(0.5, 0.5), sigma=0.35, r=0.08, K=25.0, T=1.0)
        t = temporal(m, Nx=200, Nts=(100, 200, 400, 800, 1600), american=False)
        rows.append(dict(H=[0.5, H2], beta=m.beta(), errs=t["eur_uniform"],
                         orders=orders(t["eur_uniform"], t["Nt"])[1:]))
        print(rows[-1], flush=True)
    return rows


def maturity_two():
    """Repeat the European verification for T = 2, where the valuation-date price depends on H."""
    p = dict(EXAMPLE1)
    p["T"] = 2.0
    m = GMFBM(**p)
    sp = spatial(m, Nxs=(100, 200, 400, 800), Nt=1600, american=False)
    tm = temporal(m, Nx=200, Nts=(100, 200, 400, 800, 1600), american=False)
    return dict(spatial=sp, temporal=tm, beta=m.beta())


if __name__ == "__main__":
    save("european_pointwise", pointwise())
    m1 = GMFBM(**EXAMPLE1)
    save("european_spatial", spatial(m1))
    save("european_temporal", temporal(m1))
    save("european_beta", beta_table())
    save("european_T2", maturity_two())
