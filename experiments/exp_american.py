"""American puts: reference solution and its error, pointwise comparison, semismooth Newton
diagnostics, reconstructed boundary mismatch (Tables 9, 10 and Section 9 text)."""
import sys

import numpy as np

from common import (GMFBM, EXAMPLE1, EXAMPLE2, MLSPricer, american_reference, eval_points,
                    interp_log_cubic, save, time_mesh)

PTS = np.array([0.5, 0.75, 1.0, 1.25, 1.5])


def reference_study(model):
    Se = eval_points(model.K)
    vals = {}
    for Nx, Nt in ((12000, 4000), (6000, 4000), (12000, 2000), (6000, 2000)):
        S, u = american_reference(model, Nx, Nt)
        vals[(Nx, Nt)] = interp_log_cubic(S, u, Se, model.S_ref)
    f = vals[(12000, 4000)]
    d_space = np.abs(f - vals[(6000, 4000)]).max()
    d_time = np.abs(f - vals[(12000, 2000)]).max()
    d_joint = np.abs(f - vals[(6000, 2000)]).max()
    out = dict(d_space=d_space, d_time=d_time, d_joint=d_joint,
               est_space=d_space / 3, est_time=d_time / 3, est_total=(d_space + d_time) / 3)
    out.update({k + "_rel": v / model.K for k, v in list(out.items())})
    print(out, flush=True)
    return out


def pointwise(model):
    S, u = american_reference(model)
    ref = interp_log_cubic(S, u, PTS * model.K, model.S_ref)
    p = MLSPricer(model, 800)
    res = p.american(time_mesh(model.T, 800, "twosided"))
    mls = p.eval_matrices(PTS * model.K)[0] @ res["U"]
    return dict(S=PTS * model.K, ref=ref, mls=mls)


def ssn_table():
    rows = []
    for name, ex in (("1", EXAMPLE1), ("2", EXAMPLE2)):
        m = GMFBM(**ex)
        for Nx, Nt in ((100, 200), (200, 400), (400, 800), (800, 800), (800, 1600)):
            p = MLSPricer(m, Nx)
            st = p.american(time_mesh(m.T, Nt, "twosided"))["stats"]
            rows.append(dict(ex=name, Nx=Nx, Nt=Nt, avg=float(np.mean(st["its"])), max=int(max(st["its"])),
                             res=float(max(st["res"])), red=st["reductions"], compl=st["compl"],
                             min_gap=st["min_gap"], min_dual=st["min_dual"]))
            print(rows[-1], flush=True)
    return rows


def boundary_mismatch():
    m = GMFBM(**EXAMPLE1)
    rows = []
    tau = time_mesh(m.T, 1600, "twosided")
    for Nx in (100, 200, 400, 800):
        p = MLSPricer(m, Nx)
        U = p.european(tau, "put", "exact")["U"]
        q = p.M @ U
        g0, g1 = p.boundary(m.T, "put", "exact")
        rows.append(dict(kind="european", Nx=Nx, left=abs(q[0] - g0) / m.K, right=abs(q[-1] - g1) / m.K))
        print(rows[-1], flush=True)
    for Nx in (100, 200, 400, 800):
        p = MLSPricer(m, Nx)
        U = p.american(time_mesh(m.T, Nx, "twosided"))["U"]
        q = p.M @ U
        g0, g1 = p.boundary(m.T, "american", "fin")
        rows.append(dict(kind="american", Nx=Nx, left=abs(q[0] - g0) / m.K, right=abs(q[-1] - g1) / m.K))
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__" and sys.argv[1:] == ["mismatch"]:
    save("boundary_mismatch", boundary_mismatch())
elif __name__ == "__main__":
    out = {}
    for name, ex in (("ex1", EXAMPLE1), ("ex2", EXAMPLE2)):
        m = GMFBM(**ex)
        out[name] = dict(reference=reference_study(m), pointwise=pointwise(m))
    save("american_pointwise", out)
    save("american_ssn", ssn_table())
    save("boundary_mismatch", boundary_mismatch())
