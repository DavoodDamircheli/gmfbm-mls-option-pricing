"""Greeks and tangent parameter sensitivities (Tables 11, 12)."""
import numpy as np

from common import GMFBM, EXAMPLE1, MLSPricer, eval_points, orders, save, time_mesh


def greek_table():
    m = GMFBM(**EXAMPLE1)
    Se = eval_points(m.K)
    exact = m.european_greeks(Se, m.T, "put")
    tau = time_mesh(m.T, 1600, "twosided")
    rows = []
    for Nx in (50, 100, 200, 400, 800, 1600):
        p = MLSPricer(m, Nx)
        U = p.european(tau, "put", "exact")["U"]
        g = p.greeks(U, Se, m.T)
        row = dict(Nx=Nx)
        for name, a, b in zip(("delta", "gamma", "speed", "theta"), g, exact):
            row[name] = float(np.abs(a - b).max())
        rows.append(row)
        print(row, flush=True)
    return rows


def da_dsigma(m):
    return lambda t: 2.0 / m.sigma * m.a(t)


def da_dH(m, i):
    def f(t):
        s = m.T - t
        if s <= 0:
            return 0.0
        return m.sigma ** 2 * m.alpha[i] ** 2 * s ** (2 * m.H[i] - 1) * (1 + 2 * m.H[i] * np.log(s))
    return f


def exact_sens(params, S, tau, key, eps=1e-6):
    """Central difference of the closed-form price (smooth in the parameters)."""
    def price(delta_val):
        q = dict(params)
        if key == "sigma":
            q["sigma"] = params["sigma"] + delta_val
        else:
            H = list(params["H"])
            H[1] += delta_val
            q["H"] = tuple(H)
        return GMFBM(**q).european(S, tau, "put")
    return (price(eps) - price(-eps)) / (2 * eps)


def sensitivity_table():
    params = dict(EXAMPLE1)
    m = GMFBM(**params)
    Se = eval_points(m.K)
    Nt = 800
    tau_full = time_mesh(m.T, Nt, "twosided")
    tau = tau_full[: Nt // 2 + 1]  # two-sided mesh: tau_{Nt/2} = T/2 exactly
    th = tau[-1]
    ex_s = exact_sens(params, Se, th, "sigma")
    ex_h = exact_sens(params, Se, th, "H2")
    rows = []
    for Nx in (100, 200, 400, 800):
        p = MLSPricer(m, Nx)
        sol = p.european(tau, "put", "fin", store=True, keep_lu=True)
        E = p.eval_matrices(Se)[0]
        Zs = p.tangent(tau, sol["Us"], sol["lus"], da_dsigma(m))[-1]
        Zh = p.tangent(tau, sol["Us"], sol["lus"], da_dH(m, 1))[-1]
        row = dict(Nx=Nx, sigma=float(np.abs(E @ Zs - ex_s).max()), H2=float(np.abs(E @ Zh - ex_h).max()))
        if Nx == 400:
            # consistency with central differences of the discrete solver
            def disc(key, d):
                q = dict(params)
                if key == "sigma":
                    q["sigma"] += d
                else:
                    H = list(q["H"]); H[1] += d; q["H"] = tuple(H)
                mm = GMFBM(**q)
                pp = MLSPricer(mm, Nx)
                return E @ pp.european(tau, "put", "fin")["U"]
            d = 1e-6
            row["fd_sigma"] = float(np.abs((disc("sigma", d) - disc("sigma", -d)) / (2 * d) - E @ Zs).max())
            row["fd_H2"] = float(np.abs((disc("H2", d) - disc("H2", -d)) / (2 * d) - E @ Zh).max())
        rows.append(row)
        print(row, flush=True)
    return rows


if __name__ == "__main__":
    save("greeks", greek_table())
    save("sensitivity", sensitivity_table())
