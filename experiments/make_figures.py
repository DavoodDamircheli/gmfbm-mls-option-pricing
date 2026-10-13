"""Regenerate every figure of the paper (Figures 1-8) into ../../figures/."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import GMFBM, EXAMPLE1, EXAMPLE2, MLSPricer, ROOT, load, time_mesh  # noqa: E402
from exp_hurst import exercise_boundary  # noqa: E402

FIG = os.path.join(os.path.dirname(ROOT), "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 7,
                     "lines.linewidth": 1.2, "savefig.bbox": "tight"})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def fbm_paths(Hs, n=1000, T=1.0, seed=7):
    rng = np.random.default_rng(seed)
    t = np.linspace(T / n, T, n)
    out = []
    for H in Hs:
        C = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
        Lc = np.linalg.cholesky(C + 1e-12 * np.eye(n))
        out.append(np.concatenate([[0.0], Lc @ rng.standard_normal(n)]))
    return np.concatenate([[0.0], t]), out


def fig_paths():
    m = GMFBM(**EXAMPLE1)
    t, paths = fbm_paths(m.H)
    z = sum(a * b for a, b in zip(m.alpha, paths))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    for i, (H, b) in enumerate(zip(m.H, paths)):
        ax.plot(t, b, color=COLORS[i], lw=0.8, label=r"FBM, $H=%.2f$" % H)
    ax.plot(t, z, color="k", lw=1.2, label=r"GMFBM (Example 1)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("value")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.savefig(os.path.join(FIG, "fig1_paths.pdf"))
    plt.close(fig)


def prices(ex, name, sigmas):
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    base = GMFBM(**ex)
    S = np.linspace(0.02 * base.K, 2.5 * base.K, 400)
    for ax in axs:
        ax.plot(S, np.maximum(base.K - S, 0), "k:", lw=1, label="payoff")
    for i, sg in enumerate(sigmas):
        p_ = dict(ex)
        p_["sigma"] = sg
        m = GMFBM(**p_)
        p = MLSPricer(m, 400)
        tau = time_mesh(m.T, 400, "twosided")
        E = p.eval_matrices(S)[0]
        ue = E @ p.european(tau, "put", "fin")["U"]
        ua = E @ p.american(tau)["U"]
        axs[0].plot(S, ue, color=COLORS[i], label=r"$\sigma=%.2f$" % sg)
        axs[1].plot(S, ua, color=COLORS[i], label=r"$\sigma=%.2f$" % sg)
    axs[0].set_title("European put")
    axs[1].set_title("American put")
    for ax in axs:
        ax.set_xlabel(r"$S$")
    axs[0].set_ylabel(r"$V(S,0)$")
    axs[0].legend()
    fig.savefig(os.path.join(FIG, "%s_prices.pdf" % name))
    plt.close(fig)


def boundary_curve(m, Nx, Nt):
    p = MLSPricer(m, Nx)
    tau = time_mesh(m.T, Nt, "twosided")
    res = p.american(tau, store=True)
    eta = p.payoff("american")
    Sf = np.array([exercise_boundary(p, U, eta, m.K) for U in res["Us"][1:]])
    return m.T - tau[1:], Sf, p, res


def fig_boundary_ex1():
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    for i, r in enumerate((0.04, 0.08, 0.12)):
        p_ = dict(EXAMPLE1)
        p_["r"] = r
        t, Sf, _, _ = boundary_curve(GMFBM(**p_), 1600, 800)
        ax.plot(t, Sf, color=COLORS[i], label=r"$r=%.2f$" % r)
    ax.set_xlabel(r"calendar time $t$")
    ax.set_ylabel(r"$S_f^*(t)$")
    ax.legend()
    fig.savefig(os.path.join(FIG, "ex1_boundary.pdf"))
    plt.close(fig)


def fig_convergence():
    sp = load("european_spatial")
    tm = load("european_temporal")
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
    h = np.array([r["h"] for r in sp])
    for key, lab, c, mk in (("put", "European put", COLORS[0], "o"), ("call", "European call", COLORS[1], "s"),
                            ("am", "American put", COLORS[2], "^")):
        axs[0].loglog(h, [r[key] for r in sp], marker=mk, ms=3, color=c, label=lab)
    axs[0].loglog(h, 0.07 * h ** 2, "k:", label=r"$O(h^2)$")
    axs[0].set_xlabel(r"$h$")
    axs[0].set_ylabel(r"max error / $K$")
    axs[0].set_title("space (two-sided mesh, $N_t=1600$)")
    axs[0].legend()
    Nt = np.array(tm["Nt"], dtype=float)
    for key, lab, c, ls in (("eur_uniform", "European, uniform", COLORS[0], "-"),
                            ("eur_graded", "European, graded", COLORS[0], "--"),
                            ("am_uniform", "American, uniform", COLORS[1], "-"),
                            ("am_twosided", "American, two-sided", COLORS[1], "--")):
        axs[1].loglog(1 / Nt, tm[key], ls, marker="o", ms=3, color=c, label=lab)
    axs[1].loglog(1 / Nt, 0.02 * (1 / Nt) ** 1.18, "k:", label=r"$O(\Delta\tau^{1.18})$")
    axs[1].loglog(1 / Nt, 0.02 * (1 / Nt) ** 2, "k-.", label=r"$O(\Delta\tau^{2})$")
    axs[1].set_xlabel(r"$1/N_t$")
    axs[1].set_title(r"time at $\tau=T$ ($N_x=400$)")
    axs[1].legend(fontsize=6)
    fig.savefig(os.path.join(FIG, "ex1_convergence.pdf"))
    plt.close(fig)


def fig_greeks_ex1():
    m = GMFBM(**EXAMPLE1)
    p = MLSPricer(m, 400)
    U = p.european(time_mesh(m.T, 1600, "twosided"), "put", "exact")["U"]
    S = np.linspace(0.2 * m.K, 2.5 * m.K, 300)
    d, g, _, _ = p.greeks(U, S, m.T)
    Sm = np.linspace(0.2 * m.K, 2.5 * m.K, 25)
    de, ge, _, _ = m.european_greeks(Sm, m.T, "put")
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4))
    axs[0].plot(S, d, color=COLORS[0], label="MLS")
    axs[0].plot(Sm, de, "o", ms=3, mfc="none", color="k", label="exact")
    axs[1].plot(S, g, color=COLORS[0], label="MLS")
    axs[1].plot(Sm, ge, "o", ms=3, mfc="none", color="k", label="exact")
    axs[0].set_ylabel(r"$\Delta$")
    axs[1].set_ylabel(r"$\Gamma$")
    for ax in axs:
        ax.set_xlabel(r"$S$")
        ax.legend()
    fig.savefig(os.path.join(FIG, "ex1_greeks.pdf"))
    plt.close(fig)


def fig_ex2():
    m = GMFBM(**EXAMPLE2)
    # surface and boundary
    t, Sf, p, res = boundary_curve(m, 400, 400)
    S = np.linspace(0.3 * m.K, 2.0 * m.K, 120)
    E = p.eval_matrices(S)[0]
    tau = time_mesh(m.T, 400, "twosided")
    idx = np.linspace(0, len(tau) - 1, 60).astype(int)
    V = np.array([E @ res["Us"][k] for k in idx])
    TT, SS = np.meshgrid(m.T - tau[idx], S, indexing="ij")
    fig = plt.figure(figsize=(3.4, 2.8))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(SS, TT, V, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel(r"$S$")
    ax.set_ylabel(r"$t$")
    ax.set_zlabel(r"$V$")
    ax.view_init(25, -60)
    fig.savefig(os.path.join(FIG, "ex2_surface.pdf"))
    plt.close(fig)
    t2, Sf2, _, _ = boundary_curve(m, 1600, 800)
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    ax.plot(t2, Sf2, color=COLORS[0])
    ax.set_xlabel(r"calendar time $t$")
    ax.set_ylabel(r"$S_f^*(t)$")
    fig.savefig(os.path.join(FIG, "ex2_boundary.pdf"))
    plt.close(fig)
    # American Greeks at t = 0
    S = np.linspace(0.3 * m.K, 2.0 * m.K, 600)
    d, g, _, _ = p.greeks(res["U"], S, m.T)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4))
    axs[0].plot(S, d, color=COLORS[0])
    axs[1].plot(S, g, color=COLORS[0])
    for ax in axs:
        ax.axvline(m.K, color="gray", ls="--", lw=0.7)
        ax.set_xlabel(r"$S$")
    axs[0].set_ylabel(r"$\Delta$")
    axs[1].set_ylabel(r"$\Gamma$")
    fig.savefig(os.path.join(FIG, "ex2_greeks.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_paths()
    prices(EXAMPLE1, "ex1", (0.25, 0.35, 0.45))
    prices(EXAMPLE2, "ex2", (0.4, 0.5, 0.6))
    fig_boundary_ex1()
    fig_convergence()
    fig_greeks_ex1()
    fig_ex2()
    print("figures written to", FIG)
