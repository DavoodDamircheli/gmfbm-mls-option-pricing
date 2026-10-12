#!/usr/bin/env python3
"""Figures of Section 9 (written to figures/*.pdf).  Uses results/*.json for the
convergence plot and recomputes the price/boundary/Greek curves."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmfbm_mls.model import EXAMPLE1, EXAMPLE2  # noqa: E402
from gmfbm_mls.solvers import (Discretization, greeks, solve_american_put,  # noqa: E402
                               solve_european, time_mesh)

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RES = ROOT / "results"
# categorical slots 1-3 (validated palette), always in this order; ink for payoff/reference
C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#222222", "#8a8a85"
LS = ["-", "--", "-."]
MK = ["o", "s", "^"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#e3e3df", "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight",
})
W1 = (3.3, 2.5)
NX, NT = 400, 400


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


def prices_figure(m, sigmas, name, Smax):
    d = Discretization(NX)
    S = np.linspace(0.02 * m.K, Smax * m.K, 400)
    x = np.log(S / m.K)
    E = d.eval_matrices(x, 0)[0]
    taus = time_mesh(NT, m.T, "two_sided")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, american in zip(axs, (False, True)):
        ax.plot(S, np.maximum(m.K - S, 0), color=MUTED, lw=1.0, ls=":", label="payoff")
        for i, sg in enumerate(sigmas):
            mm = m.with_(sigma=sg)
            U = (solve_american_put(mm, d, taus).U if american
                 else solve_european(mm, d, taus, "put", "financial").U)
            ax.plot(S, E @ U, color=C[i], ls=LS[i], label=fr"$\sigma={sg}$")
        ax.set_xlabel("$S$")
        ax.set_title("American put" if american else "European put")
    axs[0].set_ylabel(r"$V(S,0)$")
    axs[0].legend(loc="upper right")
    _save(fig, name)


def boundary_figure(m, rs, name, nx=1600, nt=800):
    d = Discretization(nx)
    taus = time_mesh(nt, m.T, "two_sided")
    S = d.S(m)
    phi = np.maximum(m.K - S, 0)
    fig, ax = plt.subplots(figsize=W1)
    for i, r in enumerate(rs):
        mm = m.with_(r=r)
        sol = solve_american_put(mm, d, taus, keep_history=True)
        sf = []
        x = d.x
        for U in sol.history[1:]:
            gap = d.M @ U - phi
            act = np.nonzero((gap <= 1e-9 * m.K) & (S < m.K))[0]
            if not act.size:
                sf.append(np.nan)
                continue
            j = act.max()
            # smooth pasting: gap ~ c (x - x_f)^2 just above the boundary, so sqrt(gap)
            # is linear there; its root between the last contact node and the next
            # nodes gives a sub-grid estimate of x_f.
            g1, g2 = np.sqrt(max(gap[j + 1], 0.0)), np.sqrt(max(gap[j + 2], 0.0))
            xf = x[j + 1] - g1 * (x[j + 2] - x[j + 1]) / (g2 - g1) if g2 > g1 else x[j]
            sf.append(m.K * np.exp(min(max(xf, x[j]), x[j + 1])))
        t = m.T - taus[1:]                # calendar time
        ax.plot(t, sf, color=C[i], ls=LS[i], label=fr"$r={r}$")
    ax.set_xlabel("calendar time $t$")
    ax.set_ylabel(r"$S_f^*(t)$")
    if len(rs) > 1:                       # a single series needs no legend
        ax.legend(loc="upper left")
    _save(fig, name)
    return


def greeks_figure(m, name, american):
    d = Discretization(NX, kmax=3)
    taus = time_mesh(NT, m.T, "two_sided")
    S = np.linspace(0.3 * m.K, 2.0 * m.K, 600)
    x = np.log(S / m.K)
    U = solve_american_put(m, d, taus).U if american else solve_european(m, d, taus, "put", "financial").U
    g = greeks(m, d, U, m.T, x)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, key, lab in zip(axs, ("Delta", "Gamma"), (r"$\Delta$", r"$\Gamma$")):
        ax.plot(S, g[key], color=C[0], label="MLS")
        if not american:
            gx = m.european_greeks(S[::25], m.T, "put")
            ax.plot(S[::25], gx[key], ls="none", marker="o", ms=4, mfc="white", mec=INK, mew=0.9,
                    label="exact")
            ax.legend(loc="best")
        ax.set_xlabel("$S$")
        ax.set_ylabel(lab)
        ax.axvline(m.K, color=MUTED, lw=0.6, ls=":")
    _save(fig, name)


def surface_figure(m, name):
    d = Discretization(NX)
    taus = time_mesh(200, m.T, "two_sided")
    sol = solve_american_put(m, d, taus, keep_history=True)
    S = np.linspace(0.3 * m.K, 2.0 * m.K, 120)
    E = d.eval_matrices(np.log(S / m.K), 0)[0]
    V = np.array([E @ U for U in sol.history])
    TT, SS = np.meshgrid(m.T - taus, S, indexing="ij")
    fig = plt.figure(figsize=(3.6, 2.9))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(SS, TT, V, cmap="Blues", linewidth=0, antialiased=True, rstride=4, cstride=4)
    ax.set_xlabel("$S$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$V$")
    ax.view_init(elev=24, azim=-125)
    _save(fig, name)


def convergence_figure(name):
    es = json.loads((RES / "euro_space.json").read_text())
    et = json.loads((RES / "euro_time.json").read_text())
    am = json.loads((RES / "american.json").read_text())["Example 1"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2))
    ax = axs[0]
    h = np.array(es["h"])
    ax.loglog(h, es["put"], color=C[0], marker=MK[0], ms=4, label="European put")
    ax.loglog(h, es["call"], color=C[1], ls=LS[1], marker=MK[1], ms=4, label="European call")
    ax.loglog(am["space"]["h"], am["space"]["err"], color=C[2], ls=LS[2], marker=MK[2], ms=4,
              label="American put")
    ax.loglog(h, 0.5 * es["put"][-1] * (h / h[-1]) ** 2, color=MUTED, lw=0.9, ls=":", label=r"$O(h^2)$")
    ax.set_xlabel("$h$")
    ax.set_ylabel("max error / $K$")
    ax.set_title("space (fine graded time mesh)")
    ax.legend(loc="lower right")
    ax = axs[1]
    dt = 1.0 / np.array(et["N"])
    ax.loglog(dt, et["uniform"], color=C[0], marker=MK[0], ms=4, label="European, uniform")
    ax.loglog(dt, et["two_sided"], color=C[0], ls=LS[1], marker=MK[1], ms=4, label="European, graded")
    at = am["time"]
    ax.loglog(1.0 / np.array(at["N"]), at["uniform"], color=C[1], marker=MK[0], ms=4, label="American, uniform")
    ax.loglog(1.0 / np.array(at["N"]), at["two_sided"], color=C[1], ls=LS[1], marker=MK[1], ms=4,
              label="American, graded")
    b = et["beta"]
    ax.loglog(dt, 2 * et["uniform"][-1] * (dt / dt[-1]) ** (1 + b), color=MUTED, lw=0.9, ls=":",
              label=fr"$O(\Delta\tau^{{{1 + b:.2f}}})$")
    ax.loglog(dt, 0.4 * et["two_sided"][-1] * (dt / dt[-1]) ** 2, color=INK, lw=0.9, ls=":",
              label=r"$O(\Delta\tau^{2})$")
    ax.set_xlabel(r"$1/N_t$")
    ax.set_title(r"time at $\tau=T$ ($N_x=400$)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=6.5)
    _save(fig, name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    jobs = {
        "conv": lambda: convergence_figure("ex1_convergence.pdf"),
        "ex1_prices": lambda: prices_figure(EXAMPLE1, (0.25, 0.35, 0.45), "ex1_prices.pdf", 2.5),
        "ex1_boundary": lambda: boundary_figure(EXAMPLE1, (0.04, 0.08, 0.12), "ex1_boundary.pdf"),
        "ex1_greeks": lambda: greeks_figure(EXAMPLE1, "ex1_greeks.pdf", american=False),
        "ex2_prices": lambda: prices_figure(EXAMPLE2, (0.4, 0.5, 0.6), "ex2_prices.pdf", 2.5),
        "ex2_boundary": lambda: boundary_figure(EXAMPLE2, (0.10,), "ex2_boundary.pdf"),
        "ex2_greeks": lambda: greeks_figure(EXAMPLE2, "ex2_greeks.pdf", american=True),
        "ex2_surface": lambda: surface_figure(EXAMPLE2, "ex2_surface.pdf"),
    }
    for k in (jobs if which == ["all"] else which):
        jobs[k]()
