"""Write the LaTeX bodies of all tables of Section 9 from results/*.json into ../../tables/."""
import os

import numpy as np

from common import ROOT, load, orders

TAB = os.path.join(os.path.dirname(ROOT), "tables")
os.makedirs(TAB, exist_ok=True)


def sci(x, d=2):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), d) >= 10:
        m /= 10
        e += 1
    return r"$%.*f\times10^{%d}$" % (d, m, e)


def od(o):
    return "--" if o is None else "%.2f" % o


def write(name, lines):
    with open(os.path.join(TAB, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def t_dissipativity():
    rows = load("stability_dissipativity")
    L = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{$\max_n\lambda_{\max}$, $A^h=M_h^{-1}L_h$} & \multicolumn{2}{c}{$\max_n\lambda_{\max}$, $R^h=L_hM_h^{-1}$} & & & & \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"$N_x$ & Ex.~1 & Ex.~2 & Ex.~1 & Ex.~2 & $\|M_h\|_h$ & $\|M_h^{-1}\|_h$ & $\kappa_2(M_h)$ & $C_D$ \\", r"\midrule"]
    for r in rows:
        L.append("%d & $%.5f$ & $%.5f$ & $%.5f$ & $%.5f$ & %.3f & %.2f & %.2f & %.2f \\\\" % (
            r["Nx"], r["lamA1"], r["lamA2"], r["lamB1"], r["lamB2"], r["normM"], r["normMinv"], r["kappaM"], r["CD"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_dissipativity", L)


def t_propagator():
    rows = load("stability_propagator")
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"$N_x$ & $N_t$ & $\max_nk_n/h$ & $\max_n\|P_n\|_h$ & $\max_n\operatorname{spr}(P_n)$ & $\Xi$ & bound \\", r"\midrule"]
    for r in rows:
        L.append("%d & %d & %.3f & %.7f & %.7f & %.3f & %.3f \\\\" % (
            r["Nx"], r["Nt"], r["kappa"], r["Pnorm"], r["Prho"], r["Xi"], r["bound"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_propagator", L)


def t_nodes():
    rows = load("stability_nodes")
    by = {}
    for r in rows:
        by.setdefault(r["Nx"], {})[r["nodes"]] = r
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r" & \multicolumn{1}{c}{uniform} & \multicolumn{2}{c}{clustered} & \multicolumn{3}{c}{randomly perturbed (3 draws)} \\",
         r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(lr){5-7}",
         r"$N_x$ & error & error & $\lambda_{\max}(A^h)$ & error (range) & $\lambda_{\max}(A^h)$ (max) & $\max\operatorname{Re}\operatorname{spec}(A^h)$ \\",
         r"\midrule"]
    for Nx in sorted(by):
        d = by[Nx]
        jit = [d["jitter%d" % s] for s in range(3)]
        je = [j["err"] for j in jit]
        jl = [j["lamA"] for j in jit]
        jr = [j["maxRe"] for j in jit]
        lam_c = d["sinh"]["lamA"]
        L.append("%d & %s & %s & %s & %s--%s & %s & %s \\\\" % (
            Nx, sci(d["uniform"]["err"]), sci(d["sinh"]["err"]),
            "--" if not np.isfinite(lam_c) else "$%.4f$" % lam_c,
            sci(min(je)), sci(max(je)),
            "--" if not np.all(np.isfinite(jl)) else "$%.2f$" % max(jl),
            "--" if not np.all(np.isfinite(jr)) else "$%.4f$" % max(jr)))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_nodes", L)


def t_eur_pointwise():
    d = load("european_pointwise")
    pts = [r"$0.5K$", r"$0.75K$", r"$K$", r"$1.25K$", r"$1.5K$"]
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"$S$ & Put (exact) & Put (MLS) & Abs.\ error & Call (exact) & Call (MLS) & Abs.\ error \\", r"\midrule"]
    for i, p in enumerate(pts):
        pe, pm = d["put"]["exact"][i], d["put"]["mls"][i]
        ce, cm = d["call"]["exact"][i], d["call"]["mls"][i]
        L.append("%s & %.6f & %.6f & %s & %.6f & %.6f & %s \\\\" % (p, pe, pm, sci(abs(pe - pm)), ce, cm, sci(abs(ce - cm))))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ex1_european_pointwise", L)


def t_space():
    rows = load("european_spatial")
    h = [r["h"] for r in rows]
    inv = [1 / x for x in h]
    o = {k: orders([r[k] for r in rows], inv) for k in ("put", "call", "am")}
    L = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
         r"$N_x$ & $h$ & Eur.\ put & order & Eur.\ call & order & Am.\ put & order \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append("%d & %.4f & %s & %s & %s & %s & %s & %s \\\\" % (
            r["Nx"], r["h"], sci(r["put"]), od(o["put"][i]), sci(r["call"]), od(o["call"][i]), sci(r["am"]), od(o["am"][i])))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ex1_space", L)


def t_time():
    d = load("european_temporal")
    Nt = d["Nt"]
    keys = ("eur_uniform", "eur_rannacher", "eur_graded", "am_uniform", "am_twosided")
    o = {k: orders(d[k], Nt) for k in keys}
    L = [r"\begin{tabular}{rlllll}", r"\toprule",
         r"$N_t$ & Eur.\ uniform & Eur.\ Rannacher & Eur.\ graded & Am.\ uniform & Am.\ two-sided \\", r"\midrule"]
    for i, n in enumerate(Nt):
        L.append("%d & " % n + " & ".join("%s (%s)" % (sci(d[k][i]), od(o[k][i])) for k in keys) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ex1_time", L)


def t_beta():
    rows = load("european_beta")
    tm = load("european_temporal")
    t2 = load("european_T2")
    L = [r"\begin{tabular}{lccl}", r"\toprule", r"$H$ & $\beta$ & $1+\beta$ & observed pairwise orders \\", r"\midrule"]
    for r in rows:
        L.append(r"$(%.1f,\,%s)$ & %.2f & %.2f & %s \\" % (
            r["H"][0], ("%g" % r["H"][1]), r["beta"], 1 + r["beta"], ", ".join("%.2f" % x for x in r["orders"])))
    o1 = orders(tm["eur_uniform"], tm["Nt"])[2:]
    L.append(r"Example~\ref{ex3h} & 0.18 & 1.18 & %s \\" % ", ".join("%.2f" % x for x in o1))
    o2 = orders(t2["temporal"]["eur_uniform"], t2["temporal"]["Nt"])[1:]
    L.append(r"Example~\ref{ex3h}, $T=2$ & 0.18 & 1.18 & %s \\" % ", ".join("%.2f" % x for x in o2))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_beta", L)


def t_am_pointwise():
    d = load("american_pointwise")
    pts = [r"$0.5K$", r"$0.75K$", r"$K$", r"$1.25K$", r"$1.5K$"]
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r" & $S$ & Reference & MLS & Abs.\ error \\", r"\midrule"]
    for j, (ex, lab) in enumerate((("ex1", r"Ex.~\ref{ex3h}"), ("ex2", r"Ex.~\ref{ex5h}"))):
        if j:
            L.append(r"\midrule")
        p = d[ex]["pointwise"]
        for i, s in enumerate(pts):
            L.append("%s & %s & %.6f & %.6f & %s \\\\" % (lab if i == 0 else "", s, p["ref"][i], p["mls"][i],
                                                       sci(abs(p["ref"][i] - p["mls"][i]))))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_american_pointwise", L)


def t_ssn():
    rows = load("american_ssn")
    L = [r"\begin{tabular}{crrrrrrrrr}", r"\toprule",
         r"Ex. & $N_x$ & $N_t$ & avg.\ it. & max it. & $\|\Phi\|_\infty$ & red. & compl. & $\min(q-\eta)$ & $\min(\mathcal A\mathbf U-\mathbf b)$ \\", r"\midrule"]
    for r in rows:
        L.append("%s & %d & %d & %.2f & %d & %s & %d & %s & %s & %s \\\\" % (
            r["ex"], r["Nx"], r["Nt"], r["avg"], r["max"], sci(r["res"], 1), r["red"], sci(r["compl"], 1), sci(r["min_gap"], 1), sci(r["min_dual"], 1)))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ssn", L)


def t_greeks():
    rows = load("greeks")
    Nx = [r["Nx"] for r in rows]
    o = {k: orders([r[k] for r in rows], Nx) for k in ("delta", "gamma", "speed", "theta")}
    L = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
         r"$N_x$ & $\Delta$ & order & $\Gamma$ & order & Speed & order & $\Theta$ & order \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append("%d & " % r["Nx"] + " & ".join("%s & %s" % (sci(r[k]), od(o[k][i])) for k in ("delta", "gamma", "speed", "theta")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_greeks", L)


def t_sens():
    rows = load("sensitivity")
    Nx = [r["Nx"] for r in rows]
    o = {k: orders([r[k] for r in rows], Nx) for k in ("sigma", "H2")}
    L = [r"\begin{tabular}{rrrrr}", r"\toprule",
         r"$N_x$ & $\partial_\sigma u$ error & order & $\partial_{H_2}u$ error & order \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append("%d & %s & %s & %s & %s \\\\" % (r["Nx"], sci(r["sigma"]), od(o["sigma"][i]), sci(r["H2"]), od(o["H2"][i])))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sensitivity", L)


def t_efficiency():
    rows = load("efficiency")
    Nxs = sorted({r["Nx"] for r in rows})
    meths = ("FD uniform", "MLS uniform", "MLS clustered")
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{FD, uniform} & \multicolumn{2}{c}{MLS, uniform} & \multicolumn{2}{c}{MLS, clustered} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"$N_x$ & error & time (s) & error & time (s) & error & time (s) \\", r"\midrule"]
    for kind, lab in (("european", "European put"), ("american", "American put")):
        L.append(r"\multicolumn{7}{l}{\emph{%s}} \\" % lab)
        for Nx in Nxs:
            cells = []
            for m in meths:
                r = [x for x in rows if x["Nx"] == Nx and x["method"] == m and x["kind"] == kind][0]
                cells.append("%s & %.3f" % (sci(r["err"]), r["time"]))
            L.append("%d & " % Nx + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_efficiency", L)


def t_hurst():
    rows = load("hurst")
    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{$T=1$} & \multicolumn{4}{c}{$T=0.5$} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"$H$ & European & American & premium & $S_f^*(0)$ & European & American & premium & $S_f^*(0)$ \\", r"\midrule"]
    names = []
    for r in rows:
        if r["config"] not in names:
            names.append(r["config"])
    for nm in names:
        r1 = [r for r in rows if r["config"] == nm and r["T"] == 1.0][0]
        r2 = [r for r in rows if r["config"] == nm and r["T"] == 0.5][0]
        lab = {"BM": r"$(0.5)$ (Black--Scholes)", "Example 1": r"Example~\ref{ex3h}"}.get(nm, "$" + nm + "$")
        L.append("%s & %.4f & %.4f & %.4f & %.2f & %.4f & %.4f & %.4f & %.2f \\\\" % (
            lab, r1["eur"], r1["am"], r1["eep"], r1["Sf0"], r2["eur"], r2["am"], r2["eep"], r2["Sf0"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hurst", L)


if __name__ == "__main__":
    for f in (t_dissipativity, t_propagator, t_nodes, t_eur_pointwise, t_space, t_time, t_beta, t_am_pointwise,
              t_ssn, t_greeks, t_sens, t_efficiency, t_hurst):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("missing results for", f.__name__, e)
