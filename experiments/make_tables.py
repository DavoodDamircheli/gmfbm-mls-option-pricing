#!/usr/bin/env python3
"""Build the LaTeX tables of Section 9 from results/*.json and insert them into a
manuscript at the markers  %%TABLE:<name>%%  (python make_tables.py main.tex out.tex),
or write them to tables/<name>.tex (python make_tables.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def J(name):
    return json.loads((RES / f"{name}.json").read_text())


def sci(v, d=2):
    if v is None:
        return "--"
    m, e = f"{v:.{d}e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def o(v):
    return "--" if v is None else f"{v:.2f}"


def eo(e, ordv):
    return f"{sci(e)} & {o(ordv)}"


def tab(label, caption, colspec, header, rows, size=r"\small"):
    body = "\n".join(r + r" \\" for r in rows)
    return (f"\\begin{{table}}[htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{table}}")


def t_european_pointwise():
    p = J("euro_space")["pointwise_production"]
    S = ["0.5K", "0.75K", "K", "1.25K", "1.5K"]
    rows = [f"${s}$ & {p['put']['exact'][i]:.6f} & {p['put']['mls'][i]:.6f} & {sci(p['put']['abs_err'][i])} & "
            f"{p['call']['exact'][i]:.6f} & {p['call']['mls'][i]:.6f} & {sci(p['call']['abs_err'][i])}"
            for i, s in enumerate(S)]
    return tab("tab:ex1_european_pointwise",
               r"Example~\ref{ex3h}, European options at $t=0$: exact price (Proposition~\ref{nw:prop:closed_form}) "
               r"and MLS price on the production grid ($N_x=800$, $N_t=800$, two-sided graded mesh, "
               r"financial boundary data).",
               "rrrrrrr", r"$S$ & Put (exact) & Put (MLS) & Abs.\ error & Call (exact) & Call (MLS) & Abs.\ error",
               rows, size=r"\footnotesize")


def t_american_pointwise():
    a = J("american")
    S = ["0.5K", "0.75K", "K", "1.25K", "1.5K"]
    rows = []
    for ex, lab in (("Example 1", r"Ex.~\ref{ex3h}"), ("Example 2", r"Ex.~\ref{ex5h}")):
        p = a[ex]["pointwise"]
        for i, s in enumerate(S):
            rows.append(f"{lab if i == 0 else ''} & ${s}$ & {p['benchmark'][i]:.6f} & {p['mls'][i]:.6f} & "
                        f"{sci(p['abs_err'][i])}")
        if ex == "Example 1":
            rows.append(r"\midrule")
    rows = [r for r in rows]
    body = []
    for r in rows:
        body.append(r if r == r"\midrule" else r)
    cap = (r"American put at $t=0$: reference price (centred finite differences, $12000$ subintervals, "
           r"$4000$ steps, Brennan--Schwartz) and MLS price on the production grid ($N_x=800$, $N_t=800$). "
           r"Estimated reference error (Richardson, $\|u_{12000}-u_{6000}\|_\infty/3$): "
           f"{sci(a['Example 1']['benchmark_uncertainty_over_K'] * 25, 1)} (Ex.~\\ref{{ex3h}}) and "
           f"{sci(a['Example 2']['benchmark_uncertainty_over_K'] * 100, 1)} (Ex.~\\ref{{ex5h}}).")
    txt = "\n".join((r if r == r"\midrule" else r + r" \\") for r in body)
    return (f"\\begin{{table}}[htbp]\n\\centering\n\\small\n\\caption{{{cap}}}\n\\label{{tab:american_pointwise}}\n"
            f"\\begin{{tabular}}{{llrrr}}\n\\toprule\n & $S$ & Reference & MLS & Abs.\\ error \\\\\n\\midrule\n"
            f"{txt}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}")


def t_space():
    e = J("euro_space")
    a = J("american")["Example 1"]["space"]
    rows = []
    for i, nx in enumerate(e["nx"]):
        rows.append(f"{nx} & {e['h'][i]:.4f} & {eo(e['put'][i], e['put_order'][i])} & "
                    f"{eo(e['call'][i], e['call_order'][i])} & {eo(a['err'][i], a['order'][i])}")
    return tab("tab:ex1_space",
               r"Example~\ref{ex3h}: spatial convergence at $\tau=T$, $\max_{0.2K\le S\le2.5K}|u-u_h|/K$. "
               r"European options against the exact price (exact Dirichlet traces); American put against the "
               r"finite-difference reference.  Time mesh: two-sided graded, $N_t=1600$.",
               "rrrrrrrr", r"$N_x$ & $h$ & Eur.\ put & order & Eur.\ call & order & Am.\ put & order", rows,
               size=r"\footnotesize")


def t_time():
    e = J("euro_time")
    a = J("american")["Example 1"]["time"]
    rows = []
    for i, N in enumerate(e["N"]):
        def c(errs, ords):
            return f"{sci(errs[i])} ({o(ords[i])})"
        rows.append(f"{N} & {c(e['uniform'], e['uniform_order'])} & {c(e['uniform_rannacher'], e['uniform_rannacher_order'])} & "
                    f"{c(e['graded'], e['graded_order'])} & {c(a['uniform'], a['uniform_order'])} & "
                    f"{c(a['two_sided'], a['two_sided_order'])}")
    return tab("tab:ex1_time",
               r"Example~\ref{ex3h}: temporal error at the valuation date $\tau=T$ ($N_x=400$; same-grid reference "
               r"with $N_t=12800$), $\max_{0.2K\le S\le2.5K}|u-u_h|/K$ with observed orders in parentheses. "
               r"European put: uniform steps, uniform steps with a Rannacher start, and the graded mesh "
               r"$\tau_n=T(1-(1-n/N_t)^2)$.  American put: uniform steps and the two-sided graded mesh.",
               "rlllll", r"$N_t$ & Eur.\ uniform & Eur.\ Rannacher & Eur.\ graded & Am.\ uniform & Am.\ two-sided",
               rows, size=r"\scriptsize")


def t_beta():
    e = J("euro_time")
    rows = []
    for b in e["beta_table"]:
        rows.append(f"$({b['H'][0]},\\,{b['H'][1]})$ & {b['beta']:.2f} & {1 + b['beta']:.2f} & "
                    + ", ".join(o(x) for x in b["orders"]))
    unif = e["uniform_order"][2:]
    rows.append(f"Example~\\ref{{ex3h}} & {e['beta']:.2f} & {1 + e['beta']:.2f} & " + ", ".join(o(x) for x in unif))
    return tab("tab:beta",
               r"Observed valuation-date orders of the averaged Crank--Nicolson scheme with uniform steps "
               r"against the prediction $1+\beta$ of Theorem~\ref{nw:thm:endpoint}.  Two-component rows: "
               r"$\alpha=(0.5,0.5)$, $N_x=200$, $N_t=100\to1600$; Example~\ref{ex3h}: $N_x=400$, $N_t=100\to1600$.",
               "lccl", r"$H$ & $\beta$ & $1+\beta$ & observed pairwise orders", rows)


def t_dissipativity():
    d = J("dissipativity")
    by = {}
    for r in d["production"]:
        by.setdefault(r["nx"], {})[r["example"]] = r
    rows = []
    for nx in sorted(by):
        r1, r2 = by[nx]["Example 1"], by[nx]["Example 2"]
        rows.append(f"{nx} & {r1['h']:.4f} & ${r1['lam_max']:.5f}$ & ${r2['lam_max']:.5f}$ & {r1['norm_M']:.3f} & "
                    f"{r1['norm_Minv']:.2f} & {r1['C_D']:.2f}")
    return tab("tab:dissipativity",
               r"Assumption~\ref{st:assump:dissipativity} for the assembled non-cardinal MLS operator "
               r"($m=3$, support radius $3.5h$): $\max_n\lambda_{\max}(S_n^h)$ over all time levels, the mass-matrix "
               r"bounds of Assumption~\ref{st:assump:mass} and the constant $C_D=h\|M_h^{-1}D_{1,h}\|_{h\to h}$.",
               "rrrrrrr", r"$N_x$ & $h$ & $\max_n\lambda_{\max}$ (Ex.~1) & $\max_n\lambda_{\max}$ (Ex.~2) & "
                          r"$\|M_h\|_h$ & $\|M_h^{-1}\|_h$ & $C_D$", rows, size=r"\footnotesize")


def t_propagator():
    rows = []
    for r in J("propagator"):
        rows.append(f"{r['nx']} & {r['nt']} & {r['kmax_over_h']:.3f} & {r['max_norm']:.7f} & {r['max_rho']:.7f} & "
                    f"{r['Theta']:.3f} & {r['bound']:.3f}")
    return tab("tab:propagator",
               r"Example~\ref{ex3h}: fully discrete propagator $P_n=(I-s_nA_{n+1}^h)^{-1}(I+s_nA_n^h)$ on the "
               r"two-sided graded mesh, maximum over all steps, compared with the bound "
               r"$1+\Theta\max_n|\rho_n|$ of Theorem~\ref{nw:thm:stability}.",
               "rrrrrrr", r"$N_x$ & $N_t$ & $\max_nk_n/h$ & $\max_n\|P_n\|_h$ & $\max_n\rho(P_n)$ & $\Theta$ & "
                          r"bound", rows, size=r"\footnotesize")


def t_ssn():
    a = J("american")
    rows = []
    for ex, lab in (("Example 1", "1"), ("Example 2", "2")):
        for r in a[ex]["diagnostics"]:
            rows.append(f"{lab} & {r['nx']} & {r['nt']} & {r['avg_iterations']:.2f} & {r['max_iterations']} & "
                        f"{sci(r['max_final_residual'], 1)} & {r['linesearch_reductions']} & "
                        f"{sci(r['complementarity'], 1)} & {sci(min(r['min_q_minus_phi'], 0.0), 1) if r['min_q_minus_phi'] < 0 else '0'}")
    return tab("tab:ssn",
               r"American put: Fischer--Burmeister semismooth Newton diagnostics over all time steps (average and "
               r"maximum iterations, largest final residual $\|\Phi\|_\infty$, number of Armijo step reductions) and "
               r"final-step complementarity $\|(M_h\mathbf U-\boldsymbol\eta)\odot(\mathcal A_n\mathbf U-\mathbf b^n)\|_\infty$ "
               r"and $\min(M_h\mathbf U-\boldsymbol\eta)$ on the interior nodes.  Two-sided graded time mesh.",
               "crrrrrrrr", r"Ex. & $N_x$ & $N_t$ & avg.\ it. & max it. & $\|\Phi\|_\infty$ & reductions & "
                            r"compl. & $\min(q-\eta)$", rows, size=r"\scriptsize")


def t_greeks():
    e = J("euro_space")
    rows = []
    for i, nx in enumerate(e["nx"]):
        rows.append(f"{nx} & " + " & ".join(eo(e[k][i], e[k + '_order'][i]) for k in ("Delta", "Gamma", "Speed", "Theta")))
    return tab("tab:greeks",
               r"Example~\ref{ex3h}, European put at $t=0$: maximum error of the MLS Greeks over "
               r"$0.2K\le S\le2.5K$ against the exact Greeks (two-sided graded mesh, $N_t=1600$).",
               "rrrrrrrrr", r"$N_x$ & $\Delta$ & order & $\Gamma$ & order & Speed & order & $\Theta$ & order", rows,
               size=r"\scriptsize")


def t_sens():
    s = J("sensitivity")
    ps, ph = s["params"]["sigma"], s["params"]["H1"]
    rows = []
    for i in range(len(ps["err"])):
        rows.append(f"{[100, 200, 400, 800][i]} & {eo(ps['err'][i], ps['order'][i])} & {eo(ph['err'][i], ph['order'][i])}")
    return tab("tab:sensitivity",
               r"Example~\ref{ex3h}, European put at time to maturity $\tau=T/2$: maximum error over "
               r"$0.2K\le S\le2.5K$ of the tangent-equation sensitivities $\partial u/\partial\sigma$ and "
               r"$\partial u/\partial H_2$ against the exact derivatives of Proposition~\ref{nw:prop:closed_form} "
               r"($N_t=800$, two-sided graded mesh).",
               "rrrrr", r"$N_x$ & $\partial_\sigma u$ error & order & $\partial_{H_2}u$ error & order", rows)


TABLES = {"european_pointwise": t_european_pointwise, "american_pointwise": t_american_pointwise,
          "space": t_space, "time": t_time, "beta": t_beta, "dissipativity": t_dissipativity,
          "propagator": t_propagator, "ssn": t_ssn, "greeks": t_greeks, "sensitivity": t_sens}

if __name__ == "__main__":
    if len(sys.argv) == 3:
        src = Path(sys.argv[1]).read_text()
        for k, f in TABLES.items():
            src = src.replace(f"%%TABLE:{k}%%", f())
        Path(sys.argv[2]).write_text(src)
        print("tables inserted ->", sys.argv[2])
    else:
        out = ROOT / "tables"
        out.mkdir(exist_ok=True)
        for k, f in TABLES.items():
            (out / f"{k}.tex").write_text(f() + "\n")
        print("wrote tables/*.tex")
