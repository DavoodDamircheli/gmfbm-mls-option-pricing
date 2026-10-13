"""Effect of the Hurst exponents: European and American puts, early-exercise premium and the
exercise boundary at the valuation date, for T = 1 and T = 0.5 (Table 14)."""
import numpy as np

from common import GMFBM, MLSPricer, save, time_mesh

CONFIGS = [
    ("BM", (0.5,), (1.0,)),
    ("(0.5, 0.6)", (0.5, 0.6), (2 ** -0.5,) * 2),
    ("(0.5, 0.75)", (0.5, 0.75), (2 ** -0.5,) * 2),
    ("(0.5, 0.9)", (0.5, 0.9), (2 ** -0.5,) * 2),
    ("Example 1", (0.5, 0.59, 0.68, 0.77), (0.5,) * 4),
]


def exercise_boundary(p, U, eta, K, tol=1e-9):
    """Free boundary from the reconstructed prices: sqrt(q - eta) is linear in x near the
    boundary (smooth pasting); fit a line through the first two non-contact nodes."""
    gap = np.maximum(p.M @ U - eta, 0.0)
    contact = np.where((gap[1:-1] <= tol * K) & (eta[1:-1] > 0))[0] + 1
    if contact.size == 0:
        return np.nan
    i0 = contact.max()
    x1, x2 = p.x[i0 + 1], p.x[i0 + 2]
    y1, y2 = np.sqrt(gap[i0 + 1]), np.sqrt(gap[i0 + 2])
    xf = x1 - y1 * (x2 - x1) / (y2 - y1)
    return p.model.S_ref * np.exp(xf)


def run(Nx=800, Nt=800):
    rows = []
    for T in (1.0, 0.5):
        for name, H, al in CONFIGS:
            m = GMFBM(H=H, alpha=al, sigma=0.35, r=0.08, K=25.0, T=T)
            p = MLSPricer(m, Nx)
            tau = time_mesh(T, Nt, "twosided")
            E = p.eval_matrices([m.K])[0]
            eur = float(m.european(m.K, T, "put"))
            res = p.american(tau)
            am = float((E @ res["U"])[0])
            Sf = exercise_boundary(p, res["U"], p.payoff("american"), m.K)
            rows.append(dict(T=T, config=name, beta=m.beta(), total_var=2 * float(m.theta(T)),
                             eur=eur, am=am, eep=am - eur, Sf0=float(Sf)))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    save("hurst", run())
