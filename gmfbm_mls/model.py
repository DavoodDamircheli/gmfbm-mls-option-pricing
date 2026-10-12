"""
GMFBM pricing model in log-price / forward-time variables (Section 2).

    u_tau = a(tau) (u_xx - u_x) + (r - delta) u_x - r u,     x = ln(S/K), tau = T - t,
    a(tau) = sigma^2 sum_i H_i alpha_i^2 (T - tau)^(2 H_i - 1).

The European price has the closed form of Proposition 1 (Black--Scholes with total
variance v(tau) = 2 int_0^tau a = sigma^2 sum_i alpha_i^2 (T^{2H_i} - (T-tau)^{2H_i})).
It is used as the exact benchmark for every European experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GMFBM:
    K: float = 25.0
    T: float = 1.0
    r: float = 0.08
    delta: float = 0.0
    sigma: float = 0.35
    H: tuple = (0.5, 0.59, 0.68, 0.77)
    alpha: tuple = (0.5, 0.5, 0.5, 0.5)

    def with_(self, **kw) -> "GMFBM":
        return replace(self, **kw)

    # -- coefficient and its parameter derivatives --------------------------
    def a(self, tau: float) -> float:
        s = max(self.T - tau, 0.0)
        tot = 0.0
        for H, al in zip(self.H, self.alpha):
            e = 2.0 * H - 1.0
            tot += H * al**2 * (1.0 if abs(e) < 1e-14 else (s**e if s > 0 else 0.0))
        return self.sigma**2 * tot

    def a_param(self, tau: float, p: str) -> float:
        """d a / d p for p in {'sigma', 'H<i>', 'alpha<i>'} (alpha_i independent of H_i)."""
        s = max(self.T - tau, 0.0)
        if p == "sigma":
            return 2.0 * self.a(tau) / self.sigma
        kind, i = p.rstrip("0123456789"), int(p.lstrip("Halph"))
        H, al = self.H[i], self.alpha[i]
        e = 2.0 * H - 1.0
        if kind == "alpha":
            return 2.0 * self.sigma**2 * H * al * (1.0 if abs(e) < 1e-14 else (s**e if s > 0 else 0.0))
        if kind == "H":
            if s == 0.0:
                if e <= 0:
                    raise ValueError("d a/dH_i is singular at tau = T when H_i <= 1/2")
                return 0.0
            return self.sigma**2 * al**2 * s**e * (1.0 + 2.0 * H * np.log(s))
        raise ValueError(p)

    def beta(self) -> float:
        b = [2 * H - 1 for H, al in zip(self.H, self.alpha) if H > 0.5 and al != 0]
        return min(b) if b else 1.0

    # -- closed-form European prices (Proposition 1) -------------------------
    def total_variance(self, tau: float) -> float:
        H = np.asarray(self.H)
        al = np.asarray(self.alpha)
        return self.sigma**2 * float(np.sum(al**2 * (self.T ** (2 * H) - max(self.T - tau, 0.0) ** (2 * H))))

    def dvar_dparam(self, tau: float, p: str) -> float:
        s = max(self.T - tau, 0.0)
        if p == "sigma":
            return 2.0 * self.total_variance(tau) / self.sigma
        kind, i = p.rstrip("0123456789"), int(p.lstrip("Halph"))
        H, al = self.H[i], self.alpha[i]
        if kind == "alpha":
            return 2.0 * self.sigma**2 * al * (self.T ** (2 * H) - s ** (2 * H))
        sl = s ** (2 * H) * np.log(s) if s > 0 else 0.0
        return self.sigma**2 * al**2 * 2.0 * (self.T ** (2 * H) * np.log(self.T) - sl)

    def _d(self, S, tau):
        v = self.total_variance(tau)
        sv = np.sqrt(v)
        d1 = (np.log(S / self.K) + (self.r - self.delta) * tau + 0.5 * v) / sv
        return d1, d1 - sv, v, sv

    def european(self, S, tau, kind="put"):
        S = np.asarray(S, dtype=float)
        if tau <= 0:
            return np.maximum(self.K - S, 0) if kind == "put" else np.maximum(S - self.K, 0)
        d1, d2, v, sv = self._d(S, tau)
        dfr, dfq = np.exp(-self.r * tau), np.exp(-self.delta * tau)
        if kind == "put":
            return self.K * dfr * norm.cdf(-d2) - S * dfq * norm.cdf(-d1)
        return S * dfq * norm.cdf(d1) - self.K * dfr * norm.cdf(d2)

    def european_greeks(self, S, tau, kind="put"):
        """Exact Delta, Gamma, Speed, Theta (calendar time) for the closed form."""
        S = np.asarray(S, dtype=float)
        d1, d2, v, sv = self._d(S, tau)
        dfq = np.exp(-self.delta * tau)
        n1 = norm.pdf(d1)
        delta = dfq * (norm.cdf(d1) - (1.0 if kind == "put" else 0.0))
        gamma = dfq * n1 / (S * sv)
        speed = -gamma / S * (1.0 + d1 / sv)
        u = self.european(S, tau, kind)
        a = self.a(tau)
        ux = S * delta
        uxx = S**2 * gamma + S * delta
        u_tau = a * (uxx - ux) + (self.r - self.delta) * ux - self.r * u
        return {"Delta": delta, "Gamma": gamma, "Speed": speed, "Theta": -u_tau}

    def european_param_sensitivity(self, S, tau, p, kind="put"):
        """Exact d u / d p through the total variance (same for calls and puts)."""
        S = np.asarray(S, dtype=float)
        d1, d2, v, sv = self._d(S, tau)
        dP_dv = S * np.exp(-self.delta * tau) * norm.pdf(d1) / (2.0 * sv)
        return dP_dv * self.dvar_dparam(tau, p)


EXAMPLE1 = GMFBM()
EXAMPLE2 = GMFBM(K=100.0, r=0.10, sigma=0.50, H=(0.5, 0.59, 0.68, 0.77, 0.86),
                 alpha=tuple([1 / np.sqrt(5)] * 5))
