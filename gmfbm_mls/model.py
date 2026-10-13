"""GMFBM model: diffusion coefficient, closed-form European prices and Greeks, time meshes."""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass
class GMFBM:
    H: tuple
    alpha: tuple
    sigma: float
    r: float
    K: float
    T: float = 1.0
    delta: float = 0.0  # dividend yield
    S_ref: float = None  # reference price S_* (default K)

    def __post_init__(self):
        self.H = np.asarray(self.H, dtype=float)
        self.alpha = np.asarray(self.alpha, dtype=float)
        if self.S_ref is None:
            self.S_ref = self.K

    # a(tau) = sigma^2 sum H_i alpha_i^2 (T - tau)^(2H_i - 1)
    def a(self, tau):
        t = np.maximum(self.T - np.asarray(tau, dtype=float), 0.0)
        t = np.atleast_1d(t)
        out = np.zeros_like(t)
        for h, al in zip(self.H, self.alpha):
            e = 2 * h - 1
            out += h * al ** 2 * (np.ones_like(t) if e == 0 else t ** e)
        out *= self.sigma ** 2
        return out if np.ndim(tau) else float(out[0])

    def theta(self, tau):
        """theta(tau) = int_0^tau a(s) ds (half the total variance)."""
        tau = np.asarray(tau, dtype=float)
        v = sum(al ** 2 * (self.T ** (2 * h) - np.maximum(self.T - tau, 0) ** (2 * h))
                for h, al in zip(self.H, self.alpha))
        return 0.5 * self.sigma ** 2 * v

    def beta(self):
        b = [2 * h - 1 for h, al in zip(self.H, self.alpha) if h > 0.5 and al != 0]
        return min(b) if b else 1.0

    # closed-form European prices (Proposition 2.1): Black-Scholes with total variance 2 theta
    def european(self, S, tau, kind="put"):
        S = np.asarray(S, dtype=float)
        K, r, q = self.K, self.r, self.delta
        if np.isscalar(tau) and tau == 0:
            return np.maximum(K - S, 0) if kind == "put" else np.maximum(S - K, 0)
        v = np.sqrt(2 * self.theta(tau))
        d1 = (np.log(S / K) + (r - q) * tau + 0.5 * v ** 2) / v
        d2 = d1 - v
        if kind == "put":
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    def european_greeks(self, S, tau, kind="put"):
        """Delta, Gamma, Speed, Theta (calendar, = -u_tau) of the closed-form price."""
        S = np.asarray(S, dtype=float)
        K, r, q = self.K, self.r, self.delta
        v = np.sqrt(2 * self.theta(tau))
        d1 = (np.log(S / K) + (r - q) * tau + 0.5 * v ** 2) / v
        d2 = d1 - v
        nd1 = norm.pdf(d1)
        if kind == "put":
            delta = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        else:
            delta = np.exp(-q * tau) * norm.cdf(d1)
        gamma = np.exp(-q * tau) * nd1 / (S * v)
        speed = -gamma / S * (1 + d1 / v)
        # Theta via the PDE: u_tau = a S^2 Gamma + (r-q) S Delta - r u
        u = self.european(S, tau, kind)
        a = self.a(tau)
        theta_cal = -(a * S ** 2 * gamma + (r - q) * S * delta - r * u)
        return delta, gamma, speed, theta_cal


def time_mesh(T, Nt, kind="uniform"):
    s = np.linspace(0.0, 1.0, Nt + 1)
    if kind == "uniform":
        tau = T * s
    elif kind == "graded":  # graded towards tau = T
        tau = T * (1 - (1 - s) ** 2)
    elif kind == "twosided":  # graded towards tau = 0 and tau = T
        tau = T * s ** 2 * (3 - 2 * s)
    else:
        raise ValueError(kind)
    tau[0], tau[-1] = 0.0, T
    return tau


def eval_points(K, n=401, lo=0.2, hi=2.5):
    """Error-measurement set: ln(S/K) uniform in [ln lo, ln hi]."""
    return K * np.exp(np.linspace(np.log(lo), np.log(hi), n))


EXAMPLE1 = dict(H=(0.5, 0.59, 0.68, 0.77), alpha=(0.5,) * 4, sigma=0.35, r=0.08, K=25.0, T=1.0)
EXAMPLE2 = dict(H=(0.5, 0.59, 0.68, 0.77, 0.86), alpha=(1 / np.sqrt(5),) * 5, sigma=0.5, r=0.10,
                K=100.0, T=1.0)
