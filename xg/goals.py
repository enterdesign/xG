"""Rynki golowe z oczekiwanych goli (Poisson): Over/Under 0.5/1.5/2.5/3.5, obie strzela, gol gospodarzy/gosci.

Uwaga: do GRY na tych rynkach potrzebne sa ich kursy (z wielu bukmacherow). Darmowe zrodla historyczne maja je tylko
dla Over/Under 2.5, wiec reszta rynkow jest tu wyceniana i sprawdzana pod katem kalibracji, ale nie 'obstawiana'.
"""
from __future__ import annotations

import numpy as np

LINES = (0.5, 1.5, 2.5, 3.5)


def _pmf(lam: np.ndarray, maxg: int = 12) -> np.ndarray:
    from math import factorial
    k = np.arange(maxg + 1)
    fact = np.array([factorial(int(i)) for i in k], dtype=float)
    lam = np.asarray(lam, float)[:, None]
    p = np.exp(-lam) * lam ** k / fact
    return p / p.sum(1, keepdims=True)


def goal_markets(lam_h, lam_a, maxg: int = 12) -> dict[str, np.ndarray]:
    ph, pa = _pmf(lam_h, maxg), _pmf(lam_a, maxg)
    joint = ph[:, :, None] * pa[:, None, :]
    total = np.zeros((len(ph), 2 * maxg + 1))
    for i in range(maxg + 1):
        total[:, i:i + maxg + 1] += ph[:, i:i + 1] * pa
    out = {}
    for line in LINES:
        over = total[:, int(line) + 1:].sum(1)
        out[f"ou{line}_over"], out[f"ou{line}_under"] = over, 1 - over
    btts = joint[:, 1:, 1:].sum((1, 2))
    out["btts_yes"], out["btts_no"] = btts, 1 - btts
    out["home_scores"], out["home_no_score"] = 1 - ph[:, 0], ph[:, 0]
    out["away_scores"], out["away_no_score"] = 1 - pa[:, 0], pa[:, 0]
    return out


class LambdaCalibrator:
    """Koryguje przesadzone oczekiwane gole: log(lam') = a + b*log(lam), osobno dla gospodarzy i gosci
    (regresja Poissona). Uczona wylacznie na przeszlosci."""

    def fit(self, lam_h, lam_a, goals_h, goals_a) -> "LambdaCalibrator":
        from sklearn.linear_model import PoissonRegressor
        self.mh = PoissonRegressor(alpha=0, max_iter=300).fit(np.log(np.asarray(lam_h))[:, None], goals_h)
        self.ma = PoissonRegressor(alpha=0, max_iter=300).fit(np.log(np.asarray(lam_a))[:, None], goals_a)
        return self

    def transform(self, lam_h, lam_a):
        return (self.mh.predict(np.log(np.asarray(lam_h))[:, None]), self.ma.predict(np.log(np.asarray(lam_a))[:, None]))
