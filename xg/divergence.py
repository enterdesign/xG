"""Czy rozbieznosc modelu z rynkiem cos znaczy? Dwa testy na wynikach backtestu (walk-forward):

1. Mieszanka: P = (1-w)*rynek + w*model. Jesli optymalne w > 0 i log-loss mieszanki jest istotnie lepszy od
   samego rynku (bootstrap po meczach), to model wnosi informacje, ktorej rynek nie ma.
2. Biny rozbieznosci: dla (model - rynek) w roznych przedzialach porownujemy prawdziwa czestosc zdarzenia
   z prawdopodobienstwem modelu i rynku. Kto jest blizej prawdy?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import EPS

EDGES = [-1, -0.10, -0.05, -0.02, 0.02, 0.05, 0.10, 1]


def _ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))


def analyze(bt: pd.DataFrame, model: str = "ensemble", seed: int = 0, n_boot: int = 2000) -> dict:
    df = bt.dropna(subset=["market_h", f"{model}_h"]).copy()
    y = df["y"].to_numpy(int)
    M = df[["market_h", "market_d", "market_a"]].to_numpy()
    P = df[[f"{model}_h", f"{model}_d", f"{model}_a"]].to_numpy()
    M = M / M.sum(1, keepdims=True)

    ll_m = _ll(M, y)
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    mean_ll = [float(_ll((1 - w) * M + w * P, y).mean()) for w in ws]
    best = int(np.argmin(mean_ll))
    w_best = float(ws[best])
    diff = _ll((1 - w_best) * M + w_best * P, y) - ll_m  # <0 = mieszanka lepsza od rynku
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    blend = {
        "n": int(len(df)), "best_w": w_best, "logloss_market": float(ll_m.mean()),
        "logloss_model": float(_ll(P, y).mean()), "logloss_blend": mean_ll[best],
        "gain": float(diff.mean()), "gain_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "curve": [{"w": float(w), "logloss": l} for w, l in zip(ws, mean_ll)],
    }
    blend["significant"] = bool(blend["gain_ci"][1] < 0)

    # biny rozbieznosci - wszystkie trzy wyniki razem
    gap = (P - M).ravel()
    pm, pk = P.ravel(), M.ravel()
    hit = np.eye(3)[y].ravel()
    bins = []
    cut = pd.cut(gap, EDGES)
    for iv in cut.categories:
        m = np.asarray(cut == iv)
        if m.sum() < 50:
            continue
        bins.append({"lo": float(iv.left), "hi": float(iv.right), "n": int(m.sum()),
                     "model": float(pm[m].mean()), "market": float(pk[m].mean()), "observed": float(hit[m].mean()),
                     "observed_se": float(np.sqrt(hit[m].mean() * (1 - hit[m].mean()) / m.sum()))})
    return {"blend": blend, "bins": bins, "model": model}
