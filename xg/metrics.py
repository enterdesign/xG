from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def metrics(P: np.ndarray, y: np.ndarray) -> dict:
    P = np.clip(np.asarray(P, float), EPS, 1)
    P = P / P.sum(axis=1, keepdims=True)
    y = np.asarray(y, int)
    onehot = np.eye(3)[y]
    cum = np.cumsum(P, 1) - np.cumsum(onehot, 1)
    return {
        "n": int(len(y)),
        "logloss": float(-np.log(P[np.arange(len(y)), y]).mean()),
        "brier": float(((P - onehot) ** 2).sum(1).mean()),
        "rps": float((cum[:, :2] ** 2).sum(1).mean() / 2),
        "acc": float((P.argmax(1) == y).mean()),
    }


def calibration(P: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Skalibrowanie dla wszystkich trzech wynikow razem: srednia przewidziana vs zaobserwowana czestosc."""
    P = np.asarray(P, float).ravel()
    hit = np.eye(3)[np.asarray(y, int)].ravel()
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (P >= a) & ((P < b) if b < 1 else (P <= b))
        if m.sum() >= 20:
            out.append({"pred": float(P[m].mean()), "obs": float(hit[m].mean()), "n": int(m.sum())})
    return out


def model_cols(name: str) -> list[str]:
    return [f"{name}_h", f"{name}_d", f"{name}_a"]


def table(df: pd.DataFrame, models: list[str], group: str | None = None) -> pd.DataFrame:
    rows = []
    for key, g in ([(None, df)] if group is None else df.groupby(group)):
        for m in models:
            cols = model_cols(m)
            if not set(cols) <= set(g.columns):
                continue
            sub = g.dropna(subset=cols)
            if len(sub) == 0:
                continue
            r = {"model": m, **metrics(sub[cols].to_numpy(), sub["y"].to_numpy())}
            if group is not None:
                r[group] = key
            rows.append(r)
    return pd.DataFrame(rows)
