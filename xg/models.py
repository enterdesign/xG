"""Modele: prior, poisson (z sily atak/obrona), elo_logit, lgbm (+kalibracja temperaturowa), ensemble."""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLS
from .metrics import EPS


def _ll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1)).mean())


def _norm(P):
    P = np.clip(P, EPS, 1)
    return P / P.sum(1, keepdims=True)


def _temperature(P, T):
    return _norm(np.exp(np.log(np.clip(P, EPS, 1)) / T))


class Ensemble:
    """Uczy sie na danych treningowych (chronologicznie: 90% fit, 10% walidacja - early stopping,
    temperatura i wagi ensemble). Zadnej informacji z przyszlosci wzgledem zbioru testowego."""

    def __init__(self, cols: list[str] | None = None, half_life_days: float = 4 * 365, lgb_params: dict | None = None):
        self.cols = cols or FEATURE_COLS
        self.half_life = half_life_days
        self.params = dict(n_estimators=800, learning_rate=0.03, num_leaves=8, min_child_samples=60,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                           random_state=42, verbose=-1)
        self.params.update(lgb_params or {})
        self.weights = None

    def fit(self, train: pd.DataFrame) -> "Ensemble":
        train = train.sort_values("date")
        cut = int(len(train) * 0.9)
        tr, va = train.iloc[:cut], train.iloc[cut:]
        ytr, yva = tr["y"].to_numpy(int), va["y"].to_numpy(int)
        self.prior_ = np.bincount(ytr, minlength=3) / len(ytr)

        self.logit_ = LogisticRegression(C=1.0, max_iter=500).fit(tr[["elo_diff"]] / 100, ytr)

        age = (tr["date"].max() - tr["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / self.half_life)
        self.lgbm_ = lgb.LGBMClassifier(objective="multiclass", **self.params)
        self.lgbm_.fit(tr[self.cols], ytr, sample_weight=w, eval_set=[(va[self.cols], yva)],
                       callbacks=[lgb.early_stopping(40, verbose=False)])
        raw_va = self.lgbm_.predict_proba(va[self.cols])
        res = minimize_scalar(lambda T: _ll(_temperature(raw_va, T), yva), bounds=(0.6, 2.0), method="bounded")
        self.T_ = float(res.x)

        base = self._base(va)
        names = ["lgbm", "poisson", "elo_logit"]
        best, self.weights = 1e9, None
        for a in range(11):
            for b in range(11 - a):
                wt = (a / 10, b / 10, (10 - a - b) / 10)
                P = sum(wi * base[n] for wi, n in zip(wt, names))
                ll = _ll(P, yva)
                if ll < best:
                    best, self.weights = ll, dict(zip(names, wt))
        return self

    def _base(self, df):
        return {
            "prior": np.tile(self.prior_, (len(df), 1)),
            "poisson": _norm(df[["pp_h", "pp_d", "pp_a"]].to_numpy()),
            "elo_logit": self.logit_.predict_proba(df[["elo_diff"]] / 100),
            "lgbm": _temperature(self.lgbm_.predict_proba(df[self.cols]), self.T_),
        }

    def predict(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        base = self._base(df)
        base["ensemble"] = _norm(sum(self.weights[n] * base[n] for n in self.weights))
        return base

    def importance(self) -> list[dict]:
        gain = self.lgbm_.booster_.feature_importance("gain")
        tot = gain.sum() or 1
        return sorted(({"feature": c, "gain": float(g / tot)} for c, g in zip(self.cols, gain)),
                      key=lambda r: -r["gain"])
