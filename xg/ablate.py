"""Analiza ablacyjna: o ile pogarsza sie model po usunieciu grupy cech (= ile ta grupa realnie wnosi)."""
from __future__ import annotations

import pandas as pd

from .backtest import walk_forward
from .features import FEATURE_COLS, FEATURE_GROUPS
from .metrics import metrics, model_cols


def _ll(bt):
    sub = bt.dropna(subset=model_cols("lgbm"))
    return metrics(sub[model_cols("lgbm")].to_numpy(), sub["y"].to_numpy())["logloss"]


def run(feat: pd.DataFrame, test_from=None, step_days: int = 84) -> pd.DataFrame:
    print("Ablacja: pelny zestaw cech")
    full, _ = walk_forward(feat, test_from, step_days, verbose=False)
    base = _ll(full)
    rows = []
    for g, cols in FEATURE_GROUPS.items():
        print(f"Ablacja: bez grupy '{g}'")
        keep = [c for c in FEATURE_COLS if c not in cols]
        bt, _ = walk_forward(feat, test_from, step_days, cols=keep, verbose=False)
        rows.append({"group": g, "logloss_without": _ll(bt), "delta_logloss": _ll(bt) - base})
    res = pd.DataFrame(rows).sort_values("delta_logloss", ascending=False)
    res.attrs["base"] = base
    return res
