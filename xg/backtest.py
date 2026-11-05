"""Walk-forward: dla kazdego okresu uczymy modele TYLKO na danych sprzed jego poczatku."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .metrics import model_cols
from .models import Ensemble


def walk_forward(feat: pd.DataFrame, test_from: str | pd.Timestamp | None = None, step_days: int = 28,
                 cols: list[str] | None = None, verbose: bool = True):
    played = feat[feat["y"].notna()].sort_values("date").reset_index(drop=True)
    start = pd.Timestamp(test_from) if test_from else played["date"].min() + pd.Timedelta(days=3 * 365)
    end = played["date"].max() + pd.Timedelta(days=1)
    edges = list(pd.date_range(start, end, freq=f"{step_days}D"))
    if edges[-1] < end:
        edges.append(end)
    out, last = [], None
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        train = played[played["date"] < a]
        test = played[(played["date"] >= a) & (played["date"] < b)]
        if len(test) == 0 or len(train) < 1500:
            continue
        model = Ensemble(cols=cols).fit(train)
        preds = model.predict(test)
        part = test[["match_id", "date", "div", "league", "home", "away", "fthg", "ftag", "y", "mp_h", "mp_d", "mp_a"]].copy()
        for name, P in preds.items():
            part[model_cols(name)] = P
        part["fold"] = k
        out.append(part)
        last = model
        if verbose:
            print(f"  fold {k:3d} {a.date()} train={len(train):6d} test={len(test):4d} w={model.weights}")
    if not out:
        raise SystemExit("Za malo danych do backtestu - pobierz wiecej sezonow.")
    bt = pd.concat(out, ignore_index=True)
    bt = bt.rename(columns={"mp_h": "market_h", "mp_d": "market_d", "mp_a": "market_a"})
    return bt, last
