"""Petla na zywo: predykcja nadchodzacych meczow -> zamrozenie -> rozliczenie z prawdziwym wynikiem."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .features import build_features
from .metrics import model_cols
from .models import Ensemble

LIVE_FILE = lambda: C.PRED / "live_predictions.csv"  # noqa: E731
MODELS = ["poisson", "elo_logit", "lgbm", "ensemble"]


def load_live() -> pd.DataFrame:
    p = LIVE_FILE()
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p, parse_dates=["kickoff", "created_at"])
    return pd.DataFrame()


def predict(matches: pd.DataFrame, fixtures: pd.DataFrame, now: datetime | None = None,
            horizon_days: float = 3.0) -> pd.DataFrame:
    """Zwraca zaktualizowana tabele predykcji. Predykcje meczow, ktore sie zaczely, sa zamrozone."""
    now = pd.Timestamp(now or datetime.now())
    if fixtures.empty:
        print("  Brak nadchodzacych meczow w zrodle (fixtures.csv ma je tylko w czesci tygodnia).")
        return load_live()
    fixtures = fixtures[~fixtures["match_id"].isin(matches["match_id"])]
    fixtures = fixtures[(fixtures["date"] > now) & (fixtures["date"] <= now + pd.Timedelta(days=horizon_days))]
    live = load_live()
    if fixtures.empty:
        print("  Brak nadchodzacych meczow w horyzoncie.")
        return live
    allrows = pd.concat([matches, fixtures], ignore_index=True)
    feat = build_features(allrows)
    played = feat[feat["y"].notna()]
    upcoming = feat[feat["match_id"].isin(fixtures["match_id"])].copy()
    model = Ensemble().fit(played)
    preds = model.predict(upcoming)
    new = upcoming[["match_id", "date", "div", "league", "home", "away", "lam_h", "lam_a"]].rename(columns={"date": "kickoff"})
    for name in MODELS:
        new[model_cols(name)] = preds[name]
    new["created_at"] = now
    new["fthg"] = np.nan
    new["ftag"] = np.nan
    new["y"] = np.nan
    if live.empty:
        return new
    frozen = live[(live["kickoff"] <= now) | (~live["match_id"].isin(new["match_id"]))]
    return pd.concat([frozen, new], ignore_index=True).sort_values("kickoff").reset_index(drop=True)


def settle(live: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Dopisuje prawdziwe wyniki do zamrozonych predykcji."""
    if live.empty:
        return live
    res = matches.set_index("match_id")[["fthg", "ftag"]]
    live = live.copy()
    m = live["match_id"].isin(res.index)
    live.loc[m, "fthg"] = res.loc[live.loc[m, "match_id"], "fthg"].to_numpy()
    live.loc[m, "ftag"] = res.loc[live.loc[m, "match_id"], "ftag"].to_numpy()
    done = live["fthg"].notna()
    live.loc[done, "y"] = np.where(live.loc[done, "fthg"] > live.loc[done, "ftag"], 0,
                                   np.where(live.loc[done, "fthg"] == live.loc[done, "ftag"], 1, 2))
    return live


def save(live: pd.DataFrame):
    C.PRED.mkdir(parents=True, exist_ok=True)
    live.to_csv(LIVE_FILE(), index=False)
