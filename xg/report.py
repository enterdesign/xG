"""Buduje web/data.js - dane dla strony (otwierasz web/index.html dwuklikiem, bez serwera)."""
from __future__ import annotations

import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C
from .metrics import calibration, metrics, model_cols, table

BT_FILE = lambda: C.PRED / "backtest.csv"  # noqa: E731
IMP_FILE = lambda: C.PRED / "importance.json"  # noqa: E731
ABL_FILE = lambda: C.PRED / "ablation.csv"  # noqa: E731
BT_MODELS = ["prior", "poisson", "elo_logit", "lgbm", "ensemble", "market"]


def _records(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", date_format="iso"))


def build(live: pd.DataFrame, n_matches: int, synthetic: bool = False, now: datetime | None = None) -> dict:
    now = pd.Timestamp(now or datetime.now())
    data: dict = {"meta": {"generated_at": now.isoformat(timespec="seconds"), "synthetic": synthetic,
                           "n_matches": int(n_matches), "models": BT_MODELS}}
    if BT_FILE().exists():
        bt = pd.read_csv(BT_FILE(), parse_dates=["date"])
        bt["season"] = np.where(bt["date"].dt.month >= 7, bt["date"].dt.year, bt["date"].dt.year - 1)
        bt["season"] = bt["season"].astype(str) + "/" + ((bt["season"] + 1) % 100).astype(int).astype(str).str.zfill(2)
        mk = bt.dropna(subset=model_cols("market"))
        bk = {"overall": _records(table(bt, BT_MODELS)),
              "vs_market": _records(table(mk, BT_MODELS)) if len(mk) else [],
              "by_league": _records(table(bt, BT_MODELS, "league")),
              "by_season": _records(table(bt, BT_MODELS, "season"))}
        bt["month"] = bt["date"].dt.strftime("%Y-%m")
        monthly = []
        for mth, g in bt.groupby("month"):
            row = {"month": mth, "n": len(g)}
            for m in ["ensemble", "poisson", "market"]:
                sub = g.dropna(subset=model_cols(m))
                row[m] = metrics(sub[model_cols(m)].to_numpy(), sub["y"].to_numpy())["logloss"] if len(sub) >= 10 else None
            monthly.append(row)
        bk["monthly"] = monthly
        bk["calibration"] = calibration(bt[model_cols("ensemble")].to_numpy(), bt["y"].to_numpy())
        bk["period"] = [bt["date"].min().date().isoformat(), bt["date"].max().date().isoformat()]
        data["backtest"] = bk
    if IMP_FILE().exists():
        data["importance"] = json.loads(IMP_FILE().read_text())
    if ABL_FILE().exists():
        data["ablation"] = _records(pd.read_csv(ABL_FILE()))
    for key, fname in (("replay", "replay.json"), ("goals", "goals_eval.json")):
        fp = C.PRED / fname
        if fp.exists():
            data[key] = json.loads(fp.read_text())
    live_out = {"upcoming": [], "settled": [], "summary": {}}
    if not live.empty:
        up = live[live["y"].isna() & (live["kickoff"] > now - pd.Timedelta(hours=3))].sort_values("kickoff")
        st = live[live["y"].notna()].sort_values("kickoff", ascending=False)
        cols = ["match_id", "kickoff", "league", "home", "away", "lam_h", "lam_a", "fthg", "ftag", "y"] + \
               [c for m in ["ensemble", "poisson", "elo_logit", "lgbm"] for c in model_cols(m)]
        live_out["upcoming"] = _records(up[cols])
        live_out["settled"] = _records(st[cols].head(300))
        if len(st) >= 1:
            live_out["summary"] = {m: metrics(st[model_cols(m)].to_numpy(), st["y"].to_numpy())
                                   for m in ["ensemble", "poisson", "elo_logit", "lgbm"]}
    data["live"] = live_out
    return data


def write(data: dict):
    C.WEB.mkdir(parents=True, exist_ok=True)
    (C.WEB / "data.js").write_text("window.XG_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    src = C.REPO / "web" / "index.html"
    dst = C.WEB / "index.html"
    if src.resolve() != dst.resolve():
        shutil.copy(src, dst)
