"""Podglad na danych SYNTETYCZNYCH (nie prawdziwych!). Uzycie: XG_OUT=/sciezka python scripts/demo.py"""
import numpy as np
import pandas as pd

from xg import cli, config as C, live as L, report as R
from xg.backtest import walk_forward
from xg.features import build_features
from xg.synth import make_matches
import json

all_m = make_matches(n_seasons=7, teams_per_league=12, seed=3)
cut = all_m["date"].max() - pd.Timedelta(days=60)
played = all_m[all_m["date"] <= cut]
fx = all_m[all_m["date"] > cut].copy(); fx[["fthg", "ftag"]] = np.nan
feat = build_features(played)
bt, last = walk_forward(feat, step_days=56, verbose=False)
C.PRED.mkdir(parents=True, exist_ok=True)
bt.to_csv(R.BT_FILE(), index=False); R.IMP_FILE().write_text(json.dumps(last.importance()))
now = cut + pd.Timedelta(hours=1)
live = L.predict(played, fx, now=now, horizon_days=3)
# "rozliczenie" kilku dni pozniej
later = all_m[all_m["date"] <= cut + pd.Timedelta(days=2)]
live = L.settle(live, later); L.save(live)
R.write(R.build(live, len(played), synthetic=True, now=now + pd.Timedelta(days=2)))
print("OK", C.WEB / "index.html")
