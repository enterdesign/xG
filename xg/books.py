"""Analiza bukmacherow: kto jest 'ostry', a kto 'miekki' wzgledem reszty rynku i wynikow.

Dla kazdego bukmachera b liczymy prawdopodobienstwa po usunieciu marzy (proporcjonalnie) i porownujemy z
konsensusem POZOSTALYCH bukmacherow (leave-one-out). Mieszanka P = (1-w)*konsensus + w*b:
  w* > 0  -> bukmacher wie wiecej niz reszta (ostry),
  w* ~ 0  -> tylko powtarza rynek,
  w* < 0  -> jego odchylenia od rynku to szum/blad (miekki: zawyzone/zanizone kursy sie NIE potwierdzaja wynikami).
Czesc 2: model LightGBM uczony walk-forward (rok po roku) na kursach wszystkich bukmacherow (+ opcjonalnie nasz
model) - sprawdza, czy JAKAKOLWIEK kombinacja odchylen przewiduje wynik lepiej niz sam konsensus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ingest import _parse_dates, _read_csv
from .metrics import EPS

BOOKS = ["B365", "BW", "IW", "PS", "VC", "WH"]
NAMES = {"B365": "Bet365", "BW": "Bwin", "IW": "Interwetten", "PS": "Pinnacle", "VC": "VC Bet", "WH": "William Hill"}


def load_books() -> pd.DataFrame:
    frames = []
    for path in sorted(C.RAW.glob("*_*.csv")):
        div = path.stem.split("_", 1)[1]
        raw = _read_csv(path.read_bytes())
        if not {"HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"} <= set(raw.columns):
            continue
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"])
        out = pd.DataFrame({"date": _parse_dates(raw), "div": div, "home": raw["HomeTeam"].astype(str).str.strip(),
                            "away": raw["AwayTeam"].astype(str).str.strip()}, index=raw.index)
        out["match_id"] = out["date"].dt.strftime("%Y-%m-%d") + "_" + out["home"] + "_" + out["away"]
        hg, ag = pd.to_numeric(raw["FTHG"], errors="coerce"), pd.to_numeric(raw["FTAG"], errors="coerce")
        out["y"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        for b in BOOKS:
            for phase, pre in (("o", b), ("c", b + "C")):
                cols = [f"{pre}H", f"{pre}D", f"{pre}A"]
                if all(c in raw.columns for c in cols):
                    odds = raw[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
                    ok = (odds > 1).all(axis=1)
                    inv = np.where(ok[:, None], 1 / np.where(odds > 1, odds, np.nan), np.nan)
                    ovr = inv.sum(1)
                    p = inv / ovr[:, None]
                    for k, s in enumerate("hda"):
                        out[f"{b}_{phase}_{s}"] = p[:, k]
                    out[f"{b}_{phase}_margin"] = ovr - 1
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("match_id").sort_values("date").reset_index(drop=True)


def _P(df, b, phase="o"):
    return df[[f"{b}_{phase}_h", f"{b}_{phase}_d", f"{b}_{phase}_a"]].to_numpy()


def _ll(P, y):
    P = np.clip(P, EPS, 1)
    P = P / P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(y)), y])


def book_table(df: pd.DataFrame, books=BOOKS, phase="o", seed=0, n_boot=500) -> tuple[list[dict], int]:
    cols = [f"{b}_{phase}_{s}" for b in books for s in "hda"]
    d = df.dropna(subset=cols)
    y = d["y"].to_numpy(int)
    Ps = {b: _P(d, b, phase) for b in books}
    ws = np.round(np.arange(-1.0, 1.5001, 0.05), 2)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(d), len(d)) for _ in range(n_boot)]
    rows = []
    for b in books:
        cons = np.mean([Ps[o] for o in books if o != b], axis=0)
        lls = np.array([_ll((1 - w) * cons + w * Ps[b], y) for w in ws])  # (grid, n)
        means = lls.mean(1)
        k = int(np.argmin(means))
        w_boot = [ws[int(np.argmin(lls[:, ix].mean(1)))] for ix in idx]
        base = lls[int(np.argmin(np.abs(ws)))]
        rows.append({
            "book": b, "name": NAMES.get(b, b), "n": int(len(d)),
            "margin": float(d[f"{b}_{phase}_margin"].mean()),
            "logloss": float(_ll(Ps[b], y).mean()), "logloss_consensus": float(base.mean()),
            "w_best": float(ws[k]), "w_ci": [float(np.percentile(w_boot, 2.5)), float(np.percentile(w_boot, 97.5))],
            "gain": float(base.mean() - means[k]),
            "mean_abs_dev": float(np.abs(Ps[b] - cons).mean()),
        })
    return rows, int(len(d))


def favourite_longshot(df: pd.DataFrame, books=BOOKS, phase="o") -> list[dict]:
    """Srednie odchylenie prawdopodobienstwa bukmachera od konsensusu reszty, w zaleznosci od tego jak
    prawdopodobne jest zdarzenie wg konsensusu (faworyci vs outsiderzy)."""
    cols = [f"{b}_{phase}_{s}" for b in books for s in "hda"]
    d = df.dropna(subset=cols)
    Ps = {b: _P(d, b, phase) for b in books}
    edges = [0, 0.15, 0.25, 0.40, 0.55, 1.0]
    out = []
    for b in books:
        cons = np.mean([Ps[o] for o in books if o != b], axis=0).ravel()
        dev = (Ps[b].ravel() - cons)
        cut = pd.cut(cons, edges)
        row = {"book": b, "name": NAMES.get(b, b)}
        for iv in cut.categories:
            m = np.asarray(cut == iv)
            row[f"{iv.left:.2f}-{iv.right:.2f}"] = float(dev[m].mean()) if m.sum() else None
        out.append(row)
    return out


def learned_detector(df: pd.DataFrame, ours: pd.DataFrame | None = None, first_test_year=2015, books=BOOKS,
                     seed=0, n_boot=1000) -> dict:
    """Walk-forward po latach: LightGBM na (konsensus, odchylenia kazdego bukmachera, marze[, nasz model])."""
    import lightgbm as lgb
    cols = [f"{b}_o_{s}" for b in books for s in "hda"]
    d = df.dropna(subset=cols).copy()
    if ours is not None:
        d = d.merge(ours, on="match_id", how="inner")
    P = {b: _P(d, b) for b in books}
    cons = np.mean([P[b] for b in books], axis=0)
    X = {"cons_h": cons[:, 0], "cons_d": cons[:, 1], "cons_a": cons[:, 2]}
    for b in books:
        for k, s in enumerate("hda"):
            X[f"dev_{b}_{s}"] = P[b][:, k] - cons[:, k]
        X[f"margin_{b}"] = d[f"{b}_o_margin"].to_numpy()
    if ours is not None:
        for k, s in enumerate("hda"):
            X[f"gap_model_{s}"] = d[f"ens_{s}"].to_numpy() - cons[:, k]
            X[f"model_{s}"] = d[f"ens_{s}"].to_numpy()
    X = pd.DataFrame(X, index=d.index)
    y = d["y"].to_numpy(int)
    year = d["date"].dt.year.to_numpy()
    diffs, ll_c, ll_m = [], [], []
    for yr in range(first_test_year, year.max() + 1):
        tr, te = year < yr, year == yr
        if tr.sum() < 3000 or te.sum() < 200:
            continue
        Xt, yt = X[tr], y[tr]
        cut = int(len(Xt) * 0.9)
        m = lgb.LGBMClassifier(objective="multiclass", n_estimators=600, learning_rate=0.03, num_leaves=6,
                               min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               reg_lambda=10, random_state=1, verbose=-1)
        m.fit(Xt.iloc[:cut], yt[:cut], eval_set=[(Xt.iloc[cut:], yt[cut:])], callbacks=[lgb.early_stopping(40, verbose=False)])
        pm = m.predict_proba(X[te])
        ll_m.append(_ll(pm, y[te]))
        ll_c.append(_ll(cons[te], y[te]))
    lm, lc = np.concatenate(ll_m), np.concatenate(ll_c)
    diff = lm - lc
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {"n": int(len(diff)), "logloss_consensus": float(lc.mean()), "logloss_detector": float(lm.mean()),
            "diff": float(diff.mean()), "diff_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "uses_our_model": ours is not None}
