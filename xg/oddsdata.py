"""Historyczne kursy (surowe, z marza) potrzebne do powtorki strategii: 1X2 i Over/Under 2.5."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ingest import _parse_dates, _read_csv


def _num(raw, col):
    return pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(np.nan, index=raw.index)


def _first(raw, cols):
    out = _num(raw, cols[0])
    for c in cols[1:]:
        out = out.fillna(_num(raw, c))
    return out


# nazwa -> kolejnosc kolumn zrodlowych (nowsze sezony pierwsze, starsze jako uzupelnienie)
X12 = {
    "max": ("Max{}", "BbMx{}"), "avg": ("Avg{}", "BbAv{}"), "ps": ("PS{}",), "psc": ("PSC{}",), "avgc": ("AvgC{}",),
    "maxc": ("MaxC{}",),
    "b365": ("B365{}",), "bw": ("BW{}",), "iw": ("IW{}",), "vc": ("VC{}",), "wh": ("WH{}",),
}
OU = {
    "max": ("Max{}", "BbMx{}"), "avg": ("Avg{}", "BbAv{}"), "ps": ("P{}",), "psc": ("PC{}",), "avgc": ("AvgC{}",),
    "maxc": ("MaxC{}",),
}


def load_odds() -> pd.DataFrame:
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
        hg, ag = _num(raw, "FTHG"), _num(raw, "FTAG")
        out["y"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        out["goals"] = hg + ag
        for name, pats in X12.items():
            for s, suf in zip("hda", "HDA"):
                out[f"x_{name}_{s}"] = _first(raw, [p.format(suf) for p in pats])
        for name, pats in OU.items():
            for s, suf in (("over", ">2.5"), ("under", "<2.5")):
                out[f"ou_{name}_{s}"] = _first(raw, [p.format(suf) for p in pats])
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    # najlepsza cena sposrod 5 duzych bukmacherow (bez Pinnacle) - wymagamy co najmniej 3 notowan
    for s_ in "hda":
        stack = df[[f"x_{b}_{s_}" for b in ("b365", "bw", "iw", "vc", "wh")]]
        df[f"x_best5_{s_}"] = stack.max(axis=1).where(stack.notna().sum(axis=1) >= 3)
    return df.drop_duplicates("match_id").sort_values("date").reset_index(drop=True)
