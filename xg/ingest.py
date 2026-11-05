"""Pobieranie i normalizacja danych z football-data.co.uk."""
from __future__ import annotations

import io
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C

ODDS_SETS = [  # od najlepszych (kursy zamkniecia, srednia rynku) do gorszych
    ("AvgCH", "AvgCD", "AvgCA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365CH", "B365CD", "B365CA"),
    ("B365H", "B365D", "B365A"),
]


def season_codes(n_seasons: int, today: datetime | None = None) -> list[str]:
    today = today or datetime.now()
    y = today.year if today.month >= 7 else today.year - 1
    return [f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}" for i in range(n_seasons)][::-1]


def _read_csv(data: bytes) -> pd.DataFrame:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "xg-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(n_seasons: int = 10, leagues: list[str] | None = None) -> list[str]:
    """Sciaga CSV dla sezonow/lig. Stare sezony z cache, biezacy zawsze odswiezany."""
    leagues = leagues or list(C.LEAGUES)
    C.RAW.mkdir(parents=True, exist_ok=True)
    codes = season_codes(n_seasons)
    problems = []
    for code in codes:
        for div in leagues:
            path = C.RAW / f"{code}_{div}.csv"
            if path.exists() and code != codes[-1]:
                continue
            url = f"{C.BASE_URL}/mmz4281/{code}/{div}.csv"
            try:
                path.write_bytes(_get(url))
                print(f"  ok   {code} {div}")
            except Exception as e:  # noqa: BLE001
                problems.append(f"{code} {div}: {e}")
                print(f"  FAIL {code} {div}: {e}")
    return problems


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    d = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    if "Time" in df.columns:
        t = pd.to_timedelta(df["Time"].astype(str).str.strip() + ":00", errors="coerce")
        d = d.dt.normalize() + t.fillna(pd.Timedelta(hours=C.DEFAULT_KICKOFF_HOUR))
    else:
        d = d.dt.normalize() + pd.Timedelta(hours=C.DEFAULT_KICKOFF_HOUR)
    return d


def _market_probs(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    odds = np.full((n, 3), np.nan)
    for cols in ODDS_SETS:
        if not all(c in df.columns for c in cols):
            continue
        cand = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
        mask = np.isnan(odds[:, 0]) & (cand > 1).all(axis=1)
        odds[mask] = cand[mask]
    inv = 1.0 / odds
    inv = inv / inv.sum(axis=1, keepdims=True)  # usuwa marze bukmachera
    return pd.DataFrame(inv, columns=["mp_h", "mp_d", "mp_a"], index=df.index)


def normalize(raw: pd.DataFrame, div: str) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
    out = pd.DataFrame(index=raw.index)
    out["date"] = _parse_dates(raw)
    out["div"] = div
    out["league"] = C.LEAGUES.get(div, div)
    out["home"] = raw["HomeTeam"].astype(str).str.strip()
    out["away"] = raw["AwayTeam"].astype(str).str.strip()
    for src, dst in [("FTHG", "fthg"), ("FTAG", "ftag"), ("HST", "st_h"), ("AST", "st_a"),
                     ("HS", "sh_h"), ("AS", "sh_a")]:
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
    out = pd.concat([out, _market_probs(raw)], axis=1)
    out = out.dropna(subset=["date"])
    out["match_id"] = (out["date"].dt.strftime("%Y-%m-%d") + "_" + out["home"] + "_" + out["away"])
    return out


def load_matches() -> pd.DataFrame:
    """Wczytuje wszystkie rozegrane mecze z data/raw."""
    frames = []
    for path in sorted(C.RAW.glob("*_*.csv")):
        div = path.stem.split("_", 1)[1]
        try:
            raw = _read_csv(path.read_bytes())
        except Exception as e:  # noqa: BLE001
            print(f"  pomijam {path.name}: {e}")
            continue
        if not {"HomeTeam", "AwayTeam", "Date"} <= set(raw.columns):
            continue
        frames.append(normalize(raw, div))
    if not frames:
        raise SystemExit("Brak danych w data/raw - uruchom najpierw: python -m xg download")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["fthg", "ftag"])
    df = df.drop_duplicates("match_id").sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


def fetch_fixtures() -> pd.DataFrame:
    """Nadchodzace mecze z fixtures.csv (football-data). Zamiennik/dodatek: patrz xg/external_api.py."""
    raw = _read_csv(_get(f"{C.BASE_URL}/fixtures.csv"))
    raw = raw[raw["Div"].isin(C.LEAGUES)]
    frames = [normalize(g, div) for div, g in raw.groupby("Div")]
    if not frames:
        return pd.DataFrame(columns=["match_id", "date", "div", "league", "home", "away", "fthg", "ftag"])
    fx = pd.concat(frames, ignore_index=True)
    fx[["fthg", "ftag"]] = np.nan
    return fx
