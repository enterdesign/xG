"""Darmowe wyniki pilki nożnej z football-data.co.uk (zamiast platnego endpointu /scores The Odds API)."""
from __future__ import annotations

import csv
import io
import urllib.request
from datetime import datetime

from .fixtures_src import map_names

FD = "https://football-data.co.uk"
# klucz sportu w The Odds API -> (rodzaj pliku, kod ligi w football-data)
SPORT_TO_FD = {
    "soccer_italy_serie_a": ("main", "I1"), "soccer_italy_serie_b": ("main", "I2"),
    "soccer_turkey_super_league": ("main", "T1"), "soccer_spain_la_liga": ("main", "SP1"),
    "soccer_greece_super_league": ("main", "G1"), "soccer_efl_champ": ("main", "E1"),
    "soccer_england_league1": ("main", "E2"), "soccer_france_ligue_two": ("main", "F2"),
    "soccer_netherlands_eredivisie": ("main", "N1"), "soccer_germany_bundesliga2": ("main", "D2"),
    "soccer_league_of_ireland": ("extra", "IRL"), "soccer_poland_ekstraklasa": ("extra", "POL"),
}


def fetch_text(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "xg-research/0.1"})
    data = urllib.request.urlopen(req, timeout=timeout).read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def season_code(dt: datetime) -> str:
    y = dt.year if dt.month >= 7 else dt.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _date(s: str):
    for f in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), f).date()
        except ValueError:
            pass
    return None


def parse_results(text: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        d, h, a = _date(r.get("Date", "") or ""), (r.get("HomeTeam") or r.get("Home") or "").strip(), (r.get("AwayTeam") or r.get("Away") or "").strip()
        hg, ag = (r.get("FTHG") or r.get("HG") or "").strip(), (r.get("FTAG") or r.get("AG") or "").strip()
        if d and h and a and hg.isdigit() and ag.isdigit():
            rows.append({"date": d, "home": h, "away": a, "hg": int(hg), "ag": int(ag)})
    return rows


def load_results(sport: str, now: datetime, fetch=fetch_text) -> list[dict]:
    kind, div = SPORT_TO_FD[sport]
    url = f"{FD}/mmz4281/{season_code(now)}/{div}.csv" if kind == "main" else f"{FD}/new/{div}.csv"
    return parse_results(fetch(url))


def find_result(bet: dict, rows: list[dict], tol_days: int = 1):
    """Dopasowuje zaklad do meczu po nazwach druzyn (rozmyte dopasowanie) i dacie."""
    if not rows:
        return None
    names = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
    mp, _ = map_names([bet["home"], bet["away"]], names)
    h, a = mp.get(bet["home"]), mp.get(bet["away"])
    if not (h and a):
        return None
    d = datetime.fromisoformat(bet["commence"].replace("Z", "+00:00")).date()
    for r in rows:
        if r["home"] == h and r["away"] == a and abs((r["date"] - d).days) <= tol_days:
            return r
    return None
