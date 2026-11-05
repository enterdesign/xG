"""Terminarz nadchodzacych meczow z openfootball (GitHub, darmowe, bez klucza) + tlumacz nazw druzyn
na nazwy uzywane przez football-data (np. 'Manchester United FC' -> 'Man United')."""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from . import config as C

OF_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
OF_CODES = {"en.1": "E0", "es.1": "SP1", "de.1": "D1", "it.1": "I1", "fr.1": "F1"}

ALIASES = {
    "Athletic Club": "Ath Bilbao", "Club Atlético de Madrid": "Ath Madrid", "RCD Espanyol de Barcelona": "Espanol",
    "Rayo Vallecano de Madrid": "Vallecano", "Real Sociedad de Fútbol": "Sociedad", "RC Deportivo La Coruña": "La Coruna",
    "Real Racing Club de Santander": "Santander", "Real Betis Balompié": "Betis", "RC Celta de Vigo": "Celta",
    "Deportivo Alavés": "Alaves", "Eintracht Frankfurt": "Ein Frankfurt", "Borussia Mönchengladbach": "M'gladbach",
    "1. FC Köln": "FC Koln", "FC Bayern München": "Bayern Munich", "Paris Saint-Germain FC": "Paris SG",
    "Nottingham Forest FC": "Nott'm Forest", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "FC Internazionale Milano": "Inter", "AC Milan": "Milan", "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille", "Stade Rennais FC 1901": "Rennes", "Racing Club de Lens": "Lens",
    "Stade Brestois 29": "Brest", "RC Strasbourg Alsace": "Strasbourg", "Hamburger SV": "Hamburg",
    "TSG 1899 Hoffenheim": "Hoffenheim", "Bayer 04 Leverkusen": "Leverkusen", "1. FSV Mainz 05": "Mainz",
    "Brighton & Hove Albion FC": "Brighton", "Wolverhampton Wanderers FC": "Wolves", "West Ham United FC": "West Ham",
    "Tottenham Hotspur FC": "Tottenham", "Newcastle United FC": "Newcastle", "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester", "Sheffield United FC": "Sheffield United", "Real Valladolid CF": "Valladolid",
    "RCD Mallorca": "Mallorca", "Girona FC": "Girona", "UD Las Palmas": "Las Palmas", "Cádiz CF": "Cadiz",
    "Borussia Dortmund": "Dortmund", "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum 1848": "Bochum", "1. FC Heidenheim 1846": "Heidenheim", "FC St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel", "1. FC Nürnberg": "Nurnberg", "Fortuna Düsseldorf": "Fortuna Dusseldorf",
    "Hertha BSC": "Hertha", "Hannover 96": "Hannover", "Paris FC": "Paris FC", "AS Saint-Étienne": "St Etienne",
    "Olympique Lyonnais": "Lyon", "Montpellier HSC": "Montpellier", "FC Nantes": "Nantes", "Stade de Reims": "Reims",
    "Hellas Verona FC": "Verona", "Atalanta BC": "Atalanta", "Bologna FC 1909": "Bologna", "Como 1907": "Como",
    "Parma Calcio 1913": "Parma", "US Sassuolo Calcio": "Sassuolo", "SS Lazio": "Lazio", "SSC Napoli": "Napoli",
}
_STOP = {"fc", "afc", "cf", "sc", "ac", "as", "us", "ssc", "ud", "rc", "rcd", "cd", "sv", "fsv", "tsg", "vfb", "vfl",
         "ca", "real", "club", "de", "del", "la", "le", "calcio", "sco", "ogc", "aj", "es", "hsc", "bc", "fk", "1"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)  # liczby (1909, 04...) i interpunkcja wypadaja
    return " ".join(t for t in s.split() if t not in _STOP)


def map_names(of_names: list[str], known: list[str]) -> tuple[dict, list[str]]:
    """Zwraca (mapa nazwa_openfootball -> nazwa_football-data, lista niedopasowanych)."""
    known_set, kn = set(known), {k: _norm(k) for k in known}
    out, missing = {}, []
    for name in of_names:
        if ALIASES.get(name) in known_set:
            out[name] = ALIASES[name]
            continue
        if name in known_set:
            out[name] = name
            continue
        n = _norm(name)
        exact = [k for k, v in kn.items() if v == n]
        if exact:
            out[name] = exact[0]
            continue
        sub = [k for k, v in kn.items() if v and (v in n or n in v)]
        if len(sub) == 1:
            out[name] = sub[0]
            continue
        close = difflib.get_close_matches(n, list(kn.values()), n=1, cutoff=0.75)
        if close:
            out[name] = next(k for k, v in kn.items() if v == close[0])
        else:
            missing.append(name)
    return out, missing


def season_label(today: datetime | None = None) -> str:
    today = today or datetime.now()
    y = today.year if today.month >= 7 else today.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def fetch(matches: pd.DataFrame, today: datetime | None = None, timeout: int = 60) -> pd.DataFrame:
    """Nadchodzace mecze 5 lig z openfootball, z nazwami jak w football-data."""
    today = pd.Timestamp(today or datetime.now()).normalize()
    season = season_label(today.to_pydatetime())
    rows, problems = [], []
    for code, div in OF_CODES.items():
        try:
            req = urllib.request.Request(OF_URL.format(season=season, code=code), headers={"User-Agent": "xg-research/0.1"})
            data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except Exception as e:  # noqa: BLE001
            problems.append(f"{code}: {e}")
            continue
        ms = [m for m in data["matches"] if not m.get("score") and pd.Timestamp(m["date"]) >= today]
        if not ms:
            continue
        known = sorted(set(matches.loc[matches["div"] == div, "home"]) | set(matches.loc[matches["div"] == div, "away"]))
        names = sorted({m["team1"] for m in ms} | {m["team2"] for m in ms})
        mp, missing = map_names(names, known)
        if missing:
            problems.append(f"{code}: nie dopasowano nazw {missing} (dopisz je w ALIASES) - te mecze pomijam")
        for m in ms:
            if m["team1"] in mp and m["team2"] in mp:
                t = m.get("time") or f"{C.DEFAULT_KICKOFF_HOUR}:00"
                rows.append({"date": pd.Timestamp(f"{m['date']} {t}"), "div": div, "league": C.LEAGUES[div],
                             "home": mp[m["team1"]], "away": mp[m["team2"]]})
    for p in problems:
        print("  [openfootball]", p)
    df = pd.DataFrame(rows, columns=["date", "div", "league", "home", "away"])
    df["fthg"], df["ftag"] = np.nan, np.nan
    df["st_h"] = df["st_a"] = df["sh_h"] = df["sh_a"] = np.nan
    df["mp_h"] = df["mp_d"] = df["mp_a"] = np.nan
    df["match_id"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["home"] + "_" + df["away"]
    return df
