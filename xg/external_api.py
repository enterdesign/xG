"""Adapter do Twojego API z rankingiem ELO i meczami na najblizsze 24h.

Nie znam formatu odpowiedzi, wiec to jest szkielet. Uzupelnij `_map` zgodnie z realna odpowiedzia
(klucz API ustaw w zmiennej srodowiskowej XG_API_KEY, URL w XG_API_URL - nie wpisuj ich w kod).

Oczekiwany wynik: DataFrame z kolumnami jak w ingest.normalize
(match_id, date, div, league, home, away, fthg, ftag = NaN dla nadchodzacych) + opcjonalnie `ext_elo_h`, `ext_elo_a`.
Uwaga: nazwy druzyn musza zgadzac sie z football-data (np. "Man United"), w razie potrzeby dodaj mape nazw.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pandas as pd


def fetch_upcoming() -> pd.DataFrame:
    url, key = os.environ.get("XG_API_URL"), os.environ.get("XG_API_KEY")
    if not url:
        raise RuntimeError("Ustaw XG_API_URL (i ewentualnie XG_API_KEY), albo uzyj domyslnych fixtures.csv")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read())
    return _map(payload)


def _map(payload) -> pd.DataFrame:  # TODO: dopasuj do formatu API
    raise NotImplementedError("Uzupelnij mapowanie odpowiedzi API -> kolumny (patrz docstring modulu)")
