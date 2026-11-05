"""Klient The Odds API v4 (https://the-odds-api.com). Tylko stdlib. Klucz NIGDY nie jest logowany.

Koszt kredytow (wg dokumentacji - zweryfikuj na stronie dostawcy):
  /sports, /events ........ 0 kredytow
  /odds ................... liczba rynkow x liczba regionow (na zapytanie o jedna lige/sport)
  /scores ................. 1 (2 gdy podano daysFrom)
Naglowki odpowiedzi: x-requests-remaining, x-requests-used, x-requests-last.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4"


class ApiError(RuntimeError):
    pass


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


class OddsApi:
    def __init__(self, key: str, timeout: int = 40):
        if not key:
            raise ApiError("Brak klucza API (ustaw zmienna srodowiskowa ODDS_API_KEY).")
        self._key, self.timeout = key, timeout
        self.remaining = self.used = self.last_cost = None

    def _get(self, path: str, **params):
        q = urllib.parse.urlencode({**params, "apiKey": self._key})
        req = urllib.request.Request(f"{BASE}{path}?{q}", headers={"User-Agent": "xg-research/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                h = r.headers
                self.remaining, self.used, self.last_cost = (_int(h.get("x-requests-remaining")),
                                                             _int(h.get("x-requests-used")),
                                                             _int(h.get("x-requests-last")))
                return json.loads(r.read())
        except urllib.error.HTTPError as e:  # komunikat bez URL-a (w URL jest klucz)
            raise ApiError(f"HTTP {e.code} dla {path}") from None
        except urllib.error.URLError as e:
            raise ApiError(f"Blad sieci dla {path}: {e.reason}") from None

    def sports(self):
        return self._get("/sports")

    def events(self, sport: str):
        return self._get(f"/sports/{sport}/events", dateFormat="iso")

    def odds(self, sport: str, start: str, end: str, regions: str = "eu", markets: str = "h2h", bookmakers: str | None = None):
        """Gdy podano `bookmakers` (klucze po przecinku), API zwraca tylko ich kursy i ignoruje `regions`.
        Wg dokumentacji kazda grupa 10 bukmacherow liczy sie jak 1 region - zweryfikuj na stronie dostawcy."""
        params = dict(markets=markets, oddsFormat="decimal", dateFormat="iso", commenceTimeFrom=start, commenceTimeTo=end)
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        return self._get(f"/sports/{sport}/odds", **params)

    def scores(self, sport: str, days: int = 3):
        return self._get(f"/sports/{sport}/scores", daysFrom=days, dateFormat="iso")
