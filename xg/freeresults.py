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


# ---------- tenis: tennis-data.co.uk (xlsx, czytany bez zewnetrznych bibliotek) ----------
import re as _re
import unicodedata as _ud
import zipfile as _zf
from datetime import date as _date_t, timedelta as _td
from xml.etree import ElementTree as _ET

TD = "https://www.tennis-data.co.uk"
# turnieje tenisowe, dla ktorych platne API zwraca wyniki (z tabeli pokrycia The Odds API)
TENNIS_SCORES_OK = {
    "tennis_atp_canadian_open", "tennis_atp_china_open", "tennis_atp_cincinnati_open", "tennis_atp_japan_open",
    "tennis_atp_shanghai_masters", "tennis_atp_swiss_indoors", "tennis_atp_us_open", "tennis_atp_vienna_open",
    "tennis_atp_washington_open", "tennis_wta_canadian_open", "tennis_wta_china_open", "tennis_wta_cincinnati_open",
    "tennis_wta_guadalajara_open", "tennis_wta_monterrey_open", "tennis_wta_ningbo_open", "tennis_wta_pan_pacific_open",
    "tennis_wta_singapore_open", "tennis_wta_us_open", "tennis_wta_washington_open", "tennis_wta_wuhan_open",
}


def fetch_bytes(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "xg-research/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def parse_xlsx(data: bytes) -> list[dict]:
    """Pierwszy arkusz xlsx -> lista slownikow (naglowek z pierwszego wiersza)."""
    z = _zf.ZipFile(io.BytesIO(data))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in _ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    rows = []
    for row in _ET.fromstring(z.read(sheet)).iter("{%s}row" % ns["m"]):
        cells = {}
        for c in row.findall("m:c", ns):
            col = _re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", ns)
            if c.get("t") == "inlineStr":
                val = "".join(t.text or "" for t in c.iter("{%s}t" % ns["m"]))
            elif v is None:
                continue
            else:
                val = shared[int(v.text)] if c.get("t") == "s" else v.text
            cells[col] = val
        rows.append(cells)
    if not rows:
        return []
    head = rows[0]
    return [{head[k]: v for k, v in r.items() if k in head} for r in rows[1:]]


def _norm(s: str) -> str:
    s = _ud.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace("-", " ")
    return " ".join(_re.sub(r"[^a-z ]", " ", s).split())


def _td_date(v):
    try:
        return _date_t(1899, 12, 30) + _td(days=int(float(v)))
    except (TypeError, ValueError):
        return _date(str(v or ""))


def parse_tennis(data: bytes) -> list[dict]:
    out = []
    for r in parse_xlsx(data):
        d, w, l = _td_date(r.get("Date")), (r.get("Winner") or "").strip(), (r.get("Loser") or "").strip()
        if d and w and l:
            out.append({"date": d, "winner": w, "loser": l})
    return out


def load_tennis(sport: str, now: datetime, fetch=fetch_bytes) -> list[dict]:
    folder = "w" if sport.startswith("tennis_wta") else ""
    rows = []
    for y in {now.year, (now - _td(days=20)).year}:
        rows += parse_tennis(fetch(f"{TD}/{y}{folder}/{y}.xlsx"))
    return rows


def _same_player(full: str, td_name: str) -> bool:
    """'Taylor Fritz' vs 'Fritz T.' (nazwisko + inicjal)."""
    parts = _norm(td_name).split()
    if len(parts) < 2:
        return False
    sur, ini = " ".join(parts[:-1]), parts[-1][:1]
    f = _norm(full)
    return f.endswith(" " + sur) and f.startswith(ini) or f == sur


def find_tennis_result(bet: dict, rows: list[dict], tol_days: int = 2):
    """Zwraca nazwisko zwyciezcy w nazwie z zakladu (home/away) albo None."""
    d = datetime.fromisoformat(bet["commence"].replace("Z", "+00:00")).date()
    for r in rows:
        if abs((r["date"] - d).days) > tol_days:
            continue
        for me, other in ((bet["home"], bet["away"]), (bet["away"], bet["home"])):
            if _same_player(me, r["winner"]) and _same_player(other, r["loser"]):
                return me
    return None


# ---------- ESPN (publiczne tablice wynikow, bez klucza, 0 kredytow) ----------
import json as _json

ESPN = "https://site.api.espn.com/apis/site/v2/sports"
SPORT_TO_ESPN = {
    "soccer_italy_serie_a": "soccer/ita.1", "soccer_italy_serie_b": "soccer/ita.2",
    "soccer_turkey_super_league": "soccer/tur.1", "soccer_spain_la_liga": "soccer/esp.1",
    "soccer_greece_super_league": "soccer/gre.1", "soccer_efl_champ": "soccer/eng.2",
    "soccer_england_league1": "soccer/eng.3", "soccer_france_ligue_two": "soccer/fra.2",
    "soccer_netherlands_eredivisie": "soccer/ned.1", "soccer_germany_bundesliga2": "soccer/ger.2",
    "soccer_league_of_ireland": "soccer/irl.1", "basketball_nba": "basketball/nba", "icehockey_nhl": "hockey/nhl",
}


def espn_path(sport: str):
    if sport.startswith("tennis_atp"):
        return "tennis/atp"
    if sport.startswith("tennis_wta"):
        return "tennis/wta"
    return SPORT_TO_ESPN.get(sport)


def fetch_json(url: str, timeout: int = 30):
    return _json.loads(fetch_bytes(url, timeout).decode("utf-8"))


def _espn_day(s: str):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).date()


def parse_espn(data: dict, tennis: bool) -> list[dict]:
    rows = []
    for ev in data.get("events", []):
        if tennis:
            comps = [c for g in ev.get("groupings", []) if "single" in g.get("grouping", {}).get("displayName", "").lower()
                     for c in g.get("competitions", [])]
        else:
            comps = ev.get("competitions", [])
        for c in comps:
            st = (c.get("status") or ev.get("status") or {}).get("type", {})
            if not st.get("completed"):
                continue
            d = _espn_day(c.get("date") or ev["date"])
            cp = c.get("competitors", [])
            if tennis:
                w = [x for x in cp if x.get("winner")]
                l = [x for x in cp if not x.get("winner")]
                if len(w) == 1 and len(l) == 1:
                    rows.append({"date": d, "winner": w[0]["athlete"]["displayName"], "loser": l[0]["athlete"]["displayName"]})
            else:
                h = next((x for x in cp if x.get("homeAway") == "home"), None)
                a = next((x for x in cp if x.get("homeAway") == "away"), None)
                if h and a and str(h.get("score", "")).isdigit() and str(a.get("score", "")).isdigit():
                    rows.append({"date": d, "home": h["team"]["displayName"], "away": a["team"]["displayName"],
                                 "hg": int(h["score"]), "ag": int(a["score"])})
    return rows


def load_espn(sport: str, bets: list[dict], fetch=fetch_json) -> list[dict]:
    path = espn_path(sport)
    if not path:
        return []
    tennis = path.startswith("tennis")
    days = set()
    for b in bets:
        d = datetime.fromisoformat(b["commence"].replace("Z", "+00:00")).date()
        days |= {d - _td(days=1), d, d + _td(days=1)}
    rows = []
    for d in sorted(days):
        rows += parse_espn(fetch(f"{ESPN}/{path}/scoreboard?dates={d:%Y%m%d}"), tennis)
        if tennis:  # tablica tenisa zwraca caly turniej, jedno zapytanie wystarczy
            break
    return rows


def _name_match(a: str, b: str) -> bool:
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x == y:
        return True
    xs, ys = x.split(), y.split()
    return xs[0][:1] == ys[0][:1] and (x.endswith(" " + ys[-1]) or y.endswith(" " + xs[-1]))


def find_espn_tennis(bet: dict, rows: list[dict], tol_days: int = 2):
    d = datetime.fromisoformat(bet["commence"].replace("Z", "+00:00")).date()
    for r in rows:
        if abs((r["date"] - d).days) > tol_days:
            continue
        for me, other in ((bet["home"], bet["away"]), (bet["away"], bet["home"])):
            if _name_match(me, r["winner"]) and _name_match(other, r["loser"]):
                return me
    return None
