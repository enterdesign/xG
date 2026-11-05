"""Budowa cech BEZ wycieku z przyszlosci.

Mecze przetwarzane chronologicznie, dzien po dniu: najpierw liczymy cechy dla wszystkich meczow dnia
(stan z samej przeszlosci), dopiero potem aktualizujemy stan o wyniki. Mecze bez wyniku (nadchodzace)
dostaja cechy, ale stanu nie zmieniaja.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import config as C

HOME_ADV_ELO = 60.0
ELO_K = 20.0
ELO_INIT = 1500.0
ELO_NEWCOMER = 1450.0   # druzyna pojawiajaca sie po starcie historii (np. beniaminek)
SEASON_REGRESSION = 0.2  # ile ELO wraca do sredniej na poczatku sezonu
EWMA_ALPHA = 0.08
SHRINK_K = 6.0
WINDOWS = (3, 5, 10)
_FACT = np.array([math.factorial(k) for k in range(12)], dtype=float)
_K = np.arange(12)


def poisson_1x2(lh: float, la: float) -> tuple[float, float, float]:
    ph = np.exp(-lh) * lh ** _K / _FACT
    pa = np.exp(-la) * la ** _K / _FACT
    m = np.outer(ph, pa)
    m /= m.sum()
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())


class _Team:
    __slots__ = ("elo", "hist", "home", "away", "att", "dfn", "n", "dates", "season", "sgames")

    def __init__(self, elo: float):
        self.elo = elo
        self.hist = deque(maxlen=10)   # (pts, gf, ga, sotd)
        self.home = deque(maxlen=5)    # pkt w meczach u siebie
        self.away = deque(maxlen=5)    # pkt w meczach na wyjezdzie
        self.att = 0.0
        self.dfn = 0.0
        self.n = 0
        self.dates = deque(maxlen=14)
        self.season = None
        self.sgames = 0


def _season(ts: pd.Timestamp) -> int:
    return ts.year if ts.month >= 7 else ts.year - 1


def _win(hist, n):
    items = list(hist)[-n:]
    if not items:
        return (np.nan,) * 4
    a = np.array(items, dtype=float)
    sot = np.nan if np.isnan(a[:, 3]).all() else float(np.nanmean(a[:, 3]))
    return float(a[:, 0].mean()), float(a[:, 1].mean()), float(a[:, 2].mean()), sot


def _mean(d):
    return float(np.mean(d)) if len(d) else np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: kolumny date, div, home, away, fthg, ftag, st_h, st_a (+ dowolne inne, zostaja bez zmian)."""
    df = df.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    n = len(df)
    teams: dict[str, _Team] = {}
    lgs = defaultdict(lambda: [1.5 * 20, 1.2 * 20, 20.0])  # [suma goli gosp., suma goli gosci, n] z priorem
    h2h: dict[tuple, list] = defaultdict(list)
    seen = 0

    date = df["date"].dt.normalize().to_numpy()
    ts_all = df["date"].dt.normalize().tolist()
    home, away, div = df["home"].tolist(), df["away"].tolist(), df["div"].tolist()
    hg, ag = df["fthg"].to_numpy(float), df["ftag"].to_numpy(float)
    st_h, st_a = df["st_h"].to_numpy(float), df["st_a"].to_numpy(float)
    div_codes = {d: i for i, d in enumerate(C.LEAGUES)}

    def touch(name, season):
        t = teams.get(name)
        if t is None:
            t = teams[name] = _Team(ELO_NEWCOMER if seen > 500 else ELO_INIT)
        if t.season != season:
            if t.season is not None:
                t.elo = (1 - SEASON_REGRESSION) * t.elo + SEASON_REGRESSION * ELO_INIT
            t.season, t.sgames = season, 0
        return t

    rows: list[dict | None] = [None] * n
    groups = df.groupby(date, sort=True).indices
    for day in sorted(groups):
        idx = groups[day]
        # 1) cechy (stan sprzed dnia)
        for i in idx:
            ts = ts_all[i]
            season = _season(ts)
            h, a = touch(home[i], season), touch(away[i], season)
            lg = lgs[div[i]]
            lh_avg, la_avg = lg[0] / lg[2], lg[1] / lg[2]
            g = (lh_avg + la_avg) / 2

            def eff(x, k):
                return (k * x + SHRINK_K * g) / (k + SHRINK_K) if k > 0 else g

            att_h, def_h, att_a, def_a = eff(h.att, h.n), eff(h.dfn, h.n), eff(a.att, a.n), eff(a.dfn, a.n)
            lam_h = lh_avg * (att_h / g) * (def_a / g)
            lam_a = la_avg * (att_a / g) * (def_h / g)
            pp = poisson_1x2(lam_h, lam_a)

            f = {"elo_h": h.elo, "elo_a": a.elo, "elo_diff": h.elo + HOME_ADV_ELO - a.elo,
                 "lam_h": lam_h, "lam_a": lam_a, "lam_diff": lam_h - lam_a,
                 "pp_h": pp[0], "pp_d": pp[1], "pp_a": pp[2]}
            for side, t in (("h", h), ("a", a)):
                for w in WINDOWS:
                    ppg, gf, ga, sot = _win(t.hist, w)
                    f[f"{side}_ppg{w}"], f[f"{side}_gf{w}"], f[f"{side}_ga{w}"], f[f"{side}_sotd{w}"] = ppg, gf, ga, sot
                f[f"{side}_venue_ppg5"] = _mean(t.home if side == "h" else t.away)
                f[f"{side}_rest"] = (ts - t.dates[-1]).days if t.dates else np.nan
                f[f"{side}_n14"] = sum(1 for d0 in t.dates if (ts - d0).days <= 14)
                f[f"{side}_games"] = t.sgames
                f[f"{side}_n_hist"] = t.n
            f["ppg5_diff"] = f["h_ppg5"] - f["a_ppg5"]
            f["sotd5_diff"] = f["h_sotd5"] - f["a_sotd5"]
            key = tuple(sorted((home[i], away[i])))
            meets = h2h[key][-5:]
            if meets:
                pts = [(3 if (g1 > g2) else 1 if g1 == g2 else 0) if hn == home[i]
                       else (3 if (g2 > g1) else 1 if g1 == g2 else 0) for hn, g1, g2 in meets]
                f["h2h_ppg"] = float(np.mean(pts))
            else:
                f["h2h_ppg"] = np.nan
            f["h2h_n"] = len(meets)
            f["lg_code"] = div_codes.get(div[i], -1)
            f["month"] = ts.month
            rows[i] = f

        # 2) aktualizacja stanu wynikami (tylko mecze rozegrane)
        for i in idx:
            if np.isnan(hg[i]) or np.isnan(ag[i]):
                continue
            h, a = teams[home[i]], teams[away[i]]
            lg = lgs[div[i]]
            gh, ga_ = hg[i], ag[i]
            ph_pts = 3 if gh > ga_ else 1 if gh == ga_ else 0
            pa_pts = {3: 0, 1: 1, 0: 3}[ph_pts]
            sotd_h = st_h[i] - st_a[i] if not (np.isnan(st_h[i]) or np.isnan(st_a[i])) else np.nan
            # ELO (World Football Elo: mnoznik roznicy bramek)
            dr = h.elo + HOME_ADV_ELO - a.elo
            exp_h = 1 / (1 + 10 ** (-dr / 400))
            s_h = 1.0 if gh > ga_ else 0.5 if gh == ga_ else 0.0
            delta = ELO_K * (math.log(abs(gh - ga_) + 1) + 1) * (s_h - exp_h)
            h.elo += delta
            a.elo -= delta
            for t, pts, gf, gc, sd in ((h, ph_pts, gh, ga_, sotd_h), (a, pa_pts, ga_, gh, -sotd_h)):
                t.hist.append((pts, gf, gc, sd))
                lr = max(EWMA_ALPHA, 1.0 / (t.n + 1))
                if t.n == 0:
                    t.att, t.dfn = gf, gc
                else:
                    t.att += lr * (gf - t.att)
                    t.dfn += lr * (gc - t.dfn)
                t.n += 1
                t.sgames += 1
                t.dates.append(ts_all[i])
            h.home.append(ph_pts)
            a.away.append(pa_pts)
            h2h[tuple(sorted((home[i], away[i])))].append((home[i], gh, ga_))
            lg[0] += gh
            lg[1] += ga_
            lg[2] += 1
            seen += 1

    feats = pd.DataFrame(rows)
    out = pd.concat([df, feats], axis=1)
    played = out["fthg"].notna() & out["ftag"].notna()
    out["y"] = np.where(played, np.where(out["fthg"] > out["ftag"], 0, np.where(out["fthg"] == out["ftag"], 1, 2)), np.nan)
    return out


# Grupy cech - do analizy ablacyjnej ("co naprawde ma wplyw")
FEATURE_GROUPS = {
    "elo": ["elo_h", "elo_a", "elo_diff"],
    "sila_poisson": ["lam_h", "lam_a", "lam_diff", "pp_h", "pp_d", "pp_a"],
    "forma_wyniki": [f"{s}_{m}{w}" for s in "ha" for m in ("ppg", "gf", "ga") for w in WINDOWS] + ["ppg5_diff"],
    "forma_strzaly": [f"{s}_sotd{w}" for s in "ha" for w in WINDOWS] + ["sotd5_diff"],
    "dom_wyjazd": ["h_venue_ppg5", "a_venue_ppg5"],
    "zmeczenie": ["h_rest", "a_rest", "h_n14", "a_n14"],
    "h2h": ["h2h_ppg", "h2h_n"],
    "kontekst": ["h_games", "a_games", "h_n_hist", "a_n_hist", "lg_code", "month"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
