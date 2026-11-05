"""Dane SYNTETYCZNE - wylacznie do testow mechaniki pipeline'u i podgladu strony. Nie sa wynikiem naukowym."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def make_matches(n_seasons: int = 8, teams_per_league: int = 12, seed: int = 0, last_season_start: int = 2025,
                 with_odds: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for div, league in C.LEAGUES.items():
        names = [f"{div}-T{i:02d}" for i in range(teams_per_league)]
        att = rng.normal(0, 0.25, teams_per_league)
        dfn = rng.normal(0, 0.25, teams_per_league)
        for s in range(n_seasons):
            year = last_season_start - n_seasons + 1 + s
            att += rng.normal(0, 0.08, teams_per_league)  # dryf formy
            dfn += rng.normal(0, 0.08, teams_per_league)
            fixtures = [(i, j) for i in range(teams_per_league) for j in range(teams_per_league) if i != j]
            rng.shuffle(fixtures)
            n_days = len(fixtures) // (teams_per_league // 2)
            for k, (i, j) in enumerate(fixtures):
                day = pd.Timestamp(year=year, month=8, day=10) + pd.Timedelta(days=int(k * 280 / len(fixtures)))
                lh = np.exp(0.28 + att[i] - dfn[j])
                la = np.exp(0.0 + att[j] - dfn[i])
                gh, ga = rng.poisson(lh), rng.poisson(la)
                p = _p1x2(lh, la)
                margin = 1.06
                odds = (1 / (p * margin)) if with_odds else [np.nan] * 3
                rows.append(dict(
                    date=day + pd.Timedelta(hours=int(rng.choice([13, 16, 18, 20]))), div=div, league=league,
                    home=names[i], away=names[j], fthg=gh, ftag=ga,
                    st_h=rng.poisson(max(lh * 3.2, 0.5)), st_a=rng.poisson(max(la * 3.2, 0.5)),
                    sh_h=np.nan, sh_a=np.nan,
                    mp_h=p[0] if with_odds else np.nan, mp_d=p[1] if with_odds else np.nan,
                    mp_a=p[2] if with_odds else np.nan))
    df = pd.DataFrame(rows)
    df["match_id"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["home"] + "_" + df["away"]
    return df.drop_duplicates("match_id").sort_values(["date", "match_id"]).reset_index(drop=True)


def _p1x2(lh, la):
    from .features import poisson_1x2
    return np.array(poisson_1x2(lh, la))
