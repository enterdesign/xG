"""Wirtualny bankroll: powtorka strategii 'szukanie roznic w ofertach' na historycznych kursach.

Decyzja zapada WYLACZNIE z informacji dostepnych w momencie obstawiania (kursy otwarcia). Kursy zamkniecia
i wynik sluza tylko do oceny (CLV, zysk). Strategia nie przewiduje wynikow: porownuje najlepsza cene rynku
z uczciwym prawdopodobienstwem wyliczonym z innej, referencyjnej wyceny (srednia rynku lub Pinnacle po usunieciu marzy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SELS = {"1x2": ["h", "d", "a"], "ou25": ["over", "under"]}


def _fair(df, prefix, sels, method="power"):
    """Uczciwe prawdopodobienstwa z kursow. 'prop' - proste normalizowanie (zawyza szanse outsiderow),
    'power' - potegowa metoda usuwania marzy: p_i = q_i^c, c dobrane tak, by suma = 1 (nie faworyzuje outsiderow)."""
    q = np.column_stack([1 / df[f"{prefix}_{s}"].to_numpy(float) for s in sels])
    if method == "prop":
        return q / q.sum(1, keepdims=True)
    lo, hi = np.ones(len(q)), np.full(len(q), 4.0)
    for _ in range(40):
        mid = (lo + hi) / 2
        tot = (q ** mid[:, None]).sum(1)
        lo, hi = np.where(tot > 1, mid, lo), np.where(tot > 1, hi, mid)
    return q ** ((lo + hi) / 2)[:, None]


def selections(odds: pd.DataFrame, market: str, ref: str, bet_price: str = "max", close_ref: str = "psc",
               method: str = "power") -> pd.DataFrame:
    """Format dlugi: jedna linia = jedna mozliwa selekcja. ref: 'avg' | 'ps' | kolumny wlasne (patrz model_selections)."""
    sels = SELS[market]
    pre = "x" if market == "1x2" else "ou"
    need = [f"{pre}_{bet_price}_{s}" for s in sels] + [f"{pre}_{ref}_{s}" for s in sels]
    d = odds.dropna(subset=need).copy()
    p_ref = _fair(d, f"{pre}_{ref}", sels, method)
    # kurs zamkniecia (do CLV): preferuj Pinnacle, potem srednia rynku
    close = np.full_like(p_ref, np.nan)
    for cr in (close_ref, "avgc"):
        cols = [f"{pre}_{cr}_{s}" for s in sels]
        if all(c in d.columns for c in cols):
            ok = d[cols].notna().all(1).to_numpy()
            fill = np.isnan(close[:, 0]) & ok
            if fill.any():
                close[fill] = _fair(d.loc[fill], f"{pre}_{cr}", sels, method)
    rows = []
    for k, s in enumerate(sels):
        win = (d["y"].to_numpy() == k) if market == "1x2" else ((d["goals"].to_numpy() > 2.5) == (s == "over"))
        rows.append(pd.DataFrame({"match_id": d["match_id"].to_numpy(), "date": d["date"].to_numpy(), "div": d["div"].to_numpy(),
                                  "market": market, "sel": s, "odds": d[f"{pre}_{bet_price}_{s}"].to_numpy(),
                                  "p_ref": p_ref[:, k], "p_close": close[:, k], "win": win}))
    return pd.concat(rows, ignore_index=True)


def model_selections(sel: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """Podmienia p_ref na prawdopodobienstwa naszego modelu (1X2). probs: match_id, h, d, a."""
    p = probs.melt(id_vars="match_id", var_name="sel", value_name="p_model")
    m = sel[sel["market"] == "1x2"].merge(p, on=["match_id", "sel"])
    return m.assign(p_ref=m["p_model"]).drop(columns="p_model")


def run(sel: pd.DataFrame, tau: float = 0.03, kelly: float = 0.25, cap: float = 0.03, day_cap: float = 0.15,
        bankroll0: float = 1000.0, min_odds: float = 1.2, max_odds: float = 15.0, strategy: str = "edge"):
    """strategy: 'edge' - gramy gdy odds*p_ref-1 > tau; 'favourite' - baza: zawsze faworyt (kontrola kosztu marzy)."""
    s = sel.copy()
    s["edge"] = s["odds"] * s["p_ref"] - 1
    s = s[np.isfinite(s["edge"]) & (s["p_ref"] > 0) & (s["p_ref"] < 1)]  # odrzuc wadliwe wiersze danych
    s = s[(s["odds"] >= min_odds) & (s["odds"] <= max_odds)]
    if strategy == "edge":
        s = s[s["edge"] > tau]
    else:
        s = s.loc[s.groupby(["match_id", "market"])["p_ref"].idxmax()]
    s = s.loc[s.groupby(["match_id", "market"])["edge"].idxmax()] if strategy == "edge" else s
    s = s.sort_values("date")
    day = pd.to_datetime(s["date"]).dt.normalize()
    bank, peak, mdd, ledger, curve = bankroll0, bankroll0, 0.0, [], []
    for d, g in s.groupby(day, sort=True):
        if bank <= 1:
            break
        if strategy == "edge":
            frac = np.minimum(kelly * g["edge"] / (g["odds"] - 1), cap)
        else:
            frac = pd.Series(0.01, index=g.index)
        if frac.sum() > day_cap:
            frac = frac * day_cap / frac.sum()
        stake = frac * bank
        pnl = np.where(g["win"], stake * (g["odds"] - 1), -stake)
        ledger.append(g.assign(stake=stake, pnl=pnl, bank_before=bank))
        bank += pnl.sum()
        peak = max(peak, bank)
        mdd = max(mdd, 1 - bank / peak)
        curve.append((d, bank))
    if not ledger:
        return pd.DataFrame(), {"n_bets": 0}, pd.DataFrame()
    L = pd.concat(ledger)
    turnover = L["stake"].sum()
    flat = np.where(L["win"], L["odds"] - 1, -1.0)  # zysk przy stalej stawce 1 jednostki - bez efektu procentu skladanego
    rng = np.random.default_rng(0)
    boots = [flat[rng.integers(0, len(flat), len(flat))].mean() for _ in range(1000)]
    clv = (L["odds"] * L["p_close"] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    summ = {"n_bets": int(len(L)), "turnover": float(turnover), "profit": float(L["pnl"].sum()),
            "roi_staked": float(L["pnl"].sum() / turnover), "final_bank": float(bank), "max_drawdown": float(mdd),
            "hit_rate": float(L["win"].mean()), "mean_odds": float(L["odds"].mean()), "mean_edge": float(L["edge"].mean()),
            "flat_roi": float(flat.mean()), "flat_roi_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "clv": float(clv.mean()) if len(clv) else None, "clv_n": int(len(clv))}
    return L, summ, pd.DataFrame(curve, columns=["date", "bank"])
