"""Symulator gry na zywo z wirtualna gotowka ("paper trading") - szuka ROZBIEZNOSCI W OFERTACH, nie przewiduje wynikow.

Zasada: uczciwe prawdopodobienstwo = wycena Pinnacle po usunieciu marzy (a gdy jej brak: mediana innych bukmacherow).
Gramy tam, gdzie NAJLEPSZA cena innego bukmachera jest wyraznie wyzsza od uczciwej ceny:  edge = kurs * p_uczciwe - 1.

Stawki (ryzyko vs wartosc oczekiwana):  stawka = ulamek Kelly'ego * edge/(kurs-1) * kapital, z twardymi limitami
(max % kapitalu na zaklad, na dzien, lacznie otwartych) - automat nigdy nie gra 'all-in'.

Czysty stdlib (bez pandas), zeby uruchamianie w GitHub Actions bylo szybkie.
Komendy: python -m xg.paper run | status | deposit KWOTA | withdraw KWOTA | reset [--bankroll X] | set KLUCZ WARTOSC | report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .oddsapi import ApiError, OddsApi

REPO = Path(__file__).resolve().parent.parent
PAPER = Path(os.environ.get("XG_PAPER_DIR", REPO / "paper"))
WEB = Path(os.environ.get("XG_WEB_DIR", REPO / "web"))

DEFAULT_CFG = {
    "start_bankroll": 1000.0,
    "sports": ["soccer_italy_serie_a", "soccer_italy_serie_b", "soccer_turkey_super_league", "soccer_efl_champ",
               "soccer_england_league1", "soccer_england_league2", "soccer_germany_bundesliga2",
               "basketball_nba", "icehockey_nhl", "tennis_atp_*", "tennis_wta_*"],
    "max_dynamic_per_pattern": 2,
    "regions": "eu", "markets": "h2h",
    "lookahead_hours": 10, "min_lead_minutes": 20, "max_quote_age_minutes": 90,
    "reference_book": "pinnacle", "min_books": 5,
    "min_edge_pinnacle": 0.02, "min_edge_consensus": 0.03, "max_edge": 0.25,
    "min_odds": 1.25, "max_odds": 4.5,
    "kelly_fraction": 0.25, "max_stake_pct": 0.02, "max_daily_exposure_pct": 0.10, "max_open_exposure_pct": 0.20,
    "min_stake": 1.0,
    "monthly_credit_cap": 450, "min_remaining_credits": 25,
    "settle_min_gap_hours": 10, "void_after_days": 4,
}


# ---------- pomocnicze ----------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _read(name, default):
    p = PAPER / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _write(name, obj):
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load():
    cfg = {**DEFAULT_CFG, **_read("config.json", {})}
    state = _read("state.json", None)
    if state is None:
        state = {"transactions": [{"t": iso(utcnow()), "type": "init", "amount": cfg["start_bankroll"]}],
                 "credits": {"month": None, "used": 0}, "processed": {}, "last_settle": {}, "runs": []}
    return cfg, state, _read("bets.json", []), _read("history.json", [])


def save(cfg, state, bets, hist):
    _write("config.json", cfg)
    _write("state.json", state)
    _write("bets.json", bets)
    _write("history.json", hist)


# ---------- matematyka ----------
def fair_probs(prices: list[float]) -> list[float]:
    """Usuwanie marzy metoda potegowa: p_i = q_i^c, c dobrane tak, by suma = 1 (nie zawyza szans outsiderow)."""
    q = [1.0 / p for p in prices]
    lo, hi = 1.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(x ** mid for x in q) > 1:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    return [x ** c for x in q]


def quotes_for(event: dict, now: datetime, cfg: dict) -> dict[str, dict[str, float]]:
    """bukmacher -> {nazwa wyniku: kurs}; tylko swieze notowania i spojne zestawy wynikow."""
    out, sets = {}, {}
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != cfg["markets"].split(",")[0]:
                continue
            upd = m.get("last_update") or bk.get("last_update")
            if upd and (now - parse_t(upd)) > timedelta(minutes=cfg["max_quote_age_minutes"]):
                continue  # nieaktualna linia = najczestsze zrodlo falszywych okazji
            prices = {o["name"]: float(o["price"]) for o in m.get("outcomes", []) if o.get("price")}
            if len(prices) >= 2 and all(p > 1.0 for p in prices.values()):
                out[bk["key"]] = prices
                sets[bk["key"]] = frozenset(prices)
    if not out:
        return {}
    common = max(set(sets.values()), key=list(sets.values()).count)  # najczestszy zestaw wynikow
    return {b: p for b, p in out.items() if frozenset(p) == common}


def evaluate_event(event: dict, now: datetime, cfg: dict) -> dict | None:
    q = quotes_for(event, now, cfg)
    if len(q) < cfg["min_books"]:
        return None
    names = sorted(next(iter(q.values())))
    fair = {b: dict(zip(names, fair_probs([p[n] for n in names]))) for b, p in q.items()}
    ref = cfg["reference_book"] if cfg["reference_book"] in q else None
    best = None
    for b, prices in q.items():
        if b == ref:
            continue
        for n in names:
            if ref:
                p_ref, src, min_edge = fair[ref][n], ref, cfg["min_edge_pinnacle"]
            else:
                others = [fair[o][n] for o in q if o != b]
                if len(others) < cfg["min_books"] - 1:
                    continue
                p_ref, src, min_edge = statistics.median(others), "consensus", cfg["min_edge_consensus"]
            odds = prices[n]
            edge = odds * p_ref - 1
            if not (cfg["min_odds"] <= odds <= cfg["max_odds"]) or edge < min_edge:
                continue
            cand = {"book": b, "selection": n, "odds": odds, "p_ref": p_ref, "edge": edge, "ref_source": src,
                    "n_books": len(q), "suspicious": edge > cfg["max_edge"]}
            if best is None or edge > best["edge"]:
                best = cand
    if best and best["suspicious"]:
        return {**best, "skip": "zbyt duza rozbieznosc (mozliwy blad/nieaktualna linia)"}
    return best


def diagnose(event: dict, now: datetime, cfg: dict) -> dict:
    """Dlaczego wydarzenie nie dalo zakladu: ile bukmacherow, czy jest Pinnacle, jaka byla najwieksza przewaga."""
    q = quotes_for(event, now, cfg)
    d = {"event": f"{event.get('home_team')} - {event.get('away_team')}", "n_books": len(q),
         "has_ref": cfg["reference_book"] in q, "best_edge": None, "why": None}
    if len(q) < cfg["min_books"]:
        d["why"] = f"za malo bukmacherow ({len(q)} < {cfg['min_books']})"
        return d
    names = sorted(next(iter(q.values())))
    fair = {b: dict(zip(names, fair_probs([p[n] for n in names]))) for b, p in q.items()}
    ref = cfg["reference_book"] if cfg["reference_book"] in q else None
    best = None
    for b, prices in q.items():
        if b == ref:
            continue
        for n in names:
            p_ref = fair[ref][n] if ref else statistics.median([fair[o][n] for o in q if o != b])
            e = prices[n] * p_ref - 1
            if best is None or e > best:
                best = e
    d["best_edge"] = best
    d["why"] = "brak rozbieznosci powyzej progu" if best is not None else "brak danych"
    if not ref:
        d["why"] += " (bez Pinnacle - mediana rynku)"
    return d


def kelly_fraction(edge: float, odds: float) -> float:
    return max(0.0, edge / (odds - 1))  # (p*o-1)/(o-1) przy p = p_uczciwe


def cash_and_equity(state, bets):
    deposits = sum(t["amount"] for t in state["transactions"])
    settled_pnl = sum(b.get("pnl", 0.0) for b in bets if b["status"] in ("won", "lost", "void"))
    pending = sum(b["stake"] for b in bets if b["status"] == "pending")
    cash = deposits + settled_pnl - pending
    return cash, pending, cash + pending, deposits


def plan_stakes(cands: list[dict], cfg, state, bets, now):
    """Przydziela stawki najlepszym okazjom pierwszym, pilnujac limitow. Zwraca liste zakladow."""
    cash, pending, equity, _ = cash_and_equity(state, bets)
    today = now.strftime("%Y-%m-%d")
    day_used = sum(b["stake"] for b in bets if b["placed_at"][:10] == today)
    day_cap = cfg["max_daily_exposure_pct"] * equity
    open_cap = cfg["max_open_exposure_pct"] * equity
    placed = []
    for c in sorted(cands, key=lambda x: -x["edge"]):
        f = min(cfg["kelly_fraction"] * kelly_fraction(c["edge"], c["odds"]), cfg["max_stake_pct"])
        stake = f * equity
        stake = min(stake, cash, day_cap - day_used, open_cap - pending)
        stake = math.floor(stake * 100) / 100
        if stake < cfg["min_stake"]:
            continue
        placed.append({**c, "stake": stake, "kelly_frac_used": stake / equity, "ev": stake * c["edge"]})
        cash -= stake
        pending += stake
        day_used += stake
    return placed


# ---------- krok automatu ----------
def resolve_sports(cfg, api, warn):
    try:
        active = {s["key"] for s in api.sports() if s.get("active")}
    except ApiError as e:
        warn(f"/sports: {e}")
        return []
    out = []
    for item in cfg["sports"]:
        if item.endswith("*"):
            pre = item[:-1]
            out += sorted(k for k in active if k.startswith(pre))[:cfg["max_dynamic_per_pattern"]]
        elif item in active:
            out.append(item)
    return out


def credit_guard(cfg, state, api, now) -> bool:
    """True = wolno wydac kredyty. Liczy zuzycie w miesiacu (z naglowkow API, a w razie braku lokalnie)."""
    cm = state["credits"]
    month = now.strftime("%Y-%m")
    if cm.get("month") != month:
        cm.update({"month": month, "used": 0})
    if api.remaining is not None and api.remaining < cfg["min_remaining_credits"]:
        return False
    return cm["used"] < cfg["monthly_credit_cap"]


def spend(state, api, default_cost):
    cost = api.last_cost if api.last_cost is not None else default_cost
    state["credits"]["used"] += cost


def settle(cfg, state, bets, api, now, warn):
    pend = [b for b in bets if b["status"] == "pending"]
    by_sport = {}
    for b in pend:
        if parse_t(b["commence"]) + timedelta(hours=2) < now:
            by_sport.setdefault(b["sport"], []).append(b)
    for sport, group in by_sport.items():
        last = state["last_settle"].get(sport)
        if last and now - parse_t(last) < timedelta(hours=cfg["settle_min_gap_hours"]):
            continue
        if not credit_guard(cfg, state, api, now):
            warn("limit kredytow - pomijam rozliczanie")
            break
        try:
            res = {r["id"]: r for r in api.scores(sport, 3)}
        except ApiError as e:
            warn(f"scores {sport}: {e}")
            continue
        spend(state, api, 2)
        state["last_settle"][sport] = iso(now)
        for b in group:
            r = res.get(b["event_id"])
            if r and r.get("completed") and r.get("scores"):
                sc = {s["name"]: float(s["score"]) for s in r["scores"] if s.get("score") not in (None, "")}
                if len(sc) >= 2:
                    top = max(sc.values())
                    winners = [n for n, v in sc.items() if v == top]
                    winner = winners[0] if len(winners) == 1 else "Draw"
                    _close(b, "won" if b["selection"] == winner else "lost", now, f"{b['home']} {sc.get(b['home'])}:{sc.get(b['away'])} {b['away']}")
                    continue
    for b in bets:  # niedokonczone zbyt dlugo (przelozony/odwolany) -> zwrot stawki
        if b["status"] == "pending" and now - parse_t(b["commence"]) > timedelta(days=cfg["void_after_days"]):
            _close(b, "void", now, "brak wyniku - zwrot stawki")


def _close(b, status, now, result):
    b["status"] = status
    b["pnl"] = {"won": b["stake"] * (b["odds"] - 1), "lost": -b["stake"], "void": 0.0}[status]
    b["settled_at"] = iso(now)
    b["result"] = result


def run_once(api, now=None, cfg=None, state=None, bets=None, hist=None, persist=True):
    now = now or utcnow()
    if cfg is None:
        cfg, state, bets, hist = load()
    notes = []
    warn = notes.append
    settle(cfg, state, bets, api, now, warn)

    candidates, n_events, polled, diag = [], 0, 0, []
    for sport in resolve_sports(cfg, api, warn):
        try:
            evs = api.events(sport)  # darmowe
        except ApiError as e:
            warn(f"events {sport}: {e}")
            continue
        soon = [e for e in evs if now + timedelta(minutes=cfg["min_lead_minutes"]) < parse_t(e["commence_time"])
                <= now + timedelta(hours=cfg["lookahead_hours"]) and e["id"] not in state["processed"]]
        if not soon:
            continue
        if not credit_guard(cfg, state, api, now):
            warn("limit kredytow - przerywam pobieranie kursow")
            break
        try:
            data = api.odds(sport, iso(now), iso(now + timedelta(hours=cfg["lookahead_hours"])), cfg["regions"], cfg["markets"])
        except ApiError as e:
            warn(f"odds {sport}: {e}")
            continue
        spend(state, api, len(cfg["markets"].split(",")) * len(cfg["regions"].split(",")))
        polled += 1
        for ev in data:
            if ev["id"] in state["processed"] or not (now + timedelta(minutes=cfg["min_lead_minutes"]) < parse_t(ev["commence_time"])):
                continue
            state["processed"][ev["id"]] = ev["commence_time"]
            n_events += 1
            c = evaluate_event(ev, now, cfg)
            if not c:
                diag.append({**diagnose(ev, now, cfg), "sport": sport})
            if c and "skip" in c:
                warn(f"pominieto {ev['home_team']} - {ev['away_team']}: {c['skip']} ({c['edge']:.0%})")
            elif c:
                candidates.append({**c, "event_id": ev["id"], "sport": sport, "commence": ev["commence_time"],
                                   "home": ev["home_team"], "away": ev["away_team"]})
    for p in plan_stakes(candidates, cfg, state, bets, now):
        bets.append({**p, "id": f"{p['event_id']}:{p['selection']}", "placed_at": iso(now), "status": "pending"})

    # porzadki i punkt historii
    cutoff = iso(now - timedelta(days=7))
    state["processed"] = {k: v for k, v in state["processed"].items() if v >= cutoff}
    cash, pending, equity, deposits = cash_and_equity(state, bets)
    hist.append({"t": iso(now), "equity": round(equity, 2), "cash": round(cash, 2), "pending": round(pending, 2),
                 "deposits": round(deposits, 2)})
    state["runs"] = (state["runs"] + [{"t": iso(now), "polled": polled, "events": n_events, "new_bets": len(
        [b for b in bets if b["placed_at"] == iso(now)]), "credits_used_month": state["credits"]["used"],
        "remaining": api.remaining, "notes": notes[:10], "diag": diag[:40]}])[-60:]
    if persist:
        save(cfg, state, bets, hist)
    return {"notes": notes, "polled": polled, "events": n_events}


# ---------- konto: zasilenie / reset ----------
def deposit(amount: float, kind="deposit"):
    cfg, state, bets, hist = load()
    cash, _, _, _ = cash_and_equity(state, bets)
    if kind == "withdraw":
        if amount > cash:
            raise SystemExit(f"Nie mozna wyplacic {amount:.2f} - dostepna gotowka to {cash:.2f}.")
        amount = -amount
    state["transactions"].append({"t": iso(utcnow()), "type": kind, "amount": round(amount, 2)})
    save(cfg, state, bets, hist)
    print(f"{kind}: {amount:+.2f}")


def reset(bankroll: float | None = None):
    """Archiwizuje cala dotychczasowa gre i zaczyna od nowa z podana gotowka."""
    cfg, state, bets, hist = load()
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "archive").mkdir(exist_ok=True)
    stamp = utcnow().strftime("%Y%m%d-%H%M%S")
    _write(f"archive/{stamp}.json", {"state": state, "bets": bets, "history": hist})
    bankroll = bankroll if bankroll is not None else cfg["start_bankroll"]
    state = {"transactions": [{"t": iso(utcnow()), "type": "init", "amount": bankroll}],
             "credits": state["credits"], "processed": state["processed"], "last_settle": {}, "runs": []}
    save(cfg, state, [], [])
    print(f"Reset: nowa gra z {bankroll:.2f}")


def set_cfg(key, value):
    cfg, state, bets, hist = load()
    if key not in DEFAULT_CFG:
        raise SystemExit(f"Nieznany parametr '{key}'. Dostepne: {', '.join(DEFAULT_CFG)}")
    try:
        cfg[key] = json.loads(value)
    except json.JSONDecodeError:
        cfg[key] = value
    save(cfg, state, bets, hist)
    print(f"{key} = {cfg[key]}")


# ---------- raport dla strony ----------
def summary(now=None) -> dict:
    now = now or utcnow()
    cfg, state, bets, hist = load()
    cash, pending, equity, deposits = cash_and_equity(state, bets)
    settled = [b for b in bets if b["status"] in ("won", "lost")]
    pnl = sum(b["pnl"] for b in settled)
    exp = sum(b["stake"] * b["edge"] for b in settled)
    var = 0.0
    for b in settled:
        p = (1 + b["edge"]) / b["odds"]
        var += b["stake"] ** 2 * (p * (b["odds"] - 1) ** 2 + (1 - p) - b["edge"] ** 2)
    sd = math.sqrt(var)
    peak, mdd = 0.0, 0.0
    for h in hist:
        net = h["equity"] - h["deposits"]
        peak = max(peak, net + deposits)
        mdd = max(mdd, (peak - (net + deposits)) / peak if peak else 0)
    per = {}
    for b in settled:
        d = per.setdefault(b["sport"], {"n": 0, "stake": 0.0, "pnl": 0.0})
        d["n"] += 1
        d["stake"] += b["stake"]
        d["pnl"] += b["pnl"]
    return {
        "generated_at": iso(now), "cfg": cfg,
        "account": {"cash": cash, "pending": pending, "equity": equity, "deposits": deposits, "net_pnl": equity - deposits},
        "transactions": state["transactions"][-20:],
        "stats": {"n_settled": len(settled), "n_open": len([b for b in bets if b["status"] == "pending"]),
                  "turnover": sum(b["stake"] for b in settled), "pnl": pnl, "expected_pnl": exp, "sd": sd,
                  "z": (pnl - exp) / sd if sd else None, "hit_rate": (sum(b["status"] == "won" for b in settled) / len(settled)) if settled else None,
                  "max_drawdown": mdd, "open_ev": sum(b["stake"] * b["edge"] for b in bets if b["status"] == "pending")},
        "credits": {**state["credits"], "cap": cfg["monthly_credit_cap"]},
        "per_sport": per,
        "open": sorted([b for b in bets if b["status"] == "pending"], key=lambda b: b["commence"]),
        "settled": sorted([b for b in bets if b["status"] != "pending"], key=lambda b: b.get("settled_at", ""), reverse=True)[:200],
        "history": hist[-800:], "runs": state["runs"][-12:],
    }


def write_web():
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "paper.js").write_text("window.XG_PAPER = " + json.dumps(summary(), ensure_ascii=False) + ";\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(prog="xg.paper")
    p.add_argument("cmd", choices=["run", "status", "deposit", "withdraw", "reset", "set", "report"])
    p.add_argument("args", nargs="*")
    p.add_argument("--bankroll", type=float)
    a = p.parse_args(argv)
    if a.cmd == "run":
        try:
            res = run_once(OddsApi(os.environ.get("ODDS_API_KEY", "")))
        except ApiError as e:
            raise SystemExit(f"Blad: {e}")
        print(f"Zapytania o kursy: {res['polled']}, nowe wydarzenia: {res['events']}")
        for n in res["notes"]:
            print("  -", n)
        write_web()
    elif a.cmd == "status":
        s = summary()
        ac, st = s["account"], s["stats"]
        print(f"Kapital {ac['equity']:.2f} (gotowka {ac['cash']:.2f}, w grze {ac['pending']:.2f}), wplaty {ac['deposits']:.2f}, wynik {ac['net_pnl']:+.2f}")
        print(f"Rozliczone: {st['n_settled']}, otwarte: {st['n_open']}, kredyty w tym miesiacu: {s['credits']['used']}/{s['credits']['cap']}")
    elif a.cmd in ("deposit", "withdraw"):
        deposit(float(a.args[0]), a.cmd)
        write_web()
    elif a.cmd == "reset":
        reset(a.bankroll)
        write_web()
    elif a.cmd == "set":
        set_cfg(a.args[0], a.args[1])
    elif a.cmd == "report":
        write_web()


if __name__ == "__main__":
    sys.exit(main())
