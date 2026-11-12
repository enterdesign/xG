from datetime import datetime, timedelta, timezone

import pytest

from xg import paper as P

NOW = datetime(2026, 11, 7, 10, 0, tzinfo=timezone.utc)


def ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def event(eid, hours=3, a_prices=None, minutes_old=5, n_books=6, sport="soccer_italy_serie_a"):
    """Pinnacle 1.95/1.95 (uczciwe 50/50) + n zwyklych bukmacherow; a_prices: kursy 'miekkiego' bukmachera na A."""
    upd = ts(NOW - timedelta(minutes=minutes_old))
    def bk(key, a, b): return {"key": key, "last_update": upd, "markets": [{"key": "h2h", "last_update": upd, "outcomes": [{"name": "A", "price": a}, {"name": "B", "price": b}]}]}
    books = [bk("pinnacle", 1.95, 1.95)] + [bk(f"b{i}", 1.9, 1.9) for i in range(n_books - 2)] + [bk("soft", a_prices or 1.9, 1.9)]
    return {"id": eid, "sport_key": sport, "commence_time": ts(NOW + timedelta(hours=hours)), "home_team": "A", "away_team": "B", "bookmakers": books}


class Fake:
    def __init__(self, events, scores=None):
        self.ev, self.sc = events, scores or []
        self.remaining, self.last_cost = 400, None
        self.odds_calls = 0
    def sports(self): return [{"key": "soccer_italy_serie_a", "active": True}]
    def events(self, sport): return [{"id": e["id"], "commence_time": e["commence_time"]} for e in self.ev]
    def odds(self, sport, start, end, regions="eu", markets="h2h", bookmakers=None):
        self.odds_calls += 1
        self.last_bookmakers = bookmakers
        return self.ev
    def scores(self, sport, days=3): return self.sc


def fresh(**cfg_over):
    cfg = {**P.DEFAULT_CFG, "sports": ["soccer_italy_serie_a"], "bookmakers": [], **cfg_over}
    state = {"transactions": [{"t": "x", "type": "init", "amount": 1000.0}], "credits": {"month": None, "used": 0},
             "processed": {}, "last_settle": {}, "runs": []}
    return cfg, state, [], []


def run(api, cfg, state, bets, hist, now=NOW):
    return P.run_once(api, now, cfg, state, bets, hist, persist=False)


def test_value_bet_is_placed_with_capped_stake():
    cfg, st, bets, hist = fresh()
    run(Fake([event("e1", a_prices=2.2)]), cfg, st, bets, hist)
    assert len(bets) == 1
    b = bets[0]
    assert b["selection"] == "A" and b["book"] == "soft" and b["ref_source"] == "pinnacle"
    assert b["edge"] == pytest.approx(0.10, abs=0.01)
    assert b["stake"] <= 0.02 * 1000 + 1e-9          # nigdy wiecej niz 2% kapitalu na zaklad
    assert b["stake"] < 1000 * 0.5                    # zdecydowanie nie all-in


def test_no_bet_without_discrepancy_and_stale_quotes_ignored():
    cfg, st, bets, hist = fresh()
    run(Fake([event("e1", a_prices=1.95)]), cfg, st, bets, hist)
    assert bets == []
    cfg, st, bets, hist = fresh()
    run(Fake([event("e2", a_prices=2.2, minutes_old=300)]), cfg, st, bets, hist)  # wszystkie linie nieaktualne
    assert bets == []


def test_suspicious_edge_is_skipped():
    cfg, st, bets, hist = fresh()
    res = run(Fake([event("e1", a_prices=3.2)]), cfg, st, bets, hist)  # 'edge' ~60% = pewnie blad linii
    assert bets == [] and any("zbyt duza" in n for n in res["notes"])


def test_exposure_caps_and_dedup():
    cfg, st, bets, hist = fresh()
    evs = [event(f"e{i}", a_prices=2.2) for i in range(30)]
    api = Fake(evs)
    run(api, cfg, st, bets, hist)
    total = sum(b["stake"] for b in bets)
    assert total <= 0.10 * 1000 + 1e-6               # limit dzienny 10% kapitalu
    n = len(bets)
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=1))
    assert len(bets) == n and api.odds_calls == 1    # to samo wydarzenie nie jest obstawiane/pobierane drugi raz


def test_settlement_and_accounting():
    cfg, st, bets, hist = fresh()
    api = Fake([event("e1", a_prices=2.2), event("e2", a_prices=2.2)])
    run(api, cfg, st, bets, hist)
    stakes = sum(b["stake"] for b in bets)
    cash, pending, equity, _ = P.cash_and_equity(st, bets)
    assert pending == pytest.approx(stakes) and equity == pytest.approx(1000.0)
    api.sc = [{"id": "e1", "completed": True, "scores": [{"name": "A", "score": "2"}, {"name": "B", "score": "0"}]},
              {"id": "e2", "completed": True, "scores": [{"name": "A", "score": "0"}, {"name": "B", "score": "1"}]}]
    later = NOW + timedelta(hours=8)
    cfg["free_settle_wait_days"] = 0
    def down(url): raise OSError("brak sieci")
    P.settle(cfg, st, bets, api, later, lambda m: None, fetch=down)  # darmowe zrodlo niedostepne -> zapas: API
    by = {b["event_id"]: b for b in bets}
    assert by["e1"]["status"] == "won" and by["e1"]["pnl"] == pytest.approx(by["e1"]["stake"] * 1.2)
    assert by["e2"]["status"] == "lost" and by["e2"]["pnl"] == pytest.approx(-by["e2"]["stake"])
    _, pending, equity, _ = P.cash_and_equity(st, bets)
    assert pending == 0 and equity == pytest.approx(1000 + sum(b["pnl"] for b in bets))


def test_void_after_timeout_returns_stake():
    cfg, st, bets, hist = fresh()
    run(Fake([event("e1", a_prices=2.2)]), cfg, st, bets, hist)
    P.settle(cfg, st, bets, Fake([]), NOW + timedelta(days=6), lambda m: None)
    assert bets[0]["status"] == "void" and bets[0]["pnl"] == 0
    assert P.cash_and_equity(st, bets)[2] == pytest.approx(1000.0)


def test_credit_cap_stops_polling():
    cfg, st, bets, hist = fresh(monthly_credit_cap=1)
    st["credits"] = {"month": NOW.strftime("%Y-%m"), "used": 1}
    api = Fake([event("e1", a_prices=2.2)])
    res = run(api, cfg, st, bets, hist)
    assert api.odds_calls == 0 and bets == [] and any("kredyt" in n for n in res["notes"])


def test_never_stakes_more_than_cash():
    cfg, st, bets, hist = fresh(max_stake_pct=1.0, kelly_fraction=1.0, max_daily_exposure_pct=1.0, max_open_exposure_pct=1.0)
    st["transactions"][0]["amount"] = 50.0
    run(Fake([event(f"e{i}", a_prices=2.2) for i in range(10)]), cfg, st, bets, hist)
    assert sum(b["stake"] for b in bets) <= 50.0 + 1e-9


def test_deposit_withdraw_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PAPER", tmp_path)
    cfg, st, bets, hist = P.load()
    P.save(cfg, st, bets, hist)
    P.deposit(500)
    assert P.cash_and_equity(*P.load()[1:3])[0] == pytest.approx(1500.0)
    with pytest.raises(SystemExit):
        P.deposit(99999, "withdraw")
    P.reset(250)
    cfg, st, bets, hist = P.load()
    assert P.cash_and_equity(st, bets)[2] == pytest.approx(250.0) and bets == []
    assert len(list((tmp_path / "archive").glob("*.json"))) == 1


def test_fair_probs_sum_to_one_and_remove_margin():
    p = P.fair_probs([1.8, 3.6, 4.5])
    assert sum(p) == pytest.approx(1.0) and all(0 < x < 1 for x in p)


def test_calendar_no_api_calls_when_nothing_due_and_daily_refresh():
    cfg, st, bets, hist = fresh()
    far = event("far", hours=30, a_prices=2.2)        # poza oknem 'lookahead' (10 h), ale w kalendarzu
    api = Fake([far])
    calls = {"events": 0}
    orig = api.events
    def counted(sport):
        calls["events"] += 1
        return orig(sport)
    api.events = counted
    run(api, cfg, st, bets, hist)
    assert api.odds_calls == 0 and bets == [] and len(st["calendar"]["events"]) == 1
    n = calls["events"]
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=2))      # kalendarz swiezy -> zadnych zapytan o wydarzenia
    assert calls["events"] == n and api.odds_calls == 0
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=25))     # po dobie kalendarz sie odswieza
    assert calls["events"] > n
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=22))     # mecz w oknie 10 h -> pobranie kursow
    assert api.odds_calls == 1


def test_missing_sports_reported_and_credit_reserve_for_settlement():
    cfg, st, bets, hist = fresh(sports=["soccer_italy_serie_a", "soccer_nie_ma_takiej", "tennis_atp_*"])
    run(Fake([event("e1", a_prices=2.2)]), cfg, st, bets, hist)
    assert "soccer_nie_ma_takiej" in st["sports"]["missing"] and "tennis_atp_*" in st["sports"]["missing"]
    assert "soccer_italy_serie_a" in st["sports"]["resolved"]
    # rezerwa: kursy zatrzymane przy (limit - rezerwa), rozliczanie nadal mozliwe
    cfg, st, bets, hist = fresh(monthly_credit_cap=450, settle_reserve_credits=120)
    st["credits"] = {"month": NOW.strftime("%Y-%m"), "used": 335}
    api = Fake([event("e2", a_prices=2.2)])
    run(api, cfg, st, bets, hist)
    assert api.odds_calls == 0
    assert P.credit_guard(cfg, st, api, NOW)  # pelny limit jeszcze nie wyczerpany -> rozliczanie dozwolone


def test_only_allowed_bookmakers_are_bet_and_title_is_recorded():
    cfg, st, bets, hist = fresh(bookmakers=["pinnacle", "b0", "b1", "b2", "soft"], min_books=4)
    ev = event("e1", a_prices=2.2)
    for bk in ev["bookmakers"]:
        bk["title"] = bk["key"].upper()
    # 'wyjatkowo dobra' oferta u bukmachera spoza listy (np. niedostepnego w Irlandii) ma byc zignorowana
    ev["bookmakers"].append({"key": "outsider", "title": "Outsider", "last_update": ev["bookmakers"][0]["last_update"],
                             "markets": [{"key": "h2h", "last_update": ev["bookmakers"][0]["last_update"],
                                          "outcomes": [{"name": "A", "price": 2.5}, {"name": "B", "price": 1.5}]}]})
    api = Fake([ev])
    run(api, cfg, st, bets, hist)
    assert len(bets) == 1 and bets[0]["book"] == "soft" and bets[0]["book_title"] == "SOFT"
    assert api.last_bookmakers == "pinnacle,b0,b1,b2,soft"
    assert st["credits"]["used"] == 1 and "soft" in st["books_seen"]


def test_poll_gap_limits_credit_spend_per_league():
    cfg, st, bets, hist = fresh(min_poll_gap_hours=6, calendar_refresh_hours=1)
    api = Fake([event("e1", hours=3, a_prices=1.95)])
    run(api, cfg, st, bets, hist)
    assert api.odds_calls == 1 and st["credits"]["used"] == 1
    # nowy mecz tej samej ligi pojawia sie w oknie po 2 h -> NIE pobieramy ponownie (zapas kredytow)
    api.ev = [event("e1", hours=3, a_prices=1.95), event("e2", hours=7, a_prices=1.95)]
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=2))
    assert api.odds_calls == 1 and st["credits"]["used"] == 1
    # po 6 h wolno znow (mecz e2 jest wtedy 1 h przed startem)
    run(api, cfg, st, bets, hist, NOW + timedelta(hours=6))
    assert api.odds_calls == 2 and st["credits"]["used"] == 2


def test_config_file_keeps_only_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PAPER", tmp_path)
    cfg, st, bets, hist = P.load()
    P.save(cfg, st, bets, hist)
    assert P._read("config.json", None) == {}                 # nic recznie nie zmienione -> pusty plik
    P.set_cfg("kelly_fraction", "0.1")
    assert P._read("config.json", None) == {"kelly_fraction": 0.1}
    assert P.load()[0]["kelly_fraction"] == 0.1 and P.load()[0]["lookahead_hours"] == P.DEFAULT_CFG["lookahead_hours"]


def test_resolve_sports_lists_other_active():
    class A:
        def sports(self): return [{"key": "soccer_italy_serie_a", "active": True}, {"key": "handball_x", "active": True}]
    info = {}
    P.resolve_sports({**P.DEFAULT_CFG, "sports": ["soccer_italy_serie_a"]}, A(), lambda m: None, info)
    assert info["other_active"] == ["handball_x"]


def test_rescan_reevaluates_processed_events_once_without_double_betting():
    cfg, st, bets, hist = fresh()
    api = Fake([event("e1", a_prices=1.95)])
    run(api, cfg, st, bets, hist)
    assert bets == [] and api.odds_calls == 1
    run(api, cfg, st, bets, hist, NOW + timedelta(minutes=10))
    assert api.odds_calls == 1                      # bez rescan: nic nie pobiera
    api.ev = [event("e1", a_prices=2.2)]            # kurs sie zmienil
    cfg["rescan"] = True
    run(api, cfg, st, bets, hist, NOW + timedelta(minutes=30))
    assert api.odds_calls == 2 and len(bets) == 1 and cfg["rescan"] is False
    cfg["rescan"] = True
    run(api, cfg, st, bets, hist, NOW + timedelta(minutes=40))
    assert len(bets) == 1                           # to samo wydarzenie nie jest obstawione drugi raz


def test_football_settles_for_free_from_football_data_without_api():
    csv_text = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD2,06/11/2026,18:30,Holstein Kiel,Bielefeld,1,1,D\n"
    bet = {"id": "x", "event_id": "x", "sport": "soccer_germany_bundesliga2", "home": "Holstein Kiel", "away": "Arminia Bielefeld",
           "selection": "Draw", "odds": 3.5, "stake": 2.0, "edge": 0.02, "commence": "2026-11-06T17:30:00Z", "status": "pending"}
    cfg, st, _, _ = fresh()

    class NoApi:
        remaining = last_cost = None
        def scores(self, *a, **k): raise AssertionError("platne API nie powinno byc uzyte")
    later = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
    P.settle(cfg, st, [bet], NoApi(), later, lambda m: None, fetch=lambda url: csv_text)
    assert bet["status"] == "won" and bet["pnl"] == pytest.approx(2.0 * 2.5) and "football-data" in bet["result"]
    assert st["credits"]["used"] == 0
    # brak wyniku w pliku: pilka czeka (bez API) do `free_settle_wait_days`
    bet2 = {**bet, "id": "y", "status": "pending", "away": "Nieznana Druzyna"}
    P.settle(cfg, st, [bet2], NoApi(), later, lambda m: None, fetch=lambda url: csv_text)
    assert bet2["status"] == "pending"


def _xlsx(rows):
    import io, zipfile
    strings = []
    def si(v):
        if v not in strings: strings.append(v)
        return strings.index(v)
    sheet = ""
    for ri, row in enumerate(rows, 1):
        cells = ""
        for ci, v in enumerate(row):
            ref = "ABCDEFGH"[ci] + str(ri)
            cells += f'<c r="{ref}"><v>{v}</v></c>' if isinstance(v, (int, float)) else f'<c r="{ref}" t="s"><v>{si(v)}</v></c>'
        sheet += f"<row r='{ri}'>{cells}</row>"
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>{sheet}</sheetData></worksheet>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>")
    return buf.getvalue()


def _espn_tennis(winner, loser, day="2026-11-06T17:00Z"):
    return {"events": [{"date": day, "groupings": [
        {"grouping": {"displayName": "Men's Singles"}, "competitions": [{"date": day, "status": {"type": {"completed": True}},
            "competitors": [{"winner": True, "athlete": {"displayName": winner}}, {"winner": False, "athlete": {"displayName": loser}}]}]},
        {"grouping": {"displayName": "Men's Doubles"}, "competitions": []}]}]}


def test_tennis_settles_for_free_from_espn_and_unsupported_tournaments_are_not_paid_for():
    bet = {"id": "t", "event_id": "t", "sport": "tennis_atp_paris_masters", "home": "Taylor Fritz", "away": "Carlos Alcaraz",
           "selection": "Taylor Fritz", "odds": 3.0, "stake": 2.0, "edge": 0.02, "commence": "2026-11-06T17:00:00Z", "status": "pending"}
    cfg, st, _, _ = fresh()

    class NoApi:
        remaining = last_cost = None
        def scores(self, *a, **k): raise AssertionError("platne API nie powinno byc uzyte")
    later = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
    P.settle(cfg, st, [bet], NoApi(), later, lambda m: None, fetch_json=lambda url: _espn_tennis("Taylor Fritz", "Carlos Alcaraz"))
    assert bet["status"] == "won" and bet["pnl"] == pytest.approx(4.0) and "ESPN" in bet["result"]
    assert st["tennis_free_ok"] is True and st["credits"]["used"] == 0
    bet2 = {**bet, "id": "u", "status": "pending"}  # zrodlo niedostepne, turniej bez wsparcia API: brak zapytania do API
    notes = []
    P.settle(cfg, st, [bet2], NoApi(), later, notes.append)
    assert bet2["status"] == "pending" and any("brak wynikow" in n for n in notes)


def test_football_settles_from_espn_before_football_data_and_api():
    bet = {"id": "f", "event_id": "f", "sport": "soccer_spain_la_liga", "home": "Málaga", "away": "Real Racing Club de Santander",
           "selection": "Real Racing Club de Santander", "odds": 3.0, "stake": 2.0, "edge": 0.02,
           "commence": "2026-11-07T13:00:00Z", "status": "pending"}
    data = {"events": [{"date": "2026-11-07T13:00Z", "status": {"type": {"completed": True}}, "competitions": [{
        "date": "2026-11-07T13:00Z", "status": {"type": {"completed": True}},
        "competitors": [{"homeAway": "home", "team": {"displayName": "Málaga"}, "score": "1"},
                        {"homeAway": "away", "team": {"displayName": "Racing Santander"}, "score": "2"}]}]}]}
    cfg, st, _, _ = fresh()

    class NoApi:
        remaining = last_cost = None
        def scores(self, *a, **k): raise AssertionError("platne API nie powinno byc uzyte")
    P.settle(cfg, st, [bet], NoApi(), datetime(2026, 11, 7, 20, 0, tzinfo=timezone.utc), lambda m: None,
             fetch=lambda url: "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n", fetch_json=lambda url: data)
    assert bet["status"] == "won" and "ESPN" in bet["result"] and st["credits"]["used"] == 0


def test_no_bets_on_tennis_tournaments_that_cannot_be_settled_until_free_source_works():
    ev = event("e9", a_prices=2.2)
    ev["sport_key"] = "tennis_atp_paris_masters"
    cfg, st, bets, hist = fresh(sports=["tennis_atp_paris_masters"])
    class TennisFake(Fake):
        def sports(self): return [{"key": "tennis_atp_paris_masters", "active": True}]
    api = TennisFake([ev])
    run(api, cfg, st, bets, hist)
    assert api.odds_calls == 0 and not bets
    st["tennis_free_ok"] = True   # darmowe zrodlo dziala -> mozna grac
    run(api, cfg, st, bets, hist)
    assert api.odds_calls >= 1


# ---------- rynek totals (over/under goli) ----------
def event_totals(eid, line=2.5, over_soft=2.2, sport="soccer_italy_serie_a", h2h_soft=1.9):
    """Pinnacle fair 50/50 na over/under + zwykli bukmacherzy 1.9/1.9; 'soft' ma Over po over_soft."""
    ev = event(eid, a_prices=h2h_soft, sport=sport)
    upd = ts(NOW - timedelta(minutes=5))
    for bk in ev["bookmakers"]:
        over = over_soft if bk["key"] == "soft" else 1.95 if bk["key"] == "pinnacle" else 1.9
        under = 1.95 if bk["key"] == "pinnacle" else 1.9
        bk["markets"].append({"key": "totals", "last_update": upd, "outcomes": [
            {"name": "Over", "price": over, "point": line}, {"name": "Under", "price": under, "point": line}]})
    return ev


def test_totals_bet_placed_when_enabled():
    cfg, st, bets, hist = fresh(markets="h2h,totals")
    run(Fake([event_totals("t1")]), cfg, st, bets, hist)
    assert len(bets) == 1
    b = bets[0]
    assert b["market"] == "totals" and b["selection"] == "Over 2.5" and b["line"] == 2.5
    assert b["book"] == "soft" and b["ref_source"] == "pinnacle" and b["edge"] == pytest.approx(0.10, abs=0.01)
    assert b["id"] == "t1:Over 2.5" and b["stake"] <= 0.02 * 1000 + 1e-9


def test_totals_off_by_default_and_for_non_listed_sports():
    cfg, st, bets, hist = fresh()                                    # markets = h2h
    run(Fake([event_totals("t1")]), cfg, st, bets, hist)
    assert bets == []
    cfg, st, bets, hist = fresh(markets="h2h,totals")
    run(Fake([event_totals("t2", sport="basketball_nba")]), cfg, st, bets, hist)
    assert bets == []                                                # totals tylko dla pilki (totals_sports)


@pytest.mark.parametrize("line", [2.0, 2.25, 2.75, 3.0])
def test_totals_ignores_whole_and_quarter_lines(line):
    cfg, st, bets, hist = fresh(markets="h2h,totals")
    run(Fake([event_totals("t1", line=line)]), cfg, st, bets, hist)
    assert bets == []                                                # push / polowki: nie gramy


def test_h2h_and_totals_both_possible_but_never_duplicated():
    cfg, st, bets, hist = fresh(markets="h2h,totals", min_poll_gap_hours=0)
    api = Fake([event_totals("t1", h2h_soft=2.2)])
    run(api, cfg, st, bets, hist)
    assert sorted(b.get("market", "h2h") for b in bets) == ["h2h", "totals"]
    cfg["rescan"] = True
    run(api, cfg, st, bets, hist, NOW + timedelta(minutes=30))
    assert len(bets) == 2                                            # ponowny skan nie dubluje zadnego z rynkow


def test_totals_costs_two_credits_with_up_to_ten_bookmakers():
    cfg, st, bets, hist = fresh(markets="h2h,totals", bookmakers=["pinnacle", "b0", "b1", "b2", "soft"], min_books=4)
    run(Fake([event_totals("t1")]), cfg, st, bets, hist)
    assert st["credits"]["used"] == 2                                # 2 rynki x 1 grupa (<=10 bukmacherow)


def _totals_bet(selection, line_goals_csv):
    csv_text = f"Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nD2,06/11/2026,18:30,Holstein Kiel,Bielefeld,{line_goals_csv},D\n"
    bet = {"id": "x", "event_id": "x", "sport": "soccer_germany_bundesliga2", "home": "Holstein Kiel", "away": "Arminia Bielefeld",
           "market": "totals", "line": 2.5, "selection": selection, "odds": 2.0, "stake": 4.0, "edge": 0.03,
           "commence": "2026-11-06T17:30:00Z", "status": "pending"}
    return csv_text, bet


@pytest.mark.parametrize("selection,goals,expected", [("Over 2.5", "2,1", "won"), ("Under 2.5", "2,1", "lost"),
                                                      ("Over 2.5", "1,1", "lost"), ("Under 2.5", "1,1", "won")])
def test_totals_settle_from_free_results(selection, goals, expected):
    csv_text, bet = _totals_bet(selection, goals)
    cfg, st, _, _ = fresh()

    class NoApi:
        remaining = last_cost = None
        def scores(self, *a, **k): raise AssertionError("platne API nie powinno byc uzyte")
    P.settle(cfg, st, [bet], NoApi(), datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc), lambda m: None, fetch=lambda url: csv_text)
    assert bet["status"] == expected
    assert bet["pnl"] == pytest.approx(4.0 if expected == "won" else -4.0)


def test_totals_settle_from_paid_scores_api():
    bet = {"id": "x", "event_id": "ev9", "sport": "soccer_italy_serie_a", "home": "A", "away": "B", "market": "totals", "line": 2.5,
           "selection": "Over 2.5", "odds": 2.0, "stake": 4.0, "edge": 0.03, "commence": "2026-11-01T17:30:00Z", "status": "pending"}
    cfg, st, _, _ = fresh()
    api = Fake([], scores=[{"id": "ev9", "completed": True, "scores": [{"name": "A", "score": "2"}, {"name": "B", "score": "1"}]}])
    P.settle(cfg, st, [bet], api, datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc), lambda m: None,
             fetch=lambda url: (_ for _ in ()).throw(RuntimeError("offline")))
    assert bet["status"] == "won"
