import numpy as np
import pandas as pd
import pytest

from xg import live as L
from xg.backtest import walk_forward
from xg.features import FEATURE_COLS, build_features
from xg.metrics import metrics
from xg.synth import make_matches


@pytest.fixture(scope="module")
def synth():
    return make_matches(n_seasons=7, teams_per_league=10, seed=1)


def test_no_future_leakage(synth):
    """Zmiana wynikow w przyszlosci NIE moze zmienic cech wczesniejszych meczow."""
    f1 = build_features(synth)
    cut = synth["date"].quantile(0.6)
    mod = synth.copy()
    late = mod["date"] >= cut
    rng = np.random.default_rng(5)
    mod.loc[late, "fthg"] = rng.integers(0, 6, late.sum())
    mod.loc[late, "ftag"] = rng.integers(0, 6, late.sum())
    mod.loc[late, "st_h"] = 99
    f2 = build_features(mod)
    early = f1["date"] < cut
    pd.testing.assert_frame_equal(f1.loc[early, FEATURE_COLS].reset_index(drop=True),
                                  f2.loc[early, FEATURE_COLS].reset_index(drop=True))


def test_same_day_matches_do_not_see_each_other(synth):
    f = build_features(synth)
    day = f["date"].dt.normalize()
    mod = synth.copy()
    d0 = day.iloc[len(day) // 2]
    mod.loc[mod["date"].dt.normalize() == d0, ["fthg", "ftag"]] = 7
    f2 = build_features(mod)
    same = f["date"].dt.normalize() <= d0
    pd.testing.assert_frame_equal(f.loc[same, FEATURE_COLS].reset_index(drop=True),
                                  f2.loc[same, FEATURE_COLS].reset_index(drop=True))


def test_walk_forward_beats_prior_and_is_sane(synth):
    feat = build_features(synth)
    bt, model = walk_forward(feat, step_days=90, verbose=False)
    ens = metrics(bt[["ensemble_h", "ensemble_d", "ensemble_a"]].to_numpy(), bt["y"].to_numpy())
    pri = metrics(bt[["prior_h", "prior_d", "prior_a"]].to_numpy(), bt["y"].to_numpy())
    assert ens["logloss"] < pri["logloss"]
    assert (bt[["ensemble_h", "ensemble_d", "ensemble_a"]].sum(axis=1).round(6) == 1).all()
    assert bt["date"].min() > synth["date"].min()
    assert abs(sum(model.weights.values()) - 1) < 1e-9


def test_live_freeze_and_settle(synth):
    played = synth[synth["date"] < synth["date"].max() - pd.Timedelta(days=30)].copy()
    fx = synth[synth["date"] >= synth["date"].max() - pd.Timedelta(days=30)].copy()
    fx[["fthg", "ftag"]] = np.nan
    now = fx["date"].min() - pd.Timedelta(hours=1)
    live = L.predict(played, fx, now=now, horizon_days=2)
    assert len(live) > 0 and live["y"].isna().all()
    frozen_before = live.copy()
    # nastepny dzien: mecze, ktore sie zaczely, maja zostac nietkniete
    now2 = now + pd.Timedelta(days=1)
    L_save = L.LIVE_FILE
    import xg.config as C
    C.PRED.mkdir(parents=True, exist_ok=True)
    L.save(live)
    live2 = L.predict(played, fx, now=now2, horizon_days=2)
    started = frozen_before[frozen_before["kickoff"] <= now2]
    merged = started.merge(live2, on="match_id", suffixes=("_a", "_b"))
    assert len(merged) == len(started) and np.allclose(merged["ensemble_h_a"], merged["ensemble_h_b"], atol=1e-9)
    # rozliczenie
    settled = L.settle(live2, synth)
    assert settled["y"].notna().sum() > 0


def test_team_name_mapping():
    from xg.fixtures_src import map_names
    known = ["Man United", "Ath Bilbao", "Bayern Munich", "Paris SG", "Nott'm Forest", "Sociedad", "Inter", "Bologna", "Brest"]
    of = ["Manchester United FC", "Athletic Club", "FC Bayern München", "Paris Saint-Germain FC", "Nottingham Forest FC",
          "Real Sociedad de Fútbol", "FC Internazionale Milano", "Bologna FC 1909", "Stade Brestois 29", "Nowa Druzyna FC"]
    mp, missing = map_names(of, known)
    assert mp["Manchester United FC"] == "Man United" and mp["Bologna FC 1909"] == "Bologna" and mp["Athletic Club"] == "Ath Bilbao"
    assert missing == ["Nowa Druzyna FC"]


def test_goal_markets_consistent():
    from xg.goals import goal_markets
    g = goal_markets(np.array([1.4, 2.2, 0.8]), np.array([1.1, 0.9, 1.0]))
    for a, b in (("ou2.5_over", "ou2.5_under"), ("btts_yes", "btts_no"), ("home_scores", "home_no_score")):
        assert np.allclose(g[a] + g[b], 1)
    assert (g["ou0.5_over"] > g["ou1.5_over"]).all() and (g["ou1.5_over"] > g["ou2.5_over"]).all()
    assert g["ou2.5_over"][1] > g["ou2.5_over"][0]  # wiecej oczekiwanych goli = wieksza szansa na over


def test_power_margin_removal_and_fair_odds_sim():
    from xg.sim import _fair, run
    df = pd.DataFrame({"x_a_h": [2.0, 1.5], "x_a_d": [3.5, 4.5], "x_a_a": [4.0, 7.0]})
    p = _fair(df, "x_a", ["h", "d", "a"], "power")
    assert np.allclose(p.sum(1), 1) and (p > 0).all()
    # uczciwe kursy + uczciwe wyniki => zwrot ~0 (brak 'magicznego' zysku w silniku)
    rng = np.random.default_rng(0)
    n = 40000
    pr = rng.uniform(0.2, 0.6, n)
    win = rng.random(n) < pr
    sel = pd.DataFrame({"match_id": range(n), "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) // 5, "D"),
                        "market": "1x2", "sel": "h", "odds": 1 / pr * 1.03, "p_ref": pr, "p_close": pr, "win": win})
    _, s, _ = run(sel, tau=0.0)
    assert abs(s["flat_roi"] - 0.03) < 0.03
