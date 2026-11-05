"""Uzycie: python -m xg <komenda>   (komendy: download, backtest, ablate, predict, settle, report, all, demo, serve)"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config as C
from . import live as L
from . import report as R
from .backtest import walk_forward
from .features import build_features
from .ingest import download, fetch_fixtures, load_matches
from .metrics import table


def _features():
    matches = load_matches()
    print(f"Mecze: {len(matches)} ({matches['date'].min().date()} .. {matches['date'].max().date()})")
    feat = build_features(matches)
    C.DATA.mkdir(parents=True, exist_ok=True)
    return matches, feat


def cmd_backtest(args, feat):
    bt, last = walk_forward(feat, args.test_from, args.step_days)
    C.PRED.mkdir(parents=True, exist_ok=True)
    bt.to_csv(R.BT_FILE(), index=False)
    R.IMP_FILE().write_text(json.dumps(last.importance()))
    print("\n=== Backtest walk-forward (nizszy logloss/brier/rps = lepiej) ===")
    print(table(bt, R.BT_MODELS).round(4).to_string(index=False))
    mk = bt.dropna(subset=["market_h"])
    if len(mk):
        print("\n=== To samo, tylko mecze z kursami (rynek = poprzeczka) ===")
        print(table(mk, R.BT_MODELS).round(4).to_string(index=False))


def cmd_ablate(args, feat):
    from .ablate import run
    res = run(feat, args.test_from)
    res.to_csv(R.ABL_FILE(), index=False)
    print(f"\nbazowy logloss lgbm: {res.attrs['base']:.4f}\n(delta > 0 = grupa pomaga; ~0 lub < 0 = nic nie wnosi)")
    print(res.round(4).to_string(index=False))


def cmd_live(args, matches, fixtures=None):
    fixtures = fixtures if fixtures is not None else fetch_fixtures(matches)
    live = L.predict(matches, fixtures, horizon_days=args.horizon_days)
    live = L.settle(live, matches)
    L.save(live)
    return live


def main(argv=None):
    p = argparse.ArgumentParser(prog="xg")
    p.add_argument("cmd", choices=["download", "backtest", "ablate", "predict", "settle", "report", "all", "serve"])
    p.add_argument("--seasons", type=int, default=10)
    p.add_argument("--test-from", default=None, help="poczatek okresu testowego, np. 2019-08-01")
    p.add_argument("--step-days", type=int, default=28)
    p.add_argument("--horizon-days", type=float, default=3.0)
    p.add_argument("--with-ablate", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "serve":
        import http.server
        import os
        os.chdir(C.WEB)
        print("Strona: http://localhost:8000"); http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8000)
        return
    if args.cmd in ("download", "all"):
        problems = download(args.seasons)
        if problems:
            print(f"Uwaga: {len(problems)} plikow sie nie pobralo (czesc sezonow moze nie istniec jeszcze).")
        if args.cmd == "download":
            return
    matches, feat = _features()
    if args.cmd in ("backtest", "all"):
        cmd_backtest(args, feat)
    if args.cmd == "ablate" or (args.cmd == "all" and args.with_ablate):
        cmd_ablate(args, feat)
    live = L.load_live()
    if args.cmd in ("predict", "all"):
        live = cmd_live(args, matches)
    elif args.cmd in ("settle", "report"):
        live = L.settle(live, matches)
        L.save(live) if args.cmd == "settle" else None
    if args.cmd in ("report", "all", "predict", "settle"):
        R.write(R.build(live, len(matches)))
        print(f"\nStrona gotowa: otworz {C.WEB / 'index.html'}")


if __name__ == "__main__":
    sys.exit(main())
