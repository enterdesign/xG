"""Powtorka strategii 'roznice w ofertach' na historii z wirtualnym bankrollem. Zapis: predictions/replay.json
Uzycie: python scripts/replay.py   (wymaga data/raw z `python -m xg download`)"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xg import config as C  # noqa: E402
from xg import oddsdata as O  # noqa: E402
from xg import sim as S  # noqa: E402

START = 1000.0
STRATS = [  # (id, opis, rynek, referencja, cena, od roku, do roku)
    ("1x2_best5", "1X2: najlepsza cena z 5 bukmacherów vs uczciwa wycena Pinnacle", "1x2", "ps", "best5", 2012, 2025),
    ("1x2_max", "1X2: najlepsza cena całego rynku vs uczciwa wycena Pinnacle", "1x2", "ps", "max", 2012, 2025),
    ("ou25_max", "Over/Under 2.5: najlepsza cena rynku vs Pinnacle", "ou25", "ps", "max", 2019, 2025),
]


def main():
    od = O.load_odds()
    out = {"start": START, "params": {"tau": 0.02, "kelly": 0.25, "cap": 0.03, "day_cap": 0.15, "max_odds": 4.0,
                                      "margin_method": "power"}, "strategies": []}
    for sid, desc, market, ref, price, y0, y1 in STRATS:
        sel = S.selections(od, market, ref, bet_price=price, method="power")
        sel = sel[pd.to_datetime(sel["date"]).dt.year.between(y0, y1)]
        L, sm, curve = S.run(sel, tau=0.02, kelly=0.25, cap=0.03, max_odds=4.0, bankroll0=START)
        if sm["n_bets"] == 0:
            continue
        L["year"] = pd.to_datetime(L["date"]).dt.year
        yearly = L.groupby("year").apply(lambda g: pd.Series({
            "n": len(g), "profit": g["pnl"].sum(), "flat_roi": np.where(g["win"], g["odds"] - 1, -1).mean(),
            "bank_end": g["bank_before"].iloc[-1] + g["pnl"].iloc[-1]}))
        curve["date"] = pd.to_datetime(curve["date"])
        m = curve.set_index("date")["bank"].resample("ME").last().ffill()
        halves = {}
        for name, (a, b) in {"pierwsza polowa": (y0, (y0 + y1) // 2), "druga polowa": ((y0 + y1) // 2 + 1, y1)}.items():
            sub = sel[pd.to_datetime(sel["date"]).dt.year.between(a, b)]
            _, s2, _ = S.run(sub, tau=0.02, kelly=0.25, cap=0.03, max_odds=4.0, bankroll0=START)
            halves[name] = {k: s2.get(k) for k in ("n_bets", "flat_roi", "flat_roi_ci", "clv", "final_bank")}
        out["strategies"].append({"id": sid, "desc": desc, "years": [y0, y1], "summary": sm, "halves": halves,
                                  "yearly": json.loads(yearly.reset_index().to_json(orient="records")),
                                  "equity": [{"month": d.strftime("%Y-%m"), "bank": float(v)} for d, v in m.items()]})
        print(f"{sid}: {sm['n_bets']} zakładów, flat ROI {sm['flat_roi']:+.3f} {np.round(sm['flat_roi_ci'], 3)}, "
              f"bankroll {START:.0f} -> {sm['final_bank']:.0f}, max obsunięcie {sm['max_drawdown']:.0%}, CLV {sm['clv']:+.3f}")
    C.PRED.mkdir(parents=True, exist_ok=True)
    (C.PRED / "replay.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
