# xG - predictor wynikow meczow pilkarskich

Prywatny projekt badawczy: jak dokladnie da sie przewidywac mecze (5 lig: PL, La Liga, Bundesliga, Serie A, Ligue 1).
Nie do obstawiania. Miara jakosci to **log-loss / Brier / RPS** (jakosc prawdopodobienstw), nie "trafione/nietrafione".
Poprzeczka: kursy bukmacherskie (uzywane WYLACZNIE jako punkt odniesienia, nie jako cecha modelu).

## Jak to dziala
1. `xg/ingest.py` - dane z football-data.co.uk (wyniki, strzaly, kursy).
2. `xg/features.py` - cechy bez wycieku z przyszlosci: ELO, sila atak/obrona (Poisson), forma 3/5/10 meczow, strzaly, dom/wyjazd, zmeczenie, H2H.
3. `xg/models.py` - prior, Poisson, ELO->logit, LightGBM (z kalibracja), ensemble z wagami dobranymi na walidacji.
4. `xg/backtest.py` - walk-forward: kazdy okres przewidywany modelem uczonym tylko na przeszlosci.
5. `xg/live.py` - predykcja nadchodzacych meczow, zamrozenie w chwili startu, rozliczenie z prawdziwym wynikiem.
6. `web/index.html` - strona (otwierasz dwuklikiem po `python -m xg all`).

Testy (m.in. test braku wycieku z przyszlosci): `python -m pytest -q`.

## Plan na wieczor (po powrocie do domu)
```bash
git clone https://github.com/enterdesign/xG.git && cd xG
git checkout claude/serene-meitner-2ku2pt
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # sanity check
python -m xg all --with-ablate                         # pobranie danych, backtest, ablacja, predykcje, strona
```
Potem otworz `web/index.html`. Na CPU (7800X3D) calosc to rzad minut; GPU nie jest potrzebne.

Co sprawdzic jako pierwsze:
- Zakladka **Backtest**: czy ensemble bije Poissona i ELO, jak daleko do rynku (log-loss rynku ~0.97-1.00 to typowy poziom).
- Zakladka **Cechy** + ablacja: ktore grupy cech realnie pomagaja (H2H zapewne nie).
- Kalibracja: punkty blisko przekatnej.
- Dopiero potem `python -m xg predict` raz dziennie (zbiera predykcje na zywo, ktore beda rozliczane).

Pojedyncze komendy: `download`, `backtest`, `ablate`, `predict`, `settle`, `report`, `serve`.
Opcje: `--seasons 10`, `--test-from 2019-08-01`, `--step-days 28`, `--horizon-days 3`.

## Ograniczenia (swiadome)
- Brak danych o kontuzjach/skladach i xG (Understat) - to kolejna iteracja, jesli baseline pokaze, ze warto.
- Zmeczenie liczone tylko z meczow ligowych (bez pucharow).
- ELO liczymy sami z wynikow (zewnetrzne API z ELO nie jest potrzebne). `xg/external_api.py` to nieuzywany szkielet.
- Terminarz: `openfootball` (GitHub) + `fixtures.csv` z football-data; nazwy druzyn tlumaczy `xg/fixtures_src.py` (nierozpoznane nazwy sa zglaszane - dopisz je w `ALIASES`). Daty w openfootball moga byc nieaktualne po przelozeniu meczu.
- `scripts/demo.py` generuje dane SYNTETYCZNE do podgladu strony (strona pokazuje wtedy zolty baner).
