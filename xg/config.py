import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# XG_OUT pozwala odpalic pipeline na innym katalogu (np. testy / demo)
OUT = Path(os.environ.get("XG_OUT", REPO))
DATA = OUT / "data"
RAW = DATA / "raw"
PRED = OUT / "predictions"
WEB = OUT / "web"

BASE_URL = "https://football-data.co.uk"
# Ligi docelowe: na nich oceniamy model i robimy predykcje
TARGET = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
# Ligi dodatkowe: tylko do UCZENIA (wiecej danych o sile druzyn i wariancji wynikow)
EXTRA = {
    "E1": "Championship", "E2": "League One", "E3": "League Two", "SC0": "Scottish Premiership",
    "D2": "2. Bundesliga", "I2": "Serie B", "SP2": "Segunda Division", "F2": "Ligue 2",
    "N1": "Eredivisie", "B1": "Jupiler League", "P1": "Liga Portugal", "T1": "Super Lig", "G1": "Super League Grecja",
}
LEAGUES = {**TARGET, **EXTRA}
MODEL_NAMES = ["prior", "poisson", "elo_logit", "lgbm", "ensemble"]
DEFAULT_KICKOFF_HOUR = 21  # gdy zrodlo nie podaje godziny meczu
