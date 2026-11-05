import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# XG_OUT pozwala odpalic pipeline na innym katalogu (np. testy / demo)
OUT = Path(os.environ.get("XG_OUT", REPO))
DATA = OUT / "data"
RAW = DATA / "raw"
PRED = OUT / "predictions"
WEB = OUT / "web"

BASE_URL = "https://www.football-data.co.uk"
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
MODEL_NAMES = ["prior", "poisson", "elo_logit", "lgbm", "ensemble"]
DEFAULT_KICKOFF_HOUR = 21  # gdy zrodlo nie podaje godziny meczu
