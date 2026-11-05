# Symulator gry na zywo (wirtualna gotowka)

Pliki tego katalogu tworzy automat (`python -m xg.paper run`, workflow `paper-run`):
- `config.json` - parametry (stawki, progi, lista lig/sportow, limit kredytow API). Mozna edytowac recznie na GitHubie.
- `state.json` - transakcje na koncie (wplaty/wyplaty/reset), zuzycie kredytow, obsluzone wydarzenia.
- `bets.json` - ksiega zakladow (kurs, bukmacher, uczciwe prawdopodobienstwo, przewaga, stawka, wynik).
- `history.json` - kapital po kazdym przebiegu (do wykresu).
- `archive/` - zarchiwizowane gry po resecie.

Zasilenie/reset konta: GitHub -> Actions -> **paper-admin** -> Run workflow.
Wymagany sekret repozytorium: `ODDS_API_KEY` (Settings -> Secrets and variables -> Actions).
