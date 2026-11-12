# Symulator gry na zywo (wirtualna gotowka)

Pliki tego katalogu tworzy automat (`python -m xg.paper run`, workflow `paper-run`):
- `config.json` - tylko RECZNE zmiany parametrow wzgledem domyslnych z `xg/paper.py` (stawki, progi, lista lig, limity kredytow). Pusty `{}` = wszystko domyslne. Zmiany: workflow `paper-admin` (set) albo edycja pliku.
- `state.json` - transakcje na koncie (wplaty/wyplaty/reset), zuzycie kredytow, obsluzone wydarzenia.
- `bets.json` - ksiega zakladow (kurs, bukmacher, uczciwe prawdopodobienstwo, przewaga, stawka, wynik).
- `history.json` - kapital po kazdym przebiegu (do wykresu).
- `archive/` - zarchiwizowane gry po resecie.

Zasilenie/reset konta: GitHub -> Actions -> **paper-admin** -> Run workflow.
Wymagany sekret repozytorium: `ODDS_API_KEY` (Settings -> Secrets and variables -> Actions).

## Rynki
`config.json` -> `markets`: `h2h` (zwyciezca meczu) albo `h2h,totals` (dodatkowo over/under goli, tylko pilka z `totals_sports`, tylko linie polowkowe 2.5/1.5/3.5...).
Koszt pobrania = liczba rynkow x liczba grup po 10 bukmacherow, wiec `h2h,totals` przy <=10 bukmacherach = 2 kredyty na pobranie.

## Bukmacherzy (dostepni z terenu Irlandii)
`config.json` -> `bookmakers`: lista kluczy bukmacherow wg The Odds API, u ktorych gramy, plus `pinnacle` jako cena odniesienia
(u niego nie gramy). Gdy podana, API zwraca tylko tych bukmacherow (ok. 1 kredyt na zapytanie przy do 10 kluczy).
Pusta lista = wszyscy z regionu `regions`. Lista domyslna to propozycja i wymaga potwierdzenia; `state.json` -> `books_seen`
pokazuje, jakich bukmacherow (klucz + nazwa) faktycznie zwraca API.
