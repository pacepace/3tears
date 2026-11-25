# standings

The results for the Thursday quiz league at the Fox & Anchor. After the last round each team hands in its answer
sheet, the quizmaster marks them, and `table(entries)` (in `standings.py`) puts the night's teams in order for the
board by the bar. `season(nights)` turns the nights into the league table. The rules are in the module docstring:
most points first; level on points, the team that handed in its last sheet earlier goes above; level on both, they
share the place (1, 2, 2, 4).

Until this term the quizmaster, Dee, wrote the table up by hand. `data/nights.jsonl` is this season so far, one line
per night: every team's points and the time its last sheet was handed in (`entries`), and the table as Dee published
it (`published`, each team's place). What matters to the teams is being in the right place: a night's table is as
good as the share of its teams placed where Dee placed them, and the season reads by that share across the nights.

Tests: `python -m unittest discover tests` from this directory.
