# loyalty

Points for the farm shop's loyalty card (`points.py`). At the till, `points_for(lines, tier)` works out what a basket
earns a card member: the eligible spend, the base points and the tier bonus, as a `Points` whose `total` goes on the
member's card. `statement(baskets)` adds up a member's month, `basket_lines(row)` reads a basket from the till export,
and `load_export()` reads the export.

The rules are in the module docstring. In short: a point per whole pound of eligible spend (tobacco, lottery, gift
cards and stamps don't count), and on top of that gold members always get double points, silver members get double
points on a basket of £40 or more, and bronze members get 25 bonus points on a basket of £60 or more.

We are replacing the old till software, which worked out points itself. `data/till_export.jsonl` is October's baskets
from the old tills, each with the member's tier and the points the old till put on the card (`points_awarded`), which
members have been checking against their receipts all year. The new code should award the same points on every
basket.

Tests: `python -m unittest discover tests` from this directory.
