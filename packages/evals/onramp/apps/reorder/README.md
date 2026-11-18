# reorder

The stockroom's reorder sheet for the shop (`stockroom.py`). Every Monday we review each line we stock: what it sold
over its review window, how long the supplier takes to deliver, how many days of safety stock we keep, and what we hold
and have on order. `reorder_point(line)` says the stock level at which a line should be ordered again, and
`order_quantity(line)` says how many units to order today. `order_sheet(lines)` does both for the whole review, and
`load_lines(path)` reads a review file.

The rules are the ones in the module docstring. In short: daily demand is units sold divided by the days in the window,
kept as a fraction (30 sold in 28 days is 30/28, about 1.07, a day); the reorder point is that demand times (lead days +
safety days), rounded up to a whole unit; a line is due when on hand plus on order is at or below its reorder point, and
a due line orders up to the reorder point plus 14 days of demand, in whole packs.

`data/stock_review.jsonl` is this week's review. Until now the buyer worked the sheet out by hand in a spreadsheet,
from the same figures; `sheet_reorder_point` and `sheet_order` on each line are what the spreadsheet said. Our sheet
should say the same for every line.

Tests: `python -m unittest discover tests` from this directory.
