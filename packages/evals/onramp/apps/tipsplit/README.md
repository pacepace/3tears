# tipsplit

The bill-splitting step of our table-service payment app. `split(bill)` takes one table's bill and gives what each
diner pays, in pence; `split_file(path)` does a file of bills, one JSON bill per line (`table`, `subtotal_pence`,
`tip_percent`, an optional `voucher_pence`, and `diners` in the order they are on the bill).

How a bill is worked out (the module docstring of `split.py` has the same rules):

1. The tip is a percentage of the full subtotal, before any voucher. Our staff served the whole meal, so a voucher
   never reduces their tip.
2. The voucher then comes off. The table's total is the subtotal plus the tip, less the voucher, to the nearest penny
   (half a penny rounds up), and never below zero.
3. The total is shared as evenly as whole pennies allow; when it doesn't divide exactly, the diners first on the bill
   pay a penny more each.

`data/terminal.jsonl` is a sample of tables from the card terminals at the Leith restaurant last month, before this
step was moved into the app. `charged` on each line is what the terminals charged each diner, worked by the same rules
by the floor manager. The app should charge every diner exactly that.

Tests: `python -m unittest discover tests` from this directory.
