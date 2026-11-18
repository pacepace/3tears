# tillchange

The change calculation for the deli's new till. `change_for(total_pence, tendered_pence, drawer)` says which notes
and coins to hand back for a sale; `make_change(amount_pence, drawer)` does an amount. `load_drawer(path)` reads a
float file: the count of each note and coin, keyed by the denomination in pounds as printed on it (`"20"`, `"0.50"`).

Change is given largest denomination first, as many as the amount still owed and the drawer allow, then the next
largest, down to the penny (the module docstring of `change.py` has the rule). With a properly stocked drawer that
always gives the fewest notes and coins. If the drawer can't make the amount, `CannotMakeChange` is raised and the
cashier calls a supervisor.

`data/float.json` is the standard opening float. `data/till_log.jsonl` is a day of sales from the old till, replayed
against that float when we tested the old till's change logic: `change` on each line is what the old till paid out,
which the cash office counted and agreed. The new till should pay out exactly the same notes and coins for each sale.

Tests: `python -m unittest discover tests` from this directory.
