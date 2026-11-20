# statements

The importer for our household budget spreadsheet (`importer.py`). Each month we download a CSV statement from every
account and `read_statement_file(path)` turns it into `Transaction`s (date, description, amount in pence, negative
for money going out), which the budget then sorts into categories. `read_statement(text)` does the same for a CSV
already in memory. The layouts it reads are in the module docstring; all of our accounts are with UK banks.

`data/statements/` has the first quarter's statements: the current account, the credit card, the joint account and
the savings account. Until now, Sam typed every transaction into the budget by hand from the same statements;
`data/ledger.jsonl` is that hand-kept ledger for the quarter, one line per transaction, with the statement it came
from. The importer should give the same transactions the ledger has.

Tests: `python -m unittest discover tests` from this directory.
