# contactmerge

Our contacts live in three places: the CRM, the newsletter list and the sign-up sheets from events. Each exports a
CSV, and `contacts.py` merges the exports into one address book so nobody gets three copies of the same email.
`merge_exports(paths)` reads the files and returns one `Person` per human being, with every row that is them;
`same_person(a, b)` says whether two rows (`Contact.from_row(row)`) are one person. The rules are in the module
docstring. The short version: two rows are one person when they share an email address (whatever its case) or a phone
number (whatever its formatting); a shared name alone never makes them one person.

`data/exports/` holds this spring's three exports. Before this script, Priya went through likely duplicates by hand:
`data/reviewed_pairs.jsonl` is her review, one line per pair of rows from two different exports, with `same` true
when she found them to be one person. The script should come to the same answer on every pair she reviewed.

Tests: `python -m unittest discover tests` from this directory.
