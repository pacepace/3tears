# tripcarbon

Business-travel emissions for our quarterly sustainability report. `footprint_kg(trip)` gives one traveller's trip in
kg CO2e; `Trip.from_row` reads a trip as the travel desk exports it (`trip_id`, `legs` with `mode`, `km` and, for a
flight, `cabin`, and `hotels`, one stay per hotel with its `country` and `nights`), and `footprint_file(path)` does a
whole export. How a trip is worked out is in the module docstring of `footprint.py`: each leg's distance times its
mode's conversion factor for the 2026 reporting year, flights weighted by cabin, plus a factor for each night of each
hotel stay, by country.

The report's figures are audited. `data/audited.jsonl` is last quarter's audit sample: each trip as exported, with
`audited_kg`, the auditors' own figure, worked from the 2026 factors and their own distances and booking records.
Their distances never quite match the travel desk's, so we don't expect to hit their figure exactly. The agreed
standard is that our estimate is within 5% of the audited figure for each trip; a trip further out than that is one
we have got wrong, and the auditors will ask about it.

Tests: `python -m unittest discover tests` from this directory.
