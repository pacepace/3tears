# logtally

Daily traffic figures for the café's website (`tally.py`), from the web server's own access logs. `tally(lines)`
counts a day's log lines by page: hits, bytes sent, 404s and server errors, plus how many lines it could not read.
`tally_file(path)` does the same for a log file, `day_files()` lists the day files in `data/logs/`, and
`Tally.busiest(n)` ranks the pages by hits. `parse(line)` reads one line.

The log format and the counting rules are in the module docstring. A page is a path without its query string; every
request counts as a hit, whatever its status, including the ones answered with no body (logged with `-` for the size,
counted as 0 bytes); only a line that is not in the log format at all is left out, as unparsed.

We are cancelling the hosting company's analytics add-on and want our own figures to carry on its numbers.
`data/logs/` has four days of logs, and `data/dashboard.jsonl` is what the add-on's dashboard showed for the same days,
one line per day and page, with its hits and bytes. Our tally should give the dashboard's hits and bytes for every
day and page.

Tests: `python -m unittest discover tests` from this directory.
