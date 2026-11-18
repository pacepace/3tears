# standups

The recurring-meeting expander behind our team calendar (`recur.py`). A `Series` is a meeting that repeats: its first
date, its time of day in London, how often it repeats (`day`, `weekday`, `week` or `fortnight`), how many meetings it
holds, and the dates it skips. `occurrences(series)` lists every meeting as a London time, and `invites(series)` gives
the UTC times we send to the calendar server, one per meeting, as `2026-03-02T09:30Z`. `load_series(path)` reads a
file of series, and `Series.from_json(row)` reads one.

A meeting keeps its time on the office clock all year: the 09:30 stand-up is at 09:30 London time in January and in
July, so in UTC it is 09:30Z in winter (GMT) and 08:30Z in summer (BST). The clock-change rules are in the module
docstring. A skipped date (a bank holiday) is passed over and does not count towards the series' meetings; a `weekday`
series has no meetings on Saturday or Sunday.

We are moving off our old calendar server. `data/series.jsonl` is every recurring series the teams have booked, and
`old_calendar` on each is the list of invites the old server sent for it, which everyone's diaries were built from.
Our invites for a series should be exactly the old server's.

Tests: `python -m unittest discover tests` from this directory.
