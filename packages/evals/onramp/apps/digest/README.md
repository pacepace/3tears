# digest

Digests for the weekly operations email (`digest.py`). Every incident gets a short written report from whoever was
on call; the email carries a digest of each, a few sentences long, so people can see what happened without opening
the full report. `digest(report)` picks those sentences by rule, and `digest_file(path)` reads a report from a file.
The rules are in the module docstring.

`data/reports/` holds the incident reports from January and February. For each one, the head of operations listed
the facts a reader of the email has to come away with (`data/key_facts.jsonl`: the report's path and its facts, each
a phrase as the report words it). A digest is only as good as the share of those facts it carries; three sentences
cannot always carry all of them.

Tests: `python -m pytest tests` (or `python -m unittest discover tests`) from this directory.
