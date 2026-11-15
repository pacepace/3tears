# glossator

A command-line tool that turns raw glossary terms into printable entries: `python glossator.py TERMS.jsonl OUT.jsonl`
(`-` reads stdin or writes stdout). The terms the printed glossary carries, with the headword and slug each should
get, are in `tests/fixtures/glossary.jsonl`.
