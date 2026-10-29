# W4 acceptance: the report writer's live runs on the W4 build (2026-10-29)

The fifth prototype: `write_report` run live through `analyst_cli.py write` on `claude-opus-5-5` (effort high) with the first-party Anthropic key, on the W4 build (sessions in transcript storage, the engine's reasons, the lever names). Same seven snapshots and the same blind withholds as prototype 4. `test_report_docs_accepted.py`, `test_report_writer_replay.py` and `test_report_read_doc.py` read this set; the tests that pin facts of the prototype 4 documents (`test_doc_page.py`, `test_report_doc_facts.py`, `test_misfit_page_defects.py`, the v13 record in `test_report_session_storage.py`) still read prototype 4.

| Run | Ending | Cap | Cost | Turns | Tool calls | Rounds (analyst/editor) |
|---|---|---|---|---|---|---|
| kairo | stored | $20 | $2.06 | 23 | 40 | 1/1 |
| r1-reading-only-bars | stored | $8 | $1.72 | 22 | 50 | 1/1 |
| r1-without-its-gates | stored | $8 | $1.54 | 21 | 48 | 1/1 |
| r2-with-its-gates-read-in-each-task | stored | $10 | $4.12 | 41 | 67 | 3/3 |
| r2-with-one-gate | stored | $8 | $1.99 | 21 | 43 | 1/1 |
| r2-without-its-gates | stored | $8 | $3.71 | 30 | 60 | 2/2 |
| r3-without-its-configs | stored | $10 | $2.03 | 20 | 49 | 1/1 |

The seven reports cost $17.16.
