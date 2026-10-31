# W4 acceptance: the report writer's live runs on the W4 build, re-recorded 2026-10-29 on the editor loop that revises on what blocks

The fifth prototype: `write_report` run live through `analyst_cli.py write` on `claude-opus-5-5` (effort high) with the first-party Anthropic key. Same seven snapshots and the same blind withholds as prototype 4. Re-recorded the same day on the integrated build (round 1 revises on every editor finding; later rounds only on a must_fix or a finding carried from an earlier round, a new should_fix listed but not blocking; a report stored with a blocking finding open is a draft; cost names its routing; the title code builds): the first recording's six round-one passes left should_fix findings open, which the loop no longer stores as finished. `test_report_docs_accepted.py`, `test_report_writer_replay.py` and `test_report_read_doc.py` read this set; the tests that pin facts of the prototype 4 documents (`test_doc_page.py`, `test_report_doc_facts.py`, `test_misfit_page_defects.py`, the v13 record in `test_report_session_storage.py`) still read prototype 4.

| Run | Ending | Cap | Cost | Turns | Rounds (analyst/editor) |
|---|---|---|---|---|---|
| kairo | stored | $20 | $4.14 | 43 | 3/3 |
| r1-reading-only-bars | stored | $8 | $2.87 | 32 | 2/2 |
| r1-without-its-gates | stored_with_findings | $8 | $3.76 | 44 | 3/3 |
| r2-with-its-gates-read-in-each-task | stored | $10 | $3.47 | 34 | 2/2 |
| r2-with-one-gate | stored_with_findings | $8 | $4.94 | 36 | 3/3 |
| r2-without-its-gates | stored | $8 | $3.62 | 27 | 2/2 |
| r3-without-its-configs | stored | $10 | $4.32 | 42 | 3/3 |

The seven reports cost $27.11. 2 end `stored_with_findings`: a blocking finding was still open with both revisions used, so each is stored and served as a draft.
