# W4 acceptance: the report writer's live runs on the W4 build, re-recorded 2026-10-29 on the editor loop that revises on what blocks

The fifth prototype: `write_report` run live through `analyst_cli.py write` on `claude-opus-5-5` (effort high) with the first-party Anthropic key. Same seven snapshots and the same blind withholds as prototype 4. Re-recorded the same day on the integrated build (round 1 revises on every editor finding; later rounds only on a must_fix or a finding carried from an earlier round, a new should_fix listed but not blocking; a report stored with a blocking finding open is a draft; cost names its routing; the title code builds): the first recording's six round-one passes left should_fix findings open, which the loop no longer stores as finished. Four were re-recorded on 2026-10-31: `r1-reading-only-bars`, `r2-with-its-gates-read-in-each-task` and `r2-without-its-gates`, because a should_fix first raised after round 1 now gets one more revision, and `kairo`, because each aspect's pass rule gives the Kairo snapshot a new id. `r2-without-its-gates` runs at a $10 cap, where it had $8: at $8 the extra revision's next turn could not be afforded. `test_report_docs_accepted.py`, `test_report_writer_replay.py` and `test_report_read_doc.py` read this set; the tests that pin facts of the prototype 4 documents (`test_doc_page.py`, `test_report_doc_facts.py`, `test_misfit_page_defects.py`, the v13 record in `test_report_session_storage.py`) still read prototype 4.

| Run | Ending | Cap | Cost | Turns | Rounds (analyst/editor) |
|---|---|---|---|---|---|
| kairo | stored | $20 | $3.05 | 34 | 3/3 |
| r1-reading-only-bars | stored | $8 | $3.61 | 39 | 3/3 |
| r1-without-its-gates | stored_with_findings | $8 | $3.76 | 44 | 3/3 |
| r2-with-its-gates-read-in-each-task | stored | $10 | $4.22 | 39 | 3/3 |
| r2-with-one-gate | stored_with_findings | $8 | $4.94 | 36 | 3/3 |
| r2-without-its-gates | stored | $10 | $4.74 | 39 | 3/3 |
| r3-without-its-configs | stored | $10 | $4.32 | 42 | 3/3 |

The seven reports cost $28.65. 2 end `stored_with_findings`: a blocking finding was still open with both revisions used, so each is stored and served as a draft.
