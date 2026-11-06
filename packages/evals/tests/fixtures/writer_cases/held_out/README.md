# Held-out writer cases

These are held-out acceptance cases: never dev, and never rung-5 replays.

`export_held_out_cases.py` (beside this directory) wrote them offline, calling no model. `test_held_out_cases.py` holds them.

| Case | Source | Stratum | Group | Size (bytes) | Withheld note |
|---|---|---|---|---|---|
| r1 | reference input r1 | reference | r1 | 2,948 | r1-model-selection.md |
| r2 | reference input r2 | reference | r2 | 4,577 | r2-config-sweep.md |
| r3 | reference input r3 | reference | r3 | 3,495 | r3-classifier.md |
| r4 | reference input r4 | reference | r4 | 2,943 | r4-tradeoff.md |
| r5 | reference input r5 | reference | r5 | 13,546 | none |
| s1 | shape s1, three aspects | shape | s1 | 44,402 | none |
| s2 | shape s2, a choice | shape | s2 | 17,921 | none |
| s3 | shape s3, a measurement | shape | s3 | 19,920 | none |
| s4 | shape s4, a change | shape | s4 | 13,442 | none |
| dow-router-prompts | dungeons-of-wagons rung 3, `router_prompts.py` (router-v1 against router-v2) | dow | dow-router | 28,876 | none |
| dow-forge | dungeons-of-wagons rung 2, `forge.py` (judged) | dow | dow-forge | 14,309 | none |
| dow-router | dungeons-of-wagons rung 1, `router.py` | dow | dow-router | 17,796 | none |
| toyhost | the toy host's corpus campaign | host | toyhost | 17,187 | none |
| courierhost | the courier's planner bake-off | host | courier-planner | 9,900 | none |
| hot-path-scaled | the hot-path `scaled` campaign, 41 runs | host | hot-path-scaled | 248,822 | none |

The export rewrites every case to the same bytes except the three dungeons-of-wagons cases. A replay mints a new campaign id, so those three are frozen once and kept. The export's docstring explains why.

## The acceptance run

The acceptance script reads one `--cases` directory and none of its subdirectories. So put rung 5's eleven cases and these fifteen in one directory under `.prawduct/`, from the repository root:

```bash
mkdir .prawduct/acceptance-cases
cp packages/evals/examples/5-writer/cases/*.json.gz \
   packages/evals/tests/fixtures/writer_cases/held_out/*.json.gz .prawduct/acceptance-cases/
uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance \
  --cases .prawduct/acceptance-cases --out .prawduct/acceptance-run \
  --held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium \
  --budget-usd 0.75 --turns 30 --request-s 50 --cap-usd 56
```

At 0.99, all 26 cases are held-out.

### The job budget

The plan refuses an arm when its cells could outlast the arm's job. Cells run 2 at a time. Each cell's ceiling is (turns + 14) × request ceiling, where 14 counts the request ceilings of the judge phase. The job's budget is capped at 28,800 s. So 26 cases run as 13 rounds, and each cell must fit in 28,800 / 13 = 2,215 s.

| Turns | Request ceiling | Cell ceiling | 13 rounds | Admitted |
|---|---|---|---|---|
| 36 | 90 s | 4,500 s | 58,500 s | no |
| 30 | 60 s | 2,640 s | 34,320 s | no |
| 30 | 50 s | 2,200 s | 28,600 s | yes |
| 22 | 60 s | 2,160 s | 28,080 s | yes |

At 30 turns and 60 s, the arm admits 20 cases at most (10 rounds × 2,640 s = 26,400 s), so the 25 the run needs are refused. The command above keeps 30 turns and lowers the request ceiling to 50 s. The other admitted choice is 22 turns at 60 s.

The cap of $56 gives the pairwise pass a fifth ($11.20) and each arm $22.40. Each arm's 26 reports are budgeted at 26 × $0.75 = $19.50, which leaves $2.90 for its judge calls.
