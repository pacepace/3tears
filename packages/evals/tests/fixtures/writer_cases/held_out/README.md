# Held-out writer cases

These are held-out acceptance cases: never dev, and never rung-5 replays.

`export_held_out_cases.py` (beside this directory) wrote them offline, calling no model. `test_held_out_cases.py` holds them.

| Case | Source | Stratum | Group | Size (bytes) | Withheld note |
|---|---|---|---|---|---|
| r1 | reference input r1 | reference | r1 | 2,948 | r1-model-selection.md |
| r2 | reference input r2 | reference | r2 | 4,577 | r2-config-sweep.md |
| r3 | reference input r3 | reference | r3 | 3,495 | r3-classifier.md |
| r4 | reference input r4 | reference | r4 | 2,943 | r4-tradeoff.md |
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
| discodon-dj-character-01a1b3d4 | Discodon's campaign 01a1b3d4, "Character v2 apparatus 11-07" (see below) | host | discodon-dj-character | 49,046 | none |
| discodon-judge-shortlist-01a14544 | Discodon's campaign 01a14544, "Judge shortlist" (see below) | host | discodon-judge-shortlist | 107,332 | none |

There is no r5. Reference input r5 is the Kairo round 2 campaign's report input, copied byte for byte. It would be rung 5's Kairo case a second time, and it would read the skill's own note on that campaign. `test_held_out_cases.py` refuses two cases that report on one campaign unless they share a group and withhold the same notes.

The export rewrites every case to the same bytes except the three dungeons-of-wagons cases. A replay mints a new campaign id, so those three are frozen once and kept. The export's docstring explains why.

The cases were minted on aarch64 (glibc 2.39). On another platform, s1 and hot-path-scaled rebuild under other snapshot ids: their report inputs hold statistics whose last digits differ by platform, and a snapshot id hashes them in full.

The dungeons-of-wagons cases need the clone's `src` on `PYTHONPATH`, as the export's docstring shows.

The host cases are rebuilt from the toy host's operations fixture (`ops_support`), which the export and the test import only when they build them. Deleting the planner's fixtures breaks both. Before that deletion, freeze the host cases once and keep them, as the dungeons-of-wagons cases are kept, and drop their rebuild check.

## The cases minted from a production host

The first is `discodon-dj-character-01a1b3d4`; the second, `discodon-judge-shortlist-01a14544`, is below it.

### The first

`discodon-dj-character-01a1b3d4` is the first case minted from a production host through the mint door (`python -m threetears.evals.harness mint`), not rebuilt from a fixture or a replay. It is kept as minted, like the dungeons-of-wagons cases: `export_held_out_cases.py` does not rebuild it.

- Host: Discodon, in its devcontainer, scope `brooks`. Campaign 01a1b3d4, "Character v2 apparatus 11-07": deepseek-v4.1-flash (control) against qwen3.8-flash, ten `dj_character_v2` cases over eight strata, k=1.
- Written report: claude-opus-5-5, $2.94. It is a draft: the editor left findings open.
- Export: Discodon's `tools/evals_export.py` (evals/integration 281e93624+).
- Mint: `python -m threetears.evals.harness mint ...` on evals/reset ce8f28e10.
- Withheld: nothing. No skill note names the campaign. The mint records both `reasoned_on` entries (an `eval_analysis` and an `eval_report_doc`) as unmatched.
- Snapshot `report-snapshot:de914e5e66bb9ac748f9e753210aff29`, group `discodon-dj-character`, stratum `host`.

### The second

`discodon-judge-shortlist-01a14544` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a14544, "Judge shortlist: which cheap judge agrees with the hard-negative verdicts? (38 cases)": eight judge models against the 38 authored hard-negative judge cases, 16 runs.
- Export: Discodon's campaign export from prod, `--built-by discodon.evals.host:discodon_profile`; 3,868 closure documents. The export itself is never committed.
- Written report: claude-opus-5-5, analysis 01a18b0d-663e-718f-a96f-c9ca9c9b6a47 (2026-10-30) and report doc 01a18b0d-5e4e-7577-8d85-5bbc077bba30.
- Mint, on evals/reset 06382a272, exactly as it ran (export files in `.prawduct/prod-exports/prod-01a14544/`, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a14544-3001-778c-a4b7-0abe79cb8500 --name discodon-judge-shortlist-01a14544 --group discodon-judge-shortlist --stratum host --out held_out/discodon-judge-shortlist-01a14544.json.gz`
  Discodon's venv python, because the worktree venv lacks Discodon's dependencies (`psycopg`), so `discodon.evals.host` does not import there.
- Withheld: nothing. No skill note names the campaign. The mint records both `reasoned_on` entries (an `eval_analysis` and an `eval_report_doc`) as unmatched.
- Snapshot `report-snapshot:7e6a1af7d990096595843eb8f1f4f26f`, group `discodon-judge-shortlist`, stratum `host`.

## The acceptance run

The acceptance script reads one `--cases` directory and none of its subdirectories. So put rung 5's eleven cases and these sixteen in one directory under `.prawduct/`, from the repository root:

```bash
mkdir .prawduct/acceptance-cases
cp packages/evals/examples/5-writer/cases/*.json.gz \
   packages/evals/tests/fixtures/writer_cases/held_out/*.json.gz .prawduct/acceptance-cases/
uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance \
  --cases .prawduct/acceptance-cases --out .prawduct/acceptance-run \
  --held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium \
  --budget-usd 0.75 --turns 30 --request-s 45 --cap-usd 60
```

At 0.99, all 27 cases are held-out.

### The job budget

The plan refuses an arm when its cells could outlast the arm's job. Cells run 2 at a time. Each cell's ceiling is (turns + 14) × request ceiling, where 14 counts the request ceilings of the judge phase. The job's budget is capped at 28,800 s. So 27 cases run as 14 rounds, and each cell must fit in 28,800 / 14 = 2,057 s.

| Turns | Request ceiling | Cell ceiling | 14 rounds | Admitted |
|---|---|---|---|---|
| 36 | 90 s | 4,500 s | 63,000 s | no |
| 30 | 60 s | 2,640 s | 36,960 s | no |
| 30 | 50 s | 2,200 s | 30,800 s | no |
| 30 | 45 s | 1,980 s | 27,720 s | yes |
| 22 | 60 s | 2,160 s | 30,240 s | no |
| 22 | 55 s | 1,980 s | 27,720 s | yes |

At 30 turns and 60 s, the arm admits 20 cases at most (10 rounds × 2,640 s = 26,400 s), so 27 cases are refused; so are 30 turns at 50 s. The command above keeps 30 turns and lowers the request ceiling to 45 s. The other admitted choice is 22 turns at 55 s.

### The cap

Before launch, each arm's reports and its judge calls, each priced at its ceiling, must fit the arm's part of the cap.

| Item | Figure |
|---|---|
| Cap | $60.00 |
| Pairwise pass, a fifth | $12.00 |
| Each arm, two fifths | $24.00 |
| Reports, 27 × $0.75 | $20.25 |
| Judge calls, 27 cases × 7 rubric items × 2 attempts | 378 |
| Each call at its ceiling, 2,048 tokens in and 2,048 out | $0.009216 |
| Judge calls at their ceiling | $3.48 |
| Reports and judging | $23.73 |

$23.73 fits the arm's $24.00: the cap splits into a fifth for the pairwise pass and two fifths for each of the two arms (2 × $24.00 + $12.00 = $60.00). The $56 cap leaves $22.40 per arm and refuses 27 cases at $0.75; a $0.70 report would fit it ($22.38) but changes the measured condition. The judge's rates are those rung 5's recordings imply for gemini-3.8-flash: $0.75 per million tokens in, $3.75 per million out.
