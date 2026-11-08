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
| s4 | shape s4, a change | shape | s4 | 13,497 | none |
| dow-router-prompts | dungeons-of-wagons rung 3, `router_prompts.py` (router-v1 against router-v2) | dow | dow-router | 28,876 | none |
| dow-forge | dungeons-of-wagons rung 2, `forge.py` (judged) | dow | dow-forge | 14,309 | none |
| dow-router | dungeons-of-wagons rung 1, `router.py` | dow | dow-router | 17,796 | none |
| toyhost | the toy host's corpus campaign | host | toyhost | 17,187 | none |
| courierhost | the courier's planner bake-off | host | courier-planner | 9,900 | none |
| hot-path-scaled | the hot-path `scaled` campaign, 41 runs | host | hot-path-scaled | 249,881 | none |
| discodon-dj-character-01a1b3d4 | Discodon's campaign 01a1b3d4, "Character v2 apparatus 11-07" (see below) | host | discodon-dj-character | 49,046 | none |
| discodon-judge-shortlist-01a14544 | Discodon's campaign 01a14544, "Judge shortlist" (see below) | host | discodon-judge-shortlist | 107,332 | none |
| discodon-kairo-models-pilot-01a18f50 | Discodon's campaign 01a18f50, "kairo models pilot" (see below) | host | kairo | 1,322,149 | none |
| discodon-1017-reasoning-effort-01a18d4d | Discodon's campaign 01a18d4d, "1017 reasoning effort" (see below) | host | kairo | 1,142,283 | none |
| discodon-planted-effect-01a18b10 | Discodon's campaign 01a18b10, "Planted effect (declared)" (see below) | host | kairo | 610,752 | none |
| discodon-kairo-directives-ab-01a16f8a | Discodon's campaign 01a16f8a, "Kairo directives A/B" (see below) | host | kairo | 591,904 | none |
| discodon-kairo-persona-r1-01a16bd4 | Discodon's campaign 01a16bd4, "Kairo persona model, round 1" (see below) | host | kairo | 2,112,324 | none |

There is no r5. Reference input r5 is the Kairo round 2 campaign's report input, copied byte for byte. It would be rung 5's Kairo case a second time, and it would read the skill's own note on that campaign. `test_held_out_cases.py` refuses two cases that report on one campaign unless they share a group and withhold the same notes.

The export rewrites every case to the same bytes except the three dungeons-of-wagons cases. A replay mints a new campaign id, so those three are frozen once and kept. The export's docstring explains why.

The cases were minted on aarch64 (glibc 2.39). On another platform, s1 and hot-path-scaled rebuild under other snapshot ids: their report inputs hold statistics whose last digits differ by platform, and a snapshot id hashes them in full.

The dungeons-of-wagons cases need the clone's `src` on `PYTHONPATH`, as the export's docstring shows.

The host cases are rebuilt from the toy host's operations fixture (`ops_support`), which the export and the test import only when they build them. Deleting the planner's fixtures breaks both. Before that deletion, freeze the host cases once and keep them, as the dungeons-of-wagons cases are kept, and drop their rebuild check.

## The cases minted from a production host

The first is `discodon-dj-character-01a1b3d4`; the second, `discodon-judge-shortlist-01a14544`; the third to the seventh are the five Kairo campaigns below, in rung 5's group `kairo`.

The grouping rule, for cases minted from a host's store (real material): two cases share a group when they share any test case id or template id, transitively, so shared material never falls on both sides of a split. The synthetic shapes s1 to s4 are exempt: their ids are fabricated, so they name no shared material. Compared pairwise over the test cases and templates in each case's bundle, the five Kairo campaigns (`01a18f50`, `01a18d4d`, `01a18b10`, `01a16f8a`, `01a16bd4`) and rung 5's `kairo-round-2` (group `kairo`) form one connected component: `01a18d4d`, `01a16f8a`, `01a18b10` and `kairo-round-2` share 299 test cases and one template pairwise; `01a18d4d` shares 80 with `01a18f50` and with `01a16bd4`; `01a18f50` and `01a16bd4` share 140 and two templates. So all five are in rung 5's group, `kairo`. `discodon-dj-character-01a1b3d4` (10 cases) and `discodon-judge-shortlist-01a14544` (38) share nothing with any of them or each other and keep their own groups.

**The withheld note.** The five `kairo` cases do not withhold the skill note `kairo-round2.md`, which reasons on this group's material (the 299 shared test cases). The mint's per-campaign withhold rule stands: no note names their own campaigns. Both arms read the note, so the A/B comparison stays even, while the cases are less cleanly held out (owner ruling 2026-11-08).

A sixth, `01a170ae` ("Kairo persona model, round 2"), was exported and minted, then dropped: it is the campaign rung 5's `kairo-round-2` case already reports on (group `kairo`, withholding `kairo-round2.md`), the case this directory's r5 note says it will not repeat. `test_no_two_cases_freeze_one_source` refuses it outright, whatever its group or note, since it would be rung 5's case a second time. The five here do not withhold `kairo-round2.md`; no note names their campaigns.

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

### The third

`discodon-kairo-models-pilot-01a18f50` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a18f50, "kairo models pilot": a pilot of two arms (mercury-3 against deepseek-v4.1-flash-1017), k=3, on three scored aspects, 6 runs; the pilot of the six-arm Kairo models campaign.
- Export: Discodon's campaign export from prod (v2.9.32 at 3tears 0e4155201), `--built-by discodon.evals.host:discodon_profile`; the export itself is never committed.
- Written report: analysis 01a18fc0-c380-71bd-a0df-f255f971b18c (2026-10-31, claude-opus-5-5), analysis 01a19318-0d4a-7203-8ca1-65c007cf3ab5 (2026-10-31, claude-opus-5-5), analysis 01a193db-126d-7094-ac06-8df3a9c1f766 (2026-10-31, claude-opus-5-5); report doc 01a18fc0-b54c-72d3-8e7e-11d5c0560b1a, report doc 01a19318-01ad-7440-992a-cfc82386fead, report doc 01a193db-071b-736c-9614-d787f66cdbf9. The mint records them, with the campaign's insights, as unmatched: 13 in all.
- Mint, on reset/real-cases-prod6 (c02585ad0), exactly as it ran (export files in the campaign's own directory of the prod closure exports, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a18f50-8275-7355-9ba9-1d97307dacaf --name discodon-kairo-models-pilot-01a18f50 --group kairo --stratum host --out held_out/discodon-kairo-models-pilot-01a18f50.json.gz`
  Discodon's venv python, as for the second case.
- Withheld: nothing. No skill note names the campaign.
- The skill note `kairo-round2.md` is not withheld: see "The withheld note" above.
- Snapshot `report-snapshot:e0a71713362c27521787292b03d1a15f`, group `kairo`, stratum `host`.

### The fourth

`discodon-1017-reasoning-effort-01a18d4d` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a18d4d, "1017 reasoning effort": 1017 with no reasoning parameter against low and minimal reasoning effort, three sweeps (dj_character_v2 k=2, dj_queue_discipline_v5 k=3, dj_reasoning_budget_v1 k=5), 9 runs.
- Export: Discodon's campaign export from prod (v2.9.32 at 3tears 0e4155201), `--built-by discodon.evals.host:discodon_profile`; the export itself is never committed.
- Written report: analysis 01a18e39-ea8a-72a5-9e2a-5b3a7627966b (2026-10-30, claude-opus-5-5), analysis 01a193df-d2b4-73bc-b686-9d7e3e9cf343 (2026-10-31, claude-opus-5-5); report doc 01a18e39-da2c-7042-8855-bf4ca91a5aa8, report doc 01a193df-c434-711f-8a9f-a24d3b646a67. The mint records them, with the campaign's insights, as unmatched: 11 in all.
- Mint, on reset/real-cases-prod6 (c02585ad0), exactly as it ran (export files in the campaign's own directory of the prod closure exports, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a18d4d-7c3c-75d7-b676-4c22e122095f --name discodon-1017-reasoning-effort-01a18d4d --group kairo --stratum host --out held_out/discodon-1017-reasoning-effort-01a18d4d.json.gz`
  Discodon's venv python, as for the second case.
- Withheld: nothing. No skill note names the campaign.
- The skill note `kairo-round2.md` is not withheld: see "The withheld note" above.
- Snapshot `report-snapshot:ef124642639ad38eb722f746f3621670`, group `kairo`, stratum `host`.

### The fifth

`discodon-planted-effect-01a18b10` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a18b10, "Planted effect (declared)": the detector's check: a byte-identical prompt (A/A) against a prompt sabotaged to narrate its machinery, 3 runs.
- Export: Discodon's campaign export from prod (v2.9.32 at 3tears 0e4155201), `--built-by discodon.evals.host:discodon_profile`; the export itself is never committed.
- Written report: analysis 01a18c60-fdfb-71b5-850b-0b7deb693736 (2026-10-30, claude-opus-5-5), analysis 01a193ee-77ef-7651-a58b-eca257320326 (2026-10-31, claude-opus-5-5); report doc 01a18c60-f523-74e7-b10e-8e67c00550a8, report doc 01a193ee-6ed3-77b1-990e-68107dd283a1. The mint records them, with the campaign's insights, as unmatched: 11 in all.
- Mint, on reset/real-cases-prod6 (c02585ad0), exactly as it ran (export files in the campaign's own directory of the prod closure exports, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a18b10-06b9-7794-a875-e9e59b2fb110 --name discodon-planted-effect-01a18b10 --group kairo --stratum host --out held_out/discodon-planted-effect-01a18b10.json.gz`
  Discodon's venv python, as for the second case.
- Withheld: nothing. No skill note names the campaign.
- The skill note `kairo-round2.md` is not withheld: see "The withheld note" above.
- Snapshot `report-snapshot:6aeeefc093054d1d54012f80805739a4`, group `kairo`, stratum `host`.

### The sixth

`discodon-kairo-directives-ab-01a16f8a` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a16f8a, "Kairo directives A/B: queue wording + AI lane vs today's, on character v2 (September)": the new queue-wording and AI-lane directives against today's, on dj_character_v2 under September, 4 runs.
- Export: Discodon's campaign export from prod (v2.9.32 at 3tears 0e4155201), `--built-by discodon.evals.host:discodon_profile`; the export itself is never committed.
- Written report: analysis 01a17097-c8c1-7647-96f2-8dcc8afc3c99 (2026-10-25, google/gemini-3.8-flash), analysis 01a1940a-66c8-71fd-a39e-89e47c86ba0f (2026-10-31, claude-opus-5-5); report doc 01a1940a-5ebb-740f-bbda-6e2d2c9a82ce. The mint records them, with the campaign's insights, as unmatched: 10 in all.
- Mint, on reset/real-cases-prod6 (c02585ad0), exactly as it ran (export files in the campaign's own directory of the prod closure exports, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a16f8a-28dd-7334-bdf0-e5e1817e487e --name discodon-kairo-directives-ab-01a16f8a --group kairo --stratum host --out held_out/discodon-kairo-directives-ab-01a16f8a.json.gz`
  Discodon's venv python, as for the second case.
- Withheld: nothing. No skill note names the campaign.
- The skill note `kairo-round2.md` is not withheld: see "The withheld note" above.
- Snapshot `report-snapshot:c06d084cbed5c085dccbc65d77531547`, group `kairo`, stratum `host`.

### The seventh

`discodon-kairo-persona-r1-01a16bd4` is minted the same way and kept as minted.

- Host: Discodon, scope `prod`. Campaign 01a16bd4, "Kairo persona model, round 1: September vs 1017, gemini-3.5-flash-lite, qwen3.8-flash, glm-5.3-flash across eight aspects at a 15-point margin": September against 1017, gemini-3.5-flash-lite, qwen3.8-flash and glm-5.3-flash across eight aspects, 41 runs.
- Export: Discodon's campaign export from prod (v2.9.32 at 3tears 0e4155201), `--built-by discodon.evals.host:discodon_profile`; the export itself is never committed.
- Written report: analysis 01a16c70-bb46-77c7-9a77-38cc01d60553 (2026-10-24, google/gemini-3.8-flash), analysis 01a18cdc-89c0-749f-8575-9ef2fd3b0b68 (2026-10-30, claude-opus-5-5), analysis 01a1941d-182a-77d8-a656-7217b771d02b (2026-10-31, claude-opus-5-5); report doc 01a18cdc-7051-73b9-bdd1-fda0606ae0df, report doc 01a1941c-fd51-7359-b2b5-e122a7bd2ac1. The mint records them, with the campaign's insights, as unmatched: 12 in all.
- Mint, on reset/real-cases-prod6 (c02585ad0), exactly as it ran (export files in the campaign's own directory of the prod closure exports, here `$X`):
  `PYTHONPATH=<worktree>/packages/evals/src:/opt/venv /opt/venv/.venv/bin/python -m threetears.evals.harness mint --documents $X/documents.json --profile $X/profile.json --manifest $X/manifest.json --campaign 01a16bd4-d95d-75d9-aa55-01372957d23a --name discodon-kairo-persona-r1-01a16bd4 --group kairo --stratum host --out held_out/discodon-kairo-persona-r1-01a16bd4.json.gz`
  Discodon's venv python, as for the second case.
- Withheld: nothing. No skill note names the campaign.
- The skill note `kairo-round2.md` is not withheld: see "The withheld note" above.
- Snapshot `report-snapshot:7fba39b1797487dc8ae8cf79a6e9c794`, group `kairo`, stratum `host`.

## The acceptance run

The acceptance script reads one `--cases` directory and none of its subdirectories. So put rung 5's eleven cases and these twenty-one in one directory under `.prawduct/`, from the repository root:

```bash
mkdir .prawduct/acceptance-cases
cp packages/evals/examples/5-writer/cases/*.json.gz \
   packages/evals/tests/fixtures/writer_cases/held_out/*.json.gz .prawduct/acceptance-cases/
uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance \
  --cases .prawduct/acceptance-cases --out .prawduct/acceptance-run \
  --held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium \
  --budget-usd 0.75 --turns 30 --request-s 40 --cap-usd 71
```

At 0.99, all 32 cases are held-out.

### The job budget

The plan refuses an arm when its cells could outlast the arm's job. Cells run 2 at a time. Each cell's ceiling is (turns + 14) × request ceiling, where 14 counts the request ceilings of the judge phase (7 rubric items × 2 attempts). The job's budget is capped at 28,800 s. So 32 cases run as 16 rounds, and each cell must fit in 28,800 / 16 = 1,800 s.

| Turns | Request ceiling | Cell ceiling | 16 rounds | Admitted |
|---|---|---|---|---|
| 36 | 90 s | 4,500 s | 72,000 s | no |
| 30 | 60 s | 2,640 s | 42,240 s | no |
| 30 | 45 s | 1,980 s | 31,680 s | no |
| 30 | 41 s | 1,804 s | 28,864 s | no |
| 30 | 40 s | 1,760 s | 28,160 s | yes |
| 22 | 55 s | 1,980 s | 31,680 s | no |
| 22 | 50 s | 1,800 s | 28,800 s | yes |

At 30 turns and 60 s, the arm admits 20 cases at most (10 rounds × 2,640 s = 26,400 s), so 32 cases are refused; so are 30 turns at 45 s, the previous command's ceiling. The command above keeps 30 turns and lowers the request ceiling to 40 s. The other admitted choice is 22 turns at 50 s.

### The cap

Before launch, each arm's reports and its judge calls, each priced at its ceiling, must fit the arm's part of the cap.

| Item | Figure |
|---|---|
| Cap | $71.00 |
| Pairwise pass, a fifth | $14.20 |
| Each arm, two fifths | $28.40 |
| Reports, 32 × $0.75 | $24.00 |
| Judge calls, 32 cases × 7 rubric items × 2 attempts | 448 |
| Each call at its ceiling, 2,048 tokens in and 2,048 out | $0.009216 |
| Judge calls at their ceiling | $4.13 |
| Reports and judging | $28.13 |

$28.13 fits the arm's $28.40: the cap splits into a fifth for the pairwise pass and two fifths for each of the two arms (2 × $28.40 + $14.20 = $71.00). The previous $60 cap leaves $24.00 per arm and refuses 32 cases at $0.75; the smallest cap that fits is $70.33 (two fifths of it is $28.13), so $71 is the smallest whole dollar. The judge's rates are those rung 5's recordings imply for gemini-3.8-flash: $0.75 per million tokens in, $3.75 per million out.
