# Design: judges (track T2)

Status: design, revision 3.1, 2026-11-14, read against `evals/main` at `7b2e0afeb`. Chunk A1 is nearly built; nothing
else here is built. Paths are under `packages/evals/src/threetears/evals/` unless they start with `docs/`,
`examples/` or `tests/`. Line numbers are for orientation; the builder re-reads before editing.

Bound by `information-architecture.md` (same folder), cited below as IA with its rule ids (M, V, F, T).

## Revision 3.1 (2026-11-14)

The statistician's answers to S1 to S3 (`design-judges-review.md`, "Revision 3 follow-up"), all accepted, and two
owner decisions: the IA's five subtractions are approved, and design choices are settled by principled correctness.

| Change | Where |
|---|---|
| S1: an unrecorded coordinate matches nothing, checks included. An A/A over such a run is `insufficient`, reason `identity_unrecorded`, and sets nothing aside; `corrected` is `not available`; kappa still uses those ratings; the lean leaves those pair items out | §2.4, §4.2, §5.3 |
| S2: the order this track's reasons carry into the one ranked list; looks stay below both judge reasons at every count | §8 |
| S3: `met` is an equivalence reading. A quiet check that is not inside a stated bound is `undecided`. At realistic counts most checks read `undecided` | §2.4, §4.2, §7 |
| One decision cell: a judge `met` against a reviewer shown to lean is `set aside`, `instrument_leans`, for the arm leaned toward | §5.5 |
| The evidence tier is no longer a reader-facing label and no decision reads it (owner). R1 shrinks; §9.4 lists what is not built | §2.3, §9.4 |
| Three expedient forms replaced by the principled one, each with its cost: who made a rating; the replicate coordinate; the name `sample_standing` | §1.2, §4.1, §9.5 |

## Revision 3 (2026-11-14)

The IA document is applied throughout. No statistical method changed in this revision.

| Change | Where | IA |
|---|---|---|
| One record for "how was this instrument verified": a **check** (instrument, against, sample, result). The anchor, the anchor's five standings, the A/A result and the corrected contrast are all checks. A check result is one of the four existing criterion states | §0.2, §2.3, §2.4, §4.2, §5.3 | M3, T2, T3 |
| Every reading or contrast is in one of four states (`counted`, `set aside`, `withheld`, `not available`), each non-counted state with a reason from the shared closed set. The decision table and every chunk use only these | §0.3, §5.5 | 2.3, V1, V2 |
| This track renders no first-screen line. Each part states the typed fact and reason it contributes; S5a ranks, S5b renders | §8; every part | F1, §3 |
| The corrected contrast is one field, `corrected`, with `against`: a check result on a contrast, never a second verdict or a second row | §5.3 | 1.3, B1 |
| Words: `sample` replaces "standing" for a read's sample; no reader text prints a tier value, "anchor" or bare "calibrated"; readers see "checked against people" or "checked against a model (`<id>`)"; "A/A check" and "run twice" replace "null contrast" and "replicate" in reader text | throughout | 2.1, 2.2, V5, V6 |
| Seven caveat ids become two kinds: `set_aside:<reading>` with a reason, and `looks` | §5.5, §6 | E1, D1 |
| `PersonPrefers` reads the shared clause state (`held`, `broken`, `undecided`) | §3 | V3 |
| #1178 folded into chunk A2 as `CampaignCall.made_by` | §2.9, §9 | M4, F9 |
| The evidence tier is derived from checks and read by no decision. The design works whether or not the owner accepts removing it as a reader-facing label; §9.4 says what is not built if he does | §2.3, §9.4 | T2, §6 #1 |
| Chunks resized; A1's cheap alignments listed; B0's field list rewritten in IA names | §9 | §5 |
| What this revision subtracts | §9.6 | |
| Three questions for the statistician | §13 | [stat] |

## Revision 2 (2026-11-14)

The statistical review (`design-judges-review.md`, every BLOCKER and CHANGE accepted), the owner's
reference-reviewer ruling, and the owner's answers to Q8 and Q10 were folded in: the preference verdict gated to a
finished read (B-1); skips in the sample (C-1); cases beside results (C-2); a derived seed (C-3); the reference
reviewer as a first-class part with a lean-based guard at 40 double-read pair items (B-5, C-17 to C-21); the
preference in the gains family (C-7); one null family per pair, no "within" marker, launch-group coverage (C-8 to
C-10); the corrected contrast's decision floor at 40 (B-2, C-5, C-6); the pass^k ceiling as (1 − e)^(k−1) and the
share redefined (B-3, C-11, C-12); `identical_arms_separated` beyond every override and four missing decision
cells (B-4, C-15); floors at α/L and looks wording (C-13, C-14, C-16).

## 0. Summary

### 0.1 What is missing today

1. **A sound way to get labels.** The only door is an unblinded page, or a CSV listing every answer arm by arm.
2. **A reading of the apparatus on identical arms.**
3. **A path from judge error to the decision.**

### 0.2 Words

| Word | Who sees it | What it is |
|---|---|---|
| **Blind read** | Readers | A drawn, blinded order of judged outputs, a reader's answers, and its close. The one new reader-facing noun. It replaces "labels" at rung 2 |
| **Check** | Internal record; readers see "checked against people / a model (`<id>`) / itself / identical arms" | An instrument, what it was checked against, on what sample, with one result |
| Lean | Readers | A grader's lean toward an arm. One meaning, for a judge and for a model reviewer |
| Look | Readers | One more reading of a contrast after its data changed |
| Re-score noise | Readers, one step down | A judge re-reading one answer and scoring it differently |
| Set aside | Readers | A contrast read and shown, moving no standing, for a stated reason about its instrument |
| Sheet; reference reviewer; replicate | Hosts | The renderer contract; a model answering a blind read (readers: "a model", or "reference model" where a noun is needed); the launch parameter (readers: "run twice") |
| `sample`, `anchor`/`against`, item, slot, planned items, null contrast, slot lean, accrual batch | Internal | Never in reader text or a public name |

"Standing" means an arm's standing on the question and nothing else in this design.

### 0.3 States

A reading or a contrast is in exactly one state, from the IA's shared module (owned by S5a):

| State | Reasons this track raises |
|---|---|
| `counted` | (every verdict, `not_separated` included) |
| `set aside` | `instrument_unchecked`, `instrument_below_bar`, `instrument_leans`, `people_disagree`, `identical_arms_separated` |
| `not available` | `too_few`, `not_recorded`, `replayed`, `not_drawn_at_random`, `skipped_past_limit`, `identity_unrecorded` |

This track raises no `withheld` state. A check result is a `CriterionState`: `met`, `not_met`, `undecided`,
`insufficient` (`kernel/evidence_tiers.py:139`), with a reason where it is not `met`. A clause is `held`, `broken`
or `undecided`. No other closed vocabulary is added.

### 0.4 Shape

A blind read is one seeded random order over the judged population, so any prefix answered from the top is a
random sample. The sheet carries only content and label keys, so unblinding is the existing label-key pairing. A
person or a model answers it; the two are stored apart and never pooled. Each closed read yields checks. An A/A
is the control launched again and read as a check of the rig. A judged improvement is `set aside` when its
judge's check says so.

**Limits.** The corrected reading at 40 to 60 labelled cases is valid and wide. A check against a model cannot see
a lean the judge and the model share; only people double-reading 40 pair items can. One A/A is one draw. A
preference re-read after every item is optional stopping. Blindness on the quick path is an honour system.

---

## 1. Blind read

### 1.1 The model

- The **population** is every case where each chosen arm has a judged, usable result. One result per arm per
  case is drawn. The unit is the case.
- The **order** is a seeded permutation of those cases; each case is one item.
- With one arm an item shows one output. With a control and a candidate it shows two, in slots A and B, slots a
  seeded coin in permuted blocks of two.
- The draw fixes `planned_items` (default: the population, up to 30).
- A reader answers from the top.

A prefix of a uniform random permutation is a simple random sample of cases. So "is this sample random?" is a
structural fact. The read's **`sample`** is `random` (a prefix), `selected` (anything else) or `none`. Readers see
"a random sample" or "not drawn at random". A `selected` sample's ratings still count toward agreement, as today,
and enter no corrected reading (`not available`, reason `not_drawn_at_random`).

A random sample at any stopping point is not a valid test at any stopping point (§1.7).

**There is no key to hide.** A sheet row carries the output, the case material and the output's label key
(`CalibrationRating`, `schema/models.py:1446`). Unblinding is the pairing `judge_agreement` already performs
(`analysis/agreement.py:400-462`). The draw lives on the blind read document, never in the sheet.

Byte-identical outputs across arms are left out of the pair order and counted; for the contrast they are a known
stratum (§5.3). Reads are never pooled into one preference test.

### 1.2 Stored schema

One new durable document, `BlindRead`, in `schema/blind_read.py` (new), doc type `blind_read`
(`origin="record", lifecycle="durable"`, beside `calibration_rating`, `schema/doc_types.py:115`).

| Field | Notes |
|---|---|
| `id`, `scope_id`, `name`, `created_at`, `created_by` | |
| `campaign_id` / `run_ids` | Provenance; may dangle |
| `arms` | One or two: variant key and role |
| `criteria` | Each `RubricDim` as worded, copied |
| `ask` | Subset of `verdicts`, `preference` |
| `draw` | `seed`, `population`, `population_digest`, `planned_items`, `ordinal`, `excluded` (counts by reason). All internal names |
| `reviewer` | Optional, set at creation only (§2.2) |
| `items` | In order: `item`, `outputs` (slot, result id, label keys, `RatedOutput`) |
| `answers` | `item`, `made_by` (a `Maker`; a person by default), `collected_by`, `answered_at`, `verdicts`, `prefer`, `skipped` |
| `closed_at`, `closed_by`, `closed_early`, `report` | Set once |

**Who made a rating, in the principled form.** Revision 3 stored a model's read as a third `rater_kind` value,
`reference`, with the model's id in the free-text `rater` and an optional `reviewer` beside it. That is
expedient: it encodes two facts in one enum (who judged, and by what procedure), and a rating's id derives from
the `rater` string, so two reviewer configurations of one model id would overwrite each other. The principled
form:

- One type, `Maker`, in `schema/`: `kind` (`person` or `model`), then the person's name as the host attests it, or
  the model's `ReviewerConfig` (model, served model, sent settings, prompt digest), and `on_behalf_of` where a
  model acted for a person. It is the same type `CampaignCall.made_by` uses (§2.9) and a preference carries (M4).
- `CalibrationRating` and `BlindAnswer` carry `made_by: Maker`. A model's rating id derives from the config's
  digest, not from a name.
- **Whether a model's rating is counted is a fact about procedure, not about the maker**: it is counted when its
  `source` names a closed blind read whose declared reviewer is that maker. A model's rating from any other door
  is an opinion, listed and counted nowhere, as an agent's is today.
- `rater_kind` stays as a stored field for old readers, derived on write (`person`, or `agent` for every model)
  and read by nothing new. It gains no value.

Cost: one durable schema step on `CalibrationRating` (a rating stored before it reads `made_by` from its
`rater_kind` and `rater`), a MIGRATIONS entry, and about 150 lines of rework in A1 if its `reference` value has
not shipped. A rating written by close carries `source="blind_read:<id>"` and `collected_by`, both existing.

### 1.3 Pure core, then two doors

`analysis/blind_read.py` (new):

| Function | Does |
|---|---|
| `draw_blind_read(results, traces, criteria, arms, *, seed, ask, planned_items)` | Population, order, items |
| `blind_sheet(read, *, rater=None)` | The blinded view |
| `read_sample(read, maker)` (built as `sample_standing`; renamed in A1 before it merges, §9.5) | The `sample`: kind, answered, skipped |
| `close_blind_read(read, results)` | Ratings to write, checks, preference, report |

**Skips** (C-1). A skip follows content, so a prefix with skips is missing not at random. The sample carries the
skip count; agreement and preference print it. A corrected reading over a sample with more than 10% skipped is
`not available`, reason `skipped_past_limit`.

**Host door** (`ops/blind_reads.py`, new; actions in `actions/catalogue.py`, `actions/definitions.py`). No action
name or response field uses a word the IA rules internal; responses carry `sample`, never `standing`.

| Action | Who | Does |
|---|---|---|
| `blind_read_create` | any writer | `campaign_id` or `run_ids`, `criteria`, `candidate`, `planned_items`, `name`, optional `reviewer`. No seed parameter (C-3): derived from the population digest, the arms and the count of reads already drawn over them. The response says how many reads were drawn over that population |
| `blind_read_sheet` | any reader | The sheet. Never an arm, run id, result id, judge score or check result |
| `blind_read_answer` | attested person, or the read's own review job | A person's answer asks `attested_person` (`ops/runs.py:881`) as `result_rate_person` does; unattested is refused 403. A model's answer needs no attestation, is accepted only from the read's review job, and is stored with a model `Maker` |
| `blind_read_review` | any writer | Starts the declared reviewer as a job, priced and admitted first |
| `blind_read_close` | any writer | Writes ratings through `rate_result` (`run/ratings.py:97`), stores the report, refuses later answers |
| `blind_read_get`, `blind_reads_list` | any reader | No result ids until closed |

Answers stay on the read until close. A read closed with fewer than `planned_items` answered is stored
`closed_early`, and its preference verdict is the anytime-valid one of §1.7 (B-1c, C-4).

### 1.4 The sheet contract (host-facing)

```
BlindSheet
  read_id, name, instructions
  criteria: [{dim, question, scale, levels: [{value, meaning}]}]
  items:    [{item, case_material, outputs: [{slot, text}],
              ask: {verdicts: [dim...], preference: bool}}]
  planned_items, answered: [item...], next: item | None
```

Renderer rules, checked by a conformance test in `testing/`: items in order, `next` first; show nothing but what
the sheet carries; slot labels as given; a skip is an answer; no preference count, direction or p while the read
is open.

### 1.5 The quick path, with a file

Rung 2 today writes `to_label.csv`: 120 rows in store order, arm by arm, with `run_id` and `result_id`
(`quick/result.py:1367-1401`). "Rate about 20" yields seven cases of one arm.

Change: a judged result's `write()` (`quick/result.py:725-734`) writes `blind_read.html` (one self-contained
static page: the sheet inlined, one item at a time, progress in `localStorage`, a **Download labels.csv** button,
no judge score, no running tally) and `blind_read.csv` (the same order as rows: `item, slot, dimension,
case_material, answer, label, reason, rater, made_by, model, prefer, output_fingerprint,
criterion_fingerprint`; `made_by` is `person` when blank).

`Judge(labels="labels.csv")` keeps its parameter name. `_read_labels` (`quick/result.py:1257-1298`) reads the new
columns and pairs by label key. An old-shape file still reads, as not drawn at random. The draw is recomputed,
not stored.

**The print, mid-read.** Reader words only: no tier value, no "standing", no bare "calibrated" (V5):

```
judge: 14 of 20 items read blind, in the drawn order (a random sample)
  answer.grounded   checked against people: 28 results from 14 cases, kappa 0.71 [0.38, 0.90], undecided; about 30 more results would decide it
  answer.safe       checked against people: 28 results from 14 cases, not available: you passed every answer; read further
preference: shown when the read is finished (14 of 20)
next: item 15 in blind_read.html
```

**Finished:** `preference: v2 over v1 on 13 of 18 (2 ties), sign test p 0.096: cannot tell apart`. A model's read
prints as "checked against a model (`<id>`)" on its own line, never in a people line.

`result.preference(now=True)` gives a mid-read verdict by the anytime-valid test, labelled as such.
`to_label.csv` and the word "labels" at rung 2 retire with this change (V4).

### 1.6 Close: what it produces

| Output | From |
|---|---|
| Ratings, one per answered verdict, of the answer's own kind | `rate_result` |
| One **check** per criterion, judge and "against" (people, or the model), never pooled: sample, results, cases, exact, kappa, bounds, result | `judge_agreement` (`analysis/agreement.py:462`), `criterion_state` |
| A 2×2 table per pass/fail criterion with the judge's detection rate of reader-failed answers, Wilson interval (`analysis/stats.py:328`) | new, 40 lines |
| Preference per maker: share per case, counts, test, verdict | §1.7 |
| Slot lean per rater (internal figure, one step down) | new |

The 2×2 table exists because of #848: 27 of 30 raw agreement with kappa 0; the judge caught 0 of the 3 the person
failed.

### 1.7 Statistics

**Agreement.** Kappa per rater pooled by result, score bounds, floors of 20 and 120 distinct results
(`kernel/evidence_tiers.py:112-131`). A pair item gives two results from one case, so a new simulation row decides
whether the interval's variance must be computed over per-case sums (C-2): two results per case, judge
disagreement correlated within the case, 10 to 60 cases, in `tests/test_simulated_agreement.py`. If a judge at the
bar is `met` more than 5% of the time there, chunk A1b clusters the variance (`KappaMoments`,
`analysis/stats.py:890`).

**Preference.** The unit is the case; the candidate's share is 1, 0, or 0.5 for a tie.

| When the verdict is read | Test |
|---|---|
| Finished or at `planned_items`, one rater | Exact two-sided sign test over the non-tie cases; ties counted |
| The same, several raters | `bounded_separation_p` on shares against 0.5 (`analysis/stats.py:2060`); raters treated as fixed, said on the page |
| Mid-read on request, or closed early | `bounded_separation_p(..., planned_n=M)` in drawn order, the stake fixed for the population M (`analysis/stats.py:1838-1893`). The p holds at any stop |

Verdicts are the engine's existing ones. Below 6 non-tie cases the preference is `not available`, reason
`too_few` (five cannot separate). What a finished read can separate, exact sign test at two-sided 0.05:

| Non-tie cases | 6 | 10 | 15 | 20 | 30 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| Fewest for one side | 6 | 9 | 12 | 15 | 21 | 27 | 39 |

Why the gate matters (review; no true preference, sign test re-read and stopped at first separation): after every
item from 6 to 60, 18.4%; every 5 items, 12.7%; at 20, 40, 60, 7.8%.

Assumptions, one step down on the page: the preference is over cases both arms answered; one person's preference
is that person's; a prefix is random over the population drawn. Slot balance keeps the test valid under a slot
lean (a reader choosing A 80% of the time rejects 1.1% at n = 20).

**Simulation** (`tests/test_simulated_blind_read.py`, seeded, 4,000 replicates per row):

| Claim | Passes when |
|---|---|
| The finished-read sign test holds its level (no preference; tie rate 0, 0.2, 0.5; n 6 to 60) | Separated at most 5% |
| A reader who stops on the running result gains nothing | At most 5% under the anytime-valid test; the exact test's rate under the same rule recorded beside it |
| A content-following skip rule is caught | The sample reports the skips; the corrected reading is `not available` past 10% |
| Slot balance on any prefix of 2m | Exactly m with the control in A |

### 1.8 What it contributes

Typed facts only; see §8. A closed person read contributes checks (class C when not `met` on an asked reading) and
a preference row in the gains family (§3).

### 1.9 Left out

Stratified or judge-informed sampling; several candidates in one read; person-to-person kappa; adjudication;
enforced blindness on the quick path.

---

## 2. A model as reviewer, and the check record

Owner ruling, 2026-11-14: a blind read answered by the best model available is a legitimate review; every read
says whether a model or a person did it; evals anchor to that model's judgment and adapt as person reads
accumulate.

### 2.1 What it is

A **reference reviewer** (host word) is a model that answered a drawn blind read. Its ratings carry a model
`Maker` and the read as their source (§1.2). It is never written, counted or shown as a person's; `attests_person` is untouched. Nothing in
shared code names a model.

### 2.2 How it reads

The same sheet a person reads. Each item is one call in a fresh context, through the existing completion port
with a strict reply schema (`strict_response_format`), recorded and replayable. Spend is priced and admitted
first, ledgered under the out-of-run purpose `reviewer`.

`ReviewerConfig`, on the blind read at creation and carried in each rating's `made_by`: `model` (a concrete id),
`served_model`, `sent_settings`, `prompt_digest`. The reviewer is fixed before its answers exist, and a read holds
one (C-21). The package counts the model reads per criterion and reviewer configuration over a population; the
count is a field of the check.

Quick path: `Judge(..., reviewer=<model id, Model, or completion client>)`.

A model's preference between two outputs exists only as an answer to a blind read; no engine file gains a pair
shape, so #599's decline stands. It decides nothing (§2.5), so it is shown one step down only (F6).

### 2.3 One record: the check

Everything this track learns about an instrument is one record (IA M3, T2):

```
Check
  instrument   {kind: judge | reviewer | rig, key}       # JudgeKey, ReviewerConfig, or the launch group
  criterion    dim and criterion digest (None for the rig)
  against      people | model | itself | identical_arms
  against_id   the person(s), or the model's ReviewerConfig; None otherwise
  same_family_as   [arm...] | None      # the "against" model shares a family with these arms' candidates, or with the judge
  sample       {kind: random | selected | none, results, cases, skipped, read_id}
  result       met | not_met | undecided | insufficient
  reason       a 2.3 reason when the result is not met
  figures      kappa and bounds; lean and interval; reads (count of model reads over this population)
```

The two existing criteria are checks: agreement with people is `against: people`; agreement with its own repeats
is `against: itself`. This design adds `against: model` (this section) and `against: identical_arms` (§4).

**The evidence tier is no longer a reader-facing label, and no decision reads it** (owner, 2026-11-14, approving
IA §6). `tier_of` (`kernel/evidence_tiers.py`) keeps computing the stored four-value field for old readers, from
the people check and the itself check exactly as today. A check against a model does not change it. No text
prints it, nothing new consumes it, and decision code reads checks (§5.5). §9.4 lists what is no longer built.

Which check speaks for a judge on a criterion (review, addendum (c)):

| Check against people | Check against a model | The judge is read as |
|---|---|---|
| `met` or `not_met`, over at least 20 distinct results | any | Checked against people, with that result. F7: the model check is silenced on the first screen |
| `undecided` or `insufficient` | `met` or `not_met` | Checked against a model (`<id>`), with that result; the people check shown beside it one step down |
| `undecided` or `insufficient` | `undecided` or none | Not checked |

Person and model ratings are never pooled into one kappa, and a model's rating never enters a reading named
"people". Tests hold both.

### 2.4 Three guards

**Guard 1: not the same model.** A reviewer whose served model is the judge's measures self-consistency: the read
is refused as a check against a model and reported as `against: itself`. The same family as the judge is allowed
and recorded in `same_family_as` (C-20). The family of a model is the host's to say (an optional
`EvalHost.model_family` callable); with none, the provider segment of the id; where neither reads, `not_recorded`.

**Guard 2: the reviewer is itself checked, by its lean** (B-5). The review's simulation: a reviewer that passes a
failed candidate answer 15 to 30% of the time, copied by the judge, reads kappa 0.83 to 0.92 against people, the
judge reads `met` against it, and a difference people do not see is confirmed 10 to 68% of the time. Kappa
cannot see a lean toward one arm.

So the reviewer has its own check: `instrument: reviewer`, `against: people`, over the **double-read pair items**
(items of one read that both a person and the model answered, per criterion). Its figures are the reviewer's
kappa against people and its lean: the reviewer's difference between arms minus the person's, mean and interval.
The result is a `CriterionState` (T3):

| Result | When | Effect on a judge's check against this model |
|---|---|---|
| `insufficient` (`not_recorded`) | No double-read pair items | Stands. The fact carried up: checked against a model, the model not checked against people |
| `undecided` | Anything not below: fewer than 40 items, or 40 and more with the lean interval reaching outside ±δ | Stands. The half-width h is carried, so one step down reads "a lean toward either arm is not ruled out below ±h" |
| `met` | **An equivalence reading** (S3): at least 40 double-read pair items, the lean's interval inside ±δ, and the kappa criterion `met` by its existing rule | Stands; the family marking of guard 3 lifts; the fact drops in rank (§8) |
| `not_met`, reason `instrument_leans` | Lean interval excludes zero, at 10 items or more (a floor the review did not simulate; it errs in the safe direction) | An `improved` contrast for the arm it leans toward, read by a judge checked only against this model, is `set aside` (§5.5) |
| `not_met`, reason `instrument_below_bar` | Its kappa against people shown below 0.6 | The judge reads as not checked on that criterion |

δ is the reading's declared materiality margin where it has one, since a lean smaller than the margin cannot
manufacture a material improvement; otherwise 0.10 of the scale's width.

**Most of these checks will read `undecided`, and that is the honest reading.** "Cannot tell apart" never means
"no difference". On pass/fail an unbiased reviewer lands inside ±0.10 with 80% chance at about 105 to 260
double-read pair items, and inside ±0.15 at about 50 to 115 (the statistician's arithmetic). At 40 items `met` is
rare. `undecided` changes no decision: the judge's check against the model still stands.

**Unrecorded identity** (S1). Kappa uses every rating: the check is about an instrument reading texts. The lean
compares arms, so pair items from a run with an unrecorded coordinate are left out of it, and the count left out
is shown.

What 40 buys (review): a lean of 0.16 is caught 75% of the time at 30 pair items and 99% at 60; a lean of 0.08,
23% at 30, 74% at 60, 97% at 100.

**Guard 3: shared family is a property of the check** (C-19; IA: not a caveat kind of its own). When
`same_family_as` names an arm, a judged improvement for that arm under a model check counts, as the ruling says.
The class C fact it raises carries the family property and the model-judgment marking, until the reviewer's own
check is `met` on that criterion (S3), not merely until it holds 40 items.

### 2.5 What a model's read may and may not do

| A model's read | May it? |
|---|---|
| Make a judge "checked against a model", so a judged improvement is `counted` | Yes (owner; Q8) |
| Set an improvement aside, through the lean test (`instrument_leans`) | Yes: the safe direction |
| Lift a `set aside` state (the `corrected` check `met`) | **No** (C-17): its coverage is for the model's opinion, not people's |
| Fill `ReportInput.human_pref`, satisfy `PersonPrefers`, or count as shown better | **No** (C-18) |
| Enter the cell-mean PPI or any reading named "people" | **No** |
| Be pooled with person ratings | **No** |

Tests hold all four refusals.

### 2.6 Adapting as person reads accumulate

Both kinds are ratings over stored outputs, so nothing is re-run. Each person read on items the model also read
updates guard 2. Once the people check decides a criterion, it speaks for the judge.

**First use.** The 2026-10-17 owner read is preserved under `.prawduct/artifacts/blind-read-2026-10-17/` (15 pair
items). A model reading those outputs blind gives the first reviewer check at no new human time: at 15 items its
lean interval is about ±0.25 to ±0.3, so expect `undecided`.

### 2.7 What it contributes

See §8. In short: a judge on an asked reading that is not checked, or checked against a model only, is one class
C fact. A judge checked against people contributes nothing to the first screen (F6).

### 2.8 Left out

Several reviewers voting; a reviewer chosen by the package; reviewer self-consistency as its own check;
promoting a reviewer to judge.

### 2.9 The call says who made it (#1178)

IA M4: every judgment says who made it. A recorded call needs the same signal a rating has.

`CampaignCall` (`kernel/declaration.py:1170`) gains optional `made_by`:

| Field | Notes |
|---|---|
| `made_by.kind` | `person` or `model` |
| `made_by.model`, `made_by.served_model` | Set for a model |
| `made_by.on_behalf_of` | Optional: the person a model decided for |

`campaign_record` takes it. `person` is decided by the same attestation the answer door uses
(`attested_person`): a caller the host does not attest for that person cannot record a person's call. A model's
call needs no attestation. A stored call without the field reads `not_recorded` and is never shown as a
person's. The issue's word "standing" is not used: the field is `made_by` (IA 2.2).

It lands in chunk A2, which already owns the attested door and the model/person split. Its page line is S5b's
(slot S3, F9); A2 stores and returns the field and adds it to listings' data and the CLI's JSON.

---

## 3. The clause that reads a person preference

**Build the reading now, the clause last.**

1. The bundle gains `preferences` per aspect and candidate, each with `made_by`. A person's is a contrast row; a
   model's is data one step down.
2. `ReportInput.human_pref` (`analysis/report_input.py:813`) is filled from a person's only (#1142 U9).
3. A separated preference for the candidate from a person's random sample moves the candidate's standing as a
   `counted` contrast does, and one against it likewise.

**It is one more row of the candidate's gains family** (C-7): its p enters Holm with the aspect's other gain rows.

```python
class PersonPrefers(EvalDocumentModel, CoreShapeModel):
    """The candidate is preferred to the control by a person, on a blind read of this aspect.
    One person's preference is that person's; the read names them."""
```

A member of `DecisionClause` (`kernel/declaration.py:286-340`), reading the shared clause state:

| State | When |
|---|---|
| `held` | A closed person read of the aspect, a random sample, preference separated for the candidate |
| `broken` | The same, separated for the control |
| `undecided` | Anything else, with a reason: `not_recorded` (no closed person read, a model's read only), `not_drawn_at_random`, `too_few`, or none when the read simply does not separate; and the non-tie cases at the observed share that would settle it |

It behaves as every floor does (`kernel/declaration.py:1109-1120`): a warning, never a block. A preference on an
aspect whose A/A check separated is `set aside` with it (§5.5).

---

## 4. An A/A check inside a campaign

### 4.1 How identical arms get distinct cells

A cell is `(variant_key, apparatus_class_id)` (`analysis/cells.py:509-533`). The variant key must not change
(`kernel/identity.py:154-173`). The apparatus class digests the recorded rig (`analysis/cells.py:389-426`), and
hosts already declare rig values a launch sets (`run/launch.py:1154-1158`).

**The engine declares one rig setting, `replicate`**, an integer from 1. A launch that sets `replicate=2` records
it on the run's `apparatus_settings`. Two runs of one variant at replicates 1 and 2 are two cells by the existing
rule (IA M1 lists the replicate coordinate as an instance of "same conditions"). No change to the cell key, the
variant key or `IDENTITY_VERSION`.

1. **Recorded on every run, from the identity version that introduces it.** Revision 3 let a blank read as level
   `1` when some run in the cohort recorded one, and vanish otherwise. That is expedient: a run's cell would
   depend on which other runs it is read beside, and a silence would be read as a match, which S1 forbids. The
   principled form: every launch records `replicate` (1 unless told), and `IDENTITY_VERSION` moves one step
   (25 to 26). A run stamped under the earlier version could not have been a replicate, so the version's own
   re-key rule reads it as 1, marked derived, as every predicate change is handled (`docs/design.md`, "A
   predicate change is a version bump"). Cost: one identity step, its golden vectors regenerated deliberately,
   and earlier runs comparing through the versioned path. It should ride the structure track's next identity move
   if one is planned, so the version moves once.
2. **A rig coordinate, not a confound.** Confound scans and the bisection skip it by name; it stays in the class
   id. One flag on `Sweepable`, the mirror of `no_rig_coordinate` (`kernel/host/sweepables.py:214`).
3. **The context key moves for the replicate run only**, keeping the twin out of the control's pass^k headline.

| Where | Change |
|---|---|
| `run/launch.py` `LaunchRequest` | `replicate: int | None = None` |
| `ops/sweeps.py:190-214` `SweepArm` | `replicate: int | None`; the "one arm measured twice" refusal stays for undeclared duplicates, its remedy naming `replicate` |
| `quick/contract.py:219` `Eval` | `replicate_control: bool = False` (one stem with the launch parameter, V6; revision 2's `null_control` is dropped before it ships). Readers see "the control run twice" |

### 4.2 The A/A check

`analysis/aa_check.py` (new; internal name null contrast). For each variant holding cells at two replicates, it
reads replicate n against replicate 1 on every reading the real contrasts read, through the same functions,
paired by case. Its output is a **check**: `instrument: rig`, `against: identical_arms`, one result per reading.

**Family** (C-8). One family per pair over every reading on every aspect, two-sided, no direction.

**Coverage** (C-10). The check records whether its replicate shared a launch group with each candidate, and
speaks only for those contrasts.

| Result, per reading | When | Reading-level effect |
|---|---|---|
| `met` | **An equivalence reading** (S3): `paired_equivalence` (`analysis/stats.py:2002`) holds on the differences at the reading's declared margin, at the family's level. With no declared margin it is never `met` | None. Never first-screen (F6) |
| `undecided` | The two runs did not separate and equivalence is not shown | None. At 7 to 12 cases on a 10-point margin this is always the result |
| `not_met`, reason `identical_arms_separated` | They separated | Every covered `improved` contrast on that reading is `set aside` with that reason |
| `insufficient`, reason `not_recorded` | The candidate was launched in another group | None; one step down says the A/A does not cover it |
| `insufficient`, reason `identity_unrecorded` | Either run has an unrecorded coordinate (S1) | **None: it sets nothing aside and is never read as quiet.** The difference and interval are shown one step down, labelled with the coordinate |
| `not available`, reason `replayed` | Both sides replayed a cassette | None |

Figures carried one step down: readings separated of m; each null difference with its interval, printed beside
the real contrast on that reading with no marker (C-9); the largest absolute difference, worded as a maximum over
m readings. No false-positive rate is reported: one pair is one draw.

A replicate of the control covers every candidate's contrasts in its launch group; a replicate of a candidate,
that candidate's only.

### 4.3 Simulation

`tests/test_simulated_aa_check.py`, seeded, reported per reading kind:

| Claim | Passes when |
|---|---|
| An honest rig rarely separates (7, 12, 20, 40 cases; k 1 and 3; 9 readings over 3 aspects) | Any reading separated in at most 5% of campaigns |
| The cost of the set-aside (honest rig, a true improvement on one reading) | The improvement is lost in at most 5% of campaigns |
| What the set-aside buys (a drift large enough to fake an improvement, against an honest rig) | Both firing rates recorded at 7, 12, 20, 40 cases. No bar; where close, the docs say so |

### 4.4 Cost, schema, left out

Cost: one more run of the control; +50% of run spend with one candidate, +25% with three. A judge repeat pass is
the cheap substitute for the judge's share only.

Schema, additive: a possible key in `EvalRun.apparatus_settings`; an optional field on `LaunchRequest` and
`SweepArm`; checks in the bundle.

Left out: a clause requiring an A/A; pooling the replicate into the control; splitting simulator noise from
candidate noise.

---

## 5. Judge error in the contrast

### 5.1 What is already in the interval

Noise (a judge re-reading one answer) lands in each case's mean, so the interval over cases already carries it.
Bias equal in both arms cancels. A bias that differs by arm does not, and `docs/design.md`'s "mostly cancels" is
untested.

### 5.2 The options

| Option | Verdict |
|---|---|
| **A. PPI on paired differences** (valid when the labelled cases are a random sample) | **Adopt**, as a check on the contrast; decision use from 40 labelled cases |
| B. Correction from sensitivity and specificity (assumes identical errors in both arms) | Reject |
| **C. Re-score noise as its own component** | **Adopt**, one step down, attribution only |
| **D. The contrast's state follows the judge's checks** | **Adopt**, as the decision rule |

### 5.3 The corrected reading: a check on a contrast

For cases i = 1..G shared by control and candidate, Fᵢ is the judge's difference of case means. On the labelled
cases L (a random sample of n), Yᵢ is the reader's difference on the two drawn results and Fᵢᴸ the judge's on the
same two. Rᵢ = Yᵢ − Fᵢᴸ.

    estimate = mean over G of Fᵢ  +  mean over L of Rᵢ

Standard error: the case-clustered linearisation of `prediction_powered_mean` (`analysis/stats.py:727-798`), on t
with n − 1 degrees of freedom. A sibling function, `prediction_powered_difference`, in
`analysis/people_corrected.py` (new). Identical-output cases are a known stratum: reader difference 0 with their
share of G as weight (C-6).

**It is one field on the contrast, `corrected`, with `against`** (`people` or `model`). It is never a second row,
a second verdict cell or a second contrast noun (IA 1.3). The contrast keeps one verdict, the judge's. `corrected`
is a check of that verdict:

| `corrected.result` | Meaning | When |
|---|---|---|
| `met` | Confirms | The corrected interval, at the level the row's Holm step used (C-5), excludes zero in the verdict's direction |
| `not_met`, reason `people_disagree` | Contradicts | It excludes zero the other way. The reason is raised for `against: model` too, where the word is false: a renderer reads `against` |
| `not_met`, reason `instrument_leans` | The judge leans | The lean R̄ has an interval excluding zero toward the arm the verdict favours, and the corrected interval does not confirm |
| `undecided` | Does not confirm | At least 40 labelled cases, none of the above |
| `insufficient`, reason `too_few` | Figures shown one step down, moves nothing | 20 to 39 labelled cases |

Below 20 labelled cases, or from a sample not drawn at random, or with more than 10% skipped, or where either
run has an unrecorded coordinate (S1; the contrast is already `withheld`), the field's state is `not available`
with reason `too_few`, `not_drawn_at_random`, `skipped_past_limit` or `identity_unrecorded`. No free-text reason exists.
Whether the check may move a decision is derived on read from its labelled cases; it is not stored.

The estimate, its interval, the lean and its interval are figures inside the field, shown one step down as
"people's reading" (or "a model's reading (`<id>`)"). The lean is carried in both directions, with the count of
dimensions tested.

**Assumptions.** L is a simple random sample of the cases; the population is the read's; one person's label is
taken as "people"; t is adequate for the mean of Rᵢ at the floor.

**Why 40** (B-2). A lean concentrated in a few cases broke n = 20 in the review's runs: a sample with no
disagreeing case has a rectifier of exactly zero with zero spread, the corrected interval collapses to the
judge's own, the miss rate reaches 9 to 11%, all on the confirm side. `PPI_DECISION_MIN_LABELLED_CASES = 40`, or
the smallest n that passes the simulation in the engine's own code if higher.

**Power** (review). The lean test against a lean of 0.15 on pass/fail: 37% at n = 20, 68% at 40, 86% at 60. Two
costs for the docs: a judge that exaggerates a real difference also "leans" (N-12); kappa falls on a skewed base
rate, so high-pass-rate dimensions lose improvements until about 60 labelled cases (N-15).

**Simulation** (`tests/test_simulated_people_corrected.py`, seeded, 4,000 replicates per row; G 20 to 100; k 1 and
3; n 20, 40, 60, 80, 120 and n = G; pass/fail and 1 to 5; bars fixed before the run):

| Judge | True people difference | Passes when |
|---|---|---|
| Unbiased, noisy; and biased equally in both arms | 0 | Corrected interval excludes 0 at most 6.5% |
| Leans to the candidate by 0.15 | 0 | The same; the judge-only `improved` rate recorded |
| Passes everything | 0.2 | Coverage at least 93.5% |
| One-sided rare lean (the judge passes a failed candidate answer on 3%, 6%, 10% of cases near ceiling; a 5 on 5% of answers on 1 to 5; G = 100, k = 3) | 0 | Coverage at least 93.5% at the decision floor; recorded at n = 20 |
| The confirm quantity: judge reads improved and `corrected.result` is `met` | 0 | At most 3.5%, every row, at the decision floor |
| Identical outputs on 20% of G | 0 and 0.2 | Coverage at least 93.5% with the stratum rule |
| Content-following skips at 5% and 20% | 0 | At 5% recorded; at 20% `not available` |
| Lean test over 5 dimensions | no lean | Chance of any spurious lean, recorded |
| Labels drawn by a selection rule | 0 | Fails by a recorded margin |

**The shipped cell-mean PPI: decided (Q10, yes).** `analysis/bundle/cell_reads.py:290-335` reads person ratings
from a random sample only; otherwise its state is `not available`, reason `not_drawn_at_random` (replacing its
free-text `unavailable_reason` when S5a's reason set lands). Model ratings never enter it.

### 5.4 Re-score noise (one step down)

`analysis/judge_noise.py` (new). Named re-score noise, never judge error (N-13). Every figure here prints only in
the judge's own table, one step down (IA B2, F6); "judge's share" and "pass^k ceiling" are internal names.

**The judge's share of the contrast's width** (C-11):

    share = re-score variance × (1/k_control + 1/k_candidate)  /  variance of the per-case differences

It exists at k = 1. Printed as a range. Below 20 repeated results from at least 10 cases its state is
`not available`, reason `too_few`. Re-score variance is half the mean squared difference between a score and its
repeat (`JudgeRepeat`, `schema/models.py:5974`).

**The judge's ceiling on pass^k** (B-3), on a pass/fail dimension that counts toward pass: e is the share of
repeats that fail among results whose first reading passed, with a Wilson interval over cases; the ceiling is
(1 − e)^(k−1), as a range, worded "about". (At e = 0.10, k = 3: 0.81.)

Tolerance by scale (C-12): about ±30% from 20 repeats on 1 to 5; about ±70% on pass/fail with e near 0.05.

It changes no interval, no verdict and no state.

### 5.5 The decision rule: decided (Q8, yes)

Owner, 2026-11-14: a judge shown below the bar stops counting toward "better"; one checked against a model counts.

Decision code reads checks, never the tier (T2). An `improved` contrast on a judged dimension is `counted` unless
a row below makes it `set aside`. A `regressed` contrast is never set aside.

**From the judge's checks** (the check that speaks for the judge is §2.3's):

| Check that speaks for the judge | Check against itself | Before | State of an `improved` contrast |
|---|---|---|---|
| Against people, `met` | any | counts | `counted` |
| Against a model, `met`; the model's own check not `not_met` | any | n/a | `counted` |
| Against a model, `met`; the model's own check `not_met` with `instrument_leans` | any | n/a | **`set aside`, `instrument_leans`, for the arm the model leans toward** (a judge agreeing with a leaning reviewer most likely shares the lean); other arms as the next row |
| Against a model, `met`; the model's own check `not_met` with `instrument_below_bar` | any | n/a | read as the "none" rows below |
| None (`undecided` or `insufficient`) | `met` | counts | `counted` |
| None | not `met` | set aside only if both unmeasured | `set aside`, `instrument_unchecked` |
| **Against people or a model, `not_met`** | any | counts | **`set aside`, `instrument_below_bar`** |

**From the contrast's own `corrected` check** (at least 40 labelled cases):

| `corrected` | State |
|---|---|
| `against: people`, `met` | `counted`, whatever the judge's checks say |
| `against: model`, `met` | No effect (C-17) |
| either, `not_met` with `instrument_leans` | `set aside`, `instrument_leans` |
| either, `not_met` with `people_disagree` | `set aside`, `people_disagree`. It does not also count as shown worse (default, §10) |
| `undecided`, `insufficient` | No effect |

**From the rig's check:**

| A/A check on the reading, covering this candidate | State |
|---|---|
| `not_met` | `set aside`, `identical_arms_separated`. **Nothing lifts it** (B-4), and it applies to a person preference on the aspect too. It subsumes every instrument reason on the same reading (F7) |

Where two reasons apply, one is stated, in this order: `identical_arms_separated`, `people_disagree`,
`instrument_leans`, `instrument_below_bar`, `instrument_unchecked`.

Remaining cells (C-15):

- A judge that leans against a candidate it reads as regressed: the contrast stays `counted`; the lean is in the
  `corrected` figures in that direction.
- A judged `regressed` beside a person preference separated for the candidate: both are `counted` rows; neither
  cancels the other.
- A read closed after the contrast was first read counts as a look (§6).

The caveat this raises is one kind, `set_aside:<reading>`, carrying the reason. It replaces revision 2's
`judge_short`, `judge_leans`, `people_disagree`, `null_separated` and today's uncalibrated set-aside
(`analysis/report/call.py:118-122`, `:193`). Per T1, `_judge_uncalibrated`-style recomputation goes: the caveat is
derived from the contrast's state.

A known asymmetry, left by default (N-14): "not checked, self-consistent" is `counted` while "checked, below the
bar" is not.

### 5.6 Left out

Correcting pass^k, composites or the frontier with labels; PPI with a tuned weight; several judges per criterion;
anything built on option B.

---

## 6. Looks

### 6.1 What is controllable

The writer's stopping rule works because the writer eval controls the order cases are read in. A general
campaign's evidence arrives as whole runs; there is no prefix to bet on. A look belongs to the contrast's test,
beside the family size (IA 1.2).

### 6.2 Count them, and let a campaign declare them

**A look is "the data behind a verdict changed".** For each contrast the engine counts, from stored records at
read time: the distinct launch groups (or lone runs) of its two arms from the first moment both held a result,
plus each re-judge, judge repeat pass and blind-read close after that moment (N-17). It is an upper bound.

`AspectDesign` gains `planned_looks: int | None` beside `intended_repetitions` (`kernel/declaration.py:656`). When
set to L:

- every confirmatory test of the aspect reads at α/L (a union bound);
- floors and `within_margin` read `held` on the bounded interval at 1 − α/L (C-13); a broken or undecided floor
  still warns at every look;
- past L the guarantee is gone (C-14).

`planned_looks` declared after the aspect's first result is a late criterion (`analysis/bundle/schema.py:2330`).

**One caveat kind, `looks`**, class D, carrying `planned` and `taken` (IA D1):

| Situation | Caveat | Reach |
|---|---|---|
| Taken within plan | None (F6) | |
| Taken past plan | `looks`, with `planned=L`, `taken=j`: the 5% no longer holds for the aspect's readings | report |
| No plan, more than one taken, campaign created after this feature | `looks`, `planned=None`, on every `improved` of the aspect, with the measured false-separation rate for that count | report |
| No plan, more than one taken, an earlier campaign | The same caveat at reach `method`: stored verdicts do not move and nothing reaches the first screen | method |

"Created after" is read from an internal stamp on the campaign, never printed. D1 renders no line. No verdict is
corrected automatically.

### 6.3 Simulation

`tests/test_simulated_looks.py`, seeded, 4,000 replicates per row, no true difference, stop at the first
separation:

| Set-up | Undeclared | Declared, α/L |
|---|---|---|
| Cases grow 20 to 60 over 1, 3, 5 looks, mean on t | Recorded (expected near 5%, 11%, 14%) | At most 5% |
| Repeats grow on fixed cases, 3 looks | Recorded | At most 5% |
| A pass rate on the bounded interval, 3 looks | Recorded | At most 5% |
| Unequal batches (20, +5, +35) | Recorded | At most 5% |
| A floor read `held` at any look when the true drop equals the margin | Recorded | At most 2.5% |
| Looks run to 2L | n/a | Recorded |

### 6.4 Left out

An anytime-valid test for general campaigns; a stored log of who read what; false-discovery control across
campaigns.

---

## 7. Discodon's one-time calibration

Discodon is a separate repository this design cannot see.

**First, at no human time.** A model reads the preserved 2026-10-17 outputs blind (§2.6): the first check of that
model against the owner, on 15 pair items. Expect `undecided`.

**Then, to unblock decisions.**

1. Pick the campaign carrying the next decision, with at least 40 cases both arms answered.
2. `blind_read_create` over it with a declared reviewer, verdicts and preference, on the judged dimensions that
   enter a pass rule or a guardrail, at most four (`persona.in_character`, `persona.no_machinery_narration`, plus
   what gates the campaign).
3. `blind_read_review`, then close. Judges whose check against the model is `met` count for decisions.
4. Author 10 hard negatives per pass/fail dimension (`judge_case_author`): #848's judge passed 30 of 30.

**Then, to check the model against people.** The owner answers the same read from the top (answers through
`blind_read_answer`, attested). Every item answered is a double-read pair item.

| Owner answers | Time | What it settles |
|---|---|---|
| 20 items | about 1 hour | The people check can decide at 40 results; a finished preference separates at 75%; the model's check reads `undecided` |
| **40 items** | about 2 hours | The `corrected` check may move a decision; a true-0.9 judge has an 81 to 99% chance of `met` against people at 80 results; a preference separates at 68%; a leaning model is caught (a lean of 0.16: between 75% and 99%). **The model's check will most likely read `undecided`**, with its half-width |
| 60 items | about 3 hours | A lean of 0.08 is caught 74% of the time |
| 105 to 260 items | 5 to 13 hours, accumulated over reads | The model's check can be `met` at ±0.10. This is a corpus that grows, not a sitting |

Set `planned_items` first. Then `judge_cases_freeze(from_labels=true)`, one judge campaign over the frozen cases,
and `judge_profiles_record`. Record the campaign's call with `made_by`.

**What the package reports back.** Per dimension and per "against": the 2×2 table, the check with its figures; the
model's own check with its lean interval; the preference with its test; each contrast's `corrected` check.

| Result | Consequence |
|---|---|
| A judge's check is `not_met` | Its improvements are `set aside` (`instrument_below_bar`). Reword or replace it |
| The judge catches none of what the reader failed | The dimension should leave the pass rule until it measures |
| The model's check against the owner is `not_met` | Judges checked only against it read as not checked on that criterion |
| `corrected` is `not_met` | That improvement is `set aside` (`instrument_leans` or `people_disagree`) |
| The owner's preference separates against a candidate the judge favoured | `PersonPrefers`, where declared, is `broken` |
| Checked against people, `met` | Carried to other campaigns through the stored profile |

The 60 ratings of 2026-10-17 were not drawn through a blind read: they feed kappa and read as not drawn at random.

---

## 8. What this track contributes to the first screen

**No chunk here renders a first-screen line** (F1). S5a computes the one typed first-screen value from the caveat
list; S5b renders it. This track's chunks produce reading states, checks and caveats with `class`, `reach`,
`state` and `reason`. Revision 2's nine ordered lines yield to IA §3.

| Fact | Produced by | Enters as | Class | Reason |
|---|---|---|---|---|
| A covered `improved` contrast on a reading where identical runs separated | C1 (check), E1 (state) | caveat `set_aside:<reading>` | A | `identical_arms_separated` |
| A judged `improved` contrast not counted | E1 | caveat `set_aside:<reading>` | A | `people_disagree`, `instrument_leans`, `instrument_below_bar` or `instrument_unchecked` |
| A judge on an asked reading not checked, or checked against a model only (with the model's own check result, and `same_family_as` where set) | R1 (checks), E1 (caveat) | caveat on the check | C | the check's result and reason. Silenced where the contrast it would qualify is already `set aside` for that cause (F7), and where a people check decides (F7) |
| Looks past plan, or unplanned on a new campaign | D1 | caveat `looks` | D | planned, taken |
| A person's preference | E1 | a `counted` contrast row in the gains family (slot S1 through the standing it moves) | — | — |
| `PersonPrefers` broken | E2 | a warning, as every clause | — | — |
| The call's maker | A2 | `CampaignCall.made_by` (slot S3, F9; not a caveat) | — | — |

**The order these reasons carry into the one ranked list** (S2). S5a implements one order for every track; this
is the part of it this track's reasons occupy. Looks stay below both judge reasons at every look count.

| Rank | Reason | This track's source |
|---|---|---|
| 1 | Evidence not whole | (structure track) |
| 2 | A/A separated | `set_aside`, `identical_arms_separated` |
| 3 | Identity unrecorded | (structure track); an A/A or `corrected` check made `insufficient` or `not available` by it adds no reason of its own (F7) |
| 4 | Late criterion | A `planned_looks` or `PersonPrefers` declared after the first result |
| 5 | Judge unchecked | `set_aside` with, in this order, `people_disagree`, `instrument_leans`, `instrument_below_bar`, `instrument_unchecked`; then a `counted` reading whose judge has no check |
| 6 | Judge checked against a model only | The class C fact; `same_family_as` arms first |
| 7 | Repeated undeclared looks | `looks` |
| 7½ | Judge checked against a model whose own check is `met`, no family overlap | The same class C fact, now a measured instrument |
| 8 | Few cases | (existing) |

The rates behind ranks 5 and 6 are conditional on a lean existing, and nobody has measured how often one does;
the order rests on "unbounded and unmeasured outranks bounded and measured".

Never first-screen (F6), one step down only: a quiet A/A; looks within plan; a judge checked against people; a
model's own preference; the `corrected` figures; re-score noise; the tier value; slot lean; the 2×2 table; an
earlier campaign's looks.

---

## 9. Build chunks

Sizes are source lines excluding tests and docs.

### 9.1 A1 (nearly built): what to align now, what waits

| Cheap to align in A1 now (printed words on the quick path only) | IA |
|---|---|
| No printed line uses "standing" for a sample or for a model's read: "a random sample", "not drawn at random" | 2.2 |
| No printed line prints a tier value or bare "calibrated". A person line reads "checked against people: …" with the criterion state; a model line reads "checked against a model (`<id>`): …" | V5 |
| "A model" for the reviewer in reader text; `Judge(reviewer=)` keeps its host-facing parameter name | 2.1 |
| "Blind read" in place of "labels" in printed text and rung 2 docs; `to_label.csv` gone | V4 |
| "Undecided" where a new word is needed; no new spelling | V3 |

| Waits (no change to A1's stored shape) | Until |
|---|---|
| Reasons as members of the shared closed set (A1's strings map one-to-one onto `too_few`, `not_drawn_at_random`, `skipped_past_limit`, `not_recorded`) | S5a's module |
| The check record replacing the quick path's `JudgeCalibration` | R1 |
| Any page (not print) placement of a blind-read fact | S5b |

Two things in A1 are not words and are changed before it merges, because a stored shape is cheapest to fix
before it ships (§9.5): `made_by: Maker` in place of a `reference` rater kind, and `read_sample` in place of
`sample_standing`.

### 9.2 All chunks

| Chunk | What | Owns | Size | Depends on | Done when |
|---|---|---|---|---|---|
| **A1** | Blind-read core, quick path, a model answering the sheet | `schema/blind_read.py`, `schema/maker.py` (new: `Maker`), `analysis/blind_read.py`, `quick/blind.py`, `quick/result.py`, `quick/judged.py`, `examples/2-replies/*`, `docs/ladder.md` | 2,450 | none | As revision 2, plus: no printed line contains a tier value, "standing", "anchor" or bare "calibrated" (asserted on the print's bytes) |
| A1b | Kappa variance by case, conditional | `analysis/agreement.py`, `analysis/stats.py` | 250 | A1's simulation row failing | The row passes |
| A2 | The host door; the call's maker (#1178) | `ops/blind_reads.py` (new), `actions/*`, `schema/doc_types.py`, `schema/store_port.py`, `storage/*.py`, `run/blind_review.py` (new), `ops/campaign_closure.py` (the `campaign_record` path), `testing/` sheet conformance, `docs/adopting-a-host.md` | 2,100 | A1; `CampaignCall.made_by` through S5a's declaration edit | Create takes no seed; an unattested person answer is refused 403; a model's answer is accepted only from the read's review job; an early close reports the anytime-valid verdict; responses carry `sample` and no word ruled internal; the sheet response holds no id or tally; a call recorded as a model's stores the model and reads back with it, a person's call needs attestation, a stored call without the field reads `not_recorded` and is never returned as a person's; MIGRATIONS and a "For hosts" note |
| R1 | The check record | `kernel/evidence_tiers.py` (`Check`), `analysis/agreement.py` (checks per "against"; which check speaks), `analysis/checks.py` (new: the reviewer's own check, lean with its equivalence reading, kappa), `kernel/host/` (`model_family`), `quick/result.py` (prints from checks) | 550 | A1; the shared state module from S5a | The table of §2.3 as a parametrised test; the reviewer's check returns the four criterion states from fixtures at 0, 15, 40 and 200 double-read items, `met` only with the lean interval inside ±δ, and no fifth string exists; pair items from a run with an unrecorded coordinate are left out of the lean and counted; no decision site imports the tier value (an enforcement test); the stored tier field is byte-identical to today's for a judge with no model check; `tests/test_simulated_checks.py` reproduces the shared-lean rows |
| B0 | Fields, **riding S5a** | none | 0 here | S5a | §9.3 |
| B1 | The `corrected` check; the cell-mean restriction | `analysis/people_corrected.py` (new), `analysis/bundle/cell_reads.py`, the judged-contrast filler in `analysis/bundle/comparisons.py` | 1,000 | A1, S5a | The simulation passes its fixed bars; the decision floor is the smallest n that passed, at least 40; a contrast has exactly one verdict and one `corrected` field whatever "against" (schema test); every non-`met` outcome carries a reason from the shared set and no free-text reason field exists; the cell-mean PPI is `not available` for non-random and model ratings |
| B2 | Re-score noise | `analysis/judge_noise.py` (new), its filler | 500 | S5a | The share exists at k = 1 and recovers a planted value within the per-scale tolerance; e and (1 − e)^(k−1) match the review's table; below the floor the state is `not available`, `too_few`; the figures appear in the judge's table data only |
| C1 | Replicate and the A/A check | `kernel/identity.py` (the version step), `kernel/host/sweepables.py`, `run/launch.py`, `ops/sweeps.py`, `analysis/aa_check.py` (new), `quick/contract.py` | 1,950 | S5a, R1 (`Check`) | Three cells for A, A run twice, and B; every new run records `replicate`; a run under the earlier identity version re-keys as replicate 1, marked derived, and pools as before; golden vectors regenerated with the reason stated; `met` only by `paired_equivalence` at a declared margin; a run with an unrecorded coordinate gives `insufficient`, `identity_unrecorded`, and sets nothing aside; undeclared identical arms refused; one two-sided family per pair; the result per reading is a check (`met`, `not_met`, `insufficient`) or `not available` with `replayed`; a quiet check raises no caveat; no public name contains "null"; `tests/test_simulated_aa_check.py` passes |
| D1 | Looks | `analysis/significance.py`, the reading-scope filler, `kernel/campaign.py` (the internal stamp) | 700 | S5a | The simulation's declared rows at their bars; the count includes launches, re-judges, repeat passes and read closes with no store write; exactly one caveat kind `looks` is raised, class D, with `planned` and `taken`, at reach `report` or `method` per §6.2; no page text is produced |
| E1 | Reading states and caveats | the decision read where S5a leaves it (`analysis/report/call.py` and the bundle-side caveat function), `analysis/report/decision.py`, `analysis/report_input_build.py` (`human_pref`) | 1,300 | S5a, R1, B1, C1, D1 | Every row of §5.5 as a parametrised test asserting the contrast's state and reason, the leaning-reviewer row included; `identical_arms_separated` survives every other input and sets a person preference aside; a model's reading never lifts a state, fills `human_pref` or moves a standing (three tests); the preference is a row of the gains family; each caveat is raised with its class, reach, state and reason, and E1 produces no page block; the old uncalibrated recomputation is deleted |
| E2 | `PersonPrefers` | `kernel/declaration.py` (clause), `analysis/report/decision.py` | 250 | E1, and a first closed person read at Discodon | `held`, `broken`, `undecided` from a recorded preference; a model's read alone is `undecided`, `not_recorded` |
| F1 | Docs | `docs/judges-and-calibration.md`, `measuring-soundly.md`, `design.md`, `open-problems.md`, `concepts.md`, `reading-reports.md` | docs | each part as it lands | `design.md` no longer asserts that bias cancels; every simulation's table is in the docs; the glossary gains "blind read" and loses "labels" (V4) |

Total: about 11,050 source lines in eleven code chunks (A1 +50, R1 −200, C1 +100).

**Order.**

```
A1 ─► A2
 └──► R1 ──────────────┐
S5a ─► R1, B1, B2, C1, D1 ─► E1 ─► E2      (page lines: S5b)
```

- A1 finishes now. A2's blind-read half starts when it lands; its `made_by` field needs S5a's declaration edit.
- R1 needs A1 and S5a's shared state module. B2 and D1 need only S5a and run in parallel with everything. B1
  needs A1 and S5a. C1 needs S5a and R1's `Check`.
- E1 follows R1, B1, C1, D1. E2 follows E1.
- Nothing here waits for S5b; S5b waits for these facts.

Risk, as the IA states it: S5a is now a dependency of seven of these chunks. If it slips, the temptation is a
track-local line "for now". None of this track's done-whens can be met by rendering one.

### 9.3 B0: the optional fields for the one shape move (S5a)

All optional, left out when unset, in IA names. No field carries a free-text reason: states and reasons come from
S5a's shared module.

**Bundle (`analysis/bundle/schema.py`).**

| Field | On | Shape | Filled by |
|---|---|---|---|
| `checks` | each judged dimension's judge entry (beside the stored tier), and the bundle for the rig | list of `Check`: `instrument`, `criterion`, `against`, `against_id`, `same_family_as`, `sample {kind, results, cases, skipped, read_id}`, `result`, `reason`, `figures {kappa, kappa_interval, lean, lean_interval, half_width, reads}` | R1 (judge, reviewer), C1 (rig, with `reading` and `covers_arms` per entry and `figures {delta, interval, p_adjusted}`) |
| `blind_reads` | the aspect | list of `{read_id, made_by, sample {kind, answered, skipped}, planned_items}` | E1 |
| `preferences` | the aspect, per candidate | list of `{read_id, made_by, non_tie_cases, for_candidate, ties, share, test, p, verdict, state, reason}` | E1 |
| `corrected` | each judged contrast | `{against, against_id, result, reason, state, estimate, interval, level, lean, lean_interval, labelled_cases, identical_cases, read_id} | None` | B1 |
| `rescore` | each `JudgedArm` | `{state, reason, variance, repeated_results, repeated_cases, fail_given_pass, fail_given_pass_interval, ceiling_low, ceiling_high}` | B2 |
| `rescore_share` | each judged contrast | `{low, high} | None` | B2 |
| `state`, `reason` | each contrast (if S5a does not already add them for M1 and M2) | the shared sets | E1 |
| `looks` | the aspect's `reading_scope` | `{planned, taken}` | D1 |

Dropped from revision 2's list: `anchor`, `reviewer`, `anchor_standing` (now `checks`); `prediction_powered` and
`reference_corrected` (one `corrected`); `person_preference` and `reference_preference` (one `preferences` with
`made_by`); `null_contrasts` (rig checks); `judge_noise`, `judge_share` (renamed); `accrual_batches`,
`planned_looks`, `looks_rule` (one `looks`); every `unavailable_reason`, `void_reason`, `decision_eligible`,
`closed_early`.

**Report input (`analysis/report_input.py`).**

| Field | On | Notes |
|---|---|---|
| `human_pref` | `ReportInput` | Exists (`:813`); filled from a person's preference only. `HumanPreference` gains optional `read_id`, `person`, `test`, `p`, `verdict` |
| `checks` | `Judged` | The judge's checks, projected |
| `corrected` | `Contrast` | The same one field |
| `state`, `reason` | `Contrast` | As the bundle |
| `looks` | `Aspect` | `{planned, taken}` |
| the caveat entries `set_aside:<reading>` and `looks` | `caveats[]` | With `class`, `reach`, `state`, `reason` |

A model's preference and the A/A figures reach the input as one-step-down data inside `preferences` and `checks`;
neither is in the first-screen value.

**Declaration (`kernel/declaration.py`).** `AspectDesign.planned_looks: int | None`;
`CampaignCall.made_by: {kind, model, served_model, on_behalf_of} | None`.

**Shared module (S5a's).** This track needs these members to exist: states `counted`, `set aside`,
`not available`; reasons `instrument_unchecked`, `instrument_below_bar`, `instrument_leans`, `people_disagree`,
`identical_arms_separated`, `too_few`, `not_recorded`, `replayed`, `not_drawn_at_random`, `skipped_past_limit`,
and `identity_unrecorded` usable as the reason of an `insufficient` check and a `not available` field. It adds
none. Every `made_by` in the lists above is a `Maker`.

### 9.4 The tier is no longer reader-facing: what this track does not build

Decided by the owner, 2026-11-14 (IA §6, all five subtractions approved).

| Not built, or deleted | Chunk | About |
|---|---|---|
| `JudgeEvidenceTier.anchor` and any model-aware tier value or tier sentence; `tier_sentence_in` (`analysis/agreement.py:1405`) is not extended and retires with its callers in S5b | R1 | −120 |
| The quick path's `tier` on `JudgeCalibration` and any printed tier word | R1 | −40 |
| Tier-shaped stand-in logic for a stored profile; a profile carries checks | R1 | −40 |
| A tier in any decision site: `_UNCALIBRATED` and the uncalibrated recomputation (`analysis/report/call.py:193`) are deleted, not adapted | E1 | already counted |
| "Evidence tier" in the glossary and the reading-reports section; one section, "how a judge is checked", replaces both | F1 | docs |
| A third `rater_kind` value | A1 | replaced by `Maker` |

The stored tier field keeps being written for old readers, from the people and itself checks, unchanged.

### 9.5 Expedient forms replaced by the principled one

| Was | Now | Cost |
|---|---|---|
| A model's rating as a third `rater_kind` value with the model's id in `rater` | `made_by: Maker`, one type shared by ratings, blind answers, preferences and the call; "counted" decided by the rating's source, not its maker; a model's rating id from its configuration's digest | One durable schema step on `CalibrationRating`, a MIGRATIONS entry, about 150 lines of A1 rework (§1.2) |
| A blank `replicate` read as 1 only beside runs that record one | Recorded on every run from one identity version step | `IDENTITY_VERSION` 25 to 26, golden vectors, about 100 lines in C1 (§4.1) |
| `sample_standing` kept until S5a | Renamed `read_sample` in A1 | A rename before merge |

One form is left as it is, with the reason: looks are counted as an upper bound from stored launches and passes
(§6.2). The exact count would need a record of every reading of a verdict, and most readings are unstored reads
by agents, so a stored log would be a lower bound presented as a count. An honest upper bound is the principled
choice of the two.

### 9.6 What revision 3 subtracts, against revision 2

| Subtracted | Revision 2 | Revision 3 |
|---|---|---|
| New reader-facing nouns | 6 (blind read, reference reviewer, anchor, replicate, null contrast, people-corrected contrast) | 1 (blind read); lean, look, set aside and re-score noise are existing or one-step-down words |
| New closed state vocabularies | 5 (sample standing; five anchor standings; `void`; "too few to move a decision"; clause `unclear`) plus free-text reasons | 0: four states, criterion states and clause states, all existing or shared |
| Caveat ids | 7 (`judge_short`, `judge_leans`, `people_disagree`, `reference_family`, `null_separated`, `looks_undeclared`, `looks_exceeded`) | 2 kinds (`set_aside:<reading>`, `looks`) |
| First-screen lines rendered by this track | 9 ordered lines and five per-part samples | 0 |
| Bundle and input field groups | 14 | 8 |
| Verdict cells per judged contrast | up to 3 (judge's, people-corrected, reference-corrected) | 1, with one `corrected` check |
| Source lines | about 11,850 | about 11,050 (E1 −400: no first-screen blocks, one caveat kind; R1 −350 with the tier gone; B1 −100; D1 −100; A2 +100 for #1178; A1 +50 and C1 +50 for the principled forms) |

---

## 10. Decisions and defaults

### Decided by the owner (2026-11-14)

| # | Decision | Answer |
|---|---|---|
| Q8 | A judge shown below the bar stops counting toward "better" | **Yes.** One checked against a model counts |
| Q10 | Restrict the shipped cell-mean PPI to randomly sampled ratings | **Yes** |
| — | A blind read answered by a model is a legitimate review, stored apart from person ratings | Ruling |
| — | The IA's five subtractions, the reader-facing evidence tier among them | **Approved** (§9.4) |
| — | Design choices are settled by principled correctness | Standing instruction (§9.5) |

### Awaiting the owner (from the IA; the design works either way)

| Item | Effect here |
|---|---|
| IA F4: three reasons and a remainder count amend acceptance bar A2 | None on this track's chunks; they raise caveats either way |
| IA §5.3: the presentation freeze lifted for S5a's first-screen value | Without it, this track's facts have no first-screen landing; nothing here renders one in its place |

### Defaults taken

| # | Question | Default taken |
|---|---|---|
| Q1 | Retire `to_label.csv`? | Yes |
| Q2 | Must a check against people rest on blind ratings? | Not yet; the check's sample says how many were |
| Q3 | May a reader answer out of order? | Yes; the sample is not drawn at random |
| Q4 | Does a separated person preference move a standing with no clause declared? | Yes, from a person's random sample, as a row of the gains family |
| Q5 | The A/A check's family | One per pair over every aspect, two-sided |
| Q6 | Does a replicate ever pool into the control? | Never |
| Q7 | A separated A/A check | `set aside` on that reading for covered candidates; nothing lifts it |
| Q9 | May the `corrected` check create an improvement? | No |
| Q11 | Undeclared looks | Earlier campaigns: caveat at reach `method`. New campaigns: at reach `report` on every `improved`. No automatic correction |
| Q12 | Who reads at Discodon? | A model over the whole read, then one owner over the same read, 40 items |
| Q13 | `corrected` contradicts a judged improvement: also shown worse? | No; set aside only |
| Q14 | Candidate in the reviewer's family, the reviewer's check under 40 items: does the improvement count? | Yes, with the family property on the class C fact |
| Q15 | Should "not checked, self-consistent: counted" lapse? | No lapse |
| Q16 | Default `planned_items` | The population, up to 30 |
| Q17 (new) | The quick-path parameter for an A/A check | `Eval(replicate_control=True)` |

## 11. Disputed

Every finding of the review and every amendment of the IA is applied. Objections recorded:

- **Review C-16, as it bears on Q11.** The batch count is an upper bound on looks, so a report-reach caveat tied to
  it fires on campaigns that added runs and looked once. Applied as the review states it. The IA's class D ranking
  makes this less costly than it was (it can fall below the cap), which is also statistician question S2.
- **IA 5.1, C1: "the quick-path parameter, the launch parameter and the reader word share one stem."** The IA's
  own reader word for the thing is "A/A check" and its host word is "replicate"; one stem cannot serve both.
  Applied as: both parameters share the stem `replicate`; reader text says "run twice" and "A/A check".
- **IA T3, mapping a quiet result to `met`.** Resolved by the statistician (S3): `met` is an equivalence reading.

## 12. Not verified

- That the identity version's re-key rule can read an earlier run as `replicate` 1 with no change to its stored
  key's meaning, and whether the structure track plans an identity move this can share.
- How launch groups are stored, and where a re-judge or repeat pass records its time.
- Where S5a leaves the dominance read, and whether its contrasts already carry `state` and `reason`.
- That `campaign_record` lives in `ops/campaign_closure.py`; A2's builder confirms the file.
- C-2's kappa clustering: A1's new simulation row settles it.
- Checks are re-read as ratings arrive, and a lower bound re-read until it clears is the stopping problem B-1
  fixes for the preference. It predates this work and is not measured here.
- No test, script or simulation was run for this design.

## 13. Statistician's answers (2026-11-14), all accepted

| # | Question | Answer, and where it is applied |
|---|---|---|
| S1 | May a check be read across runs with an unrecorded coordinate? | No exception to "an unrecorded coordinate matches nothing". A/A: `insufficient`, `identity_unrecorded`, sets nothing aside. `corrected`: `not available`. Kappa unchanged. The lean leaves those pair items out (§2.4, §4.2, §5.3) |
| S2 | Should looks outrank an unchecked or model-only judge? | No, at every look count. The order is §8's |
| S3 | When may a quiet check be `met`? | Only as an equivalence reading: the lean inside ±δ at 40 or more items; the A/A by `paired_equivalence` at a declared margin. Otherwise `undecided` (§2.4, §4.2) |
| — | Decision table | A judge `met` against a reviewer shown to lean is `set aside`, `instrument_leans`, for the arm leaned toward (§5.5) |

No question is open with the statistician.
