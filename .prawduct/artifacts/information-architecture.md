# Information architecture: constraints for the evals big-picture program

Status: constraint document, 2026-11-14, read against `evals/main` at `9dc3882ea`, `design-judges.md` (revision 2),
`design-structure.md` and `program-big-picture.md`. Nothing here is built. It adds no feature and rules on no
wording: it fixes which ideas exist, which words name them, and how facts compose on the first screen. Rule ids
(M, V, F, T) are for reviewers: a later diff conforms or it does not.

Marks: **[owner]** needs an owner ruling; **[stat]** the simplification's soundness is the statistician's to confirm.

## 1. The system on one page

### 1.1 One path

```
case ─► trial ─► reading ─► contrast ─► answer ─► recommendation ─► call
         │         │           │          │            │              │
       counted   one cell,   verdict,   standing     confidence     who made it,
       or lost   whole,      counted    per arm,     (three forms)  on which evidence
                 checked     or not     warnings
```

| Object | Is | Trust is established or qualified here by |
|---|---|---|
| Trial | One case, once, on one arm | It is the candidate's (passed, failed) or the rig's (`lost`, with a cause) |
| Reading | One measure of one arm over cases, **in one cell** | Identity (the cell is known and single), wholeness (the run finished, few trials lost), and the instrument's **check** |
| Contrast | Two readings over the same cases | A corrected test gives a verdict; the reading states below say whether it counts |
| Answer | The question read over its aspects: each arm's standing, each declared line that warns | Only counted contrasts move a standing; a warning never blocks |
| Recommendation | The analyst's pick and its confidence | One of three closed forms, over the qualifications left open |
| Call | The decision | Who made it (a person or a model) and the evidence it was read against |

### 1.2 Four ideas carry all of it

| # | Idea | Status | Everything in flight that is an instance of it |
|---|---|---|---|
| M1 | **Same conditions.** A reading is one cell (arm × rig). A coordinate that was not recorded matches nothing. An arm holding more than one cell has no arm-level reading | Existing ("average only what is provably comparable") | An arm answered by two models; a host measure's definition in identity; a run with no recorded definition; the replicate coordinate; "stale" (the current definition is another cell) |
| M2 | **Whole evidence.** A reading over evidence that is not whole says what is missing and gives no verdict at that level | Existing ("a failure of the rig is not a measurement") | An unfinished run; lost trials; a cost interval that cannot be computed; too few cases; a skipped-heavy blind read |
| M3 | **Every instrument is checked.** A check is: an instrument, what it was checked against (people, a model, itself, identical arms), on what sample, with one result | **New, generalising** "a judge is measured, not trusted" | Blind read (the procedure that makes a check's sample sound); reference reviewer (a model as what a judge is checked against); anchor (the "against"); the reviewer's own lean test (a check of a check); replicate and null contrast (the rig checked against identical arms); re-score noise (the judge against itself); the people-corrected contrast (a check's result carried onto one contrast); the existing did-nothing control and judge examples |
| M4 | **Every judgment says who made it.** A rating, a preference, a recommendation and a call each carry `person` or `model` | **New** | `rater_kind`; the reference reviewer's separate storage; #1178 (the call's maker); the analyst as a model |

Repeated looks are not a fifth idea: a look is one more chance for a test to separate, the same thing the Holm
family already counts ("adjusted for 16 comparisons"). It belongs to the contrast's test, beside the family size.

### 1.3 What does not fit, and what that implies

| Piece | Why it does not fit | Implication |
|---|---|---|
| The people-corrected contrast as a second estimate printed beside the first, able to restore an improvement | It is a second verdict path for one question; a reader meets two numbers and two verdicts for one contrast | It is held to M3: a check's **result on a contrast** (confirms, does not confirm, contradicts), with its estimate one step down. It is never a second row, a second verdict cell or a second contrast noun (§5, B1) |
| The reference reviewer's own preference between arms (`reference_preference`, judges §8 line 6) | By the design's own rule it decides nothing (no restore, no `human_pref`, not "shown better"), yet it is given a first-screen line | It is one step down only. A fact that can change no answer is never first-screen (F6) |
| `looks_rule` stamp, the definition "stamp", `closed_early`, `ordinal` | Bookkeeping that lets code apply M1–M3 to old records | Internal names. None becomes a reader-facing word or state |
| The campaign-kind table that gives the first screen three layouts | It is derived from arm count, not eval kind, so it breaks no ruling; but each track would have to place its line three times | The contract in §3 is one slot order for all three; kind changes only what fills the answer slot |

## 2. The vocabulary, with a budget

### 2.1 Every in-flight noun

Disposition: **R** reader-facing word; **H** host-facing only (docs for a host, API names); **I** internal name only;
**M→** merge into the named concept.

| Noun | Disposition | Ruling |
|---|---|---|
| Blind read | R | The one new reader-facing noun of the judges track. It replaces "labels" as the rung-2 word (V4) |
| Sheet | H | The renderer contract's word. Item, slot, planned items: I |
| Sample standing (`random`, `selected`, `none`) | I, renamed | "Standing" is taken (2.2). The field is `sample`; readers see "a random sample" or "not drawn at random" |
| Reference reviewer | H | Readers see "a model (`<id>`)". One reader word, **reference model**, where a noun is needed |
| Anchor | I | Readers see "checked against people" or "checked against a model". The field stays `anchor` |
| Anchor standing (five strings) | M→ check result | It is a check of the reviewer; it takes the four existing criterion states plus a reason (T3) |
| Rater kind `reference`, `agent`, `person` | M→ M4 | Stored values stay. Reader-facing there are two: a person, a model |
| Replicate | H | A launch parameter. Readers see "run twice" |
| Null contrast | I | Reader word: **A/A check**. It is a check of the rig (M3), not a kind of contrast |
| People-corrected, reference-corrected contrast | M→ check result on a contrast | Reader phrase "people's reading", one step down. One field `corrected` with `against`, in bundle and input alike |
| Judge's lean; anchor's lean | R, one meaning | A grader's lean toward an arm. **Slot lean** is I (it is slot balance, a different thing) |
| Re-score noise | R, one step down | Judge's share, pass^k ceiling: I, shown as figures in the judge's own table |
| Look | R | Accrual batch, launch group, `looks_rule`: I |
| Split arm | M→ M1 | No noun. The arm "holds two cells"; the state is `withheld`, reason `split` |
| Stands alone; stamped definition; unrecorded definition | M→ M1 | No noun. The identity state is `unrecorded` (the word `ServedModelState` already uses); stamping is a host act, H |
| Stopped short | I | Readers see "did not finish". The state is `withheld`, reason `unfinished` |
| Apparatus class, rig class, measurement context, context key | I | Reader- and host-facing there are two words for conditions: **cell** and **rig** |
| Set aside | R | One meaning only (2.3). `JudgingState.set_aside` (judging skipped under a cost cap) is a collision: it is `not_asked` when next touched |
| Caveat, disclosure, warning, flag | Two survive | **Warning**: a declared line crossed (floor, bar, guardrail, `PersonPrefers`). **Caveat**: everything else that qualifies the answer. "Disclosure" and the analyst's "flag" merge into caveat (reach decides where it prints) |

### 2.2 Collisions, ruled

| Word | Used today or proposed for | Ruling |
|---|---|---|
| Standing | An arm on the asked aspects (`AskedStanding`); a blind read's sample; an anchor; a judge ("the judge's standing"); a call (#1178); an insight (`live`, `retracted`) | **An arm's standing on the question**, nothing else. Sample → `sample`; anchor → check result; judge → "the judge's check"; call → `made_by` |
| Calibrated | Tier value meaning "agrees with people" (glossary); under R1 also "agrees with a model" | The stored value stays. Reader-facing text never prints it bare: always "checked against people" or "checked against a model (`<id>`)". A reader would be misled otherwise, so this one is not optional (V5) |
| Tier / anchor / check | Three names for "how the judge was verified" | One record: a check (against, sample, result). The tier is a summary derived from a judge's checks, never separately decided (T2) |
| Cell / rig / apparatus class / context | Four names for "the conditions" | Cell and rig (above) |
| Confidence | `ConfidenceForm` (three forms, ruled); `kernel.authored.Confidence` (four levels) | Three forms. The four-level type is a subtraction candidate (§6) |
| Undecided / unclear / undetermined / not_separated / cannot tell apart | One meaning: counted, and the evidence does not decide | Stored values and frozen wording stay. **No new spelling is added** (V3); new work picks `undecided` |

### 2.3 The one closed set of reading states

A reading or a contrast is in exactly one state. "Undecided" is not among them: it is a result of a counted
reading, not a way of not counting.

| State | Definition | Has a figure | Enters a standing | Existing and in-flight words that map onto it |
|---|---|---|---|---|
| `counted` | Read, shown, and it moves the answer | yes | yes | (the default; includes every verdict, `not_separated` among them) |
| `set aside` | Read and shown with its figures; moves no standing, for a stated reason about its instrument | yes | no | "not counted as an improvement", `judge_short`, `judge_leans`, `people_disagree`, `null_separated`, today's uncalibrated set-aside |
| `withheld` | Figures exist beneath it, but no figure or verdict is shown at this level because the evidence under it is not one whole thing; the parts are shown one step down | below only | no | split arm, stopped short, "no verdict", "stands alone" at arm level, `RemainderState.withheld`, latency withheld under concurrency |
| `not available` | No figure can be computed; says what would make one | no | no | `untested`, too few, no interval, `void`, "not shown", `unavailable_reason`, kappa undefined, `unstated`, `unreported` |

Below the reading, a trial is `lost` (with its cause). "Excluded" names nothing else.

Each non-counted state carries one reason from one closed set, shared by every track (V2):

| State | Reasons |
|---|---|
| `set aside` | `instrument_unchecked`, `instrument_below_bar`, `instrument_leans`, `people_disagree`, `identical_arms_separated` |
| `withheld` | `unfinished`, `split` (with the coordinate: served model, measure definition), `identity_unrecorded`, `measured_under_concurrency` |
| `not available` | `too_few`, `no_variation`, `not_recorded`, `replayed`, `not_drawn_at_random`, `skipped_past_limit` |

**[stat]** Whether two runs whose definition is `identity_unrecorded` may be **contrasted** with each other (the
ruling says they never pool) is not settled by the cell rule. Until ruled, a contrast across them is `withheld`.

### 2.4 Rules

| Id | Rule |
|---|---|
| V1 | A new stored or typed state is one of the four states of 2.3, a verdict, a clause state or a check result. A diff that adds another closed vocabulary for "can this be used" does not conform |
| V2 | A reason for a non-counted state is a member of the 2.3 set. Free-text `unavailable_reason`, `void_reason`, and S7's `why.kind` all draw from it or extend it in one module |
| V3 | No new spelling of "undecided" |
| V4 | The glossary does not grow: a new reader-facing word retires one (§6 lists candidates) |
| V5 | No reader-facing text prints a tier value or the word "anchor"; it says what the judge was checked against |
| V6 | No new public name carries a word ruled internal in 2.1 |

### 2.5 Budget

| Reader | Needs today | Budget | New in flight, after the merges above | Fits? |
|---|---|---|---|---|
| Rung 1 | 6 concepts + k, miss, stratum = 9 | 9 | 0 | Yes |
| Rung 2 | + labels, κ, evidence tier = 12 | 12 | Blind read replaces labels; "checked against" replaces tier; reference model is optional | Yes, only if `to_label.csv` and the tier word both leave rung 2 |
| Rung 3 | + arm, control, verdict, margin, guardrail, gate, factor = 19 | 20 | A/A check, optional | Yes |
| Decider | question, arm, control, verdict, interval, warning, confidence, caveat, recommendation, call = 10 | 12 | set aside; checked against (people, a model) | Yes. Every other in-flight fact reaches the decider as a caveat's reason, never as a noun |
| Host | About 72 glossary entries | 72, falling | cell rule covers definitions and replicates; sheet; reference reviewer; look | Only under V4 |

As designed, before these merges, the two designs and four fixes add about 14 reader-facing nouns and 20 state
strings. That does not fit any row.

## 3. The first screen as a contract

### 3.1 One structure, four slots, fixed order

| Slot | Answers | Holds | Source |
|---|---|---|---|
| S0 Gate | Can this be read at all? | Nothing when it can. Otherwise what stopped it, and S1 carries no verdict | The existing `VerdictRule` chain (`setup`, `unfinished`, `too_small`) |
| S1 Answer | What is the answer? | Warnings and the verdict or headline, as today (their relative order is not changed here) | Counted contrasts, clauses |
| S2 Trust | How far can it be trusted? | **One statement** and at most **three** reasons | The one caveat list (§4), ranked by F3 |
| S3 Change and call | What would change it; who decided | Per reason, the one act that would close it; the call with its maker | Caveat remedies; `CampaignCall` |

No slot, rule or ranking reads the eval's kind, a measure's name or any word of prose.

### 3.2 Rules

| Id | Rule (each testable from a fixture's typed first-screen value, never from text) |
|---|---|
| F1 | Every first-screen qualification is a caveat in the one list, carrying `class`, `reach`, `state` and `reason`. No track renders a first-screen line of its own |
| F2 | The trust statement is: on a written page, the confidence form; on a code-only page, the count of open report-reach caveats. Code never invents a fourth form |
| F3 | Reasons are ranked by class, then by whether they touch the decided or recommended arm on an asked reading, then by caveat id. Classes, in order: **A** the answer was changed (a contrast `set aside` or `withheld`); **B** the evidence is not what it appears (identity unrecorded, criteria set late, a confound reaching every comparison); **C** an instrument on an asked reading is unchecked or checked against a model only; **D** the test's level is loosened (looks); **E** precision is missing on the deciding measure |
| F4 | Hard cap: three reasons. The rest are one line with their count, linking one step down. A reason below the cap is never dropped, only moved |
| F5 | The gate silences S2: an answer that does not exist has no confidence and no reasons. The gate shows what stopped the read and the count of qualifications waiting below |
| F6 | A fact that changes no answer and qualifies none is never first-screen: a quiet A/A, looks within plan, a judge checked against people, a model's own preference, a caveat code proved immaterial ("checked") |
| F7 | Subsumption. A `withheld` reading silences every caveat scoped only to it. A `set aside` contrast is one reason naming its cause; the cause is not also listed (no separate "judge unchecked" beside "set aside: judge unchecked"). `identical_arms_separated` subsumes every instrument reason on the same reading. A check against people silences the check against a model on the same criterion |
| F8 | Class E appears only when the measure without precision is the one the answer turns on. A cost interval that is `not available` otherwise shows in its table cell, one step down |
| F9 | The call line always states who made it, as a person or a model (M4), and whether evidence joined or its identity changed since. It is part of S3, not a caveat, and is never ranked away |
| F10 | Both pages print the same S0, the same reasons in the same order, from one typed value in the report input. The parity guard of structure §2.2 covers it |
| F11 | The confidence form `high` requires zero open report-reach caveats (held today). Nothing else about the form is code's |

**[owner]** F4 amends acceptance bar A2 ("every report-wide caveat code assigns has a landing on the first screen")
to "is one of the three reasons or is counted in the remainder line". The kairo page's seven bullets, two of them
proved immaterial, are what A2 as written produces.

**[stat]** F3 ranks `looks` at class D, so a new campaign's `looks_undeclared` caveat "on every improved" can fall
below the cap. If the measured 11 to 14% false-separation rate should outrank an unchecked judge, D moves above C.

**Yields.** Judges §8 (nine ordered first-screen lines) yields to this section: its lines 2 and 7 become class A
reasons, line 4 class C, line 8 class D; lines 5, 6 and 9 are one step down (F6). Judges §1.8, §2.7, §4.5 and §5.6
first-screen samples are examples of content, not of placement. Structure §2.3's `caveats[]` is the list F1 names.

### 3.3 Applied (slot contents, not wording)

**Clean campaign.** Finished runs, one cell per arm, judge checked against people, one look, a person called it.

| Slot | Content |
|---|---|
| S0 | empty |
| S1 | the verdict per candidate |
| S2 | written: `high`; code-only: zero open caveats. No reasons |
| S3 | the call: arm, by a person, on the current evidence |

**Today's real campaign** ("kairo stronger model"): a host measure whose definition changed and was not recorded, one
arm decided, a model made the call.

| Slot | Content |
|---|---|
| S0 | empty (every run finished) |
| S1 | verdicts on the readings whose identity is known; every reading on the host measure is `withheld`, reason `identity_unrecorded`, and so states no rate and no failed-trial count |
| S2 | not `high`. Reason 1 (class A): the host measure's readings are withheld; its runs carry no recorded definition. Up to two further reasons by rank |
| S3 | for reason 1: re-measure or stamp the runs. The call: the arm, **made by a model**, and "the evidence's identity changed since this call" |

The stale 80% that misled the decider cannot reach S1 under M1 and F7; the model's authorship cannot be missed
under F9.

**Messy campaign**: one unfinished run, an arm split by answering model, a judge checked against a model only,
three looks.

| Case | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| The unfinished run feeds an asked aspect | which run did not finish and why; "3 qualifications below" | no verdict | silenced (F5) | resume or rerun |
| After the run is resumed and finishes | empty | the split arm's verdict `withheld`; the other candidates' verdicts | three reasons, exactly the cap: A `split` (served model) on the named arm; C judge checked against a model, not people; D looked at 3 times, none planned | per reason: rerun the arm on one model; read 40 items; declare looks. Then the call |

## 4. One state model for "can this be trusted"

| Level | Property | Closed values | Fed by | Feeds |
|---|---|---|---|---|
| Trial | Outcome | passed, failed (cause), lost (`ExclusionCause`) | the run | reading wholeness |
| Reading | State | the four of 2.3 | identity (M1), wholeness (M2), check results (M3) | contrast |
| Instrument | Check result, per "against" | `met`, `not_met`, `undecided`, `insufficient` (`CriterionState`, existing) | blind reads, repeats, A/A | reading state (`set aside`) |
| Contrast | Verdict | the five existing | corrected test (family × looks) | standing |
| Contrast | State | the four of 2.3 | its readings' states, checks | standing, caveats |
| Clause | Clause state | three (held, broken, undecided) | its own interval | warnings; caps a standing |
| Campaign | Standing per arm | `StandingWord`, existing | counted contrasts, clauses | answer |
| Campaign | Caveats | one list: id, kind, class, reach, state, reason | every non-counted state, every check not `met`, late criteria, confounds, looks | trust slot, confidence |
| Recommendation | Confidence | three forms | open caveats (analyst weighs; code holds F11) | trust slot |
| Decision | Call | arm, `made_by`, overrides, evidence read | a person or a model | S3 |

The decider reads one statement: slot S2. Everything above it reaches S2 only through the caveat list.

| Id | Rule |
|---|---|
| T1 | A trust fact enters the page through the caveat list or not at all. `_judge_uncalibrated`-style recomputation of a fact the bundle already holds does not conform |
| T2 | The evidence tier is derived from a judge's check results and is never an input to a decision. Decision code reads check results (as judges §5.5 already does) |
| T3 | A check result is a `CriterionState`. R1's five anchor standings map: unmeasured → `insufficient`; measured, undecided → `undecided`; not shown to lean → `met`; leans, below the bar → `not_met` with a reason |
| T4 | Identity has one state per coordinate: `one`, `split`, `unrecorded` (the values `ServedModelState` nearly has). Served model, measure definition and any later coordinate use it |

**[stat]** T3 calls "not shown to lean on 40 items" `met`. That is absence of evidence at a floor, not evidence of
no lean. If the statistician objects, the mapping is `undecided` below a stated power and `met` above it.

**Pairs that answer one question and should be one.**

| Question | Answered today or in flight by | Becomes |
|---|---|---|
| Has this judge been verified, and how did it go? | Evidence tier (4 values); tier criteria; R1's anchor and anchor standing (5); the `judge_uncalibrated`, `judge_partly_uncalibrated`, `judge_unstated` caveats; the set-aside list | One check record per "against"; tier and caveats derived |
| Did the arm stay on the right side of a declared line? | Floor: broken, held, unclear. Bar: cleared, missed, undecided (and `BarState` met, missed, unclear). Guardrail: held, breached, undecided. `PersonPrefers`: held, broken, unclear | One internal clause state; reader words wait for the wording thaw |
| Is this arm one thing? | `ServedModelState` (one, pooled, unrecorded); fix C's split; fix A's definition component; `DefinitionState` (current, stale, unknown, absent); "stands alone" | T4 |
| Is the evidence whole? | `StoppedShort`; `VerdictRule` `unfinished` and `setup`; the `short_run` caveat; completeness | One gate chain (S0) and the `withheld` reasons |
| How sure is the call? | `ConfidenceForm` (3); `kernel.authored.Confidence` (4) | Three forms |

## 5. Conformance: what this changes in the work in flight

### 5.1 Chunks not yet built

| Chunk | Amendment | From |
|---|---|---|
| A2 | The blind-read actions return `sample` (not `standing`) for a reader's sample | 2.2 |
| A2 | #1178 lands here as `CampaignCall.made_by` (`person` or `model`, the model's id, optional `on_behalf_of`), decided by the same attestation the answer door uses; its page line waits for S5b | M4, F9 |
| A2 | No action or response field is named with a word 2.1 rules internal (`anchor`, `null`, `ordinal` stay out of reader text) | V6 |
| R1 | `anchor_standing.state` is a `CriterionState` plus a reason, not five new strings | T3 |
| R1 | The tier is computed from check records; no decision site reads the tier value | T2 |
| R1 | The tier sentence is one-step-down text; R1 adds no first-screen line and no bare "calibrated" for a model anchor | F1, V5 |
| R1 | `reference_family:<arm>` is a property of the check ("against a model of the candidate's family"), not a caveat kind of its own | M3, F7 |
| B1 | The bundle holds one `corrected` field with `against` (`people` or `model`), as the report input already plans, not `prediction_powered` and `reference_corrected` side by side | 1.3 |
| B1 | Its outcome is the contrast's state (`counted`, or `set aside` with a 2.3 reason) and a check result; it adds no second verdict cell | 1.3, V1 |
| B1 | `unavailable_reason` is a 2.3 reason; `decision_eligible` is derived on read, not stored | V2 |
| B2 | Re-score noise, the judge's share and the ceiling print only in the judge's own table, one step down; `unavailable_reason` as B1 | F6, V2 |
| C1 | `void_reason` is state `not available`, reason `replayed`; "not covered" is a check result `insufficient` | V1, V2, T3 |
| C1 | A separated A/A is `set aside`, reason `identical_arms_separated`; a quiet A/A is never first-screen | 2.3, F6 |
| C1 | The quick-path parameter, the launch parameter and the reader word share one stem; `Eval(null_control=)` is renamed before it ships, since "null" is ruled internal | V6 |
| D1 | `looks_undeclared` and `looks_exceeded` are one caveat kind (`looks`, with planned and taken), class D; the old-campaign disclosure line is the same caveat at reach `method` | F1, F3 |
| D1 | D1 renders no page line; `looks_rule` is never printed | F1 |
| E1 | E1 raises caveats and reading states only. "The first-screen blocks" leave its scope and its done-when; "each line of §8 renders from a fixture" becomes "each caveat is raised with its class, reach, state and reason" | F1 |
| E1 | `judge_short`, `judge_leans`, `people_disagree`, `null_separated` and today's uncalibrated set-aside are one kind, `set_aside:<reading>`, with a 2.3 reason | 2.3, F7 |
| E1 | `reference_preference` is projected one step down and is in no first-screen value | 1.3, F6 |
| E2 | `PersonPrefers` reads the shared clause state; "a reference read only" is `undecided` with reason `not_recorded` (no person read) | T-table, V3 |
| S5a | Owns the shared closed sets (reading state, reason, identity state) in one numbers-side module | V1, V2, T4 |
| S5a | `caveats[]` entries carry `class`, `reach`, `state`, `reason`; `report_caveats` moves down with its ranking, cap and subsumption as one pure function | F1–F8 |
| S5a | The report input gains one typed first-screen value (gate, reasons, remainder count, call maker); B0's field list is amended as above before it lands (`blind_reads[].sample`, `anchor_standing.state`, one `corrected`, no `void_reason` or `unavailable_reason` strings) | F10 |
| S5b | The written page renders the four slots from that value and nothing else places a first-screen qualification; "checked" caveats leave the first screen | F1, F6 |
| S5b | The parity guard asserts both pages issue the same gate and the same ordered reasons | F10 |
| S5c | An absent interval is one representation in the forms table: state `not available` with its reason | 2.3, F8 |
| S7 | `why.kind` is the 2.3 reason set where the reading is not counted, and a verdict-explaining kind otherwise; the two are not mixed in one enum | V2 |
| S7 | An arm id can name one cell of a split arm, so a `withheld` arm's parts are addressable one step down | M1 |

### 5.2 Built or in review

| Work | Align now (cheap) | Leave; align in S5a or S5b |
|---|---|---|
| Judges A1 | Nothing in stored shape. If not yet merged: the printed word for a model's read does not use "standing" | `sample_standing` function name; `rater_kind` values (stored; stay) |
| Fix D (merged) | Nothing | `StoppedShort` becomes the S0 gate's `unfinished` reason; its per-surface openers are replaced by the one value |
| Fix C (in review) | The withheld arm-level reading is typed as state `withheld`, reason `split`, coordinate `served_model`, not a served-model-only flag | Its line in `campaign/warnings.py` moves into the caveat list (a split is a caveat, not a warning) |
| Fix A (rework) | The unrecorded state is spelled `unrecorded`, and its arm-level effect uses fix C's mechanism with coordinate `measure_definition` | Its two page changes fold into S2 |
| Follow-on "cells honour measure definitions" | Built as one split mechanism over coordinates, not a second parallel one | — |
| Fix B (in review) | The reasons for no interval at 2 to 4 cases and at the floor are typed reasons (`too_few`, `no_variation`), if not already sentences pinned by tests | Where pinned as sentences: S7 |
| S1–S3 | Nothing | Nothing |

No sound work above is asked to be redone for a name, except the two places a reader would be misled: bare
"calibrated" for a model anchor (V5) and a second verdict beside a contrast (B1).

### 5.3 The one place

**S5a computes the first-screen value; S5b renders it.** S5a is already where the caveat list moves to the numbers
side and where every B0 field lands; adding the ranking there is a sort over a list it already owns. This is
presentation-adjacent work inside a freeze: **[owner]** it needs the freeze lifted for exactly this, on the ground
that five tracks otherwise each add first-screen lines.

| Must wait for S5a and S5b | May proceed |
|---|---|
| Any first-screen line from R1, C1, D1, E1 and #1178; the page halves of fixes A and C; a first-screen statement of an absent cost interval | Every stored field, statistic, simulation and decision rule in A2, R1, B1, B2, C1, D1; E1's decision tables; E2 |

Risk to state plainly: S5a is now the largest chunk and the dependency of nine others. If it slips, tracks will
render their own lines "for now". The integration lead should refuse a merge that adds a first-screen line
outside the caveat list.

## 6. What to subtract (recommendations; each is an owner decision)

| # | Remove | Why it is redundant or not load-bearing | Replaced by |
|---|---|---|---|
| 1 | The evidence tier as a reader-facing word and a four-value label (`calibrated`, `separation`, `incidental`, `undetermined`), with the glossary entry and the reading-reports section built on it | After Q8 the decision reads criterion states, not the tier; `incidental` and a below-bar `separation` now have the same effect; with a model anchor `calibrated` no longer means what the glossary says | "Checked against people / a model / itself", from check records (T2). The stored field stays for old readers |
| 2 | First-screen row 4 ("each report-wide caveat, once"), the written page's unbounded caveat bullets including "(checked)" ones, the glossary's "Disclosure", and the analyst's separate "flag" line | They are four routes for one thing, each unranked; together they are the caveat wall | Slot S2: one statement, three ranked reasons, a count |
| 3 | `kernel.authored.Confidence` (`very_high`, `high`, `medium`, `low`) | A second answer to "how sure is the call", against the ruling of three closed forms. It appears to serve analyses a retired writer stored; confirm before removal | `ConfidenceForm` |
| 4 | The three clause-state trios and `BarState` as separate typed vocabularies | One question, four spellings; `PersonPrefers` would be a fifth | One clause state; reader wording at the thaw |
| 5 | Six glossary entries for conditions (apparatus, apparatus class, apparatus settings, measurement context, seat, and "rig" as a synonym pair) and `ServedModelState`, `DefinitionState`, "stands alone" as separate ideas | The cell rule (M1) with one identity state per coordinate explains all of them | Cell, rig, and T4 |

Also already decided and worth holding to: `to_label.csv` and the word "labels" at rung 2 retire with A1 (V4).

## 7. Not verified

- No test, script or build was run. Counts of glossary entries (72 `####` headings), states and words are from
  reading `evals/main`; branch contents of fixes A, B and C were read by commit summary and file list only.
- That `report_caveats` can carry `class` and ranking without the dominance read it calls today (`_outcome`)
  moving down with it. If it cannot, the ranking's "touches the decided arm" key needs the dominance read on the
  numbers side first, which S5a already plans by moving caveats down.
- "One arm set" in today's campaign was read as "one arm was decided". If it meant one arm was set aside, the
  applied table gains a class A reason and is otherwise unchanged.
- Whether Discodon reads any field this document asks to rename before it ships (`anchor_standing`, `standing`,
  `prediction_powered` on a contrast). None is released, so the cost should be nil; ask before B0 lands.

## 8. The public surface: namespaces, kinds and budgets

From `packages/evals/api/surface.json`: 1,779 exported names, 503 declared (245 `host-contract`, 258 `stable`).
Kind counts are a mechanical pass over the snapshot's `kind` field and root; about 40 are judgement calls (±10).

### 8.1 Five kinds; every declared name is exactly one

| Kind | Definition (the test that places a name) | Declared today | Where they sit today |
|---|---|---|---|
| `typed` | A newcomer writes it in an eval file: a ladder entry point | 24 | package root 11, `quick` 10, `harness` 3 |
| `bound` | A host implements it or hands it to the engine: a protocol, port, registry, store, client, kind seam | 97 | `kernel` 23, `run` 23, `schema` 20, `kernel.host` 7, `ops` 7, `providers` 5, others 12 |
| `invoked` | An operation called by name, from host code or by an agent through the catalogue | 105 | `run` 27, `analysis` 22, `ops` 16, `kernel` 11, `schema` 10, `negotiation` 7, others 12 |
| `received` | A value someone is handed and reads: result and stored-document shapes, literals, constants, errors | 259 | `schema` 75, `kernel` 55, `run` 32, `ops` 27, `kernel.host` 23, `analysis` 14, others 33 |
| `tests` | A conformance kit, a recorder or a guard a host runs in its own tests | 18 | `testing` 13, `kernel.host` 3, `actions` 2 |

Tie-breaks, in order: under `testing` or a conformance check is `tests`; a protocol is `bound`; a function is
`typed` at the root, `quick` or `harness` and `invoked` elsewhere; an exception, constant, literal, enum, model or
dataclass is `received`; any other class is `bound`. **Half the declared surface (259) is shapes**; a newcomer
types 24 names, and the five ladder examples import 11 between them.

### 8.2 Budget what costs attention; govern the rest

| Kind | Discipline | Cap or rule |
|---|---|---|
| `typed` | **Budgeted**, per audience | A first eval (rungs 0 to 1): **8**. Through rung 3: **12**. Every rung: **25** (24 today). A new capability arrives as a parameter of an existing name before it arrives as a name |
| `bound` | **Budgeted**, for a host | Ports a host must implement to run: **20** (18 on the hold-back ledger today). All `bound`: **100** (97 today), ratcheting down |
| `invoked` | **Budgeted by parity, capped loosely** | **110** (105 today). Each operation is either in the action catalogue or carries a stated reason it is Python-only; no operation exists in two spellings (`ops.x` and `run.x`) |
| `received` | **Governed, no total** | (a) reached from a `typed`, `bound` or `invoked` signature or a stored document: zero unreached; (b) documented by generation, never by hand; (c) additive-only once stable; (d) no near-duplicate: two shapes with the same field set under two names fail review; (e) reported as **shapes per concept** (declared shapes ÷ glossary concepts that own one), tracked per wave, with any concept over 8 listed |
| `tests` | **Governed** | One kit per port a host implements; a kit with no port, or a port with no kit, is the finding |

| Id | Rule |
|---|---|
| P1 | The tier map records a kind for every declared name; a name without one fails `test_api_tiers` |
| P2 | `DECLARED_CEILING` is replaced by the per-kind caps above. The total is printed, not gated |
| P3 | A `received` shape is never hidden or demoted to protect a count. It is exported, classified and held to 8.2 (a) to (e) |
| P4 | A new `typed` name needs a rung that teaches it; a new `bound` name needs a host that implements it; both name what they retire or why nothing can |

**The replacement for "under 400"** is the caps above and 8.2 (a) to (e), spelled out in 8.5's last row. Hiding
28 reachable shapes would have met the old number and failed P3.

### 8.3 Namespaces by role

| Role root | For | Holds | Today's roots that feed it |
|---|---|---|---|
| `threetears.evals` | Writing an eval | every `typed` name | root; `quick` (10 typed); `harness` (3) |
| `threetears.evals.host` | Embedding the engine | every `bound` name and the Python `invoked` operations | `kernel.host`, `kernel` (ports, seams), `run`, `storage`, `providers`, `negotiation`, `gen`, `ops`, `transports.fastmcp`, `analysis` (operations) |
| `threetears.evals.shapes` | Reading what you are handed | every `received` name | `schema` (75), `kernel` (55), `run` (32), `ops` (27), `kernel.host` (23), `analysis` (14), others |
| `threetears.evals.testing` | Proving a host | every `tests` name | `testing`; 3 from `kernel.host`; 2 from `actions` |
| `threetears.evals.actions` | An agent calling the engine | the catalogue, its mount and its outcome types | `actions`; the mount from `transports.fastmcp` |

Fifteen layer-shaped roots become five role-shaped ones; the internal packages keep their layout.

**Cost.** The production host has about 512 imports across the fifteen roots, 306 from `run`, `kernel` and `schema`.
A full re-rooting touches nearly all, and its `owner_module` strings and logger names are a second, silent dependency.

**The migration that never breaks a pinned host.**

| Step | What | Host imports moved | Decision |
|---|---|---|---|
| 0 | Kind in the tier map (P1); `surface.md` generated by role, then rung, then kind, with shapes folded under the name that reaches them; per-kind ceilings (P2); `surface_tier` returns the kind too | 0 | Proceed now |
| 1 | The five role roots exist and **re-export declared names from their current homes**. Declared public names only, so the alias-shim ban on private names is not touched; `reexports.toml` gains the role roots as a class, not name by name. Docs, examples and the reference then teach only role roots | 0 | **[owner]** |
| 2 | Per old root, one deprecation cycle: the role root becomes the home, the layer root the re-export, recorded in `MIGRATIONS.toml`; rides S8 and S9, which already move modules and leave one-minor shims | the host's, at its own pace, one root per pin | **[owner]** |
| 3 | Layer roots leave `PUBLIC_ROOTS` | none left | **[owner]**, after the host's import gate reads zero |

Step 0 delivers most of the benefit and changes no import. Steps 1 to 3 are the owner's call, worth taking only if
T6 (the middle rung and docs trunk) confirms hosts are written against roles.

### 8.4 How an agent adopter finds its ten names

| Question the agent has | What answers it | Generated from |
|---|---|---|
| What do I read first? | One short file shipped in the wheel and the repo: the five roles, the rungs, and for each rung the names to import, with one line each. Under 200 lines | The tier map with kind and rung |
| Which names matter for my task? | The same data as a query: by rung, or by role and kind, returning at most the cap for that audience (8, 12, 20) with signatures | `surface.json` + kind + rung; `surface_tier` and `declared_names` already ship |
| What will I be handed back? | For each returned name, the shapes its signature reaches, one level deep | The closed-surface test's reachability walk, which already computes it |
| What can I call without writing Python? | The action catalogue's own listing, each action naming the operation it wraps | `eval_catalogue` |

All four are generated from the tier map and snapshot, never hand-written; a test fails when they disagree.

### 8.5 Amendments to work in flight

| Work | Amendment |
|---|---|
| S5a-1 (shape fields) | Its 28 schema types are `received`. They are exported, recorded with that kind, and count against no cap; the chunk shows 8.2 (a) reachability for each and (d) that none duplicates an existing shape. It adds no `typed`, `bound` or `invoked` name |
| Blind-read slice (A1, then A2) | `Judge(reviewer=)`, `result.preference(now=)` and `Eval` options are parameters and cost no `typed` budget: keep them so. `BlindRead`, its sheet and answer shapes are `received`. A2's actions are `invoked` and enter with catalogue parity; rung 2's typed count stays within 12 |
| The surface page and `DECLARED_CEILING` | Step 0 is one small chunk owned by the structure track, ahead of S8, so every later chunk classifies as it adds |
| T7 "Subtract" bar, surface clause | Replace "Declared surface under 400 names (503 at start)" with: "typed ≤ 25 (8 first eval, 12 through rung 3); host ports ≤ 20, bound ≤ 100; invoked ≤ 110 with catalogue parity; zero unreached and zero near-duplicate shapes; shapes per concept reported each wave" |

Not verified: the 512 host imports are the coordinator's figure; 8.1's boundary cases were placed by rule, not read
one by one; catalogue actions were not counted against the 105 `invoked` names, so the parity gap is unknown.

## 9. Candidate simplifications, tested against the code

Read against `evals/main` at `83ec886a2` and `bp-k` at `4107a244a` (`bp-s5a1`, `bp-j-a1` are merged). Nothing was run.

### 9.1 Verdicts

| Idea | Verdict | Real simplification, or a new layer? | When |
|---|---|---|---|
| A Coordinate registry | **Adopt, as an extension of `Sweepable`, in two steps** | Real, because the registry already exists for the host's coordinates; a second registry beside it would be a layer | Step 1 with K's merge or straight after; step 2 as the first half of C1 |
| B One attest door, one maker | **Adopt now at the action; stored fields later** | Real but small | The action: before Discodon is sent a SHA. `made_by`: A2 |
| C Field kinds on stored models | **Adopt later, generation-first** | Real only if the hand lists are then deleted; until then it is a second source of truth | After S7 |
| D Remedies as invocable actions | **Adopt now** | Real: it joins three things that exist | S5a-2 (the value), S5b (the slot), A2 and K (action names) |
| E The report is the first screen | **Adopt the default now; the analyst's demotion is an owner decision, later** | The default is real and nearly free; the demotion removes one of two pipelines, not the stack | Default: S5b. Demotion: after T4 and T5 report |
| F Generated understanding | **Adopt for tables; reject for the glossary and concept page** | Tables: real. Generated prose: a rename of the reference page | With package U (surface by kind) and S5b |

### 9.2 Evidence, what each replaces, cost and risk

**A. Coordinates.** True to the code, and more than the coordinator saw: the host's rig coordinates are already
declared once, as `Sweepable` (`kernel/host/sweepables.py`), with `indeterminate_when_blank` (what absence means)
and `result_level` (recorded per result or per run). The engine's own coordinates are not declared; each is built
by hand:

| Hand-built today | Count |
|---|---|
| Split functions in `analysis/cells.py` (`split_on_served_model`, `split_on_measure_definitions`), each about 70 lines, with private dimension names and "no value" constants | 2 |
| Readers that re-apply the rule outside `pool_observations`: `lenses/comparison_sets.py` (its own `measure_definitions_differ` badge), `lenses/frontier.py`, `bundle/observations.py`, `bundle/assemble.py`, `bundle/comparisons.py`, `reporting.py` (`ServedModelState`) | 6 |
| Source files naming `measure_definitions` / the served-model split | 17 / 10 |
| The closed `Coordinate` literal in `reading_states.py`; per coordinate, a census entry, a disclosure sentence (`report/aspect/method.py`) and an upgrade note, all kept in step by hand | 1 + 3 per coordinate |

- Step 1 (no new abstraction): every pooling read goes through `pool_observations`; the six readers stop
  re-deriving. Perhaps 400 lines removed. This alone addresses the integration defects between the four builds.
- Step 2: served model, measure definitions and replicate become engine-declared `Sweepable` entries (attestable or
  not, what they split), and the two split functions, the `Coordinate` literal, the reason's coordinate, the
  manifest membership and the disclosure's facts read the declarations. About 700 changed, net about −300.
- **The test: C1 adds `replicate` as one declaration plus a test.** If it needs a third split function, step 2
  has failed and is reverted. The judges design already assumes this ("one flag on `Sweepable`").
- Risk: a slip here moves stored keys. `compute_context_components` (about fifteen hand-passed parameters,
  `IDENTITY_VERSION` 25) is **not** re-derived; the context key stays pinned by `test_identity_golden_keys.py`, and
  "derive the additive rule" applies to new coordinates only. Served model is recorded per result and definitions
  per run; `result_level` carries that grain and the registry must not hide it.

**B. One door, one maker.** True: `run/remeasure.py` (644 lines) holds `attest_measure_definitions` and
`attest_served_model` with the same contract (who, when, idempotent, refused over a record, all checked before any
write), two actions (`runs_attest_definitions`, `runs_attest_served_model`), two result shapes, two run fields.
The writers genuinely differ (one stamps the run; the other marks the run, then each silent usage row), so the
functions stay two.

- Now: one action, `runs_attest`, taking a coordinate; replaces 2 actions and 2 result shapes; about 150 lines;
  cheap only until a host calls the old names, which Discodon is about to be told to do.
- Later, with A step 2: one `attestations` map on the run replaces two fields (a core schema minor).
- `made_by`: `Maker` exists in `reading_states.py`; A2 carries it onto the call. **Do not** fold a coordinate's
  attestation into `Maker`: `attested_by` is an operator's statement about the past, `Maker` says whether a
  judgment was a person's. The existing `attested_person` is how `Maker` is established, not a third door.

**C. Field kinds.** True: `bundle/evidence.py` (1,091 lines) is a parallel hand description of the stored models,
about 25 paired maps (`_RUN` / `_RUN_NOT`, `_RESULT` / `_RESULT_NOT`, each exclusion with its reason). Beside it:
8 `*_PRESENTATION_FIELDS` sets, the frozen version-2 pre-image list, `test_core_schema_pins.py`,
`test_durable_schema_pins.py`, `test_identity_golden_keys.py`. Five hand lists answer "what is this field".

- Why not now: the census merged today, Critic-verified, with goldens regenerated. A slip moves every `v3:`
  fingerprint. The annotation must also sit outside the JSON schema (type metadata, not `json_schema_extra`), or it
  moves the shape pins it is meant to replace.
- How, when it is done: annotate, **generate the census from the annotations, and assert the generated maps equal
  today's** before deleting a hand map. No default kind: an unannotated stored field fails. About 1,500 lines
  touched (`schema/models.py` is over 6,000), about −500 net.
- Rejected part: a stored report's staleness is not derivable from a field kind alone. And a few fields are
  evidence for one digest and not another (a price source); if more than a handful need a per-use override, stop.

**D. Remedies.** True, and half built. `analysis/caveats.py` maps all 16 reasons to 13 closed `Remedy` kinds;
`kernel/remedies.py` already holds about 20 refusal remedies with a test that every action a remedy names is in the
catalogue; `cells.py` already emits `NextExperiment` for a merge an unobserved coordinate blocks (so "nothing
recommends which cell to run next" is not quite true: it is computed and reaches no first screen); tiers carry
`results_needed`.

- The change: `remedy()` returns a typed step: an action name with arguments filled from the caveat's own arms,
  runs and coordinate, or a plan (cells, cases, repeats). 13 remedies; about 400 lines; one test like the refusal
  one. It replaces 13 worded-only remedies and gives `NextExperiment` and `results_needed` their one landing.
- **[stat] Risk, and it is the serious one:** "apply remedies, read again" is repeated looking, mechanised. A
  remedy that adds cases must state a planned size and count as a declared look (D1), and `add_cases` on a
  not-separated contrast must never read "until it separates". Without that, the agent loop manufactures
  improvements at the rates judges §6.3 measures. A step that spends carries its priced bound and is never auto-applied.

**E. The report.** Measured: `analysis/report` 53,829; `analysis/analyst` 8,130; `ops/report_writer` 8,229;
`harness` 4,614: 74,800 lines, not 89,000. The first half of E is already the architecture: the code-only page is
the default, the analysis is optional and paid, and the program's one recorded decision was made from the
code-only page.

- Adopt now: the four slots plus an evidence appendix are the default page on every door, the quick path's print
  included. This is S5b's scope stated as a product rule.
- The demotion (the analyst elaborates within slots; cannot add, drop or re-rank) reverses a standing principle in
  `docs/design.md`: "the analyst chooses the story ... Code never chooses structure, emphasis or framing". Owner
  decision. What it would remove is one of the two page pipelines and the machinery that polices a free-form
  writer: `report/doc_page` (4,037), the bundle-only code-page modules it would supersede or merge with (about
  16,000, per structure §0.4), the editor pass, and most of what rung 5 evaluates (`harness`, pairwise promotion).
  A fair estimate is 20,000 to 30,000 lines, not the stack: `report/keys` (8,791), charts, forms and blocks stay.
- What the owner gives up: a written argument (why not the others, what to watch) and cross-aspect synthesis.
  Decide on evidence: T4 (is any writer measurably better) and T5 (does the decider act from the slots alone).

**F. Generation.** Docs are 12,334 lines; `reference.md` (2,045) is generated already. Generate, from the sets
that now exist in code: the surface by role and rung (127), the states-reasons-remedies table, the coordinate
table, the action list: about 500 lines that are hand-kept or missing today. **Reject** a generated glossary and
concept page: a registry yields a term and a link, not an explanation. Deletable docs come from fewer concepts
(§6, V4), not from generation: roughly a third of the 72 glossary entries, and most of `reading-reports.md` (1,166)
once the page is four slots. Estimate, rough: 500 generated, 1,500 to 2,500 deletable, the rest worth hand-writing.

### 9.3 Two the coordinator missed

| Idea | Evidence | Replaces | Size, when, risk |
|---|---|---|---|
| **G. Retire the mapped vocabularies; the mapping is a debt register.** S5a-1 mapped 37 reading-state vocabularies (155 words) onto four states in `analysis/state_words.py` (245 lines). A mapping keeps both sides alive: 37 vocabularies plus a table | The 142-vocabulary count; `state_words.py` | Each mapped vocabulary, as the chunk that owns its module lands (stored values degrade on read through the table, which then holds only stored spellings) | A ratchet test: the count of mapped vocabularies in live code only falls. No chunk of its own. Risk: a stored value loses its reader; the table is that reader |
| **H. "Unrecorded" is for old runs only.** Every attest door, `identity_unrecorded`, and its remedy exist because a coordinate can go unrecorded. For a run launched on this build they need not: definitions come from the host's registry at launch, and a scripted or replayed completion can record what answered (already ruled) | The owner's ruling of 2026-11-14; `launch.py` records definitions | Attest and re-measure as concepts a newcomer meets: they become an upgrade procedure, out of the ladder and the glossary | One test: a fresh run on every shipped provider, recording and example never yields `identity_unrecorded`. Small. **[owner]** what a live provider that names no model does: refuse, or record "not named" as a value that matches only itself |

### 9.4 The two that most serve a non-expert, person or agent

1. **D**, with its looks guard: the only idea here that changes what an agent adopter can do unaided.
2. **E's first half**: one default page of four slots on every door, needing no model, from rung 1 to a host.

A, B, C and G make the system cheaper to keep and harder to break; an adopter meets none of them. The simpler idea under all six: **delete the second copy before declaring a registry for it** (A step 1, G, E's
single pipeline). Each time the evidence was two implementations of one rule; one caller path needs no new noun.
