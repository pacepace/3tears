# Kairo claim-set answer-key facts (for 3tears #1287)

## WARNING: these are in-sample keys
Every fact here was derived from the same Kairo claims it will be used to grade. A grader scored against
these keys is being tested on whether it matches a claim to its fact. It is **not** being tested on key
coverage: on a new run Kairo will say things no fact here covers. Do not report a key-based score on
these results (or on reruns of these cases) as evidence of how good the keys are.

## Files
- `keys.json`: `{template: {case: ReferenceKey}}` for `dj_enumerate_v1` and `dj_producer_set_v1`, 5 cases each.
  Every entity, name and alias is unchanged from the seed templates. The seed facts are kept, and the new facts are added after them.
- `context_facts.jsonl`: facts about entities that are not in that case's key, with `in_answer_set: false`.
- `pointers.jsonl`: one line per claim (954). Fields: `claim_id`, `key_fact` `{entity, claim}` or null, `where`
  (`core|acceptable|context|none`), `label_used` (`first`, `both` or `second`), `why_null`. Extra fields: `label`,
  `rewritten` (whether the fact text differs from the claim text), and `deduped_into` (the claim whose fact this one merged into).
- `work/`: the per-claim decisions (`dec/*.txt`) and the scripts that build the outputs from them, kept for audit.

## Which claims became facts
- supported, first label (800); or first label unverifiable and second label supported (8).
- contradicted, both labels contradicted (57); or first label unverifiable and second label contradicted (17).
  For every contradicted claim, the fact is the **correct** statement from the evidence notes, never Kairo's claim.
  The evidence gave a clear correct value in all 74 cases, so no trusted contradicted claim was dropped for lack of one.
- `source_url`: the first label's `evidence_url`, or the second label's where the second label was used. In 3 'both'
  cases (`br-df5f2d2d-26`, `ll-c48477f7-16`, `cc-4dad1a57-08`), only the second label's page states the corrected value, so that URL is used.

## Counts
Facts on key entities, per case. "Claims to key" also counts the 19 claims that matched an existing seed fact, and duplicate claims.

| template | case | claims | new key facts | context facts | claims to key | claims to context | null |
|---|---|---|---|---|---|---|---|
| dj_enumerate_v1 | mclaren-dive | 172 | 73 | 58 | 97 | 65 | 10 |
| dj_enumerate_v1 | broadrick-bands | 150 | 71 | 48 | 86 | 59 | 5 |
| dj_enumerate_v1 | connelly-bands | 161 | 70 | 42 | 103 | 51 | 7 |
| dj_enumerate_v1 | lunch-groups | 122 | 63 | 32 | 82 | 39 | 1 |
| dj_enumerate_v1 | rhodes-acts | 87 | 35 | 26 | 49 | 34 | 4 |
| dj_producer_set_v1 | ogilvie-productions | 43 | 15 | 23 | 16 | 24 | 3 |
| dj_producer_set_v1 | fryer-productions | 57 | 15 | 35 | 17 | 39 | 1 |
| dj_producer_set_v1 | burgess-productions | 50 | 10 | 35 | 11 | 39 | 0 |
| dj_producer_set_v1 | gareth-jones-productions | 52 | 15 | 27 | 19 | 30 | 3 |
| dj_producer_set_v1 | kramer-productions | 23 | 13 | 7 | 14 | 8 | 1 |
| | **total** | 917 | **380** | **333** | 494 (core 329, acceptable 165) | 388 | 35 |

The table covers 917 claims. The other 37 claims are `prod_paul_barker_arc`, and all of them are null (see below).

Totals over all 954 pointers: 494 to key entities, 388 to context, 72 null. Of the 494 pointers to key entities, 34 come from contradicted claims. The other 40 trusted contradicted claims point to context.
150 pointers merged into an earlier fact that says the same thing. Most are repeats across runs, and the merge goes by text, word-set or a hand-marked paraphrase.
19 claims matched a seed fact that already exists (for example "Chris Connelly was a member of Cocksure"), so they point at it and add nothing.
I checked the facts produced by the schema's own claim+source_url dedupe: it removed none of them.

Null pointers by reason (72):
- `case_not_in_either_template`: 37. These are all the `prod_paul_barker_arc` claims (the prod-live run). That case maps
  to `dj_career_arc_v1` / `barker-arc`, which is outside this brief's two templates, so none of it is keyed. It has 34 supported and 3 contradicted claims, and it can be done the same way if wanted.
- `unverifiable` (both labels unverifiable): 21.
- `labels_disagree_contradicted_vs_supported`: 10.
- `labels_disagree_contradicted_vs_unverifiable`: 4.

## How entities were chosen (TRAP 2)
- A fact goes on a key entity only when the claim is about that entity itself, named by its name, an alias, or an obvious short form.
  The short forms accepted were "Teenage Jesus", "Dexys", "the Ants", "Adam Ant's band" for Adam and the Ants, and "McLaren" for Malcolm McLaren.
- Some claims are about a track or record of a key entity but do not name the entity, for example "Scum was released in 1987", "'Like Rats' is on Streetcleaner" or "'Down in It' was produced by...".
  These go to **context** under the track or record's name. This is literal and leaves useful facts off the key (Scum, Streetcleaner, Buffalo Gals, Duck Rock and Gangsters are all in context). Moving them is a one-line change per claim in `work/dec/`.
- When a claim names two key entities, the fact goes on the one the claim is about. "JoBoxers included Subway Sect's rhythm section" goes on JoBoxers, and "Ashman left Adam and the Ants for Bow Wow Wow" goes on Bow Wow Wow.
- In mclaren-dive, "Malcolm McLaren" is a key entity, so claims about McLaren himself go on it (for example his label threatening him over Buffalo Gals, or C30 C60 C90 Go written by McLaren with the band).
- In gareth-jones-productions, `Erasure` is the 1995 album. Facts about the band are in context as `Erasure (band)`.
  In the Ogilvie case, Ministry and The Mind Is a Terrible Thing to Taste are not key entities, so they are context.
- No context entity name folds to a key entity's name or alias. This was checked with `fold_name`.

## Supported claims whose fact text was rewritten (18)
These are slashed or loose claims that the first label accepted. For each, the fact states only the part the note supports:
`mm-a6d16112-25` ("Buffalo Gals / Duck Rock 1983" became "Duck Rock was released in 1983"), `mm-326ba4a2-30` and `mm-74d6f7c8-19` (Buffalo Gals 1982),
`ll-c48477f7-04` and `ll-2930ebab-07`, which the labels supported as 1976 and as 1977. Both now read "formed in 1976 or 1977 (sources differ)".
`ll-2930ebab-21` and `ll-a4a4259b-21` are the Harry Crews claims, supported as 1988 and as 1990: one fact is now "active only in 1988", and the album fact reads "1989 or 1990 (sources differ)".
The rest were slashes resolved or wording made clearer: `cc-d7ce46ac-19`, `ll-2930ebab-10`, `mm-a6d16112-29`, `mm-326ba4a2-33`, `ib-4ccd78f5-17`, `ib-4ccd78f5-21`, `ib-ea5c6881-06`,
`do-c4a7eb6d-04`, `do-c4a7eb6d-09`, `do-f61794a5-07`, `br-6371e04c-09`.

## Skipped contradicted claims (17), and why
Labels disagree, so no fact was written:
- `jb-fbd8bd2f-08` 'Flight of the Hermaphrodite' is about eleven minutes long. The second label says supported (10:56).
- `jb-1c94fe2f-29`, `jb-5a00f63a-24` Head of David were a Birmingham band. The second label says supported, since they were from Dudley, a few miles away.
- `jb-5a00f63a-23` Half of Scum's tracks are under a minute. The second label says supported (12 of 28).
- `jb-5a00f63a-07` Broadrick invented industrial metal with Godflesh. The second label says unverifiable, since the credit is shared and contested.
- `cc-d7ce46ac-16`, `cc-d7ce46ac-29`, `cc-4dad1a57-27` Pigface's Gub is from 1990. The second label says unverifiable: Wikipedia and Connelly's site give 1991, while AllMusic and the cassette give 1990.
- `mm-a6d16112-36` Waltz Darling produced Top 40 hits. The second label says supported (UK No. 31 and No. 29).
- `mm-a93abc28-24` Annabella Lwin was 14 when found. The second label says supported, close enough.
- `mm-a93abc28-26` The dry cleaner's was in Kilburn. The second label says supported, since West Hampstead is next door.
- `do-c4a7eb6d-12` Antichrist Superstar is multi-platinum. The second label says supported (2x Platinum in Canada).
- `jf-034ab4e3-11` Clan of Xymox's 'A Day' was recorded at Blackwing. The second label says supported in part, since it was mixed there.
- `gj-0464fe46-06` Metamatic brought Gareth Jones to Daniel Miller's attention. The second label says supported, indirectly through Foxx.

Out of scope (case `prod_paul_barker_arc`, not in either template), each with a clear correction in its evidence:
- `pb-loki-202-15` Barker had left Ministry when he made Lead Into Gold's 'Idiot' (1988). He was in Ministry from 1986 to 2003.
- `pb-loki-202-25` Chris Connelly is from Glasgow. He was born in Edinburgh.
- `pb-loki-202-37` Frankie Knuckles co-wrote 'Your Love'. Jamie Principle is the credited writer.

## Validation
`keys.json` was loaded with the real models, `threetears.evals.schema.reference.ReferenceKey.model_validate`, in two environments:
- the Discodon venv's pinned install (`/opt/venv/.venv/bin/python -I work/validate.py`);
- the 3tears `evals-main` worktree source (`cd /opt/3tears/.claude/wt/evals-main && uv run --frozen --package 3tears-evals python -I work/validate.py`).
  The `/opt/3tears` develop checkout has no `schema/reference.py`.

Both environments accept all 10 keys. The script also checks three things. The entity list, names and aliases match the seed key after the model canonicalises both.
Every seed fact is still present. The model's own dedupe removed no added fact (380 added).
The validator was mutation-checked by adding an empty `claim`, an extra `id` field and an alias that collides with another entity's name. The model refused each one.
