# Candidate skill: headline-answers

A copy of the shipped analyst skill whose editor raises five more must_fix findings, each from the owner's reading of
the rung-5 pages on 2026-11-06 (findings H1 to H14 in the reset's presentation review), and whose `SKILL.md` tells the
analyst the same rules where it says what the first screen holds.

## Files changed

| File | Change |
|---|---|
| `SKILL.md` | the headline and decision rows, and a paragraph on the first screen |
| `editor.md` | five must_fix findings |
| `README.md` | this record, in place of the shipped skill's notes for its maintainers; no session reads it |

Every other file is the shipped skill's, byte for byte.

## Rules

| Rule | Findings it answers |
|---|---|
| The headline answers the question asked, in one sentence of at most 25 words, with the answer's figure | H1, H9 |
| An absence is a caveat, never headline matter and never a sentence of its own | H5, H9, H10, H12 |
| Watch lists only what could change the call; when every alternative read worse and cost more, it says so in one line | H3 |
| A rate on the first screen carries its count | H11 |
| A term the reader did not bring is defined at first use or replaced | H2 |

## Digests

| Skill | Skill digest | Analyst text digest | Editor text digest |
|---|---|---|---|
| shipped | `4e14e276b568647e68450b3b025204213d2bddcc3a2ff6b838550224149788b8` | `063768a69822c5655b17d434470f3a655506810d0cc5334d6f4c3216c3ba368b` | `07e15b95eed5b308c0986f019b0f88fb56bce14d18f99f0fa84b77cb295d5611` |
| headline-answers | printed by the run | `1487881bcf1ab92bbf270a831a7fa4ce9019ab9c5b9ff04c4679382b9165b4b2` | `e14e57db525bafabcd88da4b1a3cbeef7b95e4ba4c095c0b21974c8352253edf` |

The candidate's skill digest covers this file, so this file cannot hold it: the acceptance readout prints it, and
`acceptance.json` records it under `candidate_skill`. The analyst's text is read with no note withheld.

## Running it

From the repository root, over the 25 held-out cases the held-out README gathers into `.prawduct/acceptance-cases`, at
the $1.50 report budget and $104 cap of the last no-editor record:

```bash
uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance \
  --candidate-skill packages/evals/tests/fixtures/writer_cases/candidates/headline-answers \
  --cases .prawduct/acceptance-cases --out .prawduct/acceptance-headline-answers \
  --held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium \
  --budget-usd 1.5 --turns 30 --request-s 50 --cap-usd 104
```

## The diff

```diff
--- a/SKILL.md
+++ b/SKILL.md
@@ -26,10 +26,10 @@
 | Field | Purpose |
 |---|---|
 | `story` | The dominant form (`stories.md`). It describes the report; it selects nothing. |
-| `headline` | The call and its reason in one sentence, or the trade, or what is unresolved. Never the eval's question restated. |
+| `headline` | The answer to the question the campaign asked, with the figure it turns on, in one sentence of at most 25 words: the call and its reason, or the trade, or what is unresolved. Never the eval's question restated, and never an absence. |
 | `lede` | At most two sentences: the strongest supporting fact, and what would change the answer. |
 | `confidence` | `high`, `high_if_immaterial` or `low`, and every open caveat with its `material_if` clause (`principles.md`). |
-| `decision` | Four slots from the shared vocabulary that fit the story: recommendation, why_not, watch, next; the_trade, where_it_flips, our_lean, what_settles_it; what_is_tangled, known_despite_it. |
+| `decision` | Four slots from the shared vocabulary that fit the story: recommendation, why_not, watch, next; the_trade, where_it_flips, our_lean, what_settles_it; what_is_tangled, known_despite_it. `watch` lists only what could change the call; when every alternative read worse and cost more, it says so in one line and lists nothing. |
 | `claims` | The story. Order is page order. Each has `text`, `weight` (lead, support, detail) and `evidence`: one block kind, a tool query, a caption. The first claim best carries the headline. |
 | `explore` | Controls for business choices (a margin, a floor, a price per point), what each binds, and in words where the outcome flips, or that it can't. |
 | `omitted` | Each substantial thing you saw and left out, with a one-line reason. The editor checks these. |
@@ -37,6 +37,8 @@
 | `method` | Term and definition pairs: cases and repeats, what pass means, intervals and correction, the judge, cost basis, what wasn't compared, the source. |
 
 A block's `query` uses the tools' query language, so the renderer resolves it from the data and a block can never show a figure that doesn't exist.
+
+**The first screen** (the headline, lede, confidence and decision block) says what the report is, never what it is not. An absence (no margin declared, no recommendation, no question) is a caveat in `confidence`, never the headline and never a sentence of its own. A rate there carries its count through its own key ("20 of 20", never "100%" alone). A term the reader did not bring is defined in plain words at first use, or replaced by them. The editor raises each of these as a must_fix.
 
 ## Tools
 
--- a/editor.md
+++ b/editor.md
@@ -47,6 +47,11 @@
   - one measure under two names where a reader would take them for two measures;
   - a claim that ranks models by cost with no provider named: a cost is the provider routing's, not the model's, so a cost comparison names who served each model (the `cost_served_by` fact states it, per arm, where the page cites a cost).
   - a confidence that reads high over an open caveat it does not stand behind: one that could plausibly be material, or whose clause does not say what would make it matter and why the call holds until then. The fix is to say which findings the high applies to and what it leaves open, or to make the form low.
+  - a headline that does not answer the question the campaign asked, with the figure the answer turns on, in one sentence of at most 25 words. A headline that restates the question, leads with what is missing, or runs past 25 words fails the reader at the first sentence they read.
+  - an absence stated as headline matter or as a sentence of its own: "nothing declares whether that pass rate is enough", "this report recommends nothing", "no recommendation was written", "no question was declared". What the campaign did not declare or decide is a caveat, carried in the confidence with what would make it matter. The page says what it is, never what it is not.
+  - a Watch item that could not change the call, the weakest margin included when no move within its interval would change it. When every alternative read worse and cost more, Watch says so in one line and lists nothing.
+  - a rate on the first screen without its count: "20 of 20", never "100%" alone, since a rate over 3 cases and one over 300 read alike without it.
+  - a term the reader did not bring, such as "unnamed provider routing", neither defined in plain words at first use nor replaced by them.
 - **`should_fix`**: the page would be clearer, tighter or better supported, but a reader acting on it now would not be misled.
 
 What sends the report back, while revisions remain:
```
