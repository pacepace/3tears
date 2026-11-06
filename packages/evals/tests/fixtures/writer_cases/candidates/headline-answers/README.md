# Candidate skill: headline-answers

A copy of the shipped analyst skill whose editor raises five more must_fix findings, each from the owner's reading of
the rung-5 pages on 2026-11-06 (findings H1 to H14 in the reset's presentation review), and whose `SKILL.md` tells the
analyst the same rules where it says what the first screen holds. `stories.md` changes where it told the analyst to
state an absence as a sentence of its own, so the analyst writes the page the editor reviews it for.

## Files changed

| File | Change |
|---|---|
| `SKILL.md` | the headline and decision rows, and a paragraph on the first screen |
| `editor.md` | five must_fix findings |
| `stories.md` | the tradeoff form's absences, each one caveat line in the confidence |
| `README.md` | this record, in place of the shipped skill's notes for its maintainers; no session reads it |

Every other file is the shipped skill's, byte for byte.

## Rules

| Rule | Findings it answers |
|---|---|
| The headline answers the question asked, in one sentence of at most 25 words, with the answer's figure | H1, H9 |
| An absence is a caveat line, never headline matter and never a sentence of its own | H5, H9, H10, H12 |
| Watch lists only what could change the call; when every alternative read worse and cost more, it says so in one line | H3 |
| A rate on the first screen carries its count | H11 |
| A term the reader did not bring is defined at first use or replaced | H2 |

## Digests

| Skill | Skill digest | Analyst text digest | Editor text digest |
|---|---|---|---|
| shipped | `4e14e276b568647e68450b3b025204213d2bddcc3a2ff6b838550224149788b8` | `063768a69822c5655b17d434470f3a655506810d0cc5334d6f4c3216c3ba368b` | `07e15b95eed5b308c0986f019b0f88fb56bce14d18f99f0fa84b77cb295d5611` |
| headline-answers | printed by the run | `f0be24e1cc01650ebe017ec9505b9cfe3c3905853bf01e1cc5f628ad3ca29aa8` | `e14e57db525bafabcd88da4b1a3cbeef7b95e4ba4c095c0b21974c8352253edf` |

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
--- a/stories.md
+++ b/stories.md
@@ -22,7 +22,7 @@
 
 **Slots:** the_trade, where_it_flips (the price bands where the best option changes), our_lean (labelled yours, with the band it holds for), what_settles_it.
 
-**What it owes:** one axis priced in the other at real volume (`price_tradeoff`), over the two to four options a decider would actually weigh. Too many variants make the bands too thin to act on. If the loss concentrates in one stratum, a mix that avoids it (route that stratum to the stronger option) may be the most useful option, but nobody ran it, so no figure states its cost or gain. Never price it. Recommend running it as a next step, say what its measurement would settle, and name the seam that suggests it. Show any evidence of what users notice. If there is none, say so in Watch. Say plainly that nothing was declared in advance, and name who makes the call.
+**What it owes:** one axis priced in the other at real volume (`price_tradeoff`), over the two to four options a decider would actually weigh. Too many variants make the bands too thin to act on. If the loss concentrates in one stratum, a mix that avoids it (route that stratum to the stronger option) may be the most useful option, but nobody ran it, so no figure states its cost or gain. Never price it. Recommend running it as a next step, say what its measurement would settle, and name the seam that suggests it. Show any evidence of what users notice. Where there is none, and where nothing declared in advance ranks the axes, each is one caveat line in the confidence, never a sentence of its own, a Watch item or the headline. Name who makes the call.
 
 **Usual failure:** deciding anyway. A lean presented as a recommendation hides the fact that the call is a business choice.
 
```
