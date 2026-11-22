# Addendum to design-judges.md: a model as a blind reviewer (owner ruling, 2026-11-14)

**Owner's ruling, in substance.** Requiring a person for every blind read has blocked development: human reads are
expensive, slow and delay work. A blind read done by the best model available (today Fable) is a legitimate review.
Every read keeps a signal saying whether a model or a person did it. Evals then anchor to that model's judgment; as
a corpus of person-reviewed reads grows, the system should adapt to it.

**Coordinator's design reading (proposals, the owner may overrule).**

1. **A third rater standing, never a person rating.** Today a rating is `person` (attested) or `agent` (an opinion
   listed beside people's and counted nowhere). Add `reference` (reader name: "reference reviewer"): a model that
   answered a drawn blind sheet. It records the model id, the served model, the effort, and the read it came from.
   It is never written or shown as a person's rating, and `attests_person` is untouched.
2. **Same sheet, same blindness.** A reference reviewer answers the same drawn order a person would: no arm, run or
   result id, no judge score, a fresh context per item (or per sheet, items independent), the rubric criterion as a
   person sees it. Nothing in shared code names a model: the reviewer is whatever model the host or quick path is
   given; "the best model accessible" is configuration.
3. **Two agreement readings, and a tier that says which.** A judge's agreement is read against reference ratings
   and against person ratings separately. A tier earned on reference ratings reads "calibrated against a reference
   model (<model>)", one earned on person ratings reads as today. Where both exist for a criterion the person
   reading decides the tier and the reference reading is shown beside it.
4. **It unblocks decisions.** For the decision rules (design §4.5, Q8), a judge calibrated against a reference
   reviewer counts as calibrated, so a judged improvement counts, and the first screen names the anchor: "judged by
   X, checked against <model>, not yet against people". The people-corrected contrast (§4.3) may run on reference
   ratings under the name "reference-corrected"; it estimates what the reference model would say.
5. **Three guards, because agreement with a model is not agreement with people.**
   - *Not the same model.* A reference reviewer that is the judge's own model (or the same served model) measures
     self-consistency; such a read is refused as calibration and reads as a repeat.
   - *The anchor is itself measured.* Wherever person ratings exist on outputs the reference reviewer also read, the
     package reports reference-against-person agreement per criterion, and a reference reviewer shown below the bar
     against people on a criterion stops anchoring that criterion. No person ratings: the page says the anchor is
     unmeasured against people. Nothing waits for it.
   - *Shared bias is stated.* A model reviewer and a model judge can share a lean (toward a model family, length,
     style). When an arm's candidate model is in the reference reviewer's family, the first screen says so. The A/A
     control (§3) and a person read are what test it.
6. **Adapting as person reads accumulate.** Person ratings always outrank reference ratings on the same criterion;
   each person read also updates the anchor's own measured agreement (guard 2). No re-run is needed: both are
   ratings over stored outputs.
7. **First use.** The 2026-10-17 owner blind read was found and preserved under
   `.prawduct/artifacts/blind-read-2026-10-17/` (key, pair key, judge scores, 15 pair ratings; 30 + 24 owner-authored
   judge cases are in Discodon's seed). Run the reference reviewer blind over those same outputs: its agreement with
   the owner is the first measurement of the anchor, at no new human time.

**Impact on chunks.** A1: the labels file's `rater` carries a kind (`person` default, `reference`) and, for a
reference, the model id; the print line names the anchor; the quick path gets a one-call way to have a model answer
the sheet through the existing completion port (replayable, recorded like any model call). A2: the answer action
takes a reference reviewer without `attests_person` and records it as `reference`. B1/E1: per points 3–5. Schema:
additive (`rater_kind` gains a value; optional reviewer fields).

**To verify statistically (for the design review):** whether treating reference ratings as the calibration target is
valid for contrasts between arms when reviewer and judge errors are correlated; what guard 2 needs (how many
person-and-reference double-read items) before it can demote an anchor; whether "person outranks reference" needs a
minimum person n.
