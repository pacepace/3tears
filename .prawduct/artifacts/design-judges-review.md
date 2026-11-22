# Statistical review: design-judges.md (§1.7, 2, 3, 4, 5) and the reference-reviewer addendum

2026-11-14, read against `evals/main` at `7b2e0afeb`. Read-only; no repo test or script was run.
Simulations are standalone, seeded, standard library only, 2,000 to 20,000 replicates per row
(Monte Carlo error about ±0.4 to ±1 point). Scripts: `ppi.py`, `harsh.py`, `misc.py` under
`/tmp/claude-1000/-opt-3tears/cc05b866-8b6a-462f-b7a1-d2cc7087c69a/scratchpad/`. They implement the design's
formulas as written, not the engine's code, and use t in place of the bounded interval for the judge-only read.

Classes: **BLOCKER** (wrong or false as designed), **CHANGE** (sound, needs a named amendment), **NOTE**.

## Verdict by chunk

| Chunk | Verdict | Why |
|---|---|---|
| A1 | Build the draw, sheet, standing, close and CSV as designed. **Amend the preference verdict first** (B-1) | A mid-read preference p with "keep going" is an optional-stopping loop |
| A2 | Build, with C-3 and C-4 | Seed is a caller parameter; the closer sees partial answers |
| B0 | Hold the shape until the amendments below are accepted | Several add fields; the shape should still move once |
| B1 | **Wait** (B-2) | Floor of 20 fails its own bar on a one-sided rare lean; override level uncorrected |
| B2 | **Wait** (B-3) | The pass^k ceiling formula is wrong by about a factor of two in the exponent |
| C1 | Replicate plumbing not reviewed statistically. `null_contrast.py` needs C-8, C-9, C-10 | Family scope, the noise-floor marker, launch window |
| D1 | Build, with C-13 and C-14 | Floors are left out of the α/L rule; wording past L |
| E1 | **Wait** (B-4, B-5) | Two rules conflict; the reference anchor needs a lean test, not kappa |
| E2 | After E1, with C-7 | The preference adds an uncorrected door to "shown better" |

## 1. Blind read (§1.1, §1.7)

**The unit is right.** The order is a permutation of cases, one item per case, one seeded result per arm per
case. A prefix is a simple random sample of cases, and of case pairs, which is the unit the paired contrast
needs. The concern about drawing over outputs does not apply to the design as written.

**B-1 BLOCKER (A1). The preference test is not valid under the stopping the quick path invites.** The print
shows the direction, the count and the p after every partial prefix, then says "keep going from item 15". A
labeller who stops when it separates gets:

| Sign test re-read, no true preference | Falsely separated |
|---|---|
| after every item, 6 to 60 | 18.4% |
| every 5 items, 10 to 60 | 12.7% |
| at 10, 20, 30, 40 | 9.1% |
| at 20, 40, 60 | 7.8% |

The print also un-blinds the labeller to which arm is ahead while they are still reading. "12 answered and 60
answered are both random samples" is true of the sample and false of the test's level.
*Amendment.* (a) Before the read is complete or has reached a length fixed at draw time (`planned_items`,
stored in the draw; default the population up to 30), the print gives items read and agreement only:
"preference: shown when the read is finished (14 of 20)". (b) If a mid-read verdict is wanted, the drawn
order is exactly the structure `bounded_separation_p(..., planned_n=M)` was built for: bet on the shares
against 0.5 in drawn order with the stake fixed for the population M. That p holds at any stop. The exact
sign test is then reserved for a read that is complete or at its planned length. (c) The host door needs the
same rule: `blind_read_close` is open to any writer, who can read the partial answers against the store and
choose when to close. Close before the planned length reports the anytime-valid p, never the exact one.

**C-1 CHANGE (A1). Skips make a prefix non-random for anything estimated from it.** A skip follows content
(hard or long items). "Counted and left out of every figure" leaves a sample missing not at random that still
reads `random`. *Amendment:* standing carries the skip count (`random, 3 skipped`); the preference and
agreement print it; B1 reads `not available` when skips exceed 10% of the prefix, and otherwise says its
population is "cases this person answered".

**C-2 CHANGE (A1). Agreement counts results, and a pair item gives two results from one case.** The print's
"28 pairs" from 14 items, and the floors of 20 and 120 distinct results, treat two outputs of one case as two
items. The house rule is that the case is the unit. Judge errors on one case's two outputs are unlikely to be
independent (a hard case is hard for both arms). *Amendment:* print cases beside results ("28 results from
14 cases"); add a row to `test_simulated_agreement.py` with within-case correlated disagreement at 10 to 60
cases, two results each; if the kappa interval under-covers there, compute its variance over per-case cost
sums (the per-item costs are already kept on `KappaMoments`). Not simulated here.

**C-3 CHANGE (A2). The seed must not be a free parameter.** `blind_read_create` takes `seed`, and its caller
can see judge scores. Fifty seeds give fifty first-20 prefixes to choose from. *Amendment:* derive the seed
from the population digest and the read's ordinal over that population; record and print how many reads were
drawn over the same population and arms. The quick path's fixed seed is fine.

**C-4 CHANGE (A2).** Close rule as in B-1(c).

**N-1.** Ties dropped and reported: the conditional sign test, valid. The design's table is exact (checked:
6/6, 9/10, 12/15, 15/20, 21/30, 27/40, 39/60). "Too few below 5" should be 6: five non-tie cases cannot
separate (minimum p 0.0625).
**N-2.** Position balance is safe. With permuted blocks of two and a labeller who picks slot A 80% of the
time regardless of content, the sign test rejects 1.1% at n = 20: conservative, never inflated. A slot lean
costs power, so the reported slot lean is the right companion.
**N-3.** Several pairs from one case cannot arise inside one read. They arise across reads (a second
candidate, a redraw). The design must say reads are never pooled into one sign test; with two raters the
bounded test on shares treats raters as fixed, which the page should say.
**N-4.** "Label a failing example to test it" invites outcome-dependent stopping. Stopping at the third
failure (minimum 10 items) biases a fail rate of 0.10 to 0.132 and 0.20 to 0.236. Small, and out-of-order
hunting already reads `selected`, but the cell-mean PPI inherits it. Word the hint as "read further".
**N-5.** Byte-identical pairs are excluded from the pair order. Fine for preference; see C-6 for the contrast.

**Simulation table (§1.7).** Rows 1 and 3 test the right thing. Row 2 (planted trend in case id) is too easy:
any shuffle passes it. Missing: a labeller who stops on the running result (B-1; the bar is at most 5% under
the amended rule, and the table above for the unamended one, recorded); a skip rule that follows content
(C-1); two results per case with correlated judge error (C-2).

## 2. PersonPrefers and Q4 (§2)

**C-7 CHANGE (E1, E2). A separated preference is a new, uncorrected route to "shown better".** A candidate's
judged rows on an aspect sit in a Holm family at 0.05. Q4 adds a second test of the same question at an
unshared 0.05, so the candidate is wrongly shown better on that aspect up to about twice as often.
*Amendment:* either the preference joins the candidate's gains family as one more row, or the page's
statement of what its correction covers names the preference as outside it. The first is the house standard.
**N-6.** `min_cases = 10` on the clause and "too few below 5" on the reading are two floors for one test;
use one. One person's preference stays that person's; the design says so, and the clause's docstring should.
**N-7.** A preference read on outputs from a rig whose A/A separated is as exposed as a judged contrast.
See B-4.

## 3. Null contrast (§3.2 to §3.4)

**What one A/A can show.** One draw from the null distribution of the whole rig, at a family-wise 5%. A quiet
one at 7 to 12 cases is weak evidence; the design says so. A separated one has a likelihood ratio near
power/0.05, which at small n and moderate drift is 1 to 3. It is a prompt to look, not a measurement.

**C-8 CHANGE (C1). The family is under-specified and, read literally, too loose.** "Its own family per
aspect" gives a campaign of A aspects up to A × 5% of spurious separations, while §3.4's bar is "any reading
separated in at most 5% of campaigns". The gains/drops split has no meaning for two labels of one arm.
*Amendment:* one family per null pair over every reading on every aspect, two-sided, no direction, as a
choice's pairs are read. That is what the simulation bar already assumes.

**C-9 CHANGE (C1). Drop the "within what identical arms showed" marker.** It compares a real difference to
the absolute value of one null draw. Simulated with a single null draw of the same standard error:

| True effect, in standard errors | Marked "within" |
|---|---|
| 0 | 50% |
| 1 | 36% |
| 2 | 14% |
| 3 | 3% |

Half of pure-noise differences escape the mark and a seventh of 2-SE effects earn it. *Amendment:* print the
null difference with its interval beside the real one and no marker. "Largest absolute null difference" is a
maximum over m readings and must be worded as one.

**C-10 CHANGE (C1). Say which window the null speaks for.** Interleaving absorbs drift across the arms of one
launch, so an interleaved A/A tests set-up level differences only. A candidate launched in a later batch (§5)
is exposed to drift the A/A never saw. *Amendment:* the null contrast records whether its replicate shared a
launch group with each candidate, and qualifies only those contrasts; for the others the page says the A/A
does not cover them.

**N-8. The set-aside (Q7) is safe and has a stated cost.** It only removes, so it cannot raise a false
"shown better"; §3.4's third row is therefore true by construction and proves nothing. The cost is that at
most 5% of honest campaigns lose any true improvement that falls on the separated reading. That is
acceptable if written down. Replace the third row with two measured ones: how often a true improvement is
lost on an honest rig (bar: at most 5% of campaigns), and how often the set-aside fires on a drift large
enough to fake an improvement against how often on an honest rig, at 7, 12, 20 and 40 cases. If the two are
close at small n, the docs must say the set-aside there is nearly a coin weighted 5%.
**N-9.** On pass rates the real contrast reads the bounded interval, which is conservative at small n; an A/A
on it will almost never separate. The first row's 5% bar passes trivially there. Report the detection table
per reading kind.
**N-10.** A replicate on the candidate sets aside that candidate's improvement only; a control replicate sets
aside every candidate's. The design should state it.

## 4. Judge error in the contrast (§4.3 to §4.5)

### 4.1 Coverage of the people-corrected contrast, as designed

Estimator and variance exactly as §4.3: per-case residuals summed, n/(n−1), t on n−1. Miss rate is how often
the 95% interval excludes the truth. "Restore" is how often the judge reads improved and the corrected
interval lies above zero when people see no difference (one direction; nominal 2.5%).

| Scenario (true people difference) | G, k, n | Miss | Judge reads improved | Restore |
|---|---|---|---|---|
| Unbiased noisy judge, pass/fail (0) | 20 to 100, 1 and 3, 20 to 60 | 4.2 to 5.3% | 2.0 to 2.8% | 0.2 to 0.7% |
| Equal bias both arms (0) | same | 3.8 to 5.1% | 1.9 to 2.8% | 0.2 to 0.7% |
| Leans to candidate about 0.15 (0) | same | 4.0 to 5.4% | 17 to 99% | 2.3 to 4.1% |
| Passes everything (0.2) | same | 4.8 to 6.6% | 0% | 0% |
| Good judge (0.2) | same | 2.9 to 5.8% | 21 to 100% | n/a |
| 1 to 5, noisy, leaning, constant | 20 to 100, 1 and 3, 20 and 60 | 3.9 to 5.8% | up to 100% | 0 to 2.9% |
| **Near-ceiling pass rates, judge passes a failed candidate answer on 6% of cases (0)** | 100, 3, **20** | **10.8%** | 58% | **10.6%** |
| same, 10% of cases | 100, 3, 20 | 9.7% | 91% | 9.4% |
| same, 6% | 100, 1, 20 | 9.2% | 41% | 8.9% |
| same, 6% and 10% | 100, 3, **40 and 60** | 4.3 to 4.8% | 60 to 91% | 1.4 to 3.4% |
| 1 to 5, judge gives the candidate a 5 on 5% of answers (0) | 100, 3, 20 | 9.3% | 29% | 9.1% |

Findings:

- **The variance estimate is right** where labelled and unlabelled items share cases, at k = 1 and k = 3 and
  at n = G. Summing residuals within a case carries the covariance.
- **Ordinary judges, including a constant one and a strongly leaning one, hold about 95%** from n = 20. The
  "passes everything" row at G = 40, n = 20 measured 6.6%, on the edge of the 93.5% bar.
- **A lean concentrated in a few cases breaks n = 20.** A sample with no disagreeing case has a rectifier of
  exactly zero with zero spread, so the corrected interval is the judge's own. The miss rate doubles and all
  of it lands on the restore side.

**B-2 BLOCKER (B1). The floor of 20 and the simulation row meant to test it.** The design's coarse row is
symmetric (±2 on 10%), which passes and hides the failure; its fallback ("the floor rises") therefore never
triggers. *Amendment:* (a) replace that row with a one-sided rare lean at 3%, 6% and 10% of cases, on
near-ceiling pass rates and on 1 to 5, at G = 100, k = 3; (b) set the floor for any use in a decision rule at
the smallest n that passes it, which these runs put at 40 labelled cases; (c) from 20 to that floor the
estimate and the lean may print, marked as too few to move a decision; (d) add the bar on the restore
quantity itself: judge improved and corrected interval above zero at most 3.5% under no true difference.

**C-5 CHANGE (B1, E1). "Restores but never creates" is creation in substance, so hold it to the family's
level.** When the judge is shown below the bar, its contrast carries no warrant and the corrected interval is
the only evidence behind the restored "better". It is read at an unshared 95% on t, while the row it restores
was tested at its Holm step on the bounded interval. *Amendment:* the override reads the corrected interval at
the level the row's Holm step used. The docs sentence "never a wrong safe" is right; it should add "it can be
the sole warrant for a restored better".
The design is right to refuse outright creation (Q9): a corrected reading that separates where the judge's
does not is a t interval on coarse values acting alone.

**C-6 CHANGE (B1). Identical-output cases are in G and never in L.** Their person difference is known to be
zero. Leaving them out of the labelled set applies the rectifier of the differing cases to all of G.
*Amendment:* treat them as a known stratum (person difference 0, weight their share of G), or compute over
the read's population only and name it.

**N-11. The lean test behaves.** Its false alarm toward the candidate is 1.7 to 3.3% per dimension. Its power
against a lean of 0.15 on pass/fail is 37% at n = 20, 68% at 40, 86% at 60. With several judged dimensions
the chance of one spurious lean grows; the cost is a lost improvement, the safe direction, and the page
should give the count of dimensions tested.
**N-12.** A judge that exaggerates a real difference in the right direction also "leans to the candidate".
Rule 2 then sets a real improvement aside whenever the corrected interval is too wide to separate, which at
n under 60 is most of the time. Conservative, and worth one sentence in the docs.

### 4.2 Judge-noise share (§4.4)

**B-3 BLOCKER (B2). The pass^k ceiling is mis-derived.** A measured flip rate f between two readings of one
answer is 2e(1−e), where e is the judge's chance of failing a passing answer. The ceiling is (1−e)^(k−1),
not (1−f)^(k−1):

| e | f | k | Design prints | Correct |
|---|---|---|---|---|
| 0.05 | 0.095 | 3 | 0.82 | 0.90 |
| 0.10 | 0.18 | 2 | 0.82 | 0.90 |
| 0.10 | 0.18 | 3 | 0.67 | 0.81 |
| 0.20 | 0.32 | 3 | 0.46 | 0.64 |

The raw flip rate also averages over failing answers, which do not bear on a consistently passing candidate.
*Amendment:* estimate e directly as the share of repeats that fail among results whose first reading passed,
with a Wilson interval over cases; print the ceiling as a range from that interval and as "about", not "at
most".

**C-11 CHANGE (B2). The share is not defined at k = 1 and answers the wrong question at k = 2 to 3.** With
one judge call per trial, within-case variance is candidate plus simulator plus judge, and only judge repeats
on stored answers separate the judge's part. That is the extra data, and the design has it. But at k = 1
there is no within-case variance to divide by, and at k = 2 to 3 with 7 to 20 cases the denominator has 7 to
40 degrees of freedom, so the ratio of two rough variances swings by a factor of two or more and the cap at 1
hides estimates above it. The reader's question is "repeats or cases", which is about the contrast's width.
*Amendment:* report judge re-score variance × (1/k_control + 1/k_candidate) over the variance of the per-case
differences the contrast already uses. It exists at k = 1. Print it as a range, and `not available` below 20
repeated results from at least 10 cases.
**C-12 CHANGE (B2).** "±30% at 20" holds for a roughly normal score. On pass/fail with a flip rate near 0.1,
twenty repeats give about two flips: ±70%. State tolerance per scale, and add a pass/fail row to the recovery
test.
**N-13.** A deterministic or cached judge repeats itself exactly and reads a share of zero while still being
wrong. Say "re-score noise", never "judge error".

### 4.3 Decision table (§4.5)

**B-4 BLOCKER (E1). Two rules conflict.** §3.3 sets an improvement aside when the A/A separated. §4.5 says
people's corrected reading "overrides all four". The correction repairs the judge. It reads the same outputs
from the same rig and says nothing about a rig that separates identical arms. *Amendment:* the override lifts
the three judge reasons only (unmeasured, below the bar, leans). `null_separated` stands whatever people's
reading says, and applies to a person preference on that aspect as well (N-7).

**C-15 CHANGE (E1). Cells the table does not have.**

| Situation | Missing rule | Amendment |
|---|---|---|
| Corrected reading separates against a judged improvement | "Does not separate" is ambiguous | Set aside, with its own caveat; the owner decides whether it also counts as shown worse (the safe direction, so t is tolerable) |
| Judge leans against the candidate and reads regressed | Regressions are never set aside | Keep the rule; print `judge_leans` in this direction too, or a decider is told only half of what was measured |
| Judge reads regressed, person preference separated for the candidate | Two asked readings disagree | Show both; neither cancels the other; state it |
| Labels arrive after the contrast was first read | A set-aside changes a verdict later | Count a closed read as a look in §5's disclosure |

**N-14. Row 2 pays people not to label.** "People not asked, self-consistent: counts" beside "people asked,
below the bar: set aside" means the first 20 labels can only cost a host an improvement. The design's own
complaint about today's rule applies in mirror image. The first-screen line helps. An owner question, not a
defect: should row 2 lapse after a stated number of campaigns?
**N-15.** Kappa falls with a skewed base rate even when the judge ranks arms correctly. At 20 to 40 labels the
override that could rescue such a judge is too wide to fire. Expect high-pass-rate dimensions to lose
improvements until about 60 labelled cases, and say so where the tier is explained.

**Simulation table (§4.3).** Bars are the right quantities except as amended in B-2. Missing: the restore
quantity; a labelled set with content-dependent skips; n = G; identical-output cases at 20% of G; the lean
test's false alarm with 5 dimensions.

## 5. Looks (§5)

**The α/L rule is valid.** A union bound needs no spending function and no information fractions, and is
indifferent to unequal batches, to growth by repeats (highly correlated looks, where it is conservative) and
to the test used. A look is correctly "the data changed", so accrual batches bound it from above.

**C-13 CHANGE (D1). Floors and `within_margin` must take α/L too.** The house reads floors uncorrected across
aspects because a warning should not weaken as floors are added. Looks are different: a decider who re-reads
until a floor says `held` inflates a claim of safety, the one error the house treats as worst. *Amendment:*
under `planned_looks = L`, `held` and `within_margin` read the bounded interval at 1 − α/L. A broken or
unclear floor still warns at every look.

**C-14 CHANGE (D1). Past L the guarantee is gone, and the page must say that.** "Batches beyond L are named
as unplanned" understates it. *Amendment:* wording "look 5 of 3 planned: the 5% no longer holds for this
reading", with a caveat id. Optionally offer the open-ended alternative: look j at α/(j(j+1)), which sums to
α over any number of looks and needs no L (α/2, α/6, α/12, ...; worse than α/3 from the third look).

**C-16 CHANGE (D1, Q11). Disclose-only is honest for stored campaigns and not for new ones.** The design's
own expectation is 11% and 14% where the page claims 5%, and the house standard is that a claimed rate is a
proven one. The objection to automatic correction is that it changes stored verdicts. *Amendment:* campaigns
created before the feature keep disclose-only. For campaigns created after it, an undeclared contrast with
more than one batch carries a report-wide caveat on every `improved` it shows, with the measured inflation
for its batch count; or `planned_looks` defaults to the open-ended sequence above. Either removes the
unqualified 5%.

**N-16.** The expected 11% and 14% match the standard figures for 3 and 5 equally spaced looks. The test
grows cases only. Add: growth by repeats on fixed cases; a pass-rate reading on the bounded interval;
unequal batches (20, +5, +35); a floor read `held` at any look when the true drop equals the margin (bar: at
most 2.5% declared, recorded undeclared); and looks run to 2L, recorded.
**N-17.** Re-judges, judge repeats and blind-read closes change verdicts without a launch. The disclosure
count should include them or say that it does not.

## 6. Addendum: a model as reference reviewer

The ruling is taken as settled. What follows is about the coordinator's reading of it.

**(a) Correlated errors, simulated.** People see no difference between arms. The reference reviewer passes a
failed candidate answer with probability s (a shared style lean); the judge copies the reviewer with 5%
noise; the reviewer read every case; people double-read m of them.

| s (lean in the contrast) | G | Judge–reference kappa | Reference–person kappa | Judge reads improved | Reference-corrected interval above 0 |
|---|---|---|---|---|---|
| 0.15 (about 0.08) | 40 | 0.90 | 0.91 | 10% | 12% |
| 0.15 | 100 | 0.90 | 0.92 | 20% | 23% |
| 0.30 (about 0.16) | 40 | 0.90 | 0.83 | 29% | 34% |
| 0.30 | 100 | 0.90 | 0.84 | 58% | 68% |

The judge is "calibrated", the anchor agrees with people at kappa 0.83 to 0.92, and a difference people do
not see is confirmed up to 68% of the time. Calibration against a reference shows the judge measures what the
reference measures. It bounds nothing about a lean the two share, and a reference-corrected contrast has
correct coverage for the reviewer's opinion and none for people's.

**B-5 BLOCKER (E1, B1). Guard 2 as designed cannot see the failure it exists for.** Kappa is an aggregate
agreement figure; in every row above the anchor's kappa against people fell below 0.6 in at most 1% of runs.
A differential lean is invisible to it. *Amendment:* guard 2 is the lean test of §4.3 run on reference against
person over the double-read pair items (reference difference minus person difference, with its interval),
reported beside kappa. An anchor stops anchoring a criterion when that interval excludes zero, or when its
kappa is shown below the bar.

**(b) How many double-read items.**

| Test | Truth | Double-read items needed | Source |
|---|---|---|---|
| Lean interval excludes zero | lean 0.16 | 30 pair items: 75%; 60: 99% | simulated |
| same | lean 0.08 | 30: 23%; 60: 74%; 100: 97% | simulated |
| Kappa shown below 0.6 | true kappa 0.3 | about 40 results | arithmetic, large-sample standard error |
| same | true kappa 0.45 | about 140 results | arithmetic |

The 2026-10-17 read has 15 pair items. Its lean interval will be about ±0.25 to ±0.3 wide: it can demote only
a grossly leaning or near-chance anchor. Below 40 double-read pair items the page should print the interval
and say "this anchor's lean toward either arm is not ruled out below ±h", never "agrees with people". Guard 2
should not read as passed below that count; "unmeasured" and "measured on m, undecided" are different states.

**(c) Person outranks reference: yes, a minimum is needed.** Read literally, three person labels replace a
reference-calibrated tier with a person tier of `insufficient`, so adding person evidence removes a standing.
*Amendment:* the person reading decides the tier once it is itself decided (met or not met at the existing
floor of 20 distinct results, counted with their cases per C-2). Below that the reference tier stands and the
person reading is shown beside it. A person lean outranks from the B-2 floor (40 labelled cases).

**(d) Guards that are missing.**

- **C-17 CHANGE (E1). Reference readings may set aside, never restore.** A reference-corrected contrast must
  not fire override 1 of §4.5. Its separation is the table above. It may fire override 2, the safe direction.
- **C-18 CHANGE (E1, E2). A reference preference is a pairwise model judge.** It must never fill
  `human_pref`, satisfy `PersonPrefers`, or count under Q4. The addendum does not say.
- **C-19 CHANGE (E1). Family overlap needs a rule, not a sentence.** When a candidate's model is in the
  reviewer's family, the simulated failure is the expected case, and it is the model-upgrade campaign the
  feature is for. *Amendment:* the improvement counts as the ruling says, with a report-wide caveat id and
  the verdict worded as model judgment, until the lean test of B-5 has at least 40 double-read pair items on
  that criterion. Whether it should count at all before then is the owner's call; the statistics give no
  support for it.
- **C-20 CHANGE (A2, E1). Same family, not only same model, between judge and reviewer.** Shared training
  inflates their kappa. Refuse the same served model as designed; disclose the same family on the tier line.
- **C-21 CHANGE (A2). Reviewer shopping.** Model reads are cheap, so a host can try reviewers and prompts
  until the judge reads calibrated. Record the count of reference reads per criterion and reviewer
  configuration, print it, and fix the reviewer before the judge's agreement with it is shown.
- **N-18.** A reviewer answers every item, so n = G and the reference-corrected interval is narrow. Narrow
  about the wrong target reads as certainty; print the anchor on the same line as the interval.
- **N-19.** Model reviewers carry strong position bias. Slot balance keeps the test valid (N-2); report the
  slot lean per reviewer. Blindness hides labels, not style, so family self-preference survives it.
- **N-20.** Person and reference ratings are never pooled into one kappa, and a reference rating never enters
  the cell-mean PPI under the name "people". The addendum implies both; make them tests.

**A1 slice of the addendum: safe to build now**, on three conditions. The rater kind is stored per row and
defaults to `person`. A reference reviewer's preference prints under its own name and never in the "People
read" line or with the person verdict wording. The same-model refusal compares the served model, and a
same-family match prints a notice. B-1 applies to a reference read as well, trivially, since it reads the
whole sheet.

## Open items this review did not settle

- C-2's kappa clustering was reasoned, not simulated.
- The judge-only columns above use t on per-case differences; the engine reads rates on the bounded interval,
  so its "judge reads improved" rates will be lower. The corrected-interval figures do not depend on that.
- §3.1's identity mechanics and §10's unverified items are engineering questions outside this review.

## Revision 3 follow-up

2026-11-14, against revision 3 of the design and `information-architecture.md`. No new simulation was run; the
figures cited are this review's earlier ones, or arithmetic marked as such.

### S1. Checks across runs with an unrecorded coordinate

**Rule: "an unrecorded coordinate matches nothing" applies, with no exception for checks.**

| Check | Either run has an unrecorded coordinate | Reports |
|---|---|---|
| A/A (`against: identical_arms`) | Result `insufficient`, reason `identity_unrecorded`, per reading | The difference and its interval one step down, labelled with the coordinate. It sets nothing aside and is never read as quiet |
| `corrected` on a contrast | State `not available`, reason `identity_unrecorded` | Nothing. The contrast it would check is already `withheld` |
| Judge or reviewer check (`against: people` or `model`): kappa | Ratings on those outputs still count | Unchanged. The check is about an instrument reading texts, and the judge's key is recorded |
| The same check: lean | Pair items from such a run are left out | The lean compares arms, so it needs the arms known; the count left out is shown |

*Reason:* the A/A's whole claim is that the two arms are identical. With a coordinate unrecorded that is
asserted, not known: a separation may be a real difference (two served models) and a quiet result is two
unknowns agreeing.

### S2. Ranking the first-screen reasons

**Rule: looks stay below both judge reasons at every look count. No look-count switch.**

| Rank | Reason | How it makes a decider wrong | Rests on |
|---|---|---|---|
| 1 | Evidence not whole (unfinished, withheld) | The answer shown is not the answer | By definition |
| 2 | A/A separated | An improvement was removed; the rig moved identical arms | By definition (it is a `set aside`) |
| 3 | Identity unrecorded (where it has not already withheld) | The arms may not be the things named; no bound | Argument, not measurement |
| 4 | Late criterion | The test was chosen after the data; up to 1 − 0.95^m over m candidate criteria | Arithmetic; not measured here |
| 5 | Judge unchecked | A lean of 0.15 read `improved` 17 to 99% of the time with no true difference; more cases make it worse | This review, §4.1 |
| 6 | Judge checked against a model only | 10 to 68% when the lean is shared; same-family arms rank above others within this row | This review, §6(a) |
| 7 | Repeated undeclared looks | 8 to 18% against a stated 5%; bounded near 20% | This review, B-1 table; the design's expected 11% and 14% |
| 8 | Few cases | It makes the answer `undecided`, rarely wrong; the interval already carries it | House simulations of interval coverage |

*Reason:* a judge's lean is a bias. It is unbounded, invisible to the interval and grows more convincing with
n. Looks are a known, bounded loosening, and the batch count overstates them. The honest caveat on rows 5 and
6: their rates are conditional on a lean existing, and nobody has measured how often one does. The ordering
rests on "unbounded and unmeasured outranks bounded and measured".

One refinement a builder can implement: a model-only check whose reviewer check is `met` under S3's rule, with
no family overlap, drops to rank 8's neighbour, below looks. It is then a measured instrument.

### S3. When a quiet check is `met`

**Rule: `met` is an equivalence reading. A quiet interval that is not inside a stated bound is `undecided`.**

| Check | `met` when | Otherwise |
|---|---|---|
| Reviewer's lean | At least 40 double-read pair items, the lean's interval lies inside ±δ, and the kappa criterion is `met` by its existing rule | `undecided`, carrying the half-width h |
| A/A, per reading | `paired_equivalence` on the null differences holds at the reading's declared margin, at the family's level | `undecided`. With no declared margin it is never `met` |

- **δ for a lean:** the reading's declared materiality margin where it has one, since a lean smaller than the
  margin cannot manufacture a material improvement; otherwise 0.10 of the scale's width.
- **What it costs (arithmetic, pass/fail, sd of the pair difference 0.32 to 0.5):** an unbiased reviewer lands
  inside ±0.10 with 80% chance at about 105 to 260 double-read pair items, and inside ±0.15 at about 50 to 115.
  At 40 items `met` is rare. That is the honest result at this scale.
- **A/A at 7 to 12 cases:** always `undecided` on a 10-point margin, as the house already says of floors.
- **No decision changes.** `undecided` already "stands" in §2.4 and has no effect in §4.2. What changes is the
  word, and two triggers: guard 3's family marking lifts at `met`, not at 40 items; and S2's refinement.

*Reason:* the house rule is that "cannot tell apart" never means "no difference", and `paired_equivalence`
already exists for exactly this claim.

### The decision table in four states

The three accepted rules are preserved. A separated A/A is `set aside` and nothing lifts it, a person
preference included. A `corrected` check `against: model` that is `met` has no effect, so a model's reading
never restores. A `corrected` `met` is defined only in the direction of an existing verdict and acts only on an
`improved` contrast, so it creates none; the 40-case floor and the Holm-step level are both carried. One cell
changed meaning and should change back: **"against a model, `met`; the model's own check `not_met`" falls to
the "none" rows, where a self-consistent judge is `counted`.** A judge that agrees at kappa 0.9 with a reviewer
shown to lean toward an arm most likely shares that lean, so demoting the reviewer then changes nothing.
Amendment: when the reviewer's `not_met` reason is `instrument_leans`, an `improved` contrast for the arm it
leans toward is `set aside` with `instrument_leans`; only `instrument_below_bar` falls to the "none" rows. Two
smaller points: the reason `people_disagree` is also raised for `against: model`, where the word is false, so
the renderer must read `against`; and the reviewer's lean demotes from 10 items, a floor this review did not
simulate (it errs in the safe direction).
