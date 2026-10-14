# Building an inventory

**For** anyone about to write their first eval. **Answers:** what to test before a line of code: how any app builds
its eval inventory in an afternoon. The running example is an email-triage assistant; swap in yours.

**You need:** your bug tracker or support inbox, your system prompt and tool descriptions, 50–100 real transcripts,
and one person who knows what "good" looks like.

**1. Mine incidents (45 min).** Collect every time the product did the wrong thing that a user or operator *saw*.
Look in closed bugs, support tickets, Slack threads with "why did it…", and post-mortems. Write each incident as one
line: *what it did · what it should have done · where you read it.* Throw out pure infrastructure failures that the
model could not have avoided; those belong in unit tests. Also read 30 transcripts yourself and note every failure
in your own words (open coding).

> Triage example: "Marked the CEO's 'quick question' as low priority · should be urgent · ticket #412."

**2. Grow a behaviour tree (45 min).** Group what you found by surface: replying, routing, tool use, safety,
latency/cost. Under each surface, write the product's **promises** as short leaves, from three places:
- the incidents;
- every rule in your prompts ("never…", "always…", "when X do Y");
- what users visibly rely on.

Mark each leaf **tested / partly / untested**, with the evidence beside it. The untested leaves are your backlog,
and the tree *is* your coverage map. The tree is your document: keep it in your repository beside the cases. The
package does not store a behaviour tree yet, so nothing here checks it against your cases.

> Example leaves: "Never auto-archive a message from a VIP sender." "Ask before replying on the user's behalf."

**3. Pick a grader per leaf (20 min).** If code can see it in the outcome (a field, a call, a final state), grade it
with code. Grade the **end state**, not the step order. Use a judge only for properties of prose. A judge checks
**one failure mode**, answers **pass, fail or can't tell**, replies in a fixed structure, and is told what evidence
outranks its own memory. The engine's judge does the last three for you: it asks one dimension per call, allows
"can't tell", and fixes the reply's shape and the evidence rule in its own instructions
([judges and calibration](judges-and-calibration.md)).

**4. Write cases, two per leaf at minimum (60 min).**
- **Write a case for each direction.** A "should" case and a "should not" case. A suite that only tests one direction
  rewards a candidate that always acts, or one that never does.
- **Write one regression case per incident.** It replays the incident's state and the user's words, and it is graded
  by the incident's fail condition.
- **Decide the reading per case, now.** Regression cases (things that must not recur) should be read as **pass^k**:
  every one of k attempts passes. Capability cases (things you are still improving) should be read as a pass rate.
  The engine reports both, but it does not yet let you declare which reading a case is held to; that is coming.
  Record the choice beside the case today, so that declaring it later is a copy, not a decision.
- **Add hard negatives for every judge.** One output that looks right but must fail, and one that looks wrong but must
  pass. They are your first calibration items. Hand a quick-path `Judge` its `examples=[Example(...)]`, or write
  them on a host with the `judge_case_author` action
  ([seeding hard negatives](judges-and-calibration.md#seeding-hard-negatives)).

> Hard negative: "Polite, accurate summary that silently drops the one action item addressed to the user" must fail
> `summary.keeps_asks`.

**5. Calibrate before you trust (ongoing, about 2 hours per judge).** Rate outputs per judge yourself, **without
looking at the judge's verdict**: the engine's floor for a `calibrated` tier is 20 person-rated results, and about 50
is a realistic plan for a good judge. Read κ with its lower bound, not raw agreement; the engine reports this as an
evidence tier. Until a judge is `calibrated`, its numbers are directional. An uncalibrated judge is not dropped from
pass^k or any other reading: the reading is flagged with the tier it stands on, and the report says so under Not
tested. Freeze the judge's wording first, because rewording a criterion orphans its labels: a label is keyed on the
criterion's wording, so a reworded dimension is a new criterion that no earlier label claims. Label guardrails first.

**6. Publish the gaps (10 min).** Every report carries a **Not tested** section. It lists what the campaign could not
speak to: levers declared and not swept, cases an arm has no usable trial of, goal checks not shown to beat doing
nothing, measures tested with no margin declared, guardrails left undecided, cost not measured, and each judge not yet
calibrated against people ([reading reports](reading-reports.md#screen-5-not-tested)). The behaviour tree's untested
leaves are not in that list, because the package does not hold the tree: publish them yourself, beside the report. A
reader should learn what the eval cannot see before they learn what it found.

**Then:** run one baseline of your current production configuration at k=3, with as many cases per behavior as you can
write (a contrast on fewer than 5 cases is reported `not tested`; 20 or more gives a usable interval), and read the
bars it proposes (`bars_propose`, or `propose_bars`). A baseline proposes a bar on each measure it measured, seeded
from where the incumbent performs. It never adopts one: you write the bars you accept into your host's registrations,
and a looser bar than the one registered is refused. From then on, a worse release is a red line, not a debate. Every
new incident becomes a regression case before its fix merges.
