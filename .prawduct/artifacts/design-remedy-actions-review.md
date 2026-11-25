# Review: design-remedy-actions.md

Reviewed 2026-11-15 against `evals/bp-s5a2` @1fdc83582 (the code the design cites). Read-only. Code paths are under
`packages/evals/src/threetears/evals/` unless they start with `tests/`. Three hats: **S** statistician, **A** API
designer, **H** host operator.

## Verdict: BUILD WITH THESE CHANGES

The core move is right: one engine procedure keyed by the caveat, one planner, one spend action and its estimate,
pairing enforced by a minted case set. It subtracts the right public surface. But five things a builder would get
wrong as written are statistical or contractual, not polish. Two would put a wrong number in front of a reader:
the headline A/B's plan, and the extend's stated lift rate. One makes R0's done-when impossible: the digest moves.
One makes the spend lift nothing on existing campaigns: "as launched" is not "would pool". Fix them in the design
before R0 starts. None of them needs a redesign.

## Findings, by severity

### Blocking (fix in the design before a builder starts)

**F1 [S] The headline A/B's held cases leave the contrast once it is extended, and the plan counted them.**
`contrast_cases` (analysis/stats.py:1597) pairs over the shared cases when there are at least two, else it reads
each side unpaired. Today's one-model, two-prompt A/B is unpaired when its arms came from separate launches (Critic
V3-1: their generated cases are disjoint). The extend gives both arms m ≥ 2 shared new cases. From then on the
contrast reads **only those m pairs**, and every held case is silently dropped. But `_settling(fresh=False)`
(caveats.py:1121) sized m as `needed − held`. The reading after the extend rests on fewer cases than the plan
assumed, and the reader sees n fall. The same holds for a partly paired contrast: its unshared cases were never
read. This is §8's own example.
*Smallest change:* plan on the design that will be read after the extend. Where the held contrast is unpaired or
partly paired, the first move is a **cross-fill**: run each arm on the held cases only the other arm ran. That
needs no generation, and it makes every held case a pair. Size any remainder from there. If cross-fill is refused
(a replayed arm, drift), plan `fresh=True` and say, typed, that the held cases leave the contrast. Add a driven test
for an unpaired held A/B.

**F2 [S] The extend's stated lift rate is not the rate it lifts at, and R0 adds a second projection.**
(a) The held cases are fixed, and the truth is taken as today's estimate. Adding m = N − n₀ cases then clears with
probability about Φ(z_β·√(N/m)), not POWER. That is 0.88 at m = N/2 and 0.95 at m = N/4. The "needed − held" rule
over-asks, and the band [0.78, 0.84] in R0 and R1-2 cannot hold for an honestly simulated extend.
(b) §6's two-point projection is a second sizing rule. `bounded_cases_to_clear` (analysis/stats.py:2975), used by
`project_rate` (analysis/floors.py:1362), already re-reads the real bounded interval at each N from the observed
per-case values, paired or unpaired. Its docstring says 1/√n scaling "overstates the cases it needs, by up to about
twice". The 2.7σ/√n premise in §6 conflicts with that. A two-point sample also misstates the shape the betting
interval reacts to: paired pass/fail differences take three values.
*Smallest change:* one projection. Extend the repeated-sample projection to hold the held cases fixed, draw the m
new ones like them, and require the low end to clear with probability POWER (a conditional-power search). Write
down exactly what the seeded simulation fixes: truth = today's estimate, held sample fixed, new cases drawn. Hold
that in [0.78, 0.84]. Record the unconditional rate (a fresh initial sample each replicate) as a second, reported
row. Include n₀ of 3 and 5, where the spread estimate is noise. Use it for gate figures (`_figure_plan`) too.

**F3 [S] The looks numbers assume fixed increments. The planner chooses the increment from the data.**
"About 8% at two looks" (§5) comes from equal, pre-fixed batches (design-judges.md §6.3). Here the increment is
sized from look 1: small when look 1 nearly cleared, huge when it was near zero. That is the direction in which
re-sizing on the interim result inflates a naive pooled test (Proschan–Hunsberger). The same objection reaches D1's
declared case. α/L by union bound needs each look's test to hold α/L on its own, and a pooled test at a
data-chosen N does not. So "on a planned aspect within plan, it raises nothing" is unproven for remedy extends.
*Smallest change:* add rows to D1's simulation (`tests/test_simulated_looks.py`). Run planner-sized extends ×1, ×2
and ×3, undeclared and declared at L = 2 and 3, on a pass rate under the bounded interval, with no true difference.
The estimate's `looks` block and the caveat then quote the measured rate for *this* procedure. If the declared rows
exceed α, say so and do not ship "raises nothing". Note for D1 the principled exit. Each stage's betting wealth,
with stakes fixed before that stage (stage 1 at its launched size, stage 2 at the planned m), multiplies into one
test supermartingale. By Ville that holds α under any planner-chosen continuation and any number of extends, with
no bet order across stages to fix in advance. Its power cost needs the same simulation.

**F4 [A/S] `Reading.pair_sd` moves the digest.** `report_input_document_digest` (report_input.py:1454) drops only
**top-level** keys in `_OUTSIDE_THE_DIGEST`. A field on the nested `Reading`, even one defaulting to None, changes
every canonical dump. Then every `report_input_digest`, snapshot id and held-out golden moves, against R0's own
done-when. *Smallest change:* put the derived fact in a new top-level part (for example `spreads`, keyed by reading
id), listed in `_OUTSIDE_THE_DIGEST` as `caveat_list` and `first_screen` are. Under F2 that part holds what the
one projection needs. That may be the per-case paired differences or the projected size, not one sd.

**F5 [H/S] "As launched" uses resume's drift check, but an extend's new runs must pool, which is a stricter
contract.** `_resume_drift` (run/resume.py:329) does not count unrecorded measure definitions as drift, because a
take-up continues the same run. Its own comment says such a run "pools with nothing … until its host attests or
re-measures it". New runs from an extend are separate runs. They record definitions and the old ones did not, so on
every pre-definitions campaign they land in another cell and add nothing to the reading. Since K, readers also
refuse to pool across a served-model coordinate. *Smallest change:* in the estimate, check "would pool". Run the
reader's own identity and coordinate comparison between the run a launch would make now and the arm's cell, and
refuse `would_not_pool`, naming the coordinate. After the runs end, check the served model and record an outcome
`split` instead of `completed`. Also stop writing a new settings table (§2.6). `admit_resumed_run`
(run/launch.py:2278–2330) already maps a stored run to its launch. It also carries `declared_margins`,
`max_metered_calls` and `cassette_corpus_id`, which §2.6 omits. Factor that mapping into one "launch of a stored
run", used by resume and remedy alike, with the case set as the only argument that differs. Two mappings would
drift (criterion 3).

### Major

**F6 [H] A spend can exceed its estimate, and nothing caps the total.** `max_cost_usd` is per run. The spend
re-plans, and its plan may be larger than the one estimated if evidence arrived in between. A writer that returns
too few cases is asked again, and those calls are extra. A per-run cap that stops an arm part-way leaves its frozen
ids equal to the set's but its results short, so §4's post-condition (on `test_case_ids`) passes on an unpaired
outcome. *Change:* make `max_total_usd` required on the spend. It is the person's consent, and the step lists it in
`needs`. Refuse when the spend's own quote exceeds it, and enforce it as a running total across runs and generation.
Drop the per-run `max_cost_usd` parameter. A run that is `budget_stopped` fails the job and joins nothing. Assert
pairing on completed results, not on frozen ids.

**F7 [A] `closes` is not stable across re-reads.** `_fact_id` (caveats.py:815) writes `<state>:<reading>` alone,
then adds `:<reason>`, then `:<coordinate>`, depending on which sibling caveats exist. The arms grouped under one id
can also change. A sibling raised between the estimate and the spend turns the call into `caveat_not_raised` for a
caveat that is still there. *Change:* resolve `closes` by the caveat's identity (state, reading, reason,
coordinate), accept any id form that names it, and return the canonical id in `RemedyEstimate`. Give `Gate` an id
of the same kind instead of inventing a `gate:<rule>` syntax (caveats.py:1715).

**F8 [S/A] An extend joins as runs are created, so a crash or a mid-flight read leaves a partial look.** §2.3
Extend step 5 joins each run at creation. A `report_read` during the job reads a partial extension, and a person
who acts on it has looked, while D1 counts nothing. A remedy that dies leaves partial runs in the campaign. A new
call is then refused only while that remedy is live, so it re-plans on the partial data. *Change:* join once, at the
end, as the re-run already does (one write, idempotent by run id). A remedy that is failed or lost blocks a new one
on that aspect until it is resumed or cancelled (`remedy_unfinished`, naming `job_resume remedy:<id>`).

**F9 [S] The looks caveat would not reach the first screen on the campaigns that exist today.** D1 (design-judges
§6.2) raises unplanned looks on a campaign created before D1 at **method** reach. Every production campaign is one.
After the engine itself adds a look, its first screen would show a verdict with no looks caveat. *Change:* treat a
look added by an `EvalRemedy` as post-feature by construction, and raise it at report reach. Also state it on the
pure step, not only in the estimate (a typed `raises: [looks]` on the step). Where the input already holds D1's
`{planned, taken}` and the looks are exhausted, the step must name the confirm, not an extend that will refuse.
That is "a next step is a promise" for a fact the report holds.

**F10 [S] `looks → confirm` has no plan.** `plan_of` (caveats.py:1287) handles no `looks` kind and returns
`_NOTHING_TO_SETTLE`, so the spend refuses `unsized` and §2.4's row leads nowhere. *Change:* size it as a fresh plan
of the aspect's readings, the way `best_of_n` is sized but without the bias flag. Drive it in R3.

**F11 [S] "Unseen" is read from current membership.** §3 says a case is unseen when "no run of the campaign … ran
it". That uses today's members. Runs the re-run's swap removed (and runs `campaign_runs_remove` removed by hand)
were seen, and so was the verdict built on them. L3 fixes this for looks but not for cases. For confirm, a pick may
have been made on runs of the same variants in another campaign. *Change:* define unseen as never run by any run
that was ever a member of the campaign. For confirm, also exclude any case a run of a compared arm's variant ran on
this template.

**F12 [A] One group cannot always hold the arms.** `LaunchGroup.add` (run/launch.py:1007) refuses mixed
`measure_latency`, and a group runs one `arm_schedule`. §2.6 also misreads this. A group whose members all measure
latency is allowed and runs them sequentially (`resolve_arm_schedule`, run/arm_schedule.py:48), so a serial re-run
needs no successive groups. Keep L2 anyway. *Change:* the estimate refuses `arms_not_one_condition` when the arms
differ in `measure_latency`, and states the schedule the group imposes. The schedule is a disclosure, not identity,
so overriding it is honest only if it is said. For the `latency` and `measured_under_concurrency` re-runs, re-run
the control in the same sequential group too. Latency compared against a control measured at another hour reads
provider load. The owner's 2026-10-23 ruling ("hours apart are one experiment") is about results, not latency.

**F13 [H] The production host's launch path is unverified (§12.1), and everything depends on it.** A run whose
host assembled it outside a launch has `measure_latency` None and no `launch_group_id`. If the production host
assembles runs, or generates cases inside its kind's launcher (as the fixture does at
tests/test_caveat_steps_driven.py:158), then every remedy refuses, or `exclude_params` never applies. *Change:*
before R1, read the production host's launch path and record the answer in the design. Add an
`arm_not_launched_by_engine` refusal. If the host generates, say what it must bind, or R1 is not "nothing to
implement".

**F14 [H/process] The chunk names collide, and the cut-over is not single.** The judges track already has an R1
(design-judges.md §9.2). `EvalRemedy` in `kernel/campaign.py` is D1's file. Between R1 and R3, steps name
`campaign_remedy` while the seven fields still ship, so a host that re-pins for any other fix in that window gets a
mixed contract. *Change:* rename the chunks (M0–M3, say), and put `EvalRemedy` in `run/remedy_job.py` beside
`EvalLaunch`. Build M1–M3 on one integration branch (`evals/remedy`) and merge it to `evals/main` once, with R3's
text map. Split R1 (~1,700 lines): M1a is the record, the estimate, the refusals and the stored-run launch mapping,
and spends nothing; M1b is the extend spend and the step cut. M1a's done-when is "the estimate's plan equals what
M1b spends".

### Minor

- **F15 [S]** Best-of-n. Keep `size_bias`, but measure it. M3 adds a seeded selection simulation: N options, pick
  the best, confirm at the planned size, record the realized lift rate. The estimate states that number, not only a
  flag. A confirm of several survivors reads them under the new campaign's family correction, so size them at the
  level that campaign reads, not at `INTERVAL_LEVEL`.
- **F16 [S]** A contrast near zero with no margin. `_ON_THE_LINE` (1e-9, relative) leaves plans in the millions.
  By the question-margin ruling there is no invented cap. State the size as computed and let F6's required budget
  stop it. Add a test that the estimate prices and refuses it, and that the spend never starts it. When the spread
  is zero, the repeated-sample projection (F2) returns the observed count or the cap: no division by zero.
- **F17 [S]** Order of remedies. Extending an arm under a raised `apparatus:`, `confound:` or `split` caveat adds
  more confounded evidence: the reading settles and stays confounded. The estimate names `first: <caveat id>`
  when a caveat on the same arms, of a higher class, would leave the extension unreadable.
- **F18 [S]** `point_estimate` sized to the floor promises only that an interval exists, not that it clears.
  §1's "and clears" overstates. Say that `borderline` may follow.
- **F19 [A]** The record has one `aspect`, but a late or best-of-n confirm spans every asked aspect (§2.3 Confirm
  step 6). Make it `aspects`.
- **F20 [A]** The subtraction is overstated. `_priced_and_planned` has 14 checks, not 15, and about six go, not
  nine. A span count of what §7 lists (caveats.py:282–350, 1329–1533, 1597–1641, the arms half of `evidence_of`,
  the StepRule fields) gives about 400 lines, not 520. The chunks add about 3,700 gross, so the design is net
  +3,000 lines. The public surface does shrink: the seven fields (verified at report_input.py:1187–1283) and
  `lift_not_shown`. Say both.
- **F21 [A]** Naming fits the house pattern: `<thing>_estimate` beside the spend, as `launch_estimate` /
  `run_launch`. No change.

## The owner questions

**Q1, extending a campaign that planned no looks. I agree with the coordinator: allow it, state the looks caveat
truthfully, and do not default to two planned looks.** A default of two is an invented constant that taxes every
campaign (about 21% more cases: (2.24+0.84)²/(1.96+0.84)²) for an option most never use. The question-margin
ruling forbids exactly that. Refusing would turn the commonest step into a confirm at about twice the cost, for a
risk that can be stated. Allowing is truthful only if:
(i) the stated rate is the one measured for planner-sized extends (F3), not 8%;
(ii) the caveat reaches the first screen on existing campaigns (F9);
(iii) the step itself types `raises: [looks]` (F9).
Ask D1 to cost the stage-wise betting product (F3). If its power cost is modest, it removes the looks caveat for
remedy extends, and Q1 goes away by principle.

**Q2, a smaller extend within a budget. I agree: no.** It is a second planner with a second promise shape. It
would also spend a look on a design that probably will not lift, which invites the next extend and the next look.
That is the worst of both. The price shown before spending, plus F6's required consent, is the brake. A person who
will not pay takes no step, or designs a new campaign.

## Changes to make to the design before a builder starts

1. §6/R0: replace the two-point projection with one conditional-power projection built on
   `bounded_cases_to_clear`, with the held cases fixed. Define the simulation (truth = estimate, held fixed; the
   unconditional rate recorded; n₀ of 3, 5, 10, 30, 100; pass rate and bounded mean). Apply it to gate figures too
   (F2).
2. §4/§6: plan on the design read after the remedy. For an unpaired or partly paired contrast, cross-fill the held
   cases first, else plan fresh and state that the held cases leave the contrast. Add a driven test for an unpaired
   held A/B (F1).
3. §6/R0: hold the derived spread in a new top-level part listed in `_OUTSIDE_THE_DIGEST`, not on `Reading`.
   R0's done-when stays "no digest moves" (F4).
4. §2.6: drop the settings table. Factor `admit_resumed_run`'s stored-run-to-launch mapping into one function used
   by resume and remedy. Add the "would pool" check (reader identity and coordinate), refusing `would_not_pool`,
   and a `split` outcome after the runs end (F5).
5. §2.2/§2.5: required `max_total_usd` on the spend, enforced as a running total across runs and generation;
   per-run `max_cost_usd` removed; `budget_stopped` fails the job; pairing asserted on completed results (F6).
6. §2.2: resolve `closes` by the caveat's identity tuple, return the canonical id, give `Gate` an id (F7).
7. §2.3/§2.5: the extend joins once at the end. A failed or lost remedy blocks the next one
   (`remedy_unfinished`) (F8).
8. §5: the D1 interface gains L5 (a remedy-added look is raised at report reach on any campaign) and L6 (planner-
   sized extend rows in D1's simulation, declared and undeclared). The pure step types `raises: [looks]` and names
   the confirm when the input shows looks exhausted (F3, F9).
9. §2.4: give `looks → confirm` a fresh plan, and drive it (F10).
10. §3: unseen = never run by any run that was ever a member of the campaign; for confirm, also no run of a
    compared variant on the template (F11).
11. §2.6: refuse `arms_not_one_condition` on mixed latency; state the imposed schedule; re-run the control in the
    latency re-runs; remove "successive groups" (F12).
12. §12.1 resolved before M1: read the production host's launch path; add `arm_not_launched_by_engine`; say what
    the host binds, if anything (F13).
13. §9: rename the chunks M0–M3; `EvalRemedy` in `run/remedy_job.py`; M1 split into M1a (estimate, no spend) and
    M1b; M1–M3 on one branch, one merge to `evals/main` with the text map (F14).
14. Done-when additions:
    - M1: an unpaired held A/B; an old-campaign arm that refuses `would_not_pool`; a budget-stopped arm that joins
      nothing; a spend refused above `max_total_usd`; the estimate's price ≥ the fixture's spend.
    - M3: the selection simulation (F15); the `looks` confirm (F10).
    - D1: the adaptive rows (F3).
15. Correct §7's counts (F20), §1's point-estimate row (F18) and the record's `aspects` (F19). Add the ordering
    refusal (F17) and the near-zero test (F16).

## Can R0 start now?

Yes, once changes 1–3 are in the design. Branch it from `evals/bp-s5a2` @1fdc83582: S5a-2 is verified ready, and
`_reading_plan`/`_settling` exist only there. Rebase after S5a-2 merges. The files:
- `analysis/separation.py`: the conditional-power search, calling `stats.bounded_cases_to_clear` and leaving it
  unedited;
- `analysis/report_input.py`: the new top-level part and `_OUTSIDE_THE_DIGEST`;
- `analysis/report_input_build.py`: filling it;
- `analysis/caveats.py`: `_settling`, `_reading_plan`, `_figure_plan` and the unpaired branch of `plan_of` only;
- tests: `tests/test_caveat_list.py`'s simulation, and a new simulation file.

Overlaps: D1 owns none of these files, provided R0 takes the reading's own level from the reading rather than
`INTERVAL_LEVEL`, so D1's α/L flows through untouched. S5a-3 edits other classes of `report_input.py`, so expect a
textual merge only.
