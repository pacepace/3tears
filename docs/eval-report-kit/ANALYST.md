# The analyst: foundation for campaign reports

**Status:** agreed with the owner, 2026-10-26. It supersedes the shape table as a rule (GENERATOR §1), the mandatory spine check, and two review items from the deviations page: "mode from code" and "the two-call planner". Everything else in GENERATOR remains as guidance the analyst reads.

**Terminology** (owner, 2026-10-26): the role this document calls the *critic* is named the **report editor** (`report_editor`) in code and in every plan. In this repository "Critic" means only the prawduct review of code changes.

## 1. The decision

A campaign report is written by an **expert analyst**: an Opus agent with tools, a skill and a feedback loop. It reads the whole campaign, works out what it means, and writes the report as one story.

- When the evidence supports a call, the story ends in a decision or a priced tradeoff.
- When it doesn't, the report explains what is tangled and what would untangle it.
- What doesn't matter is left out of the body; the appendix still has everything.

Code is the analyst's toolbox and its editor's checklist. **Code computes and verifies. It never chooses structure, emphasis or framing.**

### The principle that prevents hacks

**No code path branches on the eval's kind, shape or design.** Code knows about data, statistics, charts and rendering, never about eval types. When an eval arrives that nobody anticipated, the analyst reasons about it with the same tools. If it can't be shown well, add a tool or a block kind that is useful everywhere. Never add a branch. A table and prose are always available, so no input is ever unrenderable.

A review question for every PR: *does this add a condition on what kind of eval this is?* If it does, it belongs in the skill's guidance, not in code.

## 2. Roles and models

| Role | Model | Job |
|---|---|---|
| Analyst | Opus | Investigate, outline, write, render, revise |
| Critic | Opus, separate context | Review the rendered report against the rubric (§7) before it's stored; send it back with findings or pass it |
| Scorers | Cheaper models | Score stored reports on the rubric at scale for the calibration dataset; never gate a report |

The analyst and critic never share context. The critic sees what a reader would see: the rendered page, its values tables, and the report input. It doesn't see the analyst's reasoning.

## 3. The analyst skill

A skill made of modular prompt files, loaded together. They carry expertise, not rules to match. The skill ships in the evals package as data, at `packages/evals/src/threetears/evals/analysis/analyst/skill/`, and `threetears.evals.analysis.analyst.skill` assembles each role's system text from it.

```
packages/evals/src/threetears/evals/analysis/analyst/skill/
  SKILL.md            role, the workflow (§4), the output format (§5), tool overview
  principles.md       what a report is for, honest emphasis, elision, confounds, confidence, next steps
  stories.md          story forms: decision, tradeoff, diagnosis, status, exploratory, and mixing them
  charts.md           the block catalog: each kind, what question it answers, when it misleads
  statistics.md       reading intervals, margins, corrected vs. uncorrected, small strata, zero counts
  writing.md          headlines as findings, reader names, one caveat in one place, short sentences
  exemplars/
    INDEX.md          one line per exemplar: the eval, the story, the trap, and its structure
    kairo-round2.md   reasoning note for Kairo round 2 (Appendix A points at it)
    r1 … r4.md        reasoning notes for the four synthetic references
  editor.md           the report editor's system text: the rubric (§7) and the review it returns
  README.md           for the skill's maintainers: what each role reads, adding a note
```

An analyst is given every exemplar and the whole index, except the notes a run withholds (a blind run withholds the one reasoned on its own campaign). Nothing about a campaign chooses which notes it reads. Each index row's Structure column records the campaign its note came from, for readers of the notes.

GENERATOR's content moves into these files, rewritten as judgment ("lead with the change's size when the interval excludes zero, because…") rather than lookups ("if shape D then forest").

**The exemplars are the most important part.** A finished memo shows *what* a good report looks like. A reasoning note shows *how it was reached*: what the analyst noticed, what it chose to lead with, what it left out, and the trap it avoided. Judgment transfers through those notes. Add one whenever a report teaches something new, especially for evals that surprised everyone.

### Story forms (guidance, not an enum)

- **Decision.** Here is the call, and why. It ends on a recommendation.
- **Tradeoff.** Here is what you'd gain and lose, priced; the owner chooses.
- **Diagnosis.** The result can't be read cleanly because X is tangled with Y. This is what is still known, and this is the experiment that untangles it.
- **Status.** A single arm against its thresholds, or against its own history.
- **Exploratory.** No question was asked. Here is what we learned, and the question worth asking next.

Reports can mix forms: for example, a decision on cost with a diagnosis on the judged aspect. The headline carries the dominant form.

## 4. Workflow

1. **Understand.** What the eval tried to learn, who decides, what was declared before the run and what was decided after it. Read prior insights.
2. **Investigate.** Use the tools until the story is clear, or until it's clear there isn't a clean one. Typical probes:
   - Is the call sensitive to the margin?
   - Where is the loss concentrated?
   - What do the failures look like?
   - Is a confound large enough to matter?
   - What would settle an open question?
3. **Outline.** The headline, then the claims in reading order, each with the evidence it needs. Decide what to omit and why.
4. **Draft.** Write the report document (§5) with figure keys, never typed numbers.
5. **Render and look.** Render the page. Read the check results and the screenshots at desktop and phone width. Fix what's unclear, crowded or misleading. Repeat until clean, at most three rounds. Not in MVP (owner, 2026-10-26): no browser at storage time; check.py runs in integration tests.
6. **Critic.** Submit the report. If the critic returns findings, address them and resubmit, at most two rounds. Unresolved findings are stored with the report and shown in Method.

## 5. The report document

What the analyst writes, and what is stored. It replaces the slot-based plan.

```json
{
  "schema": "report-doc/1",
  "story": "decision | tradeoff | diagnosis | status | exploratory",
  "headline": "{figure-keyed prose}",
  "lede": "…",
  "confidence": {"form": "high | high_if_immaterial | low", "caveats": [{"id": "…", "material_if": "…"}]},
  "decision": [{"slot": "recommendation | why_not | watch | next | the_trade | where_it_flips | our_lean | what_settles_it | what_is_tangled | known_despite_it", "text": "…"}],
  "claims": [
    {"id": "c1", "text": "…", "weight": "lead | support | detail",
     "evidence": {"block": "<block kind>", "query": {"…": "…"}, "caption": "…"}}
  ],
  "explore": [{"control": "…", "binds": "…", "outcome": "…"}],
  "omitted": [{"what": "…", "why": "…"}],
  "next": [{"text": "…", "settles": "…"}],
  "method": [{"term": "…", "text": "…"}]
}
```

- **`claims` is the story.** Its order is the page order. The first claim is whatever best carries the headline, chosen by the analyst and judged by the critic. Code never checks what kind of block it is.
- **`decision` slots** come from a shared vocabulary. The analyst picks the four that fit the story. A diagnosis uses *What is tangled*, *Known despite it*, *What settles it* and *Next*.
- **`omitted` makes elision accountable.** Anything substantial the analyst saw and chose not to show gets a one-line reason. The critic checks those reasons. The appendix still holds the omitted data.
- **`evidence.query`** names data through the tools' query language (§6). The renderer resolves it, so a block can never show data that doesn't exist.

## 6. Tools (MCP)

One domain tool, `evals_analyst`, with one action per capability (owner, 2026-10-26: minimize tools, actions inside a domain). The analyst and the in-process loop use the same tool as MCP clients; the editor's cut leaves out the block actions. All actions are read-only except `report_snapshot_freeze`. Every figure an action returns carries a figure key, so the analyst can cite it. `help` lists the actions and the figure-key topics.

| Group | Action | Returns |
|---|---|---|
| Snapshot | `report_snapshot_freeze` | The frozen evidence every other action reads, by id |
| Facts | `campaign_overview` | Question, arms (full and short names, settings), aspects, declarations and when they were made, judges, counts |
| | `readings_list`, `contrasts_list` | The pipeline's readings and contrasts, with their levels and families |
| | `measure_read` | Per-arm or per-slice figures with intervals |
| | `examples_list` | Trials with judge reasons and excerpts |
| | `caveats_list` | Every open caveat, stated or implied by what the snapshot holds, with what it touches |
| | `figure_keys_check` | What each key a draft cites reads, as the page prints it, or why it reads nothing (analyst and editor) |
| Analyses | `decision_sensitivity` | Where the call or a reading flips (margin, floor, threshold) |
| | `measure_breakdown` | Where a difference is concentrated, labeled exploratory |
| | `options_dominance` | Options surviving gates and guardrails, and which dominate on which axes |
| | `price_tradeoff` | Price per point and the steps along the cost frontier. Modeled hybrids are out (owner, 2026-10-26: backlog #657) |
| | `arms_compare` | Any two arms, control or not: the difference and its interval, paired where a contrast holds one |
| | `settle_cases` | The cases that would settle an arm against its line, or two arms apart, and how often today's design reads the wrong way |
| | `levers_effects` | What one setting buys between arms that differ in it alone, and whether it reverses across another setting |
| | `confound_check` | Deferred until a prototype run needs it |
| Rendering | `block_render` | Values table and warnings for a block; no pixels |
| | `render_report(doc)` | Not in MVP (owner, 2026-10-26): no browser at storage time; check.py runs in integration tests |
| Reference | `block_catalog` | The block kinds, their sources and usage notes |
| Gaps | `capability_gap` | Acknowledges a missing capability the analyst worked around (§8 of the program plan) |

Add an action when an analyst run shows it reasoning around a missing capability. Never add one to handle a specific eval type.

## 7. Invariants and the critic

### Invariants (code; refuse to store on failure)

This list is fixed. It doesn't grow with new eval types.

1. Every number in prose resolves through a figure key to the data.
2. Every block's query resolves, and the block renders.
3. Every caveat that affects confidence is shown, either with its `material_if` clause or as checked immaterial.
4. The confidence form matches: no plain `high` while caveats are open.
5. No internal id, hash or measure key appears above the appendix.
6. The full-results appendix and Method are present.
7. `check.py` passes at 1200 and 390 px, in light and dark. Not in MVP (owner, 2026-10-26): no browser at storage time; check.py runs in integration tests.

### Critic rubric (Opus; each item scored 0–2 with a reason)

1. **First screen.** From the headline, decision block and first claim alone, a reader can say what to do (or what's unresolved), why, and the main risk.
2. **Coherent story.** The claims follow from one another, and nothing important contradicts the headline unaddressed.
3. **Honest emphasis.** The recommended option's weak axes are visible, and nothing is framed stronger than its interval supports.
4. **Sound elision.** Nothing omitted would change a reader's decision.
5. **Confounds handled.** Each one is either shown not to matter or carried into the story and the next steps.
6. **Actionable next steps.** Each names an experiment or a declaration and says what it would settle.
7. **Evidence fits claims.** Each block is the clearest way to show its claim, and the counterfactual holds: if the values flipped, the claim would have to change.

The critic passes a report when no item scores 0 and the total is at least 11 of 14. These thresholds are starting points; the calibration dataset sets them.

## 8. Migration from what's built

| Keep | Change | Retire |
|---|---|---|
| Kit, renderer, block model and chart kinds | Plan → report document (§5) | The shape table as a lookup (it becomes exemplars) |
| Figure keys and their resolver | Single planner call → analyst agent loop (§4) | The mandatory spine check |
| Stored reports, calibration dataset, rubric scoring | Code-computed signals, shape and mode → tool outputs the analyst reads | The "no shape" fallback (nothing needs a shape now) |
| The invariants, including conditional confidence | Planner prompt → the analyst skill (§3) | Strict-output slot counting and view trimming (tools query on demand) |
| `check.py` and the render checks | GENERATOR → guidance in `principles.md`, `charts.md`, `statistics.md` and `writing.md` | Any code that branches on eval kind |

Order of work:

1. **Tools.** Wrap the existing computations as MCP tools. Most already exist as code: readings, contrasts, flip margins, dominance.
2. **Skill.** Move GENERATOR's content into the skill files. Write the five reasoning notes; Appendix A is the first.
3. **Analyst loop.** Run it on the five references and on the misfit inputs from the spines doc. Compare the results with the references by critic score, not by structure.
4. **Critic and invariants.** Wire both in as the storage gate.
5. **Calibration.** Score every stored report. Review the owner-labeled anchors. Tune the rubric thresholds.

**Acceptance.**
- Kairo round 2 reaches the reference's decision, confidence form and caveats, and scores at least 12 of 14 with the critic.
- Every misfit input from the spines doc produces a stored, rendered report that the critic passes.
- No PR in the work adds a branch on eval kind.

## 9. Open items

- **Cost and latency.** An Opus analyst plus critic costs more per report than a single planner call. Measure it on the references in step 3 before deciding whether long reports need a budget.
- **Determinism.** Two runs can write different but equally good reports. Stored reports are what readers see; regenerating one is a deliberate act and keeps the previous version.
- **When there's no good story.** The analyst may conclude the campaign can't support any reading. That is a valid report: a diagnosis or exploratory story whose next steps name what to declare or run. It is not a failure.

---

## Appendix A. Reasoning note: Kairo round 2

The first reasoning note, and the form every exemplar takes, is shipped with the skill: [`exemplars/kairo-round2.md`](../../packages/evals/src/threetears/evals/analysis/analyst/skill/exemplars/kairo-round2.md). It is kept there only, so the note the analyst reads is the one this document points at.
