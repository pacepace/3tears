# The analyst: how a campaign report is written

Every campaign report is written this way. The planned report in [GENERATOR.md](GENERATOR.md), with its shape table, spine and code-computed mode, survives only as what the analysis reporter's plan cases judge. GENERATOR's guidance on gates, charts, words and visuals lives on in the skill (§3).

**Terminology:** the role that reviews a report before it is stored is the **report editor**. In this repository "Critic" means only the review of code changes.

## 1. The decision

A campaign report is written by an **expert analyst**: an agent on the host's analyst model (Opus in every run so far) with tools, a skill and a feedback loop. It reads the whole campaign, works out what it means, and writes the report as one story.

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
| Analyst | The host's `analyst` model: a strong reasoning model that takes tool turns (Opus) | Investigate, outline, write, render, revise |
| Editor | The same model, in a separate context | Review the rendered report against the rubric (§7) before it's stored; send it back with findings or pass it |
| Scorers | Cheaper models | Score stored reports on the rubric at scale for the calibration dataset; never gate a report |

One model role serves both the analyst and the editor, so a host configures one model. What keeps them apart is context, not model: the analyst and the editor never share it. The editor sees what a reader would see: the rendered page, its values tables, the report input, and facts code computed for it to weigh (a recommended arm failing a declared gate, the call disagreeing with dominance, an unanswered warning, and the like), none of which ever refuses a report. It doesn't see the analyst's reasoning.

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
5. **Render and check.** Draw each claim's block with `block_render` and read its values table. Then send the whole draft to `report_check`: it returns every mechanical verdict the store would refuse it for, then the page as a reader and the editor will read it, as text. Fix every verdict, reread the page as a reader and fix what's unclear, crowded or misleading, and check again. No browser runs at storage time; `check.py` holds the rendered pages in the integration tests.
6. **Submit.** A submission that fails a mechanical check comes straight back with each fault and its remedy (a bounce), and costs no review. One that passes goes to the editor. If the editor returns findings, address them and resubmit, at most two revisions. Unresolved findings are stored with the report and shown in Method.

## 5. The report document

What the analyst writes, and what is stored: `report-doc/1`. It replaces the slot-based plan. The model is `threetears.evals.kernel.report_doc.ReportDoc`, and [`schema/report-doc.schema.json`](schema/report-doc.schema.json) is its JSON schema, generated from it and held to it by a test. Every prose field names figures by key, never by typed number.

```json
{
  "schema": "report-doc/1",
  "story": "decision | tradeoff | diagnosis | status | exploratory",
  "headline": "{figure-keyed prose}, at most 240 characters",
  "lede": "…, at most 420 characters",
  "flag": "… (optional), at most 300 characters",
  "recommends": ["<arm id>"],
  "confidence": {"form": "high | high_if_immaterial | low",
                 "caveats": [{"id": "<code caveat id> | analyst:<slug>", "label": "…", "material_if": "…"},
                             {"id": "late:margin", "checked": "immaterial"}]},
  "decision": [{"slot": "recommendation | why_not | watch | next | the_trade | where_it_flips | our_lean | what_settles_it | what_is_tangled | known_despite_it", "text": "…"}],
  "claims": [
    {"id": "c1", "text": "…", "weight": "lead | support | detail",
     "evidence": [{"block": "<block kind>", "query": {"source": "…", "…": "…"}, "reference": "margin", "emphasis": "<arm id>", "caption": "…"}],
     "caveats": ["<caveat id>"], "durable": false}
  ],
  "explore": [{"control": "…", "binds": "…", "outcome": "…"}],
  "omitted": [{"what": "…", "why": "…"}],
  "next": [{"text": "…", "settles": "…"}],
  "method": [{"term": "…", "text": "…", "caveats": ["<caveat id>"]}]
}
```

| Field | Bounds | What it is |
|---|---|---|
| `story` | one of five | The analyst's label for its story (§3). No code reads it: the page and the checks are the same whatever it says. |
| `headline`, `lede`, `flag` | ≤240, ≤420, ≤300 characters; `flag` optional | The first screen. `flag` sits above the decision, for what a reader must know first. |
| `recommends` | 0 or more arm ids, each once | The call as data. The options table marks these arms. Code reads the call; it never chooses it. It holds only arms the report recommends: a tradeoff's lean stays in its prose, and an empty list is valid. |
| `confidence` | one form; each caveat once | How sure the call is, read from the method and the variance of the evidence (its design, criteria set late, the judge's standing, confounds, how well the evidence separates the arms), never from the result alone. Each caveat is open, with `material_if`, or `checked: "immaterial"`, never both. A lead's risk of reading the wrong way is stated beside the lead and does not lower the form by itself. |
| `decision` | 1–8 entries; a slot may repeat | Titled by the slot word, in list order. The analyst picks the words that fit the story. A diagnosis uses *What is tangled*, *Known despite it*, *What settles it* and *Next*. |
| `claims` | 1–12, unique ids | The story, in page order. `weight` sets type size and nothing else. `durable` marks a claim later campaigns should read as an insight. |
| `claims[].evidence` | 1 or 2 blocks | A block kind over a query, as `block_render` draws it, with every query field filled. `reference` and `emphasis` sit beside the query. |
| `explore` | 0–3 rows | What a reader could change, and what it would do, in prose. |
| `omitted` | any | What the analyst saw and left out, and why. |
| `next` | 0–6 | Each step and what it would settle. |
| `method` | 1–12 | The method appendix's terms. A caveat whose reach is the method may land here, but need not land anywhere: §7 asks that every caveat affecting confidence be shown, and a method-level one does not affect it. |

- **`claims` is the story.** Its order is the page order. The first claim is whatever best carries the headline, chosen by the analyst and judged by the editor. Code never checks what kind of block it is.
- **`omitted` makes elision accountable.** Anything substantial the analyst saw and chose not to show gets a one-line reason. The editor checks those reasons. The appendix still holds the omitted data.
- **`evidence.query`** names data through the tools' query language (§6). The renderer resolves it, so a block can never show data that doesn't exist.
- **Caveats** are named by the id code assigns (`caveats_list` lists them), or `analyst:<slug>` for one code doesn't assign. Code labels its own caveats, so their `label` is optional. An analyst caveat carries its own `label` and is declared once in `confidence`; a claim or a method item names it by id.
- **Validity comes before the invariants.** The document must name only arms and caveats its snapshot holds, or `analyst:` caveats.
- **The renderer adds the rest:** the masthead, the arms key, the options table (every arm, with `recommends` marked), the form words and the checked lines, each block's level, unit, axis and reference key, the full-results appendix, the omitted list, unresolved editor findings in Method, and the made line.
- **Stored** as `eval_report_doc`, beside the snapshot it reads, how the writing ended, the editor's unresolved findings, the cost tally per role, round and turn, and a reference to each session's transcript. It is derived and regenerable, like its snapshot. A stored document is never edited: regenerating writes a new one.
- **Sessions are kept whole.** The analyst's session and each editor round are stored in transcript storage, apart from the record, so listing reports reads none of them. A session holds the skill's identity, every message in order, thinking exactly as returned, and every tool call with its result, so it can be read back for an audit and resumed with one more turn, sending exactly what an uninterrupted session would.
- **Reads are strict.** A stored document is read whole or refused, never half-read. Report documents and their sessions carry a version of their own, and every change to the document's shape moves it: the store refuses a document from before it by its version, before its body is read; the page shows the code-only page with the line to regenerate; and regenerating (`analysis_generate`, a deliberate act) writes a new one from a fresh snapshot.

## 6. Tools (MCP)

One domain tool, `evals_analyst`, with one action per capability (owner, 2026-10-26: minimize tools, actions inside a domain). The analyst in the engine's own loop and any MCP client call the same tool: a host can mount it beside `evals` so its own agents investigate a campaign as the analyst does, read-only. The editor's cut leaves out the block actions and the freeze. All actions are read-only except `report_snapshot_freeze`. Every figure an action returns carries a figure key, so the analyst can cite it. `help` lists the actions and the figure-key topics.

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
| | `price_tradeoff` | Price per point and the steps along the cost frontier. Modeled hybrids are not built (#657) |
| | `arms_compare` | Any two arms, control or not: the difference and its interval, paired where a contrast holds one |
| | `settle_cases` | The cases that would settle an arm against its line, or two arms apart, and how often today's design reads the wrong way |
| | `levers_effects` | What one setting buys between arms that differ in it alone, and whether it reverses across another setting |
| | `confound_check` | Not built, until a run shows the analyst needing it |
| Rendering | `block_render` | Values table and warnings for a block; no pixels |
| | `report_check(doc)` | Every invariant's verdict on a draft, then the whole page as text, as the editor reads it; stores nothing. Analyst only |
| | `render_report(doc)` | Not built: no browser runs at storage time, and `check.py` runs in the integration tests |
| Reference | `block_catalog` | The block kinds, their sources and usage notes |
| Gaps | `capability_gap` | Records a gap: `missing`, a capability the analyst worked around, or `suspect`, a result that looks wrong, with the figure keys and tool calls that reproduce it (analyst and editor) |

Add an action when an analyst run shows it reasoning around a missing capability. Never add one to handle a specific eval type. The gap loop (§8) is how those runs are found.

## 7. Invariants and the editor

### Invariants (code; refuse to store on failure)

This list is fixed. It doesn't grow with new eval types.

1. Every number in prose resolves through a figure key to the data.
2. Every block's query resolves, and the block renders.
3. Every caveat that affects confidence is shown, either with its `material_if` clause or as checked immaterial.
4. The confidence form matches: no plain `high` while caveats are open.
5. No internal id, hash or measure key appears above the appendix.
6. The full-results appendix and Method are present.
7. `check.py` passes at 1200 and 390 px, in light and dark. Held by the integration tests, not at storage: no browser runs when a report is stored.

### Editor rubric (each item scored 0–2 with a reason)

1. **First screen.** From the headline, decision block and first claim alone, a reader can say what to do (or what's unresolved), why, and the main risk.
2. **Coherent story.** The claims follow from one another, and nothing important contradicts the headline unaddressed.
3. **Honest emphasis.** The recommended option's weak axes are visible, and nothing is framed stronger than its interval supports.
4. **Sound elision.** Nothing omitted would change a reader's decision.
5. **Confounds handled.** Each one is either shown not to matter or carried into the story and the next steps.
6. **Actionable next steps.** Each names an experiment or a declaration and says what it would settle.
7. **Evidence fits claims.** Each block is the clearest way to show its claim, and the counterfactual holds: if the values flipped, the claim would have to change.

Each finding carries a severity, and a `must_fix` sends the report back whatever the scores. Code adds the scores and decides the pass, never the model: a report passes when no item scores 0, the total is at least 11 of 14, and no finding is a must-fix. A report is stored finished only when its review passes with no blocking finding open, and one stored with a blocking finding open is a draft, which its page says first (owner rulings 2026-10-29). Every `must_fix` blocks, and so does every finding first raised in the first review while a later review marks it still open (each finding names the earlier finding it continues, or says `new`). A `should_fix` first raised in a later review never blocks, new or carried: it is sent back for one more revision while a revision remains, then listed open on a finished page, whose top line counts the findings left open (owner ruling 2026-10-31). Where a revision a cleared review sent ends on the budget, on its faults or on an unreadable review, the cleared report is stored finished. These thresholds are starting points; the calibration dataset sets them.

## 8. Running it in a host

`analysis_generate` starts the work as a job: freeze the snapshot, run the loop, store the document. A host supplies four things and the engine does the rest.

- **The model.** The `analyst` role's client, built from the host's key. It must take tool turns; one that cannot is refused before anything is spent, naming what it lacks.
- **The budget.** The host sets it, in dollars, or in tokens and turns for a model it has no price for. No caller can raise it per call. Every turn is admitted on its price bound before it is sent, and a refusal sends nothing. A bound built on rates the host marks as not guaranteed admits the turn on the estimate, and the report says the cap was checked against estimates.
- **The time limit.** The engine exports the worst-case writing time for its limits: every analyst round, revision, bounce and editor review, times the host's bound on one request. The host's job budget sits above it, with headroom, so a paid write is never cut short by a round number.
- **The page.** `report_read` serves the newest report in Markdown, HTML, static HTML (no script, no link, no URL) or JSON. A host frames the static page in a sandboxed iframe. A stored report this version no longer reads, or a planned report, is never refused by the page: it gets the code-only page and one line saying it was written in a shape this version no longer reads, and to regenerate it.

How a write can end:

| Ending | Stored |
|---|---|
| `stored`: a review passed with no blocking finding open | yes, finished, any should-fix findings that block nothing listed open and counted at the top |
| `stored_with_findings`: both revisions used | yes, a draft, with the open findings in Method |
| `over_budget` | the last reviewed draft, a draft, with that review's findings; nothing if no draft was reviewed |
| `refused`: bounces used up | no; the verdicts are returned |
| `review_failed`: the editor's answer unreadable twice | no |
| `failed`: a provider error after retries | no; the cost tally is kept |

On a storing ending the job links the document to the campaign as its analysis, names the recommended arm when the document recommends exactly one, and mints an insight from each claim marked `durable`. An ending that stores nothing fails the job, with its ending and reason.

**The gap loop.** Every `capability_gap` call is stored durably with its provenance: the snapshot, the report, the role, the model and the session. Two code notices join it: a key that could not be resolved and bounced a draft, and an editor item that scored 0 or 1. `analyst_gaps_list` groups them by kind and need, ranks suspect results first and must-fix-linked groups next, and returns each group as text ready to file as an issue, with its evidence. The engine posts nothing: a host's own scheduled job posts the digest and files the issues.

## 9. Open items

- **Cost.** A report costs many model turns rather than one call. Kairo round 2 took one analyst round and one editor round, 19 turns, $1.60. The host's budget bounds it; a report the budget stops keeps its last reviewed draft.
- **Determinism.** Two runs can write different but equally good reports. Stored reports are what readers see; regenerating one is a deliberate act and keeps the previous version.
- **When there's no good story.** The analyst may conclude the campaign can't support any reading. That is a valid report: a diagnosis or exploratory story whose next steps name what to declare or run. It is not a failure.
- **Calibration.** The editor's thresholds (no 0, at least 11 of 14) are starting points. Stored reports scored on the rubric, against reports people labelled, set them. The analysis reporter's cases still judge the planned report, not this one.
- **Chat with the analyst.** Not built. A reader could question a stored report's analyst, which would resume its stored session with read-only tools; ask it for a revision, which would take the draft's path and store a new version; and let what readers teach it reach the skill only through owner-approved proposals in the gap loop. It would ship as an optional extra on the agent framework, with the host supplying the chat UI, never inside the report page (#1002).

---

## Appendix A. Reasoning note: Kairo round 2

The first reasoning note, and the form every exemplar takes, is shipped with the skill: [`exemplars/kairo-round2.md`](../../packages/evals/src/threetears/evals/analysis/analyst/skill/exemplars/kairo-round2.md). It is kept there only, so the note the analyst reads is the one this document points at.
