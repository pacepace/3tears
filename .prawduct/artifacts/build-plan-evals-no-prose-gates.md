---
artifact: build-plan
version: 1
scope: evals-no-prose-gates
branch: evals/no-prose-gates
depends_on: []
partition: "serial -- one removal across one package's report and generator; the call sites interlock, so a second agent would collide in build.py"
last_validated: 2026-10-24
---

## Requirements Confidence

**Level:** High

**Why:** Owner ruling, 2026-10-24: "as long as it does not bring back code checking prose. Bad judgment is fixed by
improved prompts and/or oversight and calibration, not by code." Widened the same day: "Code checks code. Models can
check code or prose. Better models can check those models. Humans can check those better models. Brute force never
works right. Our entire level system should support use in a French or Arabic app. Code can't adapt to that." The
user confirmed the shipped gates go too.

Problem: the package scans, rewrites, drops and refuses the memo writer's prose with English word lists and regexes,
and reads a host's arm labels for English comparative words.

Success: no code path in `packages/evals/src` reads the meaning of writer prose or host-supplied text. The page prints
the writer's words as written (whitespace collapsed). The generator's repair round refuses on structure only. Each
gate's intent lives on as a fact in the writer view or a rule in the seeded prompt.

Out of scope: package-authored page text (its English sentences, glossary, page lint), which is listed below for a
later localization decision and left unchanged. Structural checks (references resolving, schema, response-schema
fields, recommendation fields against code's readings). No package version bump. No re-recording of the writer
fixture.

**Decisions:**
- [DECISION: the model-id-to-label rewrite (`MemoStyle.names_arms`) goes too | it rewrites writer prose by matching
  strings in it; the prompt already says to name an arm by its `arm` label (rule 7), and the writer view carries the
  labels | user can veto]
- [DECISION: `_stray_arm_titles` goes, though it matches host arm names rather than English words | it reads the
  writer's title to decide what it claims, which is code checking prose; rule 9 keeps "names only arms it cites a
  reading or trial of" | user can veto]
- [DECISION: `LABEL_CLAIMS` goes | an English word list over a host's arm labels ("cheaper", "faster"), which the
  widened ruling names; the scoreboard states each arm's cost and time | user can veto]
- [DECISION: `lint_prose`, `lint_page`, `BANNED_PHRASES`, `sentences` and `StyleViolation` stay | they hold
  package-authored page text to the report style rules (tests), which the brief keeps]

## Audit

### Writer prose and host text: gates (removed)

| Path | What it checked | Classification | Action |
|---|---|---|---|
| `analysis/report/memo_style.py` `MemoStyle` (whole class) | Every memo sentence passed through it before the page printed it | Writer-prose gate | Remove the module |
| `memo_style.py` `_aged` / `style.discounts_by_age` (`_AGED`, `_APART`, `_DISCOUNTED`, `_CONFOUNDED`, `_CHANGED`) | Dropped a sentence discounting a reading for when its runs ran (#919); Methods line "…which discounts a reading for when its runs ran…" | Writer-prose gate, English regex | Remove; prompt rule 11 keeps "never discount a reading for when its runs ran"; `late_criteria` already in the writer view |
| `memo_style.py` `proposes_sample` / `_planned` (`_SAMPLE_SIZE`, `_SPELLED_SIZE`, `_MULTIPLE`, `_REPEATS`, `_ACQUIRE`, `_MONEY`, `_WORD_COUNTS`) | Dropped or replaced a sentence proposing a sample size, k or cost no projection states (#846); Methods line "…a sample size or cost no code computed…" | Writer-prose gate, English regex and number-word list | Remove; preamble keeps "never state a sample size or a cost of your own"; rule 12 keeps `cases_to_settle`; rule 5 keeps `runs[].k_runs` and `repeats_per_case` |
| `memo_style.py` `_HEDGES` and `settled=` replacement | Replaced a sentence disputing a floor reading with code's sentence; Methods line "…disputes a difference a floor read either way…" | Writer-prose gate, English regex | Remove; rule 9 keeps "never dispute a reading or a floor code settled" |
| `memo_style.py` `holds_measured` (`_NUMBER`, `_N_OF_M`) and `fallback=` | Dropped a body sentence or replaced a title restating a measured number with code's row in words | Writer-prose gate | Remove; rule 9 keeps "never restate the numbers of the table under it" |
| `memo_style.py` `names_arms` | Rewrote raw model ids in memo prose into arm labels | Writer-prose rewrite | Remove; rule 7 keeps "name an arm by its `arm` label" |
| `memo_style.py` `offending` / `kept` / `or_as_written` | Linted kept memo sentences, listed them in the Methods "Style" fold and the "Memo style" methods row | Writer-prose lint | Remove |
| `style.fix_prose` | Dropped a sentence with a banned phrase or the repeated question | Writer-prose gate (its only caller is `MemoStyle`) | Remove |
| `report/build.py` `_memo_style`, `_settled_sentence`, `_row_sentence`, `_rate_of`, `memo_style.row_words`, the "Memo style" methods row | Built and fed the gate; wrote the replacement sentences | Gate plumbing | Remove; the memo's words print as written |
| `report/campaign_page.py` `_CampaignStyle` | The same gate over every aspect of a campaign page | Writer-prose gate | Remove |
| `report/decision.py` `projection_sentence`, `settles`, `settles_cost` | Code's replacement sentences for the gate | Gate plumbing, no other caller | Remove |
| `analysis/generator.py` `_unestablished_null_titles` + `analysis/null_claims.py` | Refused (paid repair) a finding title in an English null phrasing citing no established null | Writer-prose gate, English phrase list | Remove; rule 4 and rule 9 keep "an unresolved contrast is never no change" |
| `analysis/generator.py` `_stray_arm_titles` + `analysis/arm_mentions.py` (`_ROLE_WORDS`) | Refused a finding title naming an arm its finding cites nothing of | Writer-prose gate (string match over the title) | Remove; rule 9 keeps "names only arms it cites a reading or trial of" |
| `analysis/generator.py` `_reject_unsound_titles` | Joined the two above into one refusal | Gate plumbing | Remove |
| `style.LABEL_CLAIMS` + `report/build.py` `_label_claims` | English comparative words in a host's arm label read against its row ("'cheaper candidate' is not cheaper") | Host-text gate, English word list | Remove; the scoreboard states cost and time |

### Structural checks (kept)

| Path | What it checks | Classification |
|---|---|---|
| `analysis/prose_refs.py` | A `{{cell\|measure\|reading\|stat}}` figure reference resolves | Structural ref |
| `analysis/references.py` | Evidence rows name a cell and measure that exist | Structural ref |
| `generator.py` `_reject_mismatched_aspect_summaries`, `_reject_unknown_trials`, `_reject_unknown_arm`, `_reject_unpinned_recommendation` | Fields name aspects, trials, arms and findings the bundle holds | Structural |
| `generator.py` `_reject_recommendation_off_kind`, `_reject_exploratory_adoption`, `_reject_unseparated_adoption` | The recommendation's arm and confidence against code's readings and the campaign's kind | Structural (fields against readings) |
| `generator.py` `_reject_unaddressed_warnings` | Each raised warning keyed once in its field with a non-blank `why` | Structural (keys, presence) |
| `kernel/authored.py` `summary` validator; `_assemble_and_validate`; `_parse_payload` | Non-empty summary, schema, JSON | Schema |

### Package-authored text (kept)

`style.lint_prose`, `lint_page`, `BANNED_PHRASES`, `MAX_WORDS`, `GLOSSARY` and the report-style tests hold code's own
page sentences to the report style rules.

### Language assumptions in package-authored page text (reported, unchanged)

| Path | Assumption | Classification |
|---|---|---|
| `report/build.py`, `campaign_page.py`, `decision.py`, `verdict.py`, `words.py`, `svg.py`, `serialize_*.py` | Every page sentence, heading, table header and Methods line is an English f-string | Package-authored English page text |
| `style.GLOSSARY`, `CONFIDENCE_WORDS`, `NO_RECOMMENDATION_WORDS`, `SECTION_TITLES` | English term definitions and labels | Package-authored English page text |
| `style.lint_prose` / `lint_page` (`_WORD` `[A-Za-z]`, `_SENTENCE_END` needs `[A-Z]`, `MAX_WORDS`, `BANNED_PHRASES`) | Sentence splitting and word counting assume Latin-script English | English lint over package-authored text |
| `report/build.py` `_question_line` ("does X or Y do better than Z?"), `_capitalized` / `_PLAIN_WORD` (ASCII `[a-z]`) | English question built around host arm labels; capitalizes a host title only if ASCII | English composed on host strings |
| `style.is_dump` (`startswith("The world")`) | A host case label starting "The world" is read as a world dump | English matching on a host string |
| `style.humanize("case")` (`_NOT_A_NAME`: any whitespace) | A case label with a space is no name | Whitespace-delimited-language assumption on a host string |
| `style.humanize("dimension")` (`_` and `.` to spaces) | Host identifiers read as space-separated words | Latin-identifier assumption on a host string |
| `kernel/metrics.py` `"measuring spend" not in title.casefold()` | English suffix check on a host reader name | English matching on a host string |
| `report/campaign_page.py` (~469) `set_it[0].lower()` | Lowercases a host-derived sentence's first letter inside an English sentence | English composed on host strings |
| `report/serialize_html.py` `_CASE_NUMBER` ("case (\d+)") | Links the page's own English "case N" cells | English matching on package text |
| `analysis/reporter_kind.py` memo Markdown ("What changed:", "How sure:") | English headings around writer text sent to the reporter judge | Package-authored English text |
| `analysis/gen_prompt.py` | The seeded writer prompt is English and names no output language | Prompt language (operator-tunable) |

## Chunks

### Chunk 01: Remove the prose gates

- [x] Delete `report/memo_style.py`, `null_claims.py`, `arm_mentions.py`; remove `fix_prose`, `discounts_by_age` and
  its regexes, `LABEL_CLAIMS` from `style.py`; remove the gate plumbing from `build.py`, `campaign_page.py`,
  `decision.py` and `generator.py`.
- [x] Prompt: drop "code lints every sentence…" and the module docstring's account of the word checks; keep every
  rule. Report the character count.
- [x] Remove the tests that pinned the removed gates; leave structural tests alone.
- [x] MIGRATIONS.toml `unreleased` and CHANGELOG `## Unreleased`: seed moves (`eval_prompt_reset`), Methods lines gone,
  shape effects.
- [x] Regenerate (no change: no public name moved) `api/surface.json` and `docs/reference.md` if names change.

Verification: `-k` selections over the touched test modules; `./scripts/lint.sh`; `./scripts/typecheck.sh evals`.

## Status

- [x] Chunk 01 (Critic owed: the main session runs it; the writer fixture `test_generate_analysis_fixture` replay is red until re-recorded, as the prompt moved)
