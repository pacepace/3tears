# Writing

Reports may be written in any language a host wants: French, Arabic, English. This guidance is about how the text works, not about particular words. Apply it in the report's language.

## The owner's rules

These come from the owner's review of earlier reports. Each has a reason.

- **Numbers live in tables, not prose.** A sentence carries at most the one figure its claim turns on. A reader compares figures across rows, which prose makes hard.
- **One sentence of commentary per table or chart.** It says what the block shows. The block shows the rest.
- **A section header names; it never explains.** Read it as a table-of-contents entry: the place in the report, or the question the section answers ("Cost", "Where the loss lives"). No clause after a colon, no parenthetical, no account of how the section is built, no finding: the claim's sentence carries the finding. It fits on one line at phone width. About five words is a sanity check, not the rule: a header that needs more is usually carrying the finding or explaining itself.
- **No meta-commentary about the document.** Not "this report shows", not "the following section", not "it is worth noting". Say the thing.
- **No `key=value` labels in human text.** Ids and field names are for machines. Write "cost per 1,000 requests", never `cost_per_1k=0.42`.
- **Arms by their names, with the control marked.** Use the short names the tools return, the same name everywhere, and mark which arm is the control. A reader should never have to decode a configuration string.
- **Define each term of art once, or replace it.** If a reader needs "non-inferiority" to follow the page, define it in Method once. Better, say "may be up to 15 points worse".
- **Statistics as one clause.** "Clearly worse (the interval excludes zero)", not a paragraph about coverage. Interval method and correction go in Method.
- **Measured facts are stated flatly.** Uncertainty belongs to the interval, stated once. "Costs 28% more" is a measurement; don't soften it. Don't hedge a measured fact with "seems" or "appears".
- **Every number comes through a figure key, never typed.** The renderer formats it from the data, so the prose can't drift from the tables and a rerun updates it. A figure no tool returns isn't written; record the gap. **A count is a number even when written as a word**: "three of the runs", "eight of the ten cells" each come from a key, spelled through the `words` filter (`{count.…|words}`).
- **A change is cited by its gap, never by a level.** A step's price, a saving or an uplift is the gap between two arms: cite the gap's key (a `price.*` step's cost gap, a `compare.*` difference, a contrast's delta), never one arm's total, which a reader takes for what the move costs.
- **A saving reads as a saving.** Write a saving or a cost increase as its size (the `abs` filter) with its direction in words: "saves {gap|abs} a month", "costs {gap|abs} more". A signed gap says which arm it is measured from. A sign that contradicts its sentence (a "saving" printed below zero) tells the reader the opposite.
- **A volume says what it counts.** A declared monthly volume is a count of something: calls, conversations, tasks. Say which wherever the page cites it, Method included. Where the input doesn't say, write that it doesn't and log a `capability_gap`; a monthly figure on an unnamed count can be off by whatever each unit holds.
- **No internal id, hash or measure key above the appendix.** Trial and case ids may appear on example cards as references. The appendix may hold ids.
- **The page talks about the campaign, never the tools.** No word about tools, blocks, renders, revisions or review rounds anywhere on the page, Method included. If no block can show something, pick another block or state the figure in a sentence through its key, and log the gap with `capability_gap`.

## Headlines and claims are findings

A claim's text states what is true, with a subject and a direction. "Refund requests, the largest slice, carry most of the candidate's loss." Never a topic: "Failure analysis". A reader who reads only the claim sentences should get the argument.

The headline is the call (or the trade, or what is unresolved) with its reason. It never restates the question, and it never quotes the campaign's own question text, which is often written in ids.

## Names

- Reader names, not identifiers: "off-lane asks", not the stratum's id.
- **One measure, one name.** Use the reader name its key gives (the `.label` and `.name` keys), the same words everywhere: the headline, the lede, Watch, every caption. Never a synonym ("serving cost" on one line, "token cost" on the next), which reads as a second measure. A judged rate and a pass rate on the same aspect need visibly different names, or readers will merge them.
- **A gate is named as a gate on its measure** ("the 90% gate on balanced accuracy"), never by the measure's name, and the measure is never called by the gate's label.
- A column whose value is the same in every row moves into the heading.

## One caveat, one place

Each caveat is said once, beside the claim it qualifies, at the level it applies to. Report-wide caveats go in the confidence with their `material_if` clauses. A caveat on one arm goes beside that arm's claim. A caveat repeated in three places reads as three problems.

## Sentences

Short sentences, active voice, one idea each. The subject is the thing measured or the option, not "we" or "the analysis". A sentence that would survive any outcome ("results were mixed") says nothing. Cut it.

## Example cards

Cards show what failures look like. They never count them. Pick them from where the loss concentrates and say how many there are in all (`kept_from`, `count.failures.*`). Two rules keep them honest:

- **When the control shows the same failure, show one of its cards too.** Cards of candidate failures alone imply the failure is the candidates'. If the control shares it, one of its cards belongs beside theirs, and that sharing is often what a next step rests on.
- **A pattern on the cards is your reading, not a count.** No key counts the cards that show a pattern, so don't write one as a count. Name the cards that show it and any that doesn't ("the first and last cards …; the second fails another way"). How many failures were kept, and of how many, are counts: cite `count.examples.*` and `count.failures.*`. A claim about the pattern's share of all failures needs a count of all of them, which no tool gives; log a `capability_gap` and leave the share out.

## Before you submit

Run `figure_keys_check` over every key in the draft. Read the first screen alone. Can you say what to do, why, and the main risk? Then read each claim sentence and ask: does its block show this? Would the sentence change if the values flipped? Cut every sentence that fails either test.

Then check the draft against its own evidence:

- **Read the headline and each claim against its block's values table, row by row and option by option.** "Every option undercuts the baseline", "no option misses the floor", "the gain grows with the budget": find each row that would make the sentence false. If one does, rewrite the sentence. A page its own chart contradicts loses the reader's trust in every other line.
- **Watch carries the weakest margin by its `weakest.*` keys** (`principles.md`), or says why another margin matters more, in the same terms.
- **Every caveat on the deciding evidence is carried** with its direction (`statistics.md`), and no line calls a finding immune or irreversible while one is attached to it.
- **The headline's action and Next's first step agree.** A headline that says "ship" beside a Next that says "confirm first" leaves the reader two instructions. Say which: ship now and confirm in parallel, or confirm, then ship.
- **Every sign matches its words**, every measure has one name, and every count comes from a key.
