# Writing

Reports may be written in any language a host wants: French, Arabic, English. This guidance is about how the text works, not about particular words. Apply it in the report's language.

## The owner's rules

These come from the owner's review of earlier reports. Each has a reason.

- **Numbers live in tables, not prose.** A sentence carries at most the one figure its claim turns on. A reader compares figures across rows, which prose makes hard.
- **One sentence of commentary per table or chart.** It says what the block shows. The block shows the rest.
- **A section header names; it never explains.** Read it as a table-of-contents entry: the place in the report, or the question the section answers ("Cost", "Where the loss lives"). No clause after a colon, no parenthetical, no account of how the section is built, no finding: the claim's sentence carries the finding. It fits on one line at phone width. About five words is a sanity check, not the rule: a header that needs more is usually carrying the finding or explaining itself.
- **No meta-commentary about the document.** Not "this report shows", not "the following section", not "it is worth noting". Say the thing.
- **No `key=value` labels in human text.** Ids and field names are for machines. Write "cost per 1,000 replies", never `cost_per_1k=1.89`.
- **Arms by their names, with the control marked.** Use the short names the tools return, the same name everywhere, and mark which arm is the control. A reader should never have to decode a configuration string.
- **Define each term of art once, or replace it.** If a reader needs "non-inferiority" to follow the page, define it in Method once. Better, say "may be up to 15 points worse".
- **Statistics as one clause.** "Clearly worse (the interval excludes zero)", not a paragraph about coverage. Interval method and correction go in Method.
- **Measured facts are stated flatly.** Uncertainty belongs to the interval, stated once. "Costs 28% more" is a measurement; don't soften it. Don't hedge a measured fact with "seems" or "appears".
- **Every number comes through a figure key, never typed.** The renderer formats it from the data, so the prose can't drift from the tables and a rerun updates it. A figure no tool returns isn't written; record the gap.
- **No internal id, hash or measure key above the appendix.** Trial and case ids may appear on example cards as references. The appendix may hold ids.
- **The page talks about the campaign, never the tools.** No word about tools, blocks, renders, revisions or review rounds anywhere on the page, Method included. If no block can show something, pick another block or state the figure in a sentence through its key, and log the gap with `capability_gap`.

## Headlines and claims are findings

A claim's text states what is true, with a subject and a direction. "Off-lane asks, the largest stratum, carry about half of each candidate's loss." Never a topic: "Failure analysis". A reader who reads only the claim sentences should get the argument.

The headline is the call (or the trade, or what is unresolved) with its reason. It never restates the question, and it never quotes the campaign's own question text, which is often written in ids.

## Names

- Reader names, not identifiers: "off-lane asks", not the stratum's id.
- One name per measure. A judged rate and a pass rate on the same aspect need visibly different names, or readers will merge them.
- A column whose value is the same in every row moves into the heading.

## One caveat, one place

Each caveat is said once, beside the claim it qualifies, at the level it applies to. Report-wide caveats go in the confidence with their `material_if` clauses. A caveat on one arm goes beside that arm's claim. A caveat repeated in three places reads as three problems.

## Sentences

Short sentences, active voice, one idea each. The subject is the thing measured or the option, not "we" or "the analysis". A sentence that would survive any outcome ("results were mixed") says nothing. Cut it.

## Before you submit

Run `figure_keys_check` over every key in the draft. Read the first screen alone. Can you say what to do, why, and the main risk? Then read each claim sentence and ask: does its block show this? Would the sentence change if the values flipped? Cut every sentence that fails either test.
