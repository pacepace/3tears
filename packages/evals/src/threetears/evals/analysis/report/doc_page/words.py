"""The words code writes on a report document's page: every line the document has no prose for, one template each.

The analyst writes the page's sentences; code writes its frame: each decision entry's title by its slot word alone,
the label of each caveat code assigns, the titles of the parts no analyst heads (the options table's, next steps',
explore's, the appendix's), the explore table's heads, the review findings' severity words, and the lines a page read
from a stored document adds where part of it no longer reads. The confidence's form words, the options table's role
words and the Full results tables' words are the planned page's
(:mod:`~threetears.evals.analysis.report.plan_page.words`), so both pages say them alike. Held here so a translation
replaces them in one place.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "APPENDIX_TITLE",
    "CAVEAT_LABELS",
    "CAVEATS_TEMPLATE",
    "CAVEAT_REPEATED",
    "CAVEAT_REPEATED_ON",
    "CHECKED_MARK",
    "CLAIM_WORD",
    "DEGRADED_BLOCK",
    "DEGRADED_TABLE",
    "DEGRADED_PAGE",
    "EXPLORE_HEADERS",
    "EXPLORE_TITLE",
    "FINDINGS_TITLE",
    "LEVEL_WORDS",
    "METHOD_TITLE",
    "NEXT_TITLE",
    "OMITTED_TITLE",
    "PAGE_WORD",
    "SET_APART",
    "SEVERITY_WORDS",
    "SLOT_TITLES",
    "UNRESOLVED",
]

#: Each decision entry's title, by its slot word alone (ANALYST §5's shared vocabulary): the title never depends on
#: the story, the campaign or the other entries.
SLOT_TITLES: Final[dict[str, str]] = {
    "recommendation": "Recommendation",
    "why_not": "Why not the others",
    "watch": "Watch",
    "next": "Next",
    "the_trade": "The trade",
    "where_it_flips": "Where the answer flips",
    "our_lean": "Our lean",
    "what_settles_it": "What would settle it",
    "what_is_tangled": "What is tangled",
    "known_despite_it": "What we know despite it",
}

#: The slot words whose entries the decision block sets apart, as the planned page sets apart its recommendation and
#: its lean: a look for the word, as a claim's weight is a type size.
SET_APART: Final = frozenset({"recommendation", "our_lean"})

#: The label of each caveat code assigns, by its kind
#: (:func:`~threetears.evals.analysis.report_decision.every_caveat`). ``{name}`` is what the id names, as a reader is
#: shown it (an aspect's or criterion's reader name, a setting in words), never the id. A kind not listed here is
#: labelled by its kind in words.
CAVEAT_LABELS: Final[dict[str, str]] = {
    "late": "The {name} was set after the results",
    "judge_uncalibrated": "The judge is not calibrated",
    "confound": "A confound: {name}",
    "apparatus": "The apparatus may have differed: {name}",
    "short_run": "A run fell short of its usable measurements",
    "unsettled": "Left unsettled past its margin on {name}",
    "latency": "Latency is not compared",
    "unreported": "Recorded but not reported: {name}",
    "judge_unstated": "A judge is not described",
    "judge_partly_uncalibrated": "Some judges are not calibrated",
    "no_control": "No arm is the control",
    "nothing_declared": "Nothing declares what is good enough",
    "point_estimate": "{name} is cleared on a point estimate",
    "borderline": "{name} is cleared on a borderline reading",
    "best_of_n": "The pick is the best of several candidates",
}

#: The caveats a claim or a Method item names, by their labels, as the line beside it says them: each distinct label
#: once.
CAVEATS_TEMPLATE = "Caveats: {labels}."

#: A label several of the caveats beside one claim or Method item share (two runs of one arm short of their usable
#: measurements): said once, with how many caveats carry it, as a count in words, and the arms their records name.
CAVEAT_REPEATED_ON = "{label} — {count} of them, on {arms}"

#: The same, where no record among them names an arm.
CAVEAT_REPEATED = "{label} — {count} of them"

#: How a checked caveat's line is marked in Markdown and the page text: code's, not a caveat left open.
CHECKED_MARK = "(checked)"

#: What a figure key that no longer resolves prints, on a page read from a stored document.
UNRESOLVED = "—"

#: The one note a page read from a stored document carries where some of it no longer reads on its snapshot.
DEGRADED_PAGE = (
    "Part of this report no longer reads on its campaign's data as this version reads it: a figure that no longer "
    "resolves prints as —, and a block that no longer draws prints as its values or its caption alone. Regenerate the "
    "report to redraw it."
)

#: Under a block that no longer draws as its kind but whose values still do: they print as a table.
DEGRADED_TABLE = "This block no longer draws as it was written; its values are shown as a table."

#: Under a block that no longer draws at all: its caption stands alone.
DEGRADED_BLOCK = "This block no longer draws on this campaign's data; only its caption is shown."

#: The explore part's heading.
EXPLORE_TITLE = "What you could change"

#: The explore table's column heads.
EXPLORE_HEADERS: Final[dict[str, str]] = {
    "control": "What you could change",
    "binds": "What it decides",
    "outcome": "What happens as it moves",
}

#: The next steps' heading.
NEXT_TITLE = "Next steps"

#: The appendix's divider: what follows it is the record, not the story.
APPENDIX_TITLE = "Full results"

#: The omitted list's heading (``report-doc/1``: ``omitted[]``).
OMITTED_TITLE = "Left out, and why"

#: Method's heading in the appendix, apart from what was left out.
METHOD_TITLE = "Method"

#: The heading of the editor's findings left unresolved, in Method.
FINDINGS_TITLE = "Review findings left open"

#: Each finding's severity, as Method prints it.
SEVERITY_WORDS: Final[dict[str, str]] = {"must_fix": "Must fix", "should_fix": "Should fix"}

#: How a finding names the claim it is about (by its place on the page), or the page as a whole.
CLAIM_WORD = "Claim {place}"
PAGE_WORD = "The page"

#: How the page text states a block's interval level, unit, axis and reference line, where it has each.
LEVEL_WORDS: Final[dict[str, str]] = {
    "level": "level {value}",
    "unit": "unit {value}",
    "domain": "axis {value}",
    "reference": "reference line: {value}",
}
