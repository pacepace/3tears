"""An outcome bar's note prints an arm's passes at the scorecard's form of the pass rate.

A campaign's code page states each arm's pass rate twice where no trial was lost: its scorecard's cell (``44 (16 of
36 trials, 12 cases)`` under an arm's ``(%)`` head) and, folded under it, the aspect's outcome bar's note (``44%
passed · 16 of 36``). Both are the same passes of the same trials, so both print at one form: the places the
campaign's intervals on the pass rate set, even where the aspect compares nothing on the rate itself (a choice reads
its pairs on the campaign page only), never one at the ruler's own one decimal beside the other at none.

The same holds on a change: an aspect that compares the arms on a measure of its own and not on the pass rate still
states each arm's passes at the code page's form of the rate, which the campaign's readings set.
"""

from __future__ import annotations

import html
import json
import re

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.test_quick_comparison import CASES, careful, hasty
from threetears.evals import Eval
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report.figures import report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.ops import report_read

#: An outcome bar's note: the share that passed, then of how many.
_NOTE = re.compile(r"(?P<share>\S+) passed · (?P<passed>\d+) of (?P<trials>\d+)")

#: A scorecard cell of a pass rate in an arm's ``(%)`` column, with its count.
_CARD = re.compile(r"(?P<level>[\d.]+) \((?P<passed>\d+) of (?P<trials>\d+) trials\b")


def _notes(page: str) -> dict[str, str]:
    """Each arm's outcome note, by its label, from the bars the kit draws."""
    notes: dict[str, str] = {}
    for found in re.findall(r'data-kind="bars" data-chart="([^"]*)"', page):
        for row in json.loads(html.unescape(found))["rows"]:
            if row.get("note"):
                notes[row["label"]] = row["note"]
    return notes


def _scorecard(page: str) -> dict[str, str]:
    """Each arm's scorecard cell, by its label, from the page's first table that heads an arm's column ``(%)``."""
    header: list[str] = []
    cells: dict[str, str] = {}
    for line in page.splitlines():
        if not line.startswith("|"):
            header = []
            continue
        row = [cell.strip() for cell in line.strip("|").split("|")]
        if not header:
            header = row
            continue
        for head, cell in zip(header, row, strict=False):
            if head.endswith(" (%)") and _CARD.match(cell):
                cells.setdefault(head.removesuffix(" (%)"), cell)
    return cells


def test_the_outcome_note_and_the_scorecard_print_one_arms_passes_alike() -> None:
    fixture = off_loop(ops_fixture)
    read = {
        form: off_loop(report_read, fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, format=form).body
        for form in ("html", "markdown")
    }
    notes, cards = _notes(read["html"]), _scorecard(read["markdown"])
    compared = 0
    for arm, note in notes.items():
        said, card = _NOTE.fullmatch(note), _CARD.match(cards.get(arm, ""))
        assert said is not None, note
        if card is None or (card["passed"], card["trials"]) != (said["passed"], said["trials"]):
            continue
        compared += 1
        assert said["share"] == f"{card['level']}%", f"{arm}: note {note!r} beside scorecard {cards[arm]!r}"
    assert compared, f"no note stated the passes a scorecard cell states: {notes} / {cards}"
    levels = [card["level"] for cell in cards.values() if (card := _CARD.match(cell)) is not None]
    assert any(level not in ("0", "100") for level in levels), "a rate no whole share is what tells the places apart"


async def test_an_aspect_of_a_change_comparing_no_pass_rate_states_passes_at_the_code_pages_form() -> None:
    result = await Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"], k=2).run(
        store=":memory:", scope="note-forms"
    )
    assert result.page.readings.kind == "change"
    (fold,) = result.page.aspects
    bundle = fold.report.bundle
    compared = {comparison.name for family in bundle.multiple_comparisons.families for comparison in family.comparisons}
    assert TRIAL_PASS_NAME not in compared, "the aspect compares a measure of its own, not the pass rate"
    aspect = PageForms.of_aspects([bundle], result.page.readings)[bundle.aspect]
    code = PageForms.of_campaign(result.page.readings, {fold.aspect: fold.report})
    passing = report_rulers(bundle_decision_surface(bundle), bundle.measure_catalog).of(TRIAL_PASS_NAME, "trial")
    # A third of the trials passed: no whole share, so the places are what reads.
    stated = aspect.ruled_level(passing, 1 / 3, measure=TRIAL_PASS_NAME, kind="trial").text
    scorecard = code.level(TRIAL_PASS_NAME, 1 / 3, kind="trial").text
    assert (stated, scorecard) == ("33%", "33%"), f"the aspect states {stated!r}, the code page {scorecard!r}"
