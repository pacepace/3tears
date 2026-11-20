"""The number policy on a written page: Kairo's page, as design.md's before-and-after table states it, row by row.

Each test holds one Kairo row of the table under "What the policy changed" (``packages/evals/docs/design.md``, The
number policy) to the text the page prints, read off its Markdown, so a reader's view of one measure is held where
the reader meets it: every measure at one precision on the whole page, read from its uncertainty (rule 2); an
interval in its one form, the unit once (rule 3); the fixed forms of a p, a margin and an exact zero (rule 4); and
every heading stating the joint guarantee a corrected family holds at, the per-interval level left to Method.
"""

from __future__ import annotations

import json
from functools import cache

from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

#: Kairo's written page: the third prototype's document on its own snapshot.
_KAIRO = "prototype3/kairo"


#: The key that names the per-interval level, which a writer citing it gets by definition.
_FAMILY_LEVEL = "{family.level}"


def _markdown(doc: dict[str, object]) -> str:
    """Kairo's written page for a document, as Markdown."""
    held = ReportInput.model_validate(snapshot_of(_KAIRO).report_input)
    return doc_page_markdown(build(ReportDoc.model_validate(doc), held, strict=True))


@cache
def _lines() -> tuple[str, ...]:
    """Kairo's written page as Markdown, line by line."""
    return tuple(_markdown(converted_doc(_KAIRO)).splitlines())


def _row(*cells: str) -> str:
    """A Markdown table row of these cells."""
    return "| " + " | ".join(cells) + " |"


def test_the_options_table_s_pass_rate_reads_at_its_measure_s_precision_with_its_counts() -> None:
    """Options table, pass rate: ``85.3% (765 of 897)``: its narrowest interval is 7 points wide, which reads to two
    significant figures at a tenth (owner ruling, 2026-11-14; it read ``85%`` under the half-width's first figure)."""
    assert _row(
        "P · September", "Control (recommended)", "deepseek/deepseek-v4.1-flash", "85.3% (765 of 897)", "$1.89"
    ) in (_lines())


def test_the_change_chart_s_head_reads_the_pass_rate_at_its_measure_s_precision() -> None:
    """Change chart head: ``Pass rate · September 85.3%``, as the options table prints it."""
    heads = [line for line in _lines() if line.startswith("| Pass rate · September")]
    assert heads and all(line.startswith("| Pass rate · September 85.3% |") for line in heads), heads


def test_the_change_chart_reads_qwen_s_change_its_interval_and_its_note_at_one_precision() -> None:
    """Change chart, qwen3.8-flash: ``-7.6 pts · -12.9 to -1.8 pts · -7.6 pts worse``, at the pass rate's one
    precision, its values table writing the unit its values are read in (rule 3: an interval alone ends on its
    unit)."""
    assert (
        _row("Pass rate · September 85.3%", "A · qwen3.8-flash", "-7.6 pts", "-12.9 to -1.8 pts", "-7.6 pts worse")
        in _lines()
    )


def test_1017_s_change_reads_as_its_prose_does() -> None:
    """Prose, 1017: its drop of ``3.0`` points sits beside ``-3.0 pts (-7.4 to +2.6 pts)``, one precision."""
    lines = _lines()
    assert (
        _row("Pass rate · September 85.3%", "B · 1017", "-3.0 pts", "-7.4 to +2.6 pts", "-3.0 pts cannot tell apart")
        in lines
    )
    assert any("1017 (B)'s drop of 3.0 points" in line for line in lines)


def test_the_margin_reads_as_declared_on_the_chart_and_in_the_prose() -> None:
    """Margin line: ``margin -15 pts``, its value ``-15 pts``, never ``-15.0``, as the prose's ``15-point margin``."""
    lines = _lines()
    assert _row("margin -15 pts; shaded: no worse than the margin", "-15 pts", "", "") in lines
    assert any("the 15-point margin" in line for line in lines)
    assert not any("-15.0" in line for line in lines)


def test_holds_character_reads_at_a_tenth_for_every_arm() -> None:
    """Holds character, arm by measure: ``91.5% · 85.4% · 86.6%`` (its comparisons' narrowest interval is 8.1 points
    wide, a tenth at two significant figures)."""
    assert _row("Persona holds character", "91.5%", "85.4%", "86.6%") in _lines()


def test_holds_character_s_strata_read_at_its_one_precision() -> None:
    """Holds character, strata: ``65.8%, 42.9%, 45.6%`` off lane, as the measure reads everywhere on the page."""
    assert _row("off lane", "Persona holds character", "38", "65.8%", "42.9%", "45.6%") in _lines()


def test_the_send_message_check_keeps_its_decimal() -> None:
    """Send message check: ``99.7% · 98.7% · 100.0%`` and ``-1.0 pts (-3.6 to +1.6)``, unchanged: its interval is
    ±0.6 points, so it reads to a tenth."""
    lines = _lines()
    assert _row("Send message or queue voice check", "99.7%", "98.7%", "100.0%") in lines
    assert any("| Send message or queue voice check | -1.0 pts | -3.6 to +1.6 pts |" in line for line in lines)


def test_failed_reply_notices_read_at_two_figures_of_their_interval_s_width() -> None:
    """Failed-reply notices: ``0.0022 notices`` and ``-0.0022 (-0.0069 to +0.0025)``; an exact zero ``0 notices``."""
    lines = _lines()
    assert _row("Failed-reply notices", "0.0022 notices", "0 notices", "0 notices") in lines
    assert any("| Failed-reply notices | -0.0022 notices | -0.0069 to +0.0025 notices |" in line for line in lines)


def test_every_heading_states_the_joint_guarantee_and_method_keeps_the_per_interval_level() -> None:
    """Interval level: every head reads ``95% interval (adjusted for 16 comparisons)``, never ``99.7%``; the
    per-interval ``99.69%`` is Method's, at its stored precision."""
    lines = _lines()
    heads = [line for line in lines if "interval (" in line and line.startswith("| ")]
    assert heads and all("95% interval (adjusted for 16 comparisons)" in line for line in heads), heads
    assert not any("99.7% (" in line or "99.7% interval" in line for line in lines)
    assert any(line.startswith("- **Intervals**") and "each interval at 99.69%" in line for line in lines)


def test_adjusted_p_values_read_as_apa_writes_them() -> None:
    """Adjusted p: ``< 0.001 · < 0.001 · 0.50 · 1.00`` for ``< 0.000001 · 0.00008342 · 0.4996 · 1``."""
    lines = _lines()
    cost = _row("B · 1017", "Production cost, per 1,000 replies", "+$0.49", "+$0.17 to +$0.81")
    assert any(line.startswith(cost.removesuffix(" |") + " |") and "| < 0.001 |" in line for line in lines)
    assert any(
        line.startswith("| B · 1017 | Send message or queue voice check |") and "| 0.50 |" in line for line in lines
    )
    assert any(line.startswith("| B · 1017 | Outcome (1–5) |") and "| 1.00 |" in line for line in lines)
    assert not any("0.00008342" in line or "0.4996" in line or "< 0.000001" in line for line in lines)


def test_a_corrected_comparison_s_interval_writes_its_unit_once() -> None:
    """Corrected comparison interval: ``-12.9 to -1.8 pts``, never ``-12.9 pts to -1.8 pts``."""
    assert any(line.startswith("| A · qwen3.8-flash | Pass rate | -7.6 pts | -12.9 to -1.8 pts |") for line in _lines())


def test_only_the_family_level_key_prints_the_per_interval_level() -> None:
    """The per-interval 99.69% is the ``{family.level}`` key's, and Method's and the glossary's where they cite it:
    Kairo's page prints it once for each time its document cites the key, and, with every such citation written out
    in words, nowhere at all. No heading, scale line or phrase the package builds states it."""
    doc = converted_doc(_KAIRO)
    cited = json.dumps(doc).count(_FAMILY_LEVEL)
    assert cited, "Kairo's document cites the family's level, so this tests something"
    assert "\n".join(_lines()).count("99.69%") == cited
    worded = json.loads(json.dumps(doc).replace(_FAMILY_LEVEL, "the corrected level"))
    assert "99.69" not in _markdown(worded)


def test_a_preference_reads_at_its_interval_s_precision() -> None:
    """Rung 5's preference chart read ``40.000 to 60.000%``: its shares and its interval's ends read at the precision
    its interval supports, as a rate's do (number policy, rule 2), ``50%`` and ``40 to 60%``, one decimal more only
    where a figure would otherwise print on none or all."""
    forms = PageForms.off_page()
    even = forms.preference(0.5, (-0.2, 0.2))
    assert even is not None and (even.share_text(even.candidate), even.ends) == ("50", ("40", "60"))
    near_all = forms.preference(0.995, (0.98, 1.0))
    assert near_all is not None and (near_all.share_text(near_all.candidate), near_all.ends) == (
        "99.5",
        ("99.0", "100.0"),
    )


def test_a_preference_keeps_off_even_as_it_keeps_off_none_and_all() -> None:
    """Even, 50%, is a preference's boundary (the owner's "important boundaries"): an end of ``49.7`` that read ``50``
    would hide that the interval crosses even, so it takes the one decimal that keeps it off, and so does a share of
    ``49.7``; the set reads at that one precision. A share exactly even reads ``50``, as above."""
    forms = PageForms.off_page()
    crossing = forms.preference(0.6, (-0.006, 0.406))
    assert crossing is not None and (crossing.share_text(crossing.candidate), crossing.ends) == (
        "60.0",
        ("49.7", "70.3"),
    )
    short = forms.preference(0.497, (-0.3, 0.3))
    assert short is not None and (short.share_text(short.candidate), short.ends) == ("49.7", ("35.0", "65.0"))
