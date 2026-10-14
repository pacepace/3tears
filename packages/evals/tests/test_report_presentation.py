"""The report's presentation, held to the owner's verdict on the first production report and the host team's read of it.

The G2 campaign's numbers (4 cases × 3 repeats; production passed 10 of 12, the cheaper candidate 2 of 12; cost per
reply 0.003092 against 0.003301) are replayed here through the quick path with arms named as a host names them —
``deepseek-v4.1-flash (production)`` — and the campaign's own decision rule declared before the run. Every page the
suite renders here passes :func:`~threetears.evals.analysis.style.lint_page`.
"""

from __future__ import annotations

import html
import json
import re

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    Report,
    build_code_only_report,
    build_report,
    generate_analysis,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TableBlock, TrialBlock
from threetears.evals.analysis.report.decision import (
    RateReading,
    RuleReadings,
    evaluate_rule,
    parse_decision_rule,
)
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.bundle.schema import TrialCheck
from threetears.evals.analysis.style import GLOSSARY, arm_name, lint_page
from packages.evals.tests.g2_support import (
    CANDIDATE,
    CASES,
    G2_QUESTION,
    PRODUCTION,
    answering,
    declaring,
    g2_numbers,
    g2_report,
    g2_ruled_report,
)
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient


@pytest.fixture
async def g2_ruled() -> tuple[Result, Report]:
    """The G2 numbers through the quick path, with the G2 decision rule declared before the run."""
    return await g2_ruled_report()


# =============================================================================
# The first screen: result first, golden lines, a caveat never bold or first
# =============================================================================


async def test_the_first_screen_reads_the_question_the_arms_then_the_result(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    lines = report_markdown(report).splitlines()
    assert lines[0] == "# G2: can the cheaper model replace production?"
    assert (
        lines[1] == "deepseek-v4.1-flash (production, control) vs deepseek-v4-flash (cheaper candidate) · 14 Oct 2026"
    )
    assert lines[3].startswith("**Keep production: ")
    assert lines[5:8] == [
        "**Scoreboard**",
        "",
        "| Arm | Ran | Usable | Passed | Failed by check | Failed by judge | Cost per reply |",
    ]
    assert lines[9] == "| ▸ production | 12 | 12 | 10 | 2 | 0 | $0.003092 |"
    first = report_markdown(report).split("\n## ", 1)[0]
    # The rule settled keep on the counts, so the floor caveat says what four cases can and cannot do.
    caveat = (
        "4 cases per arm: enough to apply the rule, too few to estimate how large the difference is (that needs 5)."
    )
    assert caveat in first.splitlines(), "the case floor, in normal weight"
    assert "Too few cases to estimate a rate" not in first
    assert first.index("**Keep production") < first.index("**Scoreboard**") < first.index(caveat)


async def test_an_outcome_bar_repeating_the_scoreboard_prints_no_second_table(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    # Passed, failed by check and failed by judge are the scoreboard's own columns: the bar's table would repeat it.
    assert "**Outcomes**" not in report_markdown(report)
    # The page still draws the bar.
    assert report_html(report).count('data-chart="outcome_bar"') == 1


async def test_the_question_is_the_title_and_never_repeated_in_the_body(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    markdown = report_markdown(report)
    assert "does deepseek-v4-flash hold every pass/fail queue-discipline dimension" not in markdown
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_raw_model_id_is_named_at_most_once_per_section(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    # The verdict sentence aside: it names each arm by its model, since a label beside "cost more" can contradict it.
    (verdict,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    sections = re.split(r"\n## ", report_markdown(report).replace(verdict, "")) + [
        _visible_text(part).replace(verdict, "") for part in report_html(report).split("<h2>")
    ]
    assert "deepseek-v4-flash passed" in verdict
    for section in sections:
        for model in ("deepseek-v4.1-flash", "deepseek-v4-flash"):
            assert len(re.findall(rf"(?<![\w.-]){re.escape(model)}(?![\w.-])", section)) <= 1, (model, section[:80])


def _visible_text(page: str) -> str:
    """An HTML page's text as a reader sees it, the stylesheet left out."""
    return html.unescape(re.sub(r"<[^>]+>", "\n", re.sub(r"<style>.*?</style>", "", page, flags=re.S)))


# =============================================================================
# Every page the suite renders here passes the lint; the glossary covers every term
# =============================================================================


async def _pages() -> list[Report]:
    comparison = await g2_numbers()
    ruled = build_code_only_report(declaring(comparison.report.bundle, G2_QUESTION), assembled_at="2026-10-14T00:00Z")
    one_arm = await Eval(CASES, {"production": answering(10, 0.003)}, ["label"], k=3, name="queue").run(
        store=":memory:"
    )
    return [comparison.report, ruled, one_arm.report, await g2_report()]


def _arm_names(report: Report) -> set[str]:
    (scoreboard,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "scoreboard"]
    return {str(row["arm"]).removeprefix("▸ ") for row in scoreboard.values()}


async def test_every_page_passes_the_style_lint() -> None:
    for report in await _pages():
        assert lint_page(report_markdown(report), names=_arm_names(report)) == [], report.title


async def test_every_linked_term_is_defined_once_in_the_glossary() -> None:
    for report in await _pages():
        page = report_html(report)
        linked = set(re.findall(r'href="#g-([a-z-]+)"', page))
        defined = re.findall(r'<dt id="g-([a-z-]+)">', page)
        assert linked and linked == set(defined) and len(defined) == len(set(defined))
        assert set(report.terms) <= set(GLOSSARY)
        assert page.index('id="glossary"') > page.index('id="methods"'), "the glossary folds last"


async def test_one_arm_reads_n_of_n_cases_passed_first() -> None:
    one = await Eval(CASES, {"production": answering(10, 0.003)}, ["label"], k=3, name="queue").run(store=":memory:")
    (verdict,) = [b for b in one.report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert re.fullmatch(r"queue: \d/4 cases passed; \d miss(es)?: case \d(, case \d)*", verdict.text), verdict.text


# =============================================================================
# Trials: one row each, a record folded under a summary that is never the world dump
# =============================================================================


async def test_a_trial_s_summary_is_never_the_first_line_of_what_it_asked() -> None:
    world = (
        'The world as it stood before the persona\'s first turn:\n\n## music\n{\n  "catalog": [{"artist": "Anz"}]\n}'
    )
    cases = [{"text": f"{world} {index}", "label": "queued"} for index in range(2)]
    comparison = await Eval(cases, {"a": answering(4, 0.001), "b": answering(1, 0.001)}, ["label"], k=2).run(
        store=":memory:"
    )
    report = comparison.report
    trials = [block for block in report.blocks if isinstance(block, TrialBlock)]
    assert trials and all(re.fullmatch(r"case \d", trial.case_label) for trial in trials)
    summaries = re.findall(r"<summary>(.*?)</summary>", report_markdown(report)) + re.findall(
        r"<summary>(.*?)</summary>", report_html(report)
    )
    trial_summaries = [summary for summary in summaries if re.match(r"T\d+ · ", summary)]
    assert len(trial_summaries) == 2 * len(trials)
    for summary in trial_summaries:
        assert "{" not in summary and "The world" not in summary
        assert re.fullmatch(r"T\d+ · case \d · [ab] · (passed|failed|lost)", summary), summary
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "trials"]
    assert len(table.rows) == len(trials)
    assert all("{" not in str(value) for row in table.values() for value in row.values())


async def test_a_failed_check_reads_as_its_expression_in_the_trials_table(g2_ruled: tuple[Result, Report]) -> None:
    _, coded = g2_ruled
    expression = 'call_count("music.queue_track") >= 2'
    trials = list(coded.bundle.trials)
    first = next(index for index, trial in enumerate(trials) if trial.state == "failed")
    trials[first] = trials[first].model_copy(
        update={
            "cause": "check",
            "failed_checks": [TrialCheck(name=f"Pass rate of {expression}", expression=expression)],
        }
    )
    bundle = coded.bundle.model_copy(update={"trials": trials})
    report = build_code_only_report(bundle, assembled_at="2026-10-14T00:00Z")
    (table,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "trials"]
    whys = [row["why"] for row in table.values()]
    # Code, as written: no underscores or dots stripped, and no "Pass rate of" on one trial's failure.
    assert f"check: `{expression}`" in whys
    markdown = report_markdown(report)
    assert "Pass rate of" not in markdown and "call count(" not in markdown
    assert f"- `{expression}`" in markdown


# =============================================================================
# Colours: the page is framed under both colour schemes
# =============================================================================


def _root_blocks(style: str) -> list[str]:
    """Every block declaring the page's colour variables: the light one, the dark media one, the dark override."""
    return re.findall(r":root[^{]*\{([^}]*)\}", style)


async def test_every_colour_is_a_variable_both_schemes_define() -> None:
    page = report_html(await g2_report())
    (style,) = re.findall(r"<style>(.*?)</style>", page, re.S)
    roots = _root_blocks(style)
    assert len(roots) == 6, "light, dark by preference and dark by attribute, for the page and for the charts"
    rules = re.sub(r":root[^{]*\{[^}]*\}", "", style)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", rules), "a hard-coded colour outside :root"
    light = " ".join(roots[0::3])
    dark = " ".join(roots[1::3])
    used = set(re.findall(r"var\((--[\w-]+)", rules)) - {"--font-sans", "--font-mono"}
    for variable in used:
        assert f"{variable}:" in light and f"{variable}:" in dark, variable
    svg = "".join(re.findall(r"<svg\b.*?</svg>", page, re.S))
    assert not re.search(r'(fill|stroke|color|style)="[^"]*(#|rgb)', svg), "a colour drawn inline"


# =============================================================================
# The memo: fixed where code can fix it, listed where it cannot, never contradicting the rule
# =============================================================================


def test_a_memo_sentence_contradicting_the_settled_rule_is_replaced_and_disclosed() -> None:
    rule = parse_decision_rule(G2_QUESTION)
    assert rule is not None
    verdict = evaluate_rule(
        rule,
        RuleReadings(
            rates=(RateReading("the pass rate", 10, 12, 2, 12),), control_cost=0.003092, candidate_cost=0.003301
        ),
    )
    said = "Keep production: the pass rate drops at least 67 points, and cheaper candidate costs more per reply."
    style = MemoStyle(
        bundle=AnalysisContextBundle.model_construct(scoreboard=[], cell_measures=[]),
        names={"p": arm_name(f"model={PRODUCTION}", control=True), "c": arm_name(f"model={CANDIDATE}")},
        rule=verdict,
        rule_sentence=said,
    )
    unanswerable = "Unanswerable on this campaign because four cases leave both rates unresolved."
    assert style.sentence(unanswerable) == said
    assert style.replaced == [unanswerable]
    (disclosed,) = decision_disclosures(style)
    assert unanswerable in disclosed and "replaced it" in disclosed
    # A raw model id an arm carries reads as the arm's label.
    assert style.names_arms("deepseek-v4-flash queued fewer tracks than deepseek-v4.1-flash.") == (
        "cheaper candidate queued fewer tracks than production."
    )


async def test_a_scripted_memo_is_cleaned_and_what_cannot_be_fixed_is_listed(g2_ruled: tuple[Result, Report]) -> None:
    comparison, coded = g2_ruled
    bundle = coded.bundle
    aliases = {ref.split(":")[0]: alias for alias, ref in cell_aliases(bundle.cell_measures).items()}
    production, candidate = (comparison.arm_variants[name] for name in (PRODUCTION, CANDIDATE))
    long_caveat = " ".join(["Only four cases were measured"] + ["and"] * 30) + " so the interval is wide."
    memo = {
        "findings": [
            {
                "title": "Candidate model showed no measurable cost reduction over production",
                "body": (
                    "Findings are numbered F1 onward, and each recommendation names the findings it rests on. "
                    "deepseek-v4-flash answered 2 of 12 trials right. The candidate skipped the queue."
                ),
                "confidence": "low",
                "axes": [],
                "evidence": [
                    {"cell": aliases[production], "measure_id": "accuracy", "reading": "measure"},
                    {"cell": aliases[candidate], "measure_id": "accuracy", "reading": "measure"},
                ],
                "trials": [],
                "caveats": [{"kind": "sampling", "text": long_caveat}],
                "invalidates": [],
                "durable": "",
            }
        ],
        "decisions": [],
        "questions": [
            {
                "question_id": "q-g2",
                "resolution": "unanswerable",
                "answer": "Unanswerable because four cases leave the pass rates statistically unresolved.",
                "rests_on": [0],
            }
        ],
        "next": [],
    }
    client = FixturedClient(json.dumps(memo))
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at="2026-10-14T19:23:41+00:00",
        profile=comparison.host.profile,
    )
    report = build_report(analysis, campaign_name="G2: can the cheaper model replace production?")
    markdown = report_markdown(report)

    # The hedge on a measured fact is code's row instead; the meta-prose and the restated count are gone.
    assert "no measurable" not in markdown and "**F1.** Accuracy: production " in markdown
    assert "numbered F1 onward" not in markdown and "2 of 12 trials right" not in markdown
    assert "The candidate skipped the queue." in markdown
    # The answer that contradicted the settled rule reads the rule's outcome, and the methods say so.
    assert "**Answered.** Keep production: " in markdown
    assert any("“Unanswerable because" in b.text for b in report.blocks if isinstance(b, DisclosureBlock))
    # What code could not fix is kept, counted in the methods table and listed in a fold.
    assert "| Memo style | style: 1 violations |" in markdown
    (listed,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.fold == "Style"]
    assert listed.text.startswith("long_sentence: Only four cases were measured")
    assert lint_page(markdown, question=G2_QUESTION) == []
