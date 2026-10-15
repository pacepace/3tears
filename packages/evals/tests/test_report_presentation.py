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
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    ArmServedModel,
    Report,
    build_code_only_report,
    build_report,
    generate_analysis,
    report_html,
    report_markdown,
)
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.analysis.report import DisclosureBlock, TableBlock, TextBlock, TrialBlock
from threetears.evals.analysis.report.decision import RateReading, RuleReadings, evaluate_rule
from threetears.evals.analysis.report.memo_style import MemoStyle, decision_disclosures
from threetears.evals.analysis.bundle.schema import TrialCheck, TrialJudgement
from threetears.evals.analysis.style import GLOSSARY, arm_name, lint_page
from packages.evals.tests.g2_support import (
    CANDIDATE,
    CASES,
    G2_QUESTION,
    G2_RULE,
    PRODUCTION,
    answering,
    declaring,
    g2_numbers,
    g2_report,
    g2_ruled_report,
)
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient


#: The G2 rule as the page states it: one sentence code writes from the typed clauses, the arms by name.
G2_STATEMENT = (
    "Switch to cheaper candidate only if no pass rate drops by more than 10 points and cheaper candidate costs less "
    "per reply. Otherwise stay on production."
)


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
    # The title is the campaign's name; the question declared before the run is quoted whole on its own line under
    # it, the contract the verdict reads; then the subhead says what the question need not: each arm with its
    # version, and the typed rule in code's sentence. Then the verdict.
    assert lines[0] == "# G2: can the cheaper model replace production?"
    assert lines[1] == f'"{G2_QUESTION}"'
    assert lines[3] == (
        "deepseek-v4.1-flash (production, control) vs deepseek-v4-flash (cheaper candidate) · 14 Oct 2026 · "
        "4 cases × 3 repeats"
    )
    assert lines[5] == G2_STATEMENT
    # Said once: the Decision section opens on its clauses, whose title links up to the subhead.
    assert "\n## Decision\n\n**Clauses**\n" in report_markdown(report)
    assert '<h3><a href="#rule">Clauses</a></h3>' in report_html(report)
    assert f'<p class="arms" id="rule">{html.escape(G2_STATEMENT)}</p>' in report_html(report)
    assert lines[7].startswith("**Keep production: ")
    # Every arm's cost per reply is under a cent, so the cost is stated per 1,000 replies, and the header says so.
    assert lines[9:12] == [
        "**Scoreboard**",
        "",
        "| Arm | Ran | Usable | Passed | Failed by check | Failed by judge | Cost per 1,000 replies |",
    ]
    assert lines[13] == "| ▸ production | 12 | 12 | 10 | 2 | 0 | $3.092 |"
    first = report_markdown(report).split("\n## ", 1)[0]
    # The rule settled keep on the counts, so the floor caveat says what is withheld under four cases and why, in
    # the reader's terms; then the one other fact that changes how the verdict reads, the label its row contradicts.
    # Nothing else is under the scoreboard: the design is on line 2.
    caveat = (
        "4 cases per arm: keep follows from the counts above. Whether the gap holds on more cases is not estimated: "
        "an interval takes 5 independent cases."
    )
    label = "'cheaper candidate' is not cheaper: it cost more per reply than production."
    assert caveat in first.splitlines(), "the case floor, in normal weight"
    assert "Too few" not in first and "needs 5" not in first
    assert first.index("**Keep production") < first.index("**Scoreboard**") < first.index(caveat) < first.index(label)
    assert first.rstrip().endswith(label), "nothing under the scoreboard but the caveat and the label's claim"
    assert "The label" not in report_markdown(report)


async def test_the_subhead_names_each_arm_by_the_version_that_answered_it(g2_ruled: tuple[Result, Report]) -> None:
    """An arm asked for an alias and answered by one dated version is named by that version on the arms line; an
    arm whose responses named the model it asked for, or several, keeps the name it asked by."""
    comparison, report = g2_ruled
    production, candidate = (comparison.arm_variants[name] for name in (PRODUCTION, CANDIDATE))
    served = [
        ArmServedModel(
            variant_key=production,
            served_models=["deepseek/deepseek-v4.1-flash"],
            n_results=12,
            n_unrecorded=0,
            state="one",
        ),
        ArmServedModel(
            variant_key=candidate,
            served_models=["deepseek/deepseek-v4-flash-20261001"],
            n_results=12,
            n_unrecorded=0,
            state="one",
        ),
    ]
    bundle = report.bundle.model_copy(update={"arm_served_models": served})
    lines = report_markdown(build_code_only_report(bundle, assembled_at="2026-10-14T00:00Z")).splitlines()
    assert lines[3].startswith(
        "deepseek-v4.1-flash (production, control) vs deepseek-v4-flash-20261001 (cheaper candidate) · "
    )


async def test_an_outcome_bar_repeating_the_scoreboard_prints_no_second_table(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    # Passed, failed by check and failed by judge are the scoreboard's own columns: the bar's table would repeat it.
    assert "**Outcomes**" not in report_markdown(report)
    # The page still draws the bar.
    assert report_html(report).count('data-chart="outcome_bar"') == 1


async def test_the_question_is_quoted_once_under_the_title_and_never_in_the_body(
    g2_ruled: tuple[Result, Report],
) -> None:
    _, report = g2_ruled
    markdown = report_markdown(report)
    assert markdown.count("does deepseek-v4-flash hold every pass/fail queue-discipline dimension") == 1
    assert markdown.count(G2_STATEMENT) == 1
    (asked,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "basis"]
    (rule,) = [
        b
        for b in report.blocks
        if isinstance(b, DisclosureBlock) and b.source == "rule" and b.text.startswith("Switch")
    ]
    assert asked.section == "header" and asked.text == f'"{G2_QUESTION}"'
    assert rule.section == "header" and rule.text == G2_STATEMENT
    assert f'<p class="question">{html.escape(asked.text)}</p>' in report_html(report)
    assert lint_page(markdown, question=G2_QUESTION) == []


async def test_a_raw_model_id_is_named_at_most_once_per_section(g2_ruled: tuple[Result, Report]) -> None:
    _, report = g2_ruled
    # The verdict sentence aside: it names each arm by its model, since a label beside "cost more" can contradict it.
    # The quoted question aside too, the author's words whole, and the rule's sentence beside it.
    (verdict,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    (asked,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "basis"]
    (rule,) = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and b.text == G2_STATEMENT]
    quoted = report_markdown(report).replace(verdict, "").replace(asked, "").replace(rule, "")
    sections = re.split(r"\n## ", quoted) + [
        _visible_text(part).replace(verdict, "").replace(asked, "").replace(rule, "")
        for part in report_html(report).split("<h2>")
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


async def test_every_link_on_the_page_lands_on_one_id_the_page_has() -> None:
    # The only links are ``#`` anchors, and a passed trial, which has no record, is reached at its table row.
    for report in await _pages():
        page = report_html(report)
        linked = set(re.findall(r'href="#([^"]+)"', page))
        ids = re.findall(r'\sid="([^"]+)"', page)
        assert linked and linked <= set(ids), f"{report.title}: dangling {sorted(linked - set(ids))}"
        assert len(ids) == len(set(ids)), f"{report.title}: an id twice"


async def test_the_glossary_lists_only_what_the_page_uses_and_explains_the_control_mark(
    g2_ruled: tuple[Result, Report],
) -> None:
    _, report = g2_ruled
    # The question was declared, so the page offers no "exploratory" leads, and the glossary does not define it.
    assert "exploratory" not in report.terms
    # The ▸ in the scoreboard and the cases table is defined with the control.
    assert "control" in report.terms and "▸" in GLOSSARY["control"]
    # Settled says what settles each way, never that the outcome cannot change.
    assert "cannot change" not in GLOSSARY["settled"] and "interval" in GLOSSARY["settled"]


async def test_one_arm_reads_n_of_n_cases_passed_first() -> None:
    one = await Eval(CASES, {"production": answering(10, 0.003)}, ["label"], k=3, name="queue").run(store=":memory:")
    (verdict,) = [b for b in one.report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert re.fullmatch(r"queue: \d/4 cases passed; \d miss(es)?: case \d(, case \d)*", verdict.text), verdict.text


async def test_one_arm_heads_its_readings_as_readings_since_nothing_is_contrasted() -> None:
    one = await Eval(CASES, {"production": answering(10, 0.003)}, ["label"], k=3, name="queue").run(store=":memory:")
    assert one.report.section_titles["contrasts"] == "Readings"
    markdown = report_markdown(one.report)
    assert "\n## Readings\n" in markdown and "## Contrasts" not in markdown
    page = report_html(one.report)
    assert '<section id="contrasts">\n<h2>Readings</h2>' in page and '<a href="#contrasts">Readings</a>' in page


async def test_a_case_its_author_named_is_named_so_everywhere_and_numbered_otherwise() -> None:
    named = [{**case, "id": f"queue-{index}"} for index, case in enumerate(CASES)]
    one = await Eval(named, {"production": answering(10, 0.003)}, ["label"], k=3, name="queue").run(store=":memory:")
    report = one.report
    (verdict,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.source == "verdict"]
    assert re.fullmatch(r"queue: \d/4 cases passed; \d miss(es)?: queue-\d(, queue-\d)*", verdict.text), verdict.text
    (cases,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "cases"]
    assert sorted(row["case"] for row in cases.values()) == [f"queue-{index}" for index in range(4)]
    # Only a failed or lost trial has a record, so its summary is read off the first record the page lays out; a
    # passed trial names its case in the closed table instead.
    trials = [block for block in report.blocks if isinstance(block, TrialBlock)]
    assert trials and all(re.fullmatch(r"queue-\d", trial.case_label) for trial in trials)
    first = trials[0]
    assert f"{first.trial_id} · {first.case_label} · production · {first.state}" in report_markdown(report)
    (passed,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "passed_trials"]
    assert all(re.fullmatch(r"queue-\d", str(row["case"])) for row in passed.values())
    # The row anchors by the name; a numbered case keeps anchoring by its number.
    assert '<tr id="case-queue-0">' in report_html(report) and 'id="case-1"' not in report_html(report)
    comparison = await Eval(CASES, {"a": answering(4, 0.001), "b": answering(1, 0.001)}, ["label"], k=2).run(
        store=":memory:"
    )
    assert '<tr id="case-1">' in report_html(comparison.report)


def _passing(passes: list[int]) -> Any:
    """A runner that answers case ``i`` right on ``passes[i]`` of its repeats, the first ones."""
    calls: Counter[str] = Counter()

    async def run(case: Mapping[str, Any]) -> Answer:
        calls[case["text"]] += 1
        right = calls[case["text"]] <= passes[int(case["text"].rsplit(" ", 1)[1])]
        return Answer(case["label"] if right else "skipped", model="m", input_tokens=1, output_tokens=1, cost_usd=0.001)

    return run


async def test_the_flag_reads_regression_only_where_a_candidate_passed_a_case_less_often_than_the_control() -> None:
    # Prod, .30: production ✓✗✗ against the candidate's ✓✓✗ (the candidate did better), and 2 of 3 each (a tie),
    # both read "regression". Neither is one; a candidate passing 1 of 3 to the control's 3 of 3 is.
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(3)]
    arms = {"production": _passing([1, 2, 3]), "candidate": _passing([2, 2, 1])}
    comparison = await Eval(cases, arms, ["label"], k=3, name="flags").run(store=":memory:")
    (table,) = [b for b in comparison.report.blocks if isinstance(b, TableBlock) and b.name == "cases"]
    flags = {str(row["case"]): row["flag"] for row in table.values()}
    assert flags == {"case 3": "regression", "case 1": None, "case 2": None}
    assert next(iter(flags)) == "case 3", "regressions first"
    (flag,) = [column for column in table.columns if column.key == "flag"]
    assert flag.term == "flag" and "flag" in comparison.report.terms
    assert GLOSSARY["flag"] == (
        "Regression marks a case a candidate passed on a smaller share of its usable trials than the control; a tie "
        "or a gain never flags."
    )


# =============================================================================
# Trials: the census, then only what went wrong, each record under a summary that is never the world dump
# =============================================================================


def _trials_section(report: Report) -> tuple[DisclosureBlock, TableBlock | None, list[TrialBlock], TableBlock | None]:
    """The census line, the failed-and-lost table, the records and the passed table, as the page lays them out."""
    blocks = [block for block in report.blocks if block.section == "trials"]
    census = blocks[0]
    assert isinstance(census, DisclosureBlock)
    wrong = [b for b in blocks if isinstance(b, TableBlock) and b.name == "trials"]
    passed = [b for b in blocks if isinstance(b, TableBlock) and b.name == "passed_trials"]
    records = [b for b in blocks if isinstance(b, TrialBlock)]
    # The census, the table of what went wrong, the records, then the passed trials folded: in that order.
    kinds = [("census" if b is census else b.name if isinstance(b, TableBlock) else "record") for b in blocks]
    assert kinds == [
        "census",
        *(["trials"] if wrong else []),
        *(["record"] * len(records)),
        *(["passed_trials"] if passed else []),
    ]
    return census, wrong[0] if wrong else None, records, passed[0] if passed else None


async def test_the_trials_section_lists_only_what_went_wrong_and_folds_the_passed(
    g2_ruled: tuple[Result, Report],
) -> None:
    _, report = g2_ruled
    census, table, records, passed = _trials_section(report)
    assert census.text == "24 trials ran. Of them 12 passed and 12 failed."
    failed = [trial for trial in report.bundle.trials if trial.state == "failed"]
    assert table is not None and [row["trial"] for row in table.values()] == [trial.trial_id for trial in failed]
    assert {row["outcome"] for row in table.values()} == {"failed"}
    assert [block.trial_id for block in records] == [trial.trial_id for trial in failed]
    assert all(block.open for block in records), "twelve failed, every record open"
    assert passed is not None and passed.details and passed.title == "Passed trials"
    assert [column.header for column in passed.columns] == ["Trial", "Case", "Arm", "Stored result"]
    assert [row["result"] for row in passed.values()] == [
        f"`{trial.result_id}`" for trial in report.bundle.trials if trial.state == "passed"
    ]
    # The cases table's links: the failed ids first, then the passed, so a row's marks lead to its records.
    (cases,) = [b for b in report.blocks if isinstance(b, TableBlock) and b.name == "cases"]
    states = {trial.trial_id: trial.state for trial in report.bundle.trials}
    for row in cases.values():
        listed = [states[trial_id] for trial_id in str(row["trials"]).split()]
        assert listed == sorted(listed, key=lambda state: state == "passed"), row["trials"]
    # Both doors: the records open, the passed table a closed fold with its table set off by blank lines.
    markdown = report_markdown(report)
    assert f'<details id="{failed[0].trial_id}" open><summary>' in markdown
    assert "<details><summary>Passed trials</summary>\n\n| Trial | Case | Arm | Stored result |" in markdown
    assert "|\n\n</details>" in markdown
    page = report_html(report)
    assert page.count('<td class="trial-links">') == len(cases.rows)
    assert f'<details class="trial" id="{failed[0].trial_id}" data-state="failed" open>' in page
    assert page.count('<details data-table="passed_trials"><summary>Passed trials</summary>') == 1


async def test_the_records_open_up_to_a_cap_and_the_lost_follow_the_failed() -> None:
    every = await Eval(CASES, {"a": answering(0, 0.001), "b": answering(0, 0.001)}, ["label"], k=3).run(
        store=":memory:"
    )
    census, table, records, passed = _trials_section(every.report)
    assert census.text == "24 trials ran. Of them 0 passed and 24 failed."
    assert table is not None and len(table.rows) == 24 and passed is None
    assert [block.open for block in records] == [True] * 12 + [False] * 12
    assert report_markdown(every.report).count("Passed trials") == 0
    # G2 as it ran: every usable trial passed, and the lost ones are listed, each with its cause.
    census, table, records, passed = _trials_section(await g2_report())
    assert census.text == "24 trials ran. Of them 10 passed, 0 failed and 14 were lost."
    assert table is not None and {row["outcome"] for row in table.values()} == {"lost"}
    assert all(str(row["why"]).startswith("lost: ") for row in table.values())
    assert passed is not None and len(passed.rows) == 10
    assert len(records) == 14 and sum(block.open for block in records) == 12


async def test_when_every_trial_passed_the_census_line_stands_alone() -> None:
    clean = await Eval(CASES, {"a": answering(99, 0.001)}, ["label"], k=1).run(store=":memory:")
    census, table, records, passed = _trials_section(clean.report)
    assert census.text == "4 trials ran. Every one passed."
    assert table is None and records == [] and passed is not None and len(passed.rows) == 4
    markdown = report_markdown(clean.report)
    assert "| Trial | Case | Arm | Outcome | Why |" not in markdown
    assert markdown.count("<summary>Passed trials</summary>") == 1
    assert lint_page(markdown, names={"a"}) == []


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


#: A trial's transcript as a host's world renders it: its own headings, a fenced block and raw HTML of its own.
_TRANSCRIPT = (
    'The world as it stood before the persona\'s first turn:\n\n## music\n```json\n{"queue": []}\n```\n\n'
    "# Persona\n## Ben\n</details>\n<script>alert(1)</script>\n````\nnot a fence of mine\n````"
)


def _outline(markdown: str) -> list[str]:
    """The headings a CommonMark reader finds: ATX headings outside fenced code, each fence closed only by a run of its
    own character at least as long as the one that opened it."""
    headings: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.lstrip(" ")
        opening = re.match(r"(`{3,}|~{3,})", stripped) if len(line) - len(stripped) <= 3 else None
        if fence is not None:
            if opening and opening.group(1)[0] == fence[0] and len(opening.group(1)) >= len(fence):
                if not stripped[len(opening.group(1)) :].strip():
                    fence = None
            continue
        if opening:
            fence = opening.group(1)
        elif re.match(r" {0,3}#{1,6}(\s|$)", line):
            headings.append(line.strip())
    return headings


async def test_a_transcript_is_quoted_whole_and_none_of_it_is_the_report_s_structure(
    g2_ruled: tuple[Result, Report],
) -> None:
    _, coded = g2_ruled
    trials = [
        trial.model_copy(update={"first_input": _TRANSCRIPT, "excerpt": _TRANSCRIPT, "error_words": _TRANSCRIPT})
        if trial.state == "failed"
        else trial
        for trial in coded.bundle.trials
    ]
    report = build_code_only_report(
        coded.bundle.model_copy(update={"trials": trials}), assembled_at="2026-10-14T00:00Z"
    )
    markdown = report_markdown(report)
    titles = report.section_titles
    own = [f"# {report.title}", *(f"## {titles[s]}" for s in titles if f"\n## {titles[s]}\n" in markdown)]
    assert _outline(markdown) == own
    # Each transcript sits whole inside a fence longer than any run of backticks in it.
    assert "`````\nThe world as it stood before the persona's first turn:" in markdown
    assert markdown.count("not a fence of mine") == 3 * sum(trial.state == "failed" for trial in trials)
    # The page escapes the same text: in a <pre>, its tags and headings text, and no heading but the page's own.
    page = report_html(report)
    assert "<script>alert" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<pre>The world as it stood before the persona&#x27;s first turn:\n\n## music" in page
    assert not re.search(r"<h[1-6][^>]*>[^<]*(music|Persona|Ben)", page)


async def test_an_author_s_words_open_no_block_of_their_own_in_markdown(g2_ruled: tuple[Result, Report]) -> None:
    _, coded = g2_ruled
    report = build_code_only_report(coded.bundle, assembled_at="2026-10-14T00:00Z")
    body = "## music\n\n- # Persona\n- plain\n\n```\n\n> quoted\n\n1. first\n\n<details>"
    words = report.model_copy(
        update={"blocks": [*report.blocks, TextBlock(section="not_tested", role="finding_body", body=body, finding=0)]}
    )
    markdown = report_markdown(words)
    assert "\\## music\n\n- \\# Persona\n- plain\n\n\\```\n\n\\> quoted\n\n1\\. first\n\n\\<details>" in markdown
    assert _outline(markdown) == _outline(report_markdown(report))


async def test_sixty_passed_trials_and_one_failed_carry_one_record_and_one_table() -> None:
    cases = [{"text": f"queue request {index}", "label": "queued"} for index in range(61)]
    run = await Eval(cases, {"production": answering(60, 0.003)}, ["label"], k=1, name="queue").run(store=":memory:")
    census, table, records, passed = _trials_section(run.report)
    assert census.text == "61 trials ran. Of them 60 passed and 1 failed."
    assert table is not None and len(table.rows) == 1 and len(records) == 1
    assert passed is not None and len(passed.rows) == 60
    markdown = report_markdown(run.report)
    assert markdown.count('<details id="T') == 1
    passed_rows = markdown.split("<summary>Passed trials</summary>", 1)[1].split("</details>", 1)[0]
    assert len(re.findall(r"^\| T\d+ \| .* \| `[0-9a-f-]+` \|$", passed_rows, re.M)) == 60
    assert "What the case asked" not in passed_rows and "Stored result:" not in passed_rows


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


async def test_what_a_trial_quotes_leaves_the_document_s_structure_intact(g2_ruled: tuple[Result, Report]) -> None:
    """The judge's words, a check's expression and why a trial failed are quoted: a ``</details>`` the judge wrote
    closes no fold, and a backtick in an expression ends no code span."""
    _, coded = g2_ruled
    expression = "answer == `queued`"
    reason = "Closed early.</details>\n\n## Not a section"
    trials = list(coded.bundle.trials)
    first = next(index for index, trial in enumerate(trials) if trial.state == "failed")
    trials[first] = trials[first].model_copy(
        update={
            "failed_checks": [TrialCheck(name="the *label* check", expression=expression)],
            "judge": [
                TrialJudgement(dim="tone", criterion="Is it <b>kind</b>?", score=1, scale="ordinal", reason=reason)
            ],
        }
    )
    report = build_code_only_report(
        coded.bundle.model_copy(update={"trials": trials}), assembled_at="2026-10-14T00:00Z"
    )
    markdown = report_markdown(report)
    assert "- `the *label* check`: `` answer == `queued` ``" in markdown
    assert "- `tone`: 1 (ordinal). Asked: `Is it <b>kind</b>?`. `Closed early.</details>  ## Not a section`" in markdown
    # Outside every code span and fence, the folds still open and close in pairs, and each record still ends after
    # its stored result.
    unquoted = re.sub(r"(`+)(?!`).+?(?<!`)\1(?!`)", "", markdown)
    assert unquoted.count("<details") == unquoted.count("</details>")
    record = unquoted.split(f'<details id="{trials[first].trial_id}"', 1)[1].split("</details>", 1)[0]
    assert "Stored result:" in record
    assert _outline(markdown) == _outline(report_markdown(build_code_only_report(coded.bundle, assembled_at="x")))


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
    verdict = evaluate_rule(
        G2_RULE,
        RuleReadings(
            rates=(RateReading("the pass rate", 10, 12, 2, 12),), control_cost=0.003092, candidate_cost=0.003301
        ),
        control="production",
        candidate="cheaper candidate",
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
    # A role word already beside the id collapses with it into the label once: never "production production".
    assert style.names_arms(
        "On the proven check requiring at least two queued tracks, deepseek-v4-flash passed 6 of 12 of runs "
        "compared to 11 of 12 for production deepseek-v4.1-flash."
    ) == (
        "On the proven check requiring at least two queued tracks, cheaper candidate passed 6 of 12 of runs "
        "compared to 11 of 12 for production."
    )
    assert (
        style.names_arms("deepseek-v4.1-flash (production) kept the queue full.") == "production kept the queue full."
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
            },
            {
                "title": "Candidate model showed no measurable cost reduction over production",
                "body": "The candidate searched the catalog more.",
                "confidence": "low",
                "axes": [],
                "evidence": [
                    {"cell": aliases[production], "measure_id": "production_replicating_cost", "reading": "measure"},
                    {"cell": aliases[candidate], "measure_id": "production_replicating_cost", "reading": "measure"},
                ],
                "trials": [],
                "caveats": [],
                "invalidates": [],
                "durable": "",
            },
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
    assert "no measurable" not in markdown and "**Accuracy: cheaper candidate " in markdown
    assert "**F1." not in markdown
    # A title restating its one reading is code's plain sentence, the candidate first, money in dollars on the
    # scale the scoreboard states it.
    assert "**Cheaper candidate cost $3.301 per 1,000 replies to production's $3.092" in markdown
    assert " usd" not in markdown
    assert "numbered F1 onward" not in markdown and "2 of 12 trials right" not in markdown
    assert "The candidate skipped the queue." in markdown
    # The rule settled every clause, so the verdict line and the decision table answer its question: the page prints
    # no answer line for it, and the methods still say the memo's answer contradicted the rule.
    assert "Answered." not in markdown and "See the decision rule above." not in markdown
    assert "Unanswerable because" not in markdown.split("## Methods", 1)[0]
    assert markdown.count("Keep production: ") == 1
    assert any("“Unanswerable because" in b.text for b in report.blocks if isinstance(b, DisclosureBlock))
    # What code could not fix is kept, counted in the methods table and listed in a fold.
    assert "| Memo style | style: 1 violations |" in markdown
    (listed,) = [b for b in report.blocks if isinstance(b, DisclosureBlock) and b.fold == "Style"]
    assert listed.text.startswith("long_sentence: Only four cases were measured")
    assert lint_page(markdown, question=G2_QUESTION) == []
