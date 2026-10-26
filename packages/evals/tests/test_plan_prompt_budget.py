"""The report-planner prompt stays inside its declared budget, and every rule has its ledger row.

:mod:`threetears.evals.analysis.plan_prompt` declares a ceiling of :data:`PLAN_PROMPT_CHAR_BUDGET` characters and
:data:`PLAN_PROMPT_RULE_BUDGET` rules, and says the budget only falls. This measures the seed prompt as the module
builds it and fails the moment either ceiling is crossed, as ``test_gen_prompt_budget.py`` does for the writer's.

**What is measured.** The seed, :data:`EVAL_REPORT_PLAN_DEFAULT`, has no placeholders, so the text measured here is,
byte for byte, the head of both calls' system prompts, as a generation sends them
(:func:`~threetears.evals.analysis.planner.generate_report_plan`, checked below). What the planner appends after it —
the figure-key grammar, the host's register and the call's own section — is code-owned, and outside the budget.

**What counts as a rule.** A numbered entry under the ``RULES`` heading, numbered 1..n with no gap or repeat.

**Every rule has a ledger row, and every row a rule.** The rules are read from the prompt as (number, heading) pairs,
the heading being the rule's opening sentence; the ledger's rows (``analysis/PLAN_PROMPT_RULES.md``) as (``#``,
``Heading``) pairs, and the two lists must be equal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import threetears.evals.analysis as analysis_package
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.report_plan_support import prose_reply, r5_input, r5_plan, structure_reply
from packages.evals.tests.toyhost_memo import MODEL, FixturedClient, FixturedCompletion
from threetears.evals.analysis.plan_prompt import (
    EVAL_REPORT_PLAN_DEFAULT,
    PLAN_PROMPT_CHAR_BUDGET,
    PLAN_PROMPT_RULE_BUDGET,
)
from threetears.evals.analysis.planner import (
    FACTS_HEADING,
    PROSE_CALL,
    REPORT_INPUT_HEADING,
    STRUCTURE_CALL,
    generate_report_plan,
)
from threetears.evals.analysis.report.figure_keys import GRAMMAR_HEADING
from threetears.evals.analysis.report_input import ReportInput

#: Where the rules start. Everything before it is preamble, not rules.
_RULES_HEADING = "\nRULES — "

#: A rule's opening line: a number at the start of a line, a full stop and a space.
_RULE_START = re.compile(r"^(\d+)\. ", re.MULTILINE)

#: A rule's number and heading: its opening sentence, up to the first full stop that ends a sentence.
_RULE_HEADING = re.compile(r"^(\d+)\. (.*?)\.(?=\s|$)", re.MULTILINE)

#: A ledger row's number and heading, its first two cells.
_LEDGER_ROW = re.compile(r"^\| (\d+) \| ([^|]+?) \|", re.MULTILINE)

#: The rule ledger, beside the prompt.
_LEDGER = Path(analysis_package.__file__).parent / "PLAN_PROMPT_RULES.md"


def _rules_text(prompt: str) -> str:
    assert _RULES_HEADING in prompt, "the prompt lost its RULES heading, so its rules cannot be counted"
    return prompt.split(_RULES_HEADING, 1)[1]


def _prompt_rules(prompt: str) -> list[tuple[int, str]]:
    return [(int(number), heading) for number, heading in _RULE_HEADING.findall(_rules_text(prompt))]


def _ledger_rules(ledger: str) -> list[tuple[int, str]]:
    return [(int(number), heading) for number, heading in _LEDGER_ROW.findall(ledger)]


def _rule_numbers(prompt: str) -> list[int]:
    return [int(number) for number in _RULE_START.findall(_rules_text(prompt))]


def _ledger_disagreements(prompt: str, ledger: str) -> list[str]:
    rules, rows = _prompt_rules(prompt), _ledger_rules(ledger)
    return [f"rule without a ledger row: {rule}" for rule in rules if rule not in rows] + [
        f"ledger row for no rule: {row}" for row in rows if row not in rules
    ]


class _TwoReplies(FixturedClient):
    """A planner answering the structure call, then the prose call, and recording what each was sent."""

    def __init__(self, *bodies: str) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        self.completion = FixturedCompletion(content=self.bodies[len(self.calls)], stop_reason="end_turn")
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


class TestThePromptIsWithinItsBudget:
    def test_it_is_within_the_character_budget(self) -> None:
        length = len(EVAL_REPORT_PLAN_DEFAULT)
        assert length <= PLAN_PROMPT_CHAR_BUDGET, (
            f"the planner prompt is {length} characters, over its budget of {PLAN_PROMPT_CHAR_BUDGET}: tighten "
            "existing wording, or retire a rule, rather than raise the budget (it only falls)"
        )

    def test_it_is_within_the_rule_budget(self) -> None:
        count = len(_rule_numbers(EVAL_REPORT_PLAN_DEFAULT))
        assert count <= PLAN_PROMPT_RULE_BUDGET, (
            f"the planner prompt carries {count} rules, over its budget of {PLAN_PROMPT_RULE_BUDGET}: a new rule names "
            "the one it replaces, in the ledger analysis/PLAN_PROMPT_RULES.md"
        )

    def test_the_rules_are_numbered_one_to_n_so_the_count_is_honest(self) -> None:
        numbers = _rule_numbers(EVAL_REPORT_PLAN_DEFAULT)
        assert numbers == list(range(1, len(numbers) + 1))

    async def test_what_is_measured_is_the_head_of_both_calls_system_prompts(self) -> None:
        """The planner only appends; nothing it adds lands inside the budgeted text. Read off the two calls a
        generation sends, the structure's and the prose's."""
        client = _TwoReplies(json.dumps(structure_reply(r5_plan())), json.dumps(prose_reply(r5_plan())))
        await generate_report_plan(
            ReportInput.model_validate(r5_input()),
            prompt=EVAL_REPORT_PLAN_DEFAULT,
            model=MODEL,
            client=client,
            profile=toyhost_profile(),
        )
        systems = [call["system"] for call in client.calls]
        assert [system.endswith(call) for system, call in zip(systems, (STRUCTURE_CALL, PROSE_CALL))] == [True, True]
        for system in systems:
            assert system.startswith(EVAL_REPORT_PLAN_DEFAULT + "\n\n")

    def test_the_headings_and_sections_it_points_at_are_the_ones_code_sends(self) -> None:
        """The prompt names the user message's parts and the grammar's section by heading; a renamed heading in code
        would leave it pointing at nothing."""
        for heading in (REPORT_INPUT_HEADING, FACTS_HEADING, GRAMMAR_HEADING.upper(), "THIS CALL", "allowed_modes"):
            assert heading in EVAL_REPORT_PLAN_DEFAULT, heading


class TestTheRuleCountBites:
    def test_a_rule_added_past_the_ceiling_is_counted(self) -> None:
        numbers = _rule_numbers(EVAL_REPORT_PLAN_DEFAULT)
        extra = "".join(f"\n\n{n}. AN ADDED RULE. Text." for n in range(len(numbers) + 1, PLAN_PROMPT_RULE_BUDGET + 2))
        assert len(_rule_numbers(EVAL_REPORT_PLAN_DEFAULT + extra)) == PLAN_PROMPT_RULE_BUDGET + 1


class TestEveryRuleHasItsLedgerRow:
    def test_the_ledger_and_the_prompt_list_the_same_rules_in_order(self) -> None:
        ledger = _LEDGER.read_text(encoding="utf-8")
        assert _ledger_disagreements(EVAL_REPORT_PLAN_DEFAULT, ledger) == []
        assert _ledger_rules(ledger) == _prompt_rules(EVAL_REPORT_PLAN_DEFAULT)

    def test_every_counted_rule_has_a_heading_read(self) -> None:
        assert [n for n, _ in _prompt_rules(EVAL_REPORT_PLAN_DEFAULT)] == _rule_numbers(EVAL_REPORT_PLAN_DEFAULT)

    def test_a_rule_added_without_a_row_fails(self) -> None:
        added = len(_rule_numbers(EVAL_REPORT_PLAN_DEFAULT)) + 1
        prompt = EVAL_REPORT_PLAN_DEFAULT + f"\n\n{added}. AN ADDED RULE. Text."
        assert _ledger_disagreements(prompt, _LEDGER.read_text(encoding="utf-8")) == [
            f"rule without a ledger row: {(added, 'AN ADDED RULE')}"
        ]

    def test_a_rule_reworded_under_its_old_row_fails(self) -> None:
        (number, heading), *_ = _prompt_rules(EVAL_REPORT_PLAN_DEFAULT)
        prompt = EVAL_REPORT_PLAN_DEFAULT.replace(f"{number}. {heading}.", f"{number}. CHOOSE SOMETHING ELSE.", 1)
        assert len(_ledger_disagreements(prompt, _LEDGER.read_text(encoding="utf-8"))) == 2
