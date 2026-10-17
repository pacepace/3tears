"""The report-writer prompt stays inside its declared budget (#604).

:mod:`threetears.evals.analysis.gen_prompt` declares a ceiling of :data:`PROMPT_CHAR_BUDGET` characters and
:data:`PROMPT_RULE_BUDGET` rules, and says the budget only falls. Rules are each added to fix one failure, so
they accrete; without a check the constants are a wish. This measures the seed prompt as the module builds
it and fails the moment either ceiling is crossed.

**What is measured.** The seed, :data:`EVAL_ANALYSIS_GEN_DEFAULT`, is a constant with no placeholders (its
prompt type does not resolve them), so it has no inputs to vary: the text measured here is, byte for byte,
the head of every system prompt a generation sends (:func:`~threetears.evals.analysis.generator.assemble_system_prompt`,
checked below). What the generator appends after it — the figure-reference grammar, the style register and
the host's extra caveat kinds — is code-owned or host-registered, not this prompt's text, and is outside the
budget the module declares.

**What counts as a rule.** A numbered entry under the ``RULES`` heading (``1. ``, ``2. ``, …). The opening
paragraphs above that heading, and the bullets inside a rule, are not rules. The numbering must run 1..n
with no gap or repeat, so a rule cannot hide from the count by being misnumbered.

**A new rule names the one it replaces.** The budget only falls, so a rule added at the ceiling must retire
or absorb another. The ledger of which rule replaced which, and what each prevents, is the maintainer file
beside the prompt, ``packages/evals/src/threetears/evals/analysis/GEN_PROMPT_RULES.md``.

**Every rule has a ledger row, and every row a rule (#623).** The rules are read from the prompt as (number,
heading) pairs, the heading being the rule's opening sentence; the ledger's rows as (``#``, ``Heading``) pairs.
The two lists must be equal, so a rule added without its reason, a row left for a deleted rule, and a rule
renumbered or reworded under a row that still describes the old one all fail.
"""

from __future__ import annotations

import re
from pathlib import Path

import threetears.evals.analysis as analysis_package
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT, PROMPT_CHAR_BUDGET, PROMPT_RULE_BUDGET
from threetears.evals.analysis.generator import assemble_system_prompt

#: Where the rules start. Everything before it is preamble, not rules.
_RULES_HEADING = "\nRULES — "

#: A rule's opening line: a number at the start of a line, a full stop and a space.
_RULE_START = re.compile(r"^(\d+)\. ", re.MULTILINE)


#: A rule's number and heading: its opening sentence, up to the first full stop that ends a sentence.
_RULE_HEADING = re.compile(r"^(\d+)\. (.*?)\.(?=\s|$)", re.MULTILINE)

#: A ledger row's number and heading, its first two cells.
_LEDGER_ROW = re.compile(r"^\| (\d+) \| ([^|]+?) \|", re.MULTILINE)

#: The rule ledger, beside the prompt.
_LEDGER = Path(analysis_package.__file__).parent / "GEN_PROMPT_RULES.md"


def _prompt_rules(prompt: str) -> list[tuple[int, str]]:
    """Each rule's number and heading, in prompt order."""
    rules = prompt.split(_RULES_HEADING, 1)[1]
    return [(int(number), heading) for number, heading in _RULE_HEADING.findall(rules)]


def _ledger_rules(ledger: str) -> list[tuple[int, str]]:
    """Each ledger row's number and heading, in table order."""
    return [(int(number), heading) for number, heading in _LEDGER_ROW.findall(ledger)]


def _ledger_disagreements(prompt: str, ledger: str) -> list[str]:
    """What the prompt's rules and the ledger's rows disagree on; empty when every rule has its row."""
    rules, rows = _prompt_rules(prompt), _ledger_rules(ledger)
    return [f"rule without a ledger row: {rule}" for rule in rules if rule not in rows] + [
        f"ledger row for no rule: {row}" for row in rows if row not in rules
    ]


def _rule_numbers(prompt: str) -> list[int]:
    """The numbers of the rules after the ``RULES`` heading, in the order they appear."""
    assert _RULES_HEADING in prompt, "the prompt lost its RULES heading, so its rules cannot be counted"
    rules = prompt.split(_RULES_HEADING, 1)[1]
    return [int(number) for number in _RULE_START.findall(rules)]


class TestThePromptIsWithinItsBudget:
    def test_it_is_within_the_character_budget(self) -> None:
        length = len(EVAL_ANALYSIS_GEN_DEFAULT)
        assert length <= PROMPT_CHAR_BUDGET, (
            f"the writer prompt is {length} characters, over its budget of {PROMPT_CHAR_BUDGET}: tighten existing "
            "wording, or retire a rule, rather than raise the budget (it only falls)"
        )

    def test_it_is_within_the_rule_budget(self) -> None:
        count = len(_rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT))
        assert count <= PROMPT_RULE_BUDGET, (
            f"the writer prompt carries {count} rules, over its budget of {PROMPT_RULE_BUDGET}: a new rule names "
            "the one it replaces, in the ledger analysis/GEN_PROMPT_RULES.md"
        )

    def test_the_rules_are_numbered_one_to_n_so_the_count_is_honest(self) -> None:
        numbers = _rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT)
        assert numbers == list(range(1, len(numbers) + 1))

    def test_what_is_measured_is_the_head_of_what_a_generation_sends(self) -> None:
        """The generator only appends; nothing it adds lands inside the budgeted text."""
        sent = assemble_system_prompt(EVAL_ANALYSIS_GEN_DEFAULT, toyhost_profile())
        assert sent.startswith(EVAL_ANALYSIS_GEN_DEFAULT + "\n\n")


class TestTheRuleCountBites:
    """The rule check is only as good as the counting, so the counting is shown to see an added rule."""

    def test_a_rule_added_past_the_ceiling_is_counted(self) -> None:
        numbers = _rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT)
        extra = "".join(f"\n\n{n}. AN ADDED RULE. Text." for n in range(len(numbers) + 1, PROMPT_RULE_BUDGET + 2))
        assert len(_rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT + extra)) == PROMPT_RULE_BUDGET + 1


class TestEveryRuleHasItsLedgerRow:
    """The ledger names each rule by its number and heading, and the check fails on any drift (#623)."""

    def test_the_ledger_and_the_prompt_list_the_same_rules_in_order(self) -> None:
        ledger = _LEDGER.read_text(encoding="utf-8")
        assert _ledger_disagreements(EVAL_ANALYSIS_GEN_DEFAULT, ledger) == []
        assert _ledger_rules(ledger) == _prompt_rules(EVAL_ANALYSIS_GEN_DEFAULT)

    def test_every_counted_rule_has_a_heading_read(self) -> None:
        assert [n for n, _ in _prompt_rules(EVAL_ANALYSIS_GEN_DEFAULT)] == _rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT)

    def test_a_rule_added_without_a_row_fails(self) -> None:
        added = len(_rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT)) + 1
        prompt = EVAL_ANALYSIS_GEN_DEFAULT + f"\n\n{added}. AN ADDED RULE. Text."
        assert _ledger_disagreements(prompt, _LEDGER.read_text(encoding="utf-8")) == [
            f"rule without a ledger row: {(added, 'AN ADDED RULE')}"
        ]

    def test_a_rule_deleted_with_its_row_left_behind_fails(self) -> None:
        last = len(_rule_numbers(EVAL_ANALYSIS_GEN_DEFAULT))
        prompt = EVAL_ANALYSIS_GEN_DEFAULT.split(f"\n\n{last}. ", 1)[0]
        (disagreement,) = _ledger_disagreements(prompt, _LEDGER.read_text(encoding="utf-8"))
        assert disagreement.startswith(f"ledger row for no rule: ({last}, ")

    def test_a_rule_reworded_under_its_old_row_fails(self) -> None:
        (number, heading), *_ = _prompt_rules(EVAL_ANALYSIS_GEN_DEFAULT)
        prompt = EVAL_ANALYSIS_GEN_DEFAULT.replace(f"{number}. {heading}.", f"{number}. RANK ON SOMETHING ELSE.", 1)
        assert len(_ledger_disagreements(prompt, _LEDGER.read_text(encoding="utf-8"))) == 2
