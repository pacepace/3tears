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
"""

from __future__ import annotations

import re

from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT, PROMPT_CHAR_BUDGET, PROMPT_RULE_BUDGET
from threetears.evals.analysis.generator import assemble_system_prompt

#: Where the rules start. Everything before it is preamble, not rules.
_RULES_HEADING = "\nRULES — "

#: A rule's opening line: a number at the start of a line, a full stop and a space.
_RULE_START = re.compile(r"^(\d+)\. ", re.MULTILINE)


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
