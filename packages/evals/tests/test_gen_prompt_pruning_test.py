"""The writer prompt states the per-sentence pruning test once, and rule 13's cut applies it (#682).

Before, the prompt applied the test item by item -- a finding or next step "only if a decision rests on it",
delete "any sentence that repeats another" -- and never stated it, so a sentence of a kind the list did not
name passed. These pin that the test, and the reader it is applied for, are stated as a general rule near
"WRITE LESS", and that rule 13's cut points back at it rather than standing as the whole of it.

Whether the rule changes what the writer produces is for a reporter eval to measure, not for these: they pin
the seed's text, as the other prompt-text tests do.
"""

from __future__ import annotations

from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT

#: The test itself, as the prompt words it.
_TEST = "Every sentence must pass one test: does it change what that reader does or believes? If not, cut it"


def _preamble() -> str:
    """The prompt before its numbered rules."""
    return EVAL_ANALYSIS_GEN_DEFAULT.split("\nRULES — ", 1)[0]


def _write_less() -> str:
    """The paragraph that opens with WRITE LESS."""
    return next(p for p in EVAL_ANALYSIS_GEN_DEFAULT.split("\n\n") if p.startswith("WRITE LESS"))


def _rule_13() -> str:
    """Rule 13's text, up to rule 14."""
    return EVAL_ANALYSIS_GEN_DEFAULT.split("\n\n13. ", 1)[1].split("\n\n14. ", 1)[0]


def test_the_test_is_stated_once_beside_write_less_as_a_rule_for_every_sentence() -> None:
    assert _TEST in _write_less()
    assert EVAL_ANALYSIS_GEN_DEFAULT.count(_TEST) == 1
    assert "whatever kind of sentence it is" in _write_less()
    assert _TEST in _preamble(), "the test is a general rule, not one rule's clause"


def test_the_paragraph_names_the_reader_the_test_is_applied_for() -> None:
    paragraph = _write_less()
    reader = "Your reader decides what to change next"
    assert reader in paragraph
    assert paragraph.index(reader) < paragraph.index(_TEST), "the reader is named before the test that serves them"


def test_rule_13s_cut_applies_the_test_rather_than_standing_as_the_whole_of_it() -> None:
    rule = _rule_13()
    assert "Then cut, by the test under WRITE LESS." in rule
    # The old wording made the list the rule: "Delete any sentence that repeats another, ...".
    assert "Delete any sentence" not in rule
    assert "What usually fails it:" in rule
    # Kinds the old list did not name, which the general test now catches, are named as applications.
    for kind in ("a restated method", "a narrated chart", "a hedge in a finding's body"):
        assert kind in rule
