"""``calls()``: a goal check reads what the subject passed, and never matches the text it wrote.

A call's recorded parameters mix closed values (a list position, an item id) with text the model
wrote (what a spoken line says). The engine cannot tell them apart and does not try: the action's
own parameter schema — the one the model is shown — says which values are closed, and every other
string is free text a check may test for presence and length only. Each refusal below has an
accepted sibling over the SAME schema, so an inverted rule cannot pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from threetears.evals.schema.goal_grammar import DSLError, extract_text_matches, parse
from threetears.evals.kernel.dsl import (
    call_parameter_matches,
    evaluate,
    undefined_call_references,
    world_prose_matches,
)
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.run.authoring import create_template
from threetears.evals.schema.call_ledger import CallLedger
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND


_SAY = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "position": {"type": "string", "pattern": "^(head|tail|[0-9]+)$"},
        "mood": {"type": "string", "enum": ["warm", "dry"]},
        "kind": {"const": "speech"},
        "speed": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "labels": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
        "note": {"description": "no type declared"},
    },
}


def _reader(tool: str, action: str):
    return _SAY if (tool, action) == ("speech", "say") else None


def _state(*calls: tuple[str, str, dict]) -> dict:
    """The keyword arguments ``evaluate`` takes for a cell that made ``calls`` and holds no world."""
    ledger = CallLedger()
    for tool, action, params in calls:
        ledger.record(tool, action, params)
    return {"end_state": {}, "ledger": ledger, "world": None}


def _refusals(expression: str, reader=_reader) -> list[str]:
    return [reason for _, reason in call_parameter_matches(expression, reader)]


class TestEvaluation:
    def test_returns_each_matching_calls_parameters_in_ledger_order(self):
        state = _state(
            ("speech", "say", {"text": "one", "position": "head"}),
            ("shop", "add_item", {"item_ref": "r1"}),
            ("speech", "say", {"text": "two"}),
        )

        assert evaluate('calls("speech.say").length == 2', **state)
        assert evaluate('calls("speech.say")[0].position == "head"', **state)
        assert evaluate('any(it.position == "head" for it in calls("speech.say"))', **state)
        assert not evaluate('all(it.position == "head" for it in calls("speech.say"))', **state)

    def test_a_length_test_reads_presence_across_every_call(self):
        state = _state(("shop", "add_item", {"note_text": "here"}), ("shop", "add_item", {"note_text": ""}))

        assert not evaluate('all(it.note_text.length > 0 for it in calls("shop.add_item"))', **state)
        assert evaluate('any(it.note_text.length > 0 for it in calls("shop.add_item"))', **state)

    def test_no_matching_call_is_an_empty_list(self):
        state = _state(("shop", "search", {}))

        assert evaluate('calls("speech.say").length == 0', **state)
        assert not evaluate('any(it.position == "head" for it in calls("speech.say"))', **state)

    def test_the_expression_cannot_reach_the_ledger(self):
        state = _state(("speech", "say", {"text": "one"}))
        evaluate('calls("speech.say")[0].text == "one"', **state)

        assert state["ledger"].calls[0].params == {"text": "one"}

    @pytest.mark.parametrize(
        "expression",
        [
            "calls(variation.action).length > 0",
            "calls().length > 0",
            'calls("speech.say", "x").length > 0',
            'calls("speechsay").length > 0',
        ],
    )
    def test_a_spec_the_gate_could_not_name_is_refused_at_parse(self, expression):
        """A computed or dotless spec would leave the gate unable to find the action's schema."""
        with pytest.raises(DSLError):
            parse(expression)


class TestTheExtractor:
    def test_it_bound_to_calls_reports_the_parameter_and_the_action(self):
        (match,) = extract_text_matches('any(it.position == "head" for it in calls("speech.say"))')

        assert (match.call, match.operand) == ("speech.say", ("position",))

    def test_a_subscripted_call_reports_the_parameter(self):
        (match,) = extract_text_matches('calls("speech.say")[-1].text == "hi"')

        assert (match.call, match.operand) == ("speech.say", ("text",))

    def test_a_count_or_length_is_not_a_text_match(self):
        assert extract_text_matches('calls("speech.say").length >= 1') == ()
        assert extract_text_matches('all(it.text.length > 0 for it in calls("speech.say"))') == ()
        assert extract_text_matches('all(it.text != "" for it in calls("speech.say"))') == ()

    def test_the_world_gate_leaves_call_matches_to_the_call_gate(self):
        """A call parameter is not a world path, so the world vocabulary has no say over it."""
        assert world_prose_matches('any(it.text == "hi" for it in calls("speech.say"))', None) == ()


class TestTheCallParameterGate:
    @pytest.mark.parametrize("parameter", ["position", "mood", "kind", "speed", "labels"])
    def test_a_value_the_schema_closes_may_be_compared(self, parameter):
        assert _refusals(f'any(it.{parameter} == "x" for it in calls("speech.say"))') == []

    @pytest.mark.parametrize("parameter", ["text", "tags", "note"])
    def test_free_text_may_not_be_compared(self, parameter):
        assert _refusals(f'any(it.{parameter} == "x" for it in calls("speech.say"))') == [
            f"speech.say's {parameter} is free text"
        ]

    def test_contains_over_free_text_is_refused_and_over_a_closed_list_is_not(self):
        assert len(_refusals('any(contains(it.text, "hi") for it in calls("speech.say"))')) == 1
        assert _refusals('any(contains(it.labels, "a") for it in calls("speech.say"))') == []

    def test_free_text_may_be_tested_for_presence_and_length(self):
        assert _refusals('all(it.text.length > 0 for it in calls("speech.say"))') == []
        assert _refusals('all(it.text != "" for it in calls("speech.say"))') == []

    def test_an_undeclared_parameter_is_refused(self):
        assert _refusals('any(it.volume == "x" for it in calls("speech.say"))') == [
            "speech.say declares no parameter volume"
        ]

    def test_an_undescribed_action_is_refused(self):
        assert _refusals('any(it.position == "head" for it in calls("speech.shout"))') == [
            "this host describes no parameters for speech.shout"
        ]

    def test_a_host_with_no_reader_refuses_every_comparison_and_no_length_test(self):
        assert len(_refusals('any(it.position == "head" for it in calls("speech.say"))', reader=None)) == 1
        assert _refusals('calls("speech.say").length >= 1', reader=None) == []


def _create_template(definition: dict) -> object:
    """Author through the engine's own gate, with every host-side check a no-op.

    The refusals below are the engine's: the host hooks ``create_template`` takes are for
    what only a host can know (its tool catalog, its world's seed walk, its kinds), and passing
    none isolates the rule under test from any one host's answers.
    """
    storage = MagicMock()
    storage.load_template_by_name.return_value = None
    return create_template(
        toyhost_host(storage=storage),
        definition,
        scope_id="scope-a",
        require_known_tools_allowed=lambda _tools: None,
        refuse_undeclared_world_seed=lambda _template: None,
        refuse_undeliverable_template=lambda _template: None,
    )


class TestTheAuthoringGate:
    def test_a_host_describing_no_actions_refuses_the_comparison(self):
        """The engine's rule, not a host's: the toy host has no reader, so it fails closed."""
        with pytest.raises(ValidationFailedError) as refused:
            _create_template(
                {
                    "name": "p",
                    "intent": "i",
                    "candidate_kind": TOY_EXTRACTOR_KIND,
                    "goal_state_checks": ['any(it.x == "y" for it in calls("docs.extract"))'],
                }
            )

        assert "describes no parameters for docs.extract" in str(refused.value)


def _actions(tool: str):
    return frozenset({"say", "hush"}) if tool == "speech" else None


def _undefined(expression: str) -> tuple[str, ...]:
    return undefined_call_references(expression, _actions, _reader)


class TestUndefinedCallReferences:
    """A misspelled action or parameter scores False on every trial, so authoring refuses it."""

    @pytest.mark.parametrize(
        "expression",
        [
            'call_count("speech.sya") >= 1',
            'calls("speech.sya").length >= 1',
            'called_before("speech.say", "speech.sya")',
            'called_after("speech.sya", "speech.say")',
            'last_call_was("speech.sya")',
        ],
    )
    def test_every_call_builtin_refuses_an_action_the_tool_does_not_have(self, expression):
        assert _undefined(expression) == ("speech has no action 'sya' (its actions: hush, say)",)

    @pytest.mark.parametrize(
        "expression",
        [
            "all(it.txet.length > 0 for it in calls('speech.say'))",
            'calls("speech.say")[0].txet.length > 0',
            'any(it.txet == "" for it in calls("speech.say"))',
        ],
    )
    def test_a_parameter_the_schema_does_not_declare_is_refused_whether_compared_or_not(self, expression):
        assert _undefined(expression) == ("speech.say declares no parameter txet",)

    @pytest.mark.parametrize(
        "expression",
        [
            'call_count("speech.say") >= 1 and called_before("speech.hush", "speech.say")',
            "all(it.text.length > 0 for it in calls('speech.say'))",
            'calls("speech.say")[0].tags[0].length > 0',
            'calls("speech.say").length >= 1',
            'any(it.position == "head" for it in calls("speech.say"))',
        ],
    )
    def test_names_the_host_defines_pass(self, expression):
        assert _undefined(expression) == ()

    def test_a_tool_or_action_the_host_cannot_describe_passes_unverified(self):
        """No answer is not the answer "absent": refusing here would refuse every valid check on such a tool."""
        assert _undefined('call_count("chat.send") >= 1 and all(it.x.length > 0 for it in calls("chat.send"))') == ()
        assert undefined_call_references('call_count("speech.sya") >= 1', None, None) == ()

    def test_a_tool_the_host_does_not_have_offers_no_action(self):
        """An empty answer is "no such action", which a check sharing its typo with its control cannot pass."""
        reasons = undefined_call_references('call_count("musik.delete") == 0', lambda tool: frozenset(), None)

        assert reasons == (
            "musik has no action 'delete' (this host lists no actions for 'musik' — is it a tool the host has?)",
        )
