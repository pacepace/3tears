"""Contract for what the eval package needs from a model provider.

``threetears/evals/contracts/provider.py`` holds the provider trivia the engine needs, so it
names no host LLM module. This file asserts the contract itself -- what the engine depends on.
Parity with a particular host's own copies of those values is that host's test to keep, and so is the
fit of its concrete completion type to the port, which it checks with
:func:`~threetears.evals.testing.check_completion_conformance` -- the check this file runs on the toy host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

import pytest

from threetears.evals.contracts.completion import (
    COMPLETION_RESULT_ATTRIBUTES,
    JSON_OBJECT_RESPONSE_FORMAT,
    USAGE_LEDGER_ATTRIBUTES,
    StopReason,
)
from threetears.evals.contracts.provider import (
    INCOMPLETE_STOP_REASONS,
    describe_incomplete_completion,
    extract_json,
    extract_json_array,
    sum_optional_tokens,
)
from threetears.evals.testing import CompletionConformanceFailure, check_completion_conformance

from packages.evals.tests.fixtures.toyhost.judge import ToyJudgeCompletion


@dataclass
class _Completion:
    """A completion satisfying :class:`CompletionResult`, built without the host.

    The package is read by hosts it has never seen, so its own tests construct the port's
    shape rather than any host's concrete result type. Whether a given host's type still fits
    is that host's question to assert.
    """

    content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    served_model: str | None = None
    reasoning_tokens: int | None = None
    stop_reason: str = "end_turn"


class TestSumOptionalTokens:
    """Missing is not zero, and that distinction is the whole reason this exists."""

    def test_all_unreported_stays_unreported(self):
        assert sum_optional_tokens(None, None, None) is None

    def test_no_values_at_all_is_unreported(self):
        assert sum_optional_tokens() is None

    def test_all_reported_sums(self):
        assert sum_optional_tokens(100, 25, 8) == 133

    def test_partially_reported_sums_what_is_known(self):
        """Five real measurements describe a turn better than discarding them."""
        assert sum_optional_tokens(100, None, 30) == 130

    def test_a_reported_zero_is_an_observation(self):
        assert sum_optional_tokens(0, 0) == 0
        assert sum_optional_tokens(None, 0) == 0


class TestJsonObjectResponseFormat:
    def test_is_the_provider_wire_literal(self):
        assert JSON_OBJECT_RESPONSE_FORMAT == {"type": "json_object"}


class TestExtractJsonArray:
    def test_plain_array(self):
        assert extract_json_array('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]

    def test_markdown_fence(self):
        assert extract_json_array('```json\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_code_fence_no_lang(self):
        assert extract_json_array('```\n[{"a": 1}]\n```') == [{"a": 1}]

    def test_a_bare_object_becomes_a_one_element_batch(self):
        """A model asked for an array of cases that returns one case returned one case."""
        assert extract_json_array('{"a": 1}') == [{"a": 1}]

    def test_prose_around_the_array(self):
        assert extract_json_array('Sure! [{"a": 1}] — hope that helps') == [{"a": 1}]

    def test_a_trailing_note_holding_a_bracket_does_not_lose_the_batch(self):
        """An outermost-bracket slice would span the note and fail; the batch is already paid for."""
        assert extract_json_array('[{"a": 1}]\n\nNote: see [1] for the source.') == [{"a": 1}]

    def test_both_parsers_read_the_same_fence(self):
        """One fence rule, so a block one parser reads the other reads too."""
        fenced = 'Here you go:\n```json\n{"a": 1}\n```\nthanks'
        assert extract_json(fenced) == {"a": 1}
        assert extract_json_array(fenced) == [{"a": 1}]

    def test_unparseable_raises_rather_than_reading_as_an_empty_batch(self):
        """The generation is already paid for; "the parser could not read it" is not "nothing"."""
        with pytest.raises(ValueError, match="no JSON array"):
            extract_json_array("no json here")

    def test_a_scalar_document_raises(self):
        with pytest.raises(ValueError, match="not an array"):
            extract_json_array("42")

    def test_an_empty_array_is_an_empty_batch(self):
        """A model that answered with no cases answered; that is the generator's to judge."""
        assert extract_json_array("[]") == []


class TestCompletionPort:
    """The injected client must still fit the shape eval was written against."""

    def test_the_eval_call_usage_record_carries_every_attribute_the_ledger_reads(self):
        """``CallUsage`` is fed to the same ledger method as a real completion, so it carries the ledger's declared set.

        Read off the declaration the ledger itself reads, never a list typed here: a hand-typed list
        already missed ``price_source`` once, and a rename on ``CallUsage`` degraded every judge and
        simulator row to "unreported" with nothing failing.
        """
        from threetears.evals.contracts.usage_capture import CALL_USAGE_ONLY_ATTRIBUTES, CallUsage

        usage = CallUsage()
        for field in (*USAGE_LEDGER_ATTRIBUTES, *CALL_USAGE_ONLY_ATTRIBUTES):
            assert hasattr(usage, field), f"CallUsage lost {field!r}, which the ledger's duck type needs"

    def test_every_attribute_the_ledger_reads_off_a_completion_is_a_protocol_member(self):
        """The declaration names nothing the protocol does not promise, so a host held to the protocol supplies it."""
        assert set(USAGE_LEDGER_ATTRIBUTES) <= set(COMPLETION_RESULT_ATTRIBUTES)

    def test_the_protocol_member_list_is_read_off_the_protocol(self):
        """``served_model`` and ``stop_reason`` are members; ``calls`` is not — it is ``CallUsage``'s alone."""
        assert {"content", "served_model", "stop_reason", "price_source"} <= set(COMPLETION_RESULT_ATTRIBUTES)
        assert "calls" not in COMPLETION_RESULT_ATTRIBUTES

    def test_a_conforming_completion_passes_the_hosts_check(self):
        """The toy host's completion type is what a host's own suite would hand the check."""
        check_completion_conformance(
            ToyJudgeCompletion(
                content="{}", input_tokens=1, output_tokens=1, cost_usd=None, model="m", price_source=None
            )
        )

    @pytest.mark.parametrize("renamed", USAGE_LEDGER_ATTRIBUTES)
    def test_renaming_any_attribute_the_ledger_reads_fails_naming_it(self, renamed):
        """A rename the ledger would read as "unreported" in silence fails the host's check by name."""
        fields = {name: None for name in COMPLETION_RESULT_ATTRIBUTES}
        fields.update(content="{}", stop_reason="end_turn")
        fields[f"{renamed}_renamed"] = fields.pop(renamed)
        completion = type("RenamedCompletion", (), fields)()

        with pytest.raises(CompletionConformanceFailure, match=rf"missing {renamed}\b.*usage ledger"):
            check_completion_conformance(completion)

    def test_a_raw_provider_stop_reason_fails_the_check(self):
        """OpenAI's ``length`` passed through reads as finished; the check says so."""
        fields: dict[str, object] = {name: None for name in COMPLETION_RESULT_ATTRIBUTES}
        fields.update(content="{}", stop_reason="length")

        with pytest.raises(CompletionConformanceFailure, match="stop_reason 'length'"):
            check_completion_conformance(type("RawCompletion", (), fields)())

    def test_the_ledger_reads_the_served_model_never_the_requested_one(self):
        """A completion naming the alias in ``model`` and the concrete model in ``served_model`` lands both, apart."""
        from threetears.evals.contracts.usage_capture import RoleUsageLedger

        ledger = RoleUsageLedger(role="candidate")
        ledger.add_llm_result(_Completion(model="~vendor/model-latest", served_model="vendor/model-2026-03"))
        ledger.add_llm_result(_Completion(model="~vendor/model-latest", served_model="vendor/model-2026-06"))
        ledger.add_llm_result(_Completion(model="~vendor/model-latest"))

        rows = ledger.rows()
        assert [(row.model, row.served_model) for row in rows] == [
            ("~vendor/model-latest", "vendor/model-2026-03"),
            ("~vendor/model-latest", "vendor/model-2026-06"),
            # A response that named no model is not recorded — never the alias standing in for it.
            ("~vendor/model-latest", None),
        ]


class TestDescribeIncompleteCompletion:
    """The reason a caller's parse failed, when the provider already knew.

    A caller that reads only ``content`` reports whatever its parser said — "not valid
    JSON", true and causally useless. These pin the causal sentence instead, per branch.
    """

    def test_a_normal_finish_has_nothing_to_explain(self):
        """``None`` lets a caller append this unconditionally and add a clause only when there is one."""
        assert describe_incomplete_completion(_Completion(content='{"ok": true}', stop_reason="end_turn")) is None
        assert describe_incomplete_completion(_Completion(content="", stop_reason="tool_use")) is None

    def test_a_cap_hit_names_the_truncation_and_the_move_that_fixes_it(self):
        """A caller told only "not valid JSON" retries or blames the prompt; neither helps here."""
        note = describe_incomplete_completion(
            _Completion(content='{"bluf": {"headl', stop_reason="max_tokens", output_tokens=16384, reasoning_tokens=0)
        )

        assert note is not None
        assert "TRUNCATED" in note
        assert "finish_reason=length" in note
        assert "16384 output token(s), of which 0 were reasoning" in note
        # Retrying unchanged truncates again — say so, or the caller burns another generation.
        assert "retrying the same request unchanged will truncate again" in note

    def test_an_unreported_reasoning_split_says_unreported_rather_than_zero(self):
        """``reasoning_tokens=None`` is "the provider reported no split", not a measured zero.

        The port keeps those apart deliberately; rendering the unknown as ``0`` would
        publish a measurement nobody made.
        """
        note = describe_incomplete_completion(_Completion(content="x", stop_reason="max_tokens", output_tokens=500))

        assert note is not None
        assert "500 output token(s), reasoning split unreported" in note
        assert "were reasoning" not in note

    def test_a_reported_zero_reasoning_split_is_reported_as_zero(self):
        """The mirror of the case above: a reported ``0`` is an observation and must survive."""
        note = describe_incomplete_completion(
            _Completion(content="x", stop_reason="max_tokens", output_tokens=500, reasoning_tokens=0)
        )

        assert note is not None
        assert "of which 0 were reasoning" in note
        assert "unreported" not in note

    def test_an_empty_completion_is_called_out_separately_from_a_partial_one(self):
        """``0 chars`` after a cap hit is a different diagnosis from a half-written object.

        The empty case is what reached prod, and "not valid JSON (0 chars)" reads as a
        missing response rather than a budget that was fully spent before any content.
        """
        empty = describe_incomplete_completion(
            _Completion(content="", stop_reason="max_tokens", output_tokens=16384, reasoning_tokens=16384)
        )
        partial = describe_incomplete_completion(
            _Completion(content='{"partial": ', stop_reason="max_tokens", output_tokens=16384, reasoning_tokens=0)
        )

        assert empty is not None and partial is not None
        assert "completion is empty" in empty
        assert "completion is empty" not in partial

    def test_a_content_filter_says_the_request_needs_changing_not_retrying(self):
        """A filtered completion is deterministic in a way a truncation is not."""
        note = describe_incomplete_completion(_Completion(content="", stop_reason="content_filter"))

        assert note is not None
        assert "content filter" in note
        assert "not retrying" in note

    def test_a_choiceless_response_says_nothing_was_generated(self):
        """A host's no-choices branch stamps ``error``; the caller is owed that fact."""
        note = describe_incomplete_completion(_Completion(content="", stop_reason="error", output_tokens=0))

        assert note is not None
        assert "no choices at all" in note

    def test_every_cut_short_reason_gets_its_own_sentence(self):
        """Exhaustive AND distinct — "returns something" is what a fallthrough also does.

        The set is derived from the wording map, so a member with no sentence cannot exist;
        what a test still has to say is that no two members share one. A guard asserting
        only ``is not None`` passes on exactly the defect it names, because a fallthrough
        arm returns another reason's diagnosis rather than nothing.
        """
        notes = {
            reason: describe_incomplete_completion(_Completion(content="x", stop_reason=reason))
            for reason in INCOMPLETE_STOP_REASONS
        }

        assert all(note is not None for note in notes.values())
        assert len(set(notes.values())) == len(notes), f"two cut-short reasons share one diagnosis: {notes}"

    def test_the_stop_reason_vocabulary_is_one_finish_and_the_cut_short_reasons(self):
        """The literal a host maps onto and the set eval reads truncation off are one vocabulary."""
        assert set(get_args(StopReason)) == INCOMPLETE_STOP_REASONS | {"end_turn"}
        assert "end_turn" not in INCOMPLETE_STOP_REASONS

    def test_a_reason_outside_the_vocabulary_is_not_described_at_all(self):
        """The other half of totality: an unknown reason reads as finished, not as a guess.

        ``describe_incomplete_completion`` indexes the wording map directly, so this is the
        assertion that the membership test in front of it is what keeps that lookup safe.
        """
        assert describe_incomplete_completion(_Completion(content="x", stop_reason="banana")) is None
