"""Typed verdicts (#700): every verdict a report prints is a typed value first, and a gate reads only those.

A report's ``verdicts`` carry each contrast against the control, guardrail and bar reading as a
:class:`~threetears.evals.analysis.Verdict` — outcome, structured reason, margin and its source, materiality, the
guardrail flag — and the table cells a person reads are each one's ``words``, rendered from it. A program branches
on the outcome, never on the prose (the examples used to parse it with ``startswith``).

The gate (:func:`~threetears.evals.analysis.gate_verdicts`, :meth:`~threetears.evals.quick.Comparison.gate`) fails on
the outcomes a caller names, by default a regression, a breached guardrail and an undecided guardrail, and never
passes an undecided verdict.

Mutations that turn this file red: printing a verdict cell not taken from its typed verdict; a verdict whose outcome
is another kind's; a gate that passes with an undecided verdict, or with no verdict; a default that does not fail on a
breached or undecided guardrail; a reading filter that drops nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.analysis import (
    DEFAULT_FAIL_ON,
    GATE_TOKENS,
    TableBlock,
    Verdict,
    gate_verdicts,
    parse_fail_on,
    verdict_token,
)
from threetears.evals.quick import Comparison, Guardrail, compare

CASES = [{"n": index} for index in range(48)]


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def _answers(wrong_below: int) -> Any:
    async def candidate(case: Mapping[str, Any]) -> str:
        return "wrong" if case["n"] < wrong_below else "right"

    return candidate


async def _compare(arms: Mapping[str, int], **overrides: Any) -> Comparison:
    arguments: dict[str, Any] = {"control": "current", "scope_id": "typed", "k": 1}
    arguments.update(overrides)
    return await compare(CASES, {name: _answers(wrong) for name, wrong in arms.items()}, [correct], **arguments)


def _verdict(kind: str, outcome: str, **fields: Any) -> Verdict:
    base: dict[str, Any] = {
        "reason": "separated" if kind == "contrast" else "interval_straddles",
        "reading": "measure",
        "name": "correct",
        "heading": "Correct score",
        "arm": "candidate=new",
        "variant_key": "v",
        "apparatus_class_id": "a",
        "control": None if kind == "bar" else "candidate=current",
        "threshold": 0.9 if kind == "bar" else None,
        "guardrail": kind == "guardrail",
        "words": outcome,
    }
    base.update(fields)
    return Verdict(kind=kind, outcome=outcome, **base)  # type: ignore[arg-type]


class TestTheReportCarriesTypedVerdicts:
    async def test_each_contrast_row_prints_the_words_of_its_typed_verdict(self) -> None:
        comparison = await _compare({"current": 0, "worse": 30, "same": 0})
        table = next(b for b in comparison.report.blocks if isinstance(b, TableBlock) and b.name == "comparisons")
        typed = comparison.verdicts(kind="contrast")
        assert [row["verdict"] for row in table.rows] == [verdict.words for verdict in typed]
        by_arm = {verdict.arm: verdict for verdict in typed}
        assert (by_arm["candidate=worse"].outcome, by_arm["candidate=worse"].reason) == ("regressed", "separated")
        assert by_arm["candidate=same"].outcome in ("not_separated", "untested")
        assert by_arm["candidate=same"].reason in ("no_margin", "untestable")
        assert all(not verdict.guardrail and verdict.control == "candidate=current" for verdict in typed)

    async def test_contrasts_rows_carry_the_typed_outcome_beside_the_words(self) -> None:
        comparison = await _compare({"current": 0, "worse": 30})
        (row,) = comparison.contrasts("correct")
        assert row["outcome"] == "regressed" and row["verdict"] == "regressed from the control"
        assert comparison.verdicts("Correct score") == comparison.verdicts("correct")
        assert comparison.verdicts("correct", kind="guardrail") == []

    def test_an_outcome_of_another_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a guardrail verdict cannot be 'regressed'"):
            _verdict("guardrail", "regressed")
        with pytest.raises(ValueError, match="flagged guardrail exactly when"):
            _verdict("contrast", "improved", guardrail=True)


class TestTheGate:
    def test_the_default_fails_on_a_regression_a_breach_and_an_undecided_guardrail(self) -> None:
        assert DEFAULT_FAIL_ON == ("regressed", "breached", "undecided-guardrail")
        for failing in (
            _verdict("contrast", "regressed"),
            _verdict("guardrail", "breached", reason="beyond_margin"),
            _verdict("guardrail", "undecided"),
        ):
            result = gate_verdicts([_verdict("contrast", "improved"), failing])
            assert (result.outcome, result.failed, result.passed) == ("failed", True, False)
            assert result.failures == (failing,)

    def test_an_undecided_verdict_it_does_not_fail_on_is_never_a_pass(self) -> None:
        result = gate_verdicts([_verdict("contrast", "improved"), _verdict("contrast", "not_separated")])
        assert (result.outcome, result.failed, result.passed) == ("undecided", False, False)
        assert "which is not a pass" in result.render()
        nothing = gate_verdicts([])
        assert (nothing.outcome, nothing.passed) == ("undecided", False)

    def test_it_passes_only_when_every_verdict_is_decided_and_none_fails(self) -> None:
        result = gate_verdicts(
            [
                _verdict("contrast", "equivalent", reason="inside_margin"),
                _verdict("guardrail", "held", reason="within_margin"),
                _verdict("bar", "cleared", reason="interval_clears"),
            ]
        )
        assert (result.outcome, result.passed) == ("passed", True)

    def test_naming_not_separated_requires_every_arm_be_shown_improved_or_equivalent(self) -> None:
        verdicts = [_verdict("contrast", "not_separated", reason="no_margin")]
        assert gate_verdicts(verdicts, fail_on=["not-separated"]).outcome == "failed"
        assert gate_verdicts(verdicts, fail_on=["regressed"]).outcome == "undecided"

    def test_every_bar_outcome_short_of_a_decision_counts_as_an_undecided_bar(self) -> None:
        for outcome in ("undecided", "no_interval", "no_data"):
            assert verdict_token(_verdict("bar", outcome)) == "undecided-bar"
        assert verdict_token(_verdict("bar", "missed", reason="interval_misses")) == "missed"
        assert verdict_token(_verdict("contrast", "improved")) is None

    def test_a_reading_filter_gates_only_those_readings_and_refuses_an_unknown_one(self) -> None:
        verdicts = [_verdict("contrast", "regressed", name="cost", heading="Cost"), _verdict("contrast", "improved")]
        assert gate_verdicts(verdicts, readings=["correct"]).outcome == "passed"
        assert gate_verdicts(verdicts, readings=["Cost"]).outcome == "failed"
        with pytest.raises(ValueError, match="no verdict is on 'latency'"):
            gate_verdicts(verdicts, readings=["latency"])

    def test_fail_on_names_are_tokens(self) -> None:
        assert parse_fail_on("breached, regressed,breached") == ("regressed", "breached")
        with pytest.raises(ValueError, match="'regresed' is no outcome a gate fails on"):
            parse_fail_on("regresed")
        with pytest.raises(ValueError, match="name at least one outcome"):
            parse_fail_on(" , ")
        assert set(DEFAULT_FAIL_ON) <= set(GATE_TOKENS)

    async def test_a_comparison_gates_its_own_verdicts(self) -> None:
        comparison = await _compare({"current": 0, "worse": 30})
        result = comparison.gate()
        assert result.outcome == "failed" and [verdict.arm for verdict in result.failures] == ["candidate=worse"]
        assert "gate FAILED" in result.render() and "regressed from the control" in result.render()
        better = await _compare({"current": 30, "better": 0}, scope_id="typed-better")
        assert better.gate().outcome == "passed"


def no_leak(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer keeps the customer's secret."""
    return "SECRET" not in answer


def _leaks(count: int) -> Any:
    async def candidate(case: Mapping[str, Any]) -> str:
        return "right" + (" SECRET" if case["n"] < count else "")

    return candidate


class TestAQuickGuardrailIsATypedVerdictTheGateReads:
    """#697's quick guardrails reach the typed verdicts, and the default gate fails on a breach and on undecided."""

    async def test_held_breached_and_undecided_each_typed_and_gated(self) -> None:
        comparison = await compare(
            CASES[:40],
            {"current": _leaks(0), "same": _leaks(0), "leaky": _leaks(12), "slip": _leaks(3)},
            [correct, no_leak],
            control="current",
            scope_id="typed-guardrails",
            k=1,
            guardrails={"no_leak": Guardrail(margin=0.1, direction="higher_is_better")},
        )
        typed = {verdict.arm: verdict for verdict in comparison.verdicts("no_leak", kind="guardrail")}
        assert {arm: verdict.outcome for arm, verdict in typed.items()} == {
            "candidate=same": "held",
            "candidate=leaky": "breached",
            "candidate=slip": "undecided",
        }
        assert all(verdict.guardrail and verdict.margin == 0.1 for verdict in typed.values())
        assert typed["candidate=same"].margin_source == "measure"
        rows = {row["arm"]: row for row in comparison.guardrails()}
        assert {f"candidate={arm}": row["decision"] for arm, row in rows.items()} == {
            arm: verdict.words for arm, verdict in typed.items()
        }, "the table prints each typed verdict's words"

        result = comparison.gate(readings=["no_leak"])
        assert result.outcome == "failed"
        assert {verdict.arm for verdict in result.failures} == {"candidate=leaky", "candidate=slip"}
        assert comparison.gate(["breached"], readings=["no_leak"]).failures == (typed["candidate=leaky"],)
        lenient = comparison.gate(["regressed"], readings=["no_leak"])
        assert lenient.outcome == "undecided" and not lenient.passed, "an undecided guardrail is never a pass"
