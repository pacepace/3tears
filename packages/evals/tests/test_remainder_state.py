"""An attribution remainder's two states are named once, and both carriers read them the same way (#615).

The remainder is STATED (``unattributed_delta``) or WITHHELD (a sentence in ``unattributed_withheld``), never
both and never neither. Two types carry the pair, the chart's :class:`AttributionPayload` and the bundle's
:class:`ScopeDivergence`, and they once disagreed about an empty sentence: the payload tested it for truth
(an empty string explains nothing) and the bundle for ``None`` (an empty string is set). Both now ask
:func:`remainder_state`, so the same pair is accepted, refused and named alike by each.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import ScopeDivergence
from threetears.evals.analysis.remainder import remainder_state
from threetears.evals.analysis.viz.intents.attribution import attribution_intent
from threetears.evals.analysis.viz.payloads import AttributionPayload, PayloadError, parse_payload

_REASON = "tool_ms is not declared a component of total_ms, so their remainder may describe no stretch of time."


def _movement(name: str, scope: str, delta: float) -> dict[str, Any]:
    return {
        "name": name,
        "scope": scope,
        "mean_a": 100.0,
        "mean_b": 100.0 + delta,
        "delta": delta,
        "n_a": 5,
        "n_b": 5,
        "direction": "regressed",
        "materiality": "material",
    }


def _divergence(delta: float | None, withheld: str | None) -> ScopeDivergence:
    return ScopeDivergence.model_validate(
        {
            "lever": "chunk_tokens",
            "level_a": "256",
            "level_b": "512",
            "unit": "ms",
            "end_to_end": _movement("total_ms", "end_to_end", 300.0),
            "subsystem": _movement("tool_ms", "subsystem", 100.0),
            "test": "paired",
            "n_cases_a": 5,
            "n_cases_b": 5,
            "p_raw": 0.001,
            "p_adjusted": 0.002,
            "family_size": 1,
            "contained_by": "total_ms",
            "unattributed_delta": delta,
            "unattributed_withheld": withheld,
        }
    )


def _payload(delta: float | None, withheld: str | None) -> AttributionPayload:
    parsed = parse_payload(
        "attribution",
        {
            "end_to_end": {"measure": "total_ms", "delta": 300.0},
            "subsystem": {"measure": "tool_ms", "delta": 100.0},
            "unit": "ms",
            "contained_by": "total_ms",
            "unattributed_delta": delta,
            "unattributed_withheld": withheld,
        },
    )
    assert isinstance(parsed, AttributionPayload)
    return parsed


class TestThePredicate:
    def test_a_number_alone_is_stated(self) -> None:
        assert remainder_state(200.0, None) == "stated"

    def test_a_zero_is_still_a_number(self) -> None:
        assert remainder_state(0.0, None) == "stated"

    def test_a_sentence_alone_is_withheld(self) -> None:
        assert remainder_state(None, _REASON) == "withheld"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_sentence_explains_nothing(self, blank: str) -> None:
        assert remainder_state(200.0, blank) == "stated"
        with pytest.raises(ValueError, match="neither"):
            remainder_state(None, blank)

    def test_both_are_refused(self) -> None:
        with pytest.raises(ValueError, match="both set"):
            remainder_state(200.0, _REASON)


class TestBothCarriersAgree:
    """The same pair is named the same way, or refused the same way, by the chart and the bundle."""

    @pytest.mark.parametrize(
        ("delta", "withheld", "expected"),
        [(200.0, None, "stated"), (None, _REASON, "withheld"), (200.0, "", "stated"), (200.0, "  ", "stated")],
    )
    def test_an_accepted_pair_names_one_state_in_both(
        self, delta: float | None, withheld: str | None, expected: str
    ) -> None:
        assert _divergence(delta, withheld).remainder == expected
        assert _payload(delta, withheld).remainder == expected

    @pytest.mark.parametrize(("delta", "withheld"), [(None, ""), (None, "   "), (None, None), (200.0, _REASON)])
    def test_a_refused_pair_is_refused_by_both(self, delta: float | None, withheld: str | None) -> None:
        with pytest.raises(ValidationError):
            _divergence(delta, withheld)
        with pytest.raises(PayloadError):
            _payload(delta, withheld)


class TestTheChartReadsTheState:
    def test_a_stated_remainder_plots_and_discloses_the_arithmetic(self) -> None:
        intent = attribution_intent(_payload(200.0, ""))
        assert any(row.get("derived") for row in intent.data)
        assert not any(_REASON in line for line in intent.disclosures)
        assert any("minus" in line for line in intent.disclosures)

    def test_a_withheld_remainder_carries_its_sentence_and_plots_no_number(self) -> None:
        intent = attribution_intent(_payload(None, _REASON))
        assert not any(row.get("derived") for row in intent.data)
        assert _REASON in intent.disclosures
