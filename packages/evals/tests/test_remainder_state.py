"""An attribution remainder's two states are named once, and the bundle reads them by that one predicate (#615).

The remainder is STATED (``unattributed_delta``) or WITHHELD (a sentence in ``unattributed_withheld``), never
both and never neither. The bundle's :class:`ScopeDivergence` carries the pair and asks :func:`remainder_state`,
which reads an empty sentence as explaining nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import ScopeDivergence
from threetears.evals.analysis.remainder import remainder_state

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


class TestTheBundleReadsThePredicate:
    """The bundle names a pair the predicate's way, and refuses what it refuses."""

    @pytest.mark.parametrize(
        ("delta", "withheld", "expected"),
        [(200.0, None, "stated"), (None, _REASON, "withheld"), (200.0, "", "stated"), (200.0, "  ", "stated")],
    )
    def test_an_accepted_pair_names_one_state(self, delta: float | None, withheld: str | None, expected: str) -> None:
        assert _divergence(delta, withheld).remainder == expected

    @pytest.mark.parametrize(("delta", "withheld"), [(None, ""), (None, "   "), (None, None), (200.0, _REASON)])
    def test_a_refused_pair_is_refused(self, delta: float | None, withheld: str | None) -> None:
        with pytest.raises(ValidationError):
            _divergence(delta, withheld)
