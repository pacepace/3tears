"""A judge completion's temperature is part of the completion contract: a host's completion lacking it fails loudly.

``sent_temperature`` read ``CompletionResult.temperature`` through ``hasattr`` and answered "not recorded" for a
completion that had none, so a host whose completion type omitted it made every judge unrecordable as a profile
and every judge comparison "unknown", with no error anywhere. The attribute is a member of the protocol, which
``check_completion_conformance`` holds a host's type to; reading it is no longer optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from threetears.evals.run.judge import sent_temperature
from threetears.evals.schema.models import MODEL_DEFAULT_TEMPERATURE
from threetears.evals.testing import CompletionConformanceFailure, check_completion_conformance


@dataclass(frozen=True)
class _NoTemperature:
    """A host's completion that reports everything but the temperature it was sent at."""

    content: str = "{}"
    input_tokens: int | None = 1
    output_tokens: int | None = 1
    cost_usd: float | None = 0.0
    price_source: str | None = "provider"
    model: str = "judge-model"
    served_model: str | None = "judge-model"
    reasoning_tokens: int | None = None
    stop_reason: str = "end_turn"


@dataclass(frozen=True)
class _Sent(_NoTemperature):
    temperature: float | None = 0.0


def test_a_completion_with_no_temperature_is_a_defect_named_where_it_is_read() -> None:
    with pytest.raises(TypeError, match="check_completion_conformance"):
        sent_temperature(_NoTemperature())  # type: ignore[arg-type]


def test_the_conformance_check_names_it() -> None:
    with pytest.raises(CompletionConformanceFailure, match="temperature"):
        check_completion_conformance(_NoTemperature())


def test_a_temperature_sent_or_withheld_is_read() -> None:
    assert sent_temperature(_Sent(temperature=0.0)) == 0.0  # type: ignore[arg-type]
    assert sent_temperature(_Sent(temperature=None)) == MODEL_DEFAULT_TEMPERATURE  # type: ignore[arg-type]
