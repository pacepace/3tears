"""Tests for the wall-clock ceilings the engine derives over a host's requests.

:func:`~threetears.evals.analysis.generation_ceiling_s`, :func:`~threetears.evals.analysis.judge_phase_ceiling_s`
and :func:`~threetears.evals.analysis.reporter_cell_timeout_s` each count requests and multiply by the host's own
ceiling for one request (:data:`~threetears.evals.schema.RequestCeiling`). The host owns that answer because only
it knows its client: how many provider calls one request can become and how long the client sleeps between them. The
engine once reconstructed it from per-call parameters with no term for those sleeps, so a host whose client retries
could not state its own worst case, and the engine shipped a package-side calls-per-request figure a host would read
as its own.

Every expected value below is written out as arithmetic over a request ceiling that answers a DIFFERENT number for
every cap, so a formula that asks it at the wrong cap, drops a factor or adds a per-call term of its own changes the
answer rather than landing on a value that still looks reasonable.
"""

from __future__ import annotations

import inspect

import pytest

import threetears.evals.kernel as kernel
import threetears.evals.schema as schema
from threetears.evals.analysis import generation_ceiling_s, judge_phase_ceiling_s, reporter_cell_timeout_s

_GENERATOR_MAX_TOKENS = 8_000
_JUDGE_MAX_TOKENS = 3_000


class _RecordingRequestCeiling:
    """A request ceiling answering ``max_tokens / 10 + 7`` seconds and recording every cap it was asked about."""

    def __init__(self) -> None:
        self.asked: list[int] = []

    def __call__(self, max_tokens: int) -> float:
        self.asked.append(max_tokens)
        return max_tokens / 10 + 7


class TestTheGenerationCeiling:
    def test_is_two_requests_at_the_generator_cap(self) -> None:
        """A generation is the request plus its one repair round-trip, each the host's request ceiling at the cap."""
        request_s = _RecordingRequestCeiling()
        ceiling = generation_ceiling_s(request_s=request_s, generator_max_tokens=_GENERATOR_MAX_TOKENS)
        assert ceiling == 2 * (800 + 7)
        assert set(request_s.asked) == {_GENERATOR_MAX_TOKENS}

    def test_adds_nothing_of_its_own_to_a_request(self) -> None:
        """A host whose request takes a flat 100s, sleeps included, gets exactly 200s: no per-call term on top."""
        assert generation_ceiling_s(request_s=lambda _max_tokens: 100.0, generator_max_tokens=1) == 200.0


class TestTheJudgePhaseCeiling:
    def test_is_waves_times_attempts_times_one_request_at_the_judge_cap(self) -> None:
        """Five dimensions two at a time are three waves; each dimension can make two requests."""
        request_s = _RecordingRequestCeiling()
        ceiling = judge_phase_ceiling_s(
            judge_dims=5,
            judge_concurrency=2,
            judge_call_attempts=2,
            judge_max_tokens=_JUDGE_MAX_TOKENS,
            request_s=request_s,
        )
        assert ceiling == 3 * 2 * (300 + 7)
        assert set(request_s.asked) == {_JUDGE_MAX_TOKENS}

    @pytest.mark.parametrize(
        ("judge_dims", "judge_concurrency", "waves"),
        [(4, 2, 2), (5, 2, 3), (1, 8, 1), (0, 3, 0)],
    )
    def test_waves_round_up(self, judge_dims: int, judge_concurrency: int, waves: int) -> None:
        """A partial last wave is a whole wave of wall clock; no dimensions is no judging."""
        ceiling = judge_phase_ceiling_s(
            judge_dims=judge_dims,
            judge_concurrency=judge_concurrency,
            judge_call_attempts=1,
            judge_max_tokens=_JUDGE_MAX_TOKENS,
            request_s=lambda _max_tokens: 10.0,
        )
        assert ceiling == waves * 10.0


class TestTheReporterCellCeiling:
    def test_is_the_generation_then_the_judge_phase(self) -> None:
        """The two caps differ, so judging at the generator's cap (or generating at the judge's) changes the sum."""
        request_s = _RecordingRequestCeiling()
        ceiling = reporter_cell_timeout_s(
            judge_dims=5,
            judge_concurrency=2,
            generator_max_tokens=_GENERATOR_MAX_TOKENS,
            judge_call_attempts=2,
            judge_max_tokens=_JUDGE_MAX_TOKENS,
            request_s=request_s,
        )
        assert ceiling == 2 * (800 + 7) + 3 * 2 * (300 + 7)
        assert set(request_s.asked) == {_GENERATOR_MAX_TOKENS, _JUDGE_MAX_TOKENS}


class TestTheHostOwnsTheRequest:
    """No ceiling takes a per-call parameter the host would have to translate its client into."""

    @pytest.mark.parametrize(
        ("ceiling", "parameters"),
        [
            (generation_ceiling_s, {"request_s", "generator_max_tokens"}),
            (
                judge_phase_ceiling_s,
                {"judge_dims", "judge_concurrency", "judge_call_attempts", "judge_max_tokens", "request_s"},
            ),
            (
                reporter_cell_timeout_s,
                {
                    "judge_dims",
                    "judge_concurrency",
                    "generator_max_tokens",
                    "judge_call_attempts",
                    "judge_max_tokens",
                    "request_s",
                },
            ),
        ],
    )
    def test_the_request_is_the_hosts_one_input(self, ceiling: object, parameters: set[str]) -> None:
        assert callable(ceiling)
        signature = inspect.signature(ceiling)
        assert set(signature.parameters) == parameters
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in signature.parameters.values())

    def test_the_package_states_no_calls_per_request_of_its_own(self) -> None:
        """The contract names the host's answer, and no package-side count a host could mistake for its client's."""
        assert "RequestCeiling" in schema.__all__
        assert not [name for name in (*schema.__all__, *kernel.__all__) if "ATTEMPTS" in name and "REQUEST" in name]
