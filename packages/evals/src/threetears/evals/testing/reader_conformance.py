"""The reader conformance kit: every promise a host's readers make to the engine, as a case a host runs.

A host's profile carries code the engine calls and never inspects — each sweepable's ``read``, an open
family's membership test and residual reader, the variant-lever reader. The engine's every comparison,
key and confound scan is built on what they return, and their rules are stated in prose on the types
that hold them. This module states them again as checks, so a host's readers are proved rather than
read.

**The kit is plain Python, and the host's test runner parametrises it.** Each
:class:`ReaderConformanceCase` takes a :class:`ReaderSample` — the host's profile and some runs it
produced, each with its results — and either returns or raises :class:`ReaderConformanceFailure`,
whose message names the case, the rule, the reader and the run. Under pytest::

    import pytest
    from threetears.evals.testing import READER_CONFORMANCE_CASES, ReaderConformanceCase, ReaderSample

    @pytest.mark.parametrize("case", READER_CONFORMANCE_CASES, ids=lambda case: case.name)
    def test_my_readers_conform(case: ReaderConformanceCase) -> None:
        case.run(ReaderSample(profile=my_profile(), runs=my_recorded_runs()))

**Every case is mandatory**, and a sample with no runs is refused: a reader checked over nothing is
a reader nobody checked.

**What it cannot see.** That a reader reads the RIGHT thing — a judge reader returning the candidate
model would pass every case here. Behaviour on a run shape the sample does not hold: hand the kit
runs of every kind the host launches, and at least one recorded before any optional field existed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from threetears.evals.contracts.hashing import UnhashableContentError, canonical_json
from threetears.evals.contracts.identity import LeverCoordinateError, derive_variant_identity

if TYPE_CHECKING:
    from threetears.evals.contracts.host.profile import HostProfile
    from threetears.evals.contracts.host.sweepables import Sweepable
    from threetears.evals.contracts.models import EvalResult, EvalRun

__all__ = [
    "READER_CONFORMANCE_CASES",
    "ReaderConformanceCase",
    "ReaderConformanceFailure",
    "ReaderSample",
]


class ReaderConformanceFailure(AssertionError):
    """A host's reader broke a promise the engine relies on; the message names the case, the reader and the run.

    An :class:`AssertionError`, so a test runner reports it as a failed assertion rather than an error
    in the test.
    """


@dataclass(frozen=True)
class ReaderSample:
    """What the kit checks a host's readers over: the host's profile and runs it produced.

    Attributes:
        profile: The host's profile — every reader on it is checked.
        runs: ``(run, its results)`` pairs, as the host's store holds them. At least one.
    """

    profile: HostProfile
    runs: Sequence[tuple[EvalRun, Sequence[EvalResult]]]


@dataclass(frozen=True)
class ReaderConformanceCase:
    """One promise a reader makes, as a check over a sample.

    Attributes:
        name: A stable identifier, for a test id (``"sweepable.json_safe"``).
        rule: The promise, in one sentence.
        check: The check. Takes the sample; returns, or raises :class:`ReaderConformanceFailure`.
    """

    name: str
    rule: str
    check: Callable[[ReaderSample], None]

    def run(self, sample: ReaderSample) -> None:
        """Run the case over ``sample``.

        Args:
            sample: The host's profile and runs.

        Raises:
            ReaderConformanceFailure: The sample holds no run, or a reader broke the rule; the message
                names the case, the rule and what was observed. Anything else a reader raises propagates
                unchanged.
        """
        if not sample.runs:
            raise ReaderConformanceFailure(
                f"{self.name}: the sample holds no run — a reader checked over nothing is a reader nobody checked"
            )
        try:
            self.check(sample)
        except ReaderConformanceFailure as failure:
            raise ReaderConformanceFailure(f"{self.name}: {self.rule} — {failure}") from None


# --- helpers ----------------------------------------------------------------------------------------


def _json(value: Any, what: str) -> str:
    try:
        return canonical_json(value)
    except UnhashableContentError as refused:
        raise ReaderConformanceFailure(f"{what} returned {value!r}, which JSON cannot encode ({refused})") from None


def _dump(run: EvalRun, results: Sequence[EvalResult]) -> tuple[Any, list[Any]]:
    return run.model_dump(mode="json"), [result.model_dump(mode="json") for result in results]


def _declarations(sample: ReaderSample) -> tuple[Sweepable, ...]:
    return sample.profile.sweepables.declarations


def _where(declared: Sweepable, run: EvalRun) -> str:
    return f"{declared.name!r}'s reader on run {run.id!r}"


# --- the cases --------------------------------------------------------------------------------------


def _every_read_is_json_safe(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for declared in _declarations(sample):
            _json(declared.read(run, results), _where(declared, run))


def _a_read_is_deterministic(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for declared in _declarations(sample):
            first = _json(declared.read(run, results), _where(declared, run))
            second = _json(declared.read(run, results), _where(declared, run))
            if first != second:
                raise ReaderConformanceFailure(f"{_where(declared, run)} answered {first} and then {second}")


def _a_read_ignores_result_order(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for declared in _declarations(sample):
            forward = _json(declared.read(run, list(results)), _where(declared, run))
            backward = _json(declared.read(run, list(reversed(results))), _where(declared, run))
            if forward != backward:
                raise ReaderConformanceFailure(
                    f"{_where(declared, run)} answered {forward} over the results in order and {backward} reversed — "
                    "a set must come out sorted"
                )


def _a_read_mutates_nothing(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for declared in _declarations(sample):
            before = _dump(run, results)
            declared.read(run, results)
            if _dump(run, results) != before:
                raise ReaderConformanceFailure(f"{_where(declared, run)} changed the run or its results")


def _a_family_reads_a_map_of_its_own_members(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for family in sample.profile.sweepables.open_families:
            members = family.read(run, results)
            if not isinstance(members, Mapping) or not all(isinstance(name, str) for name in members):
                raise ReaderConformanceFailure(f"{_where(family, run)} returned {members!r}, not a member-name map")
            if family.owns_member is not None:
                strays = sorted(name for name in members if not family.owns_member(name))
                if strays:
                    raise ReaderConformanceFailure(
                        f"{_where(family, run)} resolved {strays}, which the family's own membership test disowns"
                    )


def _a_residual_reads_comparable_content(sample: ReaderSample) -> None:
    for run, results in sample.runs:
        for family in sample.profile.sweepables.open_families:
            if family.read_residual is None:
                continue
            members = family.read(run, results)
            removed = frozenset(members) if isinstance(members, Mapping) else frozenset()
            for taken_out in (frozenset(), removed):
                _json(family.read_residual(run, results, taken_out), f"{family.name!r}'s residual reader on {run.id!r}")


def _every_run_derives_a_variant_key(sample: ReaderSample) -> None:
    for run, _results in sample.runs:
        try:
            derive_variant_identity(run=run, profile=sample.profile)
        except LeverCoordinateError as refused:
            raise ReaderConformanceFailure(
                f"run {run.id!r}'s lever map does not match the registry: {refused}"
            ) from None


#: Every promise, as a case. Parametrise over this tuple.
READER_CONFORMANCE_CASES: tuple[ReaderConformanceCase, ...] = (
    ReaderConformanceCase(
        "sweepable.json_safe",
        "every reader returns a value JSON can encode, because every comparison is made on its canonical JSON",
        _every_read_is_json_safe,
    ),
    ReaderConformanceCase(
        "sweepable.deterministic",
        "a reader asked the same question twice gives the same answer",
        _a_read_is_deterministic,
    ),
    ReaderConformanceCase(
        "sweepable.order_independent",
        "a reader's answer does not depend on the order the run's results arrive in",
        _a_read_ignores_result_order,
    ),
    ReaderConformanceCase(
        "sweepable.mutates_nothing",
        "a reader leaves the run and its results exactly as it found them",
        _a_read_mutates_nothing,
    ),
    ReaderConformanceCase(
        "open_family.member_map",
        "an open family reads a map of member names, every one of which its own membership test admits",
        _a_family_reads_a_map_of_its_own_members,
    ),
    ReaderConformanceCase(
        "open_family.residual_json_safe",
        "an open family's residual reader returns content JSON can encode, or None",
        _a_residual_reads_comparable_content,
    ),
    ReaderConformanceCase(
        "variant_levers.match_the_registry",
        "every run's lever map names exactly the levers the registry declares, so every run derives a variant key",
        _every_run_derives_a_variant_key,
    ),
)
