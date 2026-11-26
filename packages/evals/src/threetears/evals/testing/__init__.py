"""What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

It holds four kits and two checks. The store conformance kit: every rule of the
:class:`~threetears.evals.schema.DocumentStore` port as a :class:`StoreConformanceCase`, collected
in :data:`STORE_CONFORMANCE_CASES`. An adapter's test suite hands each case a fresh, empty store and
parametrises over the tuple; a broken rule raises :class:`StoreConformanceFailure` naming the case and
the rule. Beside it, the readings conformance kit: every rule of the optional
:class:`~threetears.evals.schema.ResultReadings` capability as a :class:`ResultReadingsConformanceCase`, in
:data:`RESULT_READINGS_CONFORMANCE_CASES`, for a store that offers it; each case also takes a hook that counts or
fails the queries the store issues (:mod:`threetears.evals.testing.readings_conformance`). The reader conformance
kit: every promise a host's readers make — JSON-safe, deterministic, order-independent, side-effect free, an open family's members its own, every run deriving a variant key —
as a :class:`ReaderConformanceCase` over a :class:`ReaderSample` (the profile and runs it produced), in
:data:`READER_CONFORMANCE_CASES`. The completion check: :func:`check_completion_conformance` takes an instance of a
host's completion type and raises :class:`CompletionConformanceFailure` naming every attribute eval reads off a
completion that it lacks — the usage ledger reads its own defensively, so a rename there is otherwise
silent. The jobs conformance kit: every action in a catalogue, the host's included, as a :class:`JobsConformanceCase`
from :func:`jobs_conformance_cases` — each action's call makes no completion request, and each spend action's happy
path returns ``JobsStarted`` and its job makes at least one — watched through a :class:`CompletionGuard` the host
wraps its client factory in (:mod:`threetears.evals.testing.jobs_conformance`). Beside the kits, one client: the
recorded completions, :class:`RecordedCompletions`, which records the calls a host's real completion client answers
into a JSONL file once and replays them exactly with no network, refusing by name a prompt the recording does not hold
(:mod:`threetears.evals.testing.recorded_completions`). Beside it, the recorded tool turns,
:class:`RecordedTurns`, which keep a tool-calling loop's replies alone, per named session and in order, with the
snapshot id a placeholder, and replay them while the tools run live, a session asked one turn more than it holds
refused naming the turn (:mod:`threetears.evals.testing.recorded_turns`). And the recorded web search,
:class:`RecordedWebSearch`, which records a host's web search port once and replays it by port and query, refusing a
query it does not hold (:mod:`threetears.evals.testing.recorded_web_search`). The negotiation check:
:func:`check_refusal_negotiation` scripts
three routers that refuse a host's own client (its temperature under enforcement; any enforced request; its output
cap and effort word, naming what is taken), and raises :class:`NegotiationConformanceFailure` naming each mistake
unless the client resends in order, reports what the served request carried, and remembers what it proved and
nothing else (:mod:`threetears.evals.testing.negotiation_conformance`). All of it is plain Python and imports no
test runner. See :mod:`threetears.evals.testing.store_conformance` and
:mod:`threetears.evals.testing.reader_conformance` for an example under pytest.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``. Each name is served by the module ``__getattr__`` from the module defining it, on
first read, so the completion client, which takes the recorded replays from here, loads no jobs kit and with it no
action catalogue or operation (``tests/test_import_time.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from threetears.evals.testing.completion_conformance import (
        CompletionConformanceFailure,
        check_completion_conformance,
    )
    from threetears.evals.testing.jobs_conformance import (
        BuildConformanceHost,
        CompletionGuard,
        CompletionOutsideJob,
        JobAwaitedByItsCall,
        JobsConformanceCase,
        JobsConformanceFailure,
        JobsConformanceHost,
        jobs_conformance_cases,
    )
    from threetears.evals.testing.negotiation_conformance import (
        NegotiationConformanceFailure,
        RouterTransport,
        check_refusal_negotiation,
    )
    from threetears.evals.testing.readings_conformance import (
        RESULT_READINGS_CONFORMANCE_CASES,
        ResultReadingsConformanceCase,
    )
    from threetears.evals.testing.reader_conformance import (
        READER_CONFORMANCE_CASES,
        ReaderConformanceCase,
        ReaderConformanceFailure,
        ReaderSample,
    )
    from threetears.evals.testing.recorded_completions import (
        RecordedCompletions,
        RecordedPrompt,
        RecordingRefused,
        ReplayedCompletion,
    )
    from threetears.evals.testing.recorded_turns import RecordedTurns
    from threetears.evals.testing.recorded_web_search import RecordedWebSearch
    from threetears.evals.testing.store_conformance import (
        STORE_CONFORMANCE_CASES,
        StoreConformanceCase,
        StoreConformanceFailure,
    )

__all__ = [
    "BuildConformanceHost",
    "CompletionGuard",
    "CompletionOutsideJob",
    "JobAwaitedByItsCall",
    "JobsConformanceCase",
    "JobsConformanceFailure",
    "JobsConformanceHost",
    "jobs_conformance_cases",
    "NegotiationConformanceFailure",
    "RouterTransport",
    "check_refusal_negotiation",
    "READER_CONFORMANCE_CASES",
    "RESULT_READINGS_CONFORMANCE_CASES",
    "STORE_CONFORMANCE_CASES",
    "CompletionConformanceFailure",
    "ReaderConformanceCase",
    "ReaderConformanceFailure",
    "ReaderSample",
    "ResultReadingsConformanceCase",
    "RecordedCompletions",
    "RecordedPrompt",
    "RecordedTurns",
    "RecordedWebSearch",
    "RecordingRefused",
    "ReplayedCompletion",
    "StoreConformanceCase",
    "StoreConformanceFailure",
    "check_completion_conformance",
]

#: Each name of ``__all__``, by the module defining it: served by :func:`__getattr__` on first read, so importing this
#: root (or one recorded replay from it) loads no more of the kits than the name asked for.
_LAZY_EXPORTS: Mapping[str, str] = MappingProxyType(
    {
        "BuildConformanceHost": "threetears.evals.testing.jobs_conformance",
        "CompletionGuard": "threetears.evals.testing.jobs_conformance",
        "CompletionOutsideJob": "threetears.evals.testing.jobs_conformance",
        "JobAwaitedByItsCall": "threetears.evals.testing.jobs_conformance",
        "JobsConformanceCase": "threetears.evals.testing.jobs_conformance",
        "JobsConformanceFailure": "threetears.evals.testing.jobs_conformance",
        "JobsConformanceHost": "threetears.evals.testing.jobs_conformance",
        "jobs_conformance_cases": "threetears.evals.testing.jobs_conformance",
        "NegotiationConformanceFailure": "threetears.evals.testing.negotiation_conformance",
        "RouterTransport": "threetears.evals.testing.negotiation_conformance",
        "check_refusal_negotiation": "threetears.evals.testing.negotiation_conformance",
        "READER_CONFORMANCE_CASES": "threetears.evals.testing.reader_conformance",
        "RESULT_READINGS_CONFORMANCE_CASES": "threetears.evals.testing.readings_conformance",
        "STORE_CONFORMANCE_CASES": "threetears.evals.testing.store_conformance",
        "CompletionConformanceFailure": "threetears.evals.testing.completion_conformance",
        "ReaderConformanceCase": "threetears.evals.testing.reader_conformance",
        "ReaderConformanceFailure": "threetears.evals.testing.reader_conformance",
        "ReaderSample": "threetears.evals.testing.reader_conformance",
        "ResultReadingsConformanceCase": "threetears.evals.testing.readings_conformance",
        "RecordedCompletions": "threetears.evals.testing.recorded_completions",
        "RecordedPrompt": "threetears.evals.testing.recorded_completions",
        "RecordedTurns": "threetears.evals.testing.recorded_turns",
        "RecordedWebSearch": "threetears.evals.testing.recorded_web_search",
        "RecordingRefused": "threetears.evals.testing.recorded_completions",
        "ReplayedCompletion": "threetears.evals.testing.recorded_completions",
        "StoreConformanceCase": "threetears.evals.testing.store_conformance",
        "StoreConformanceFailure": "threetears.evals.testing.store_conformance",
        "check_completion_conformance": "threetears.evals.testing.completion_conformance",
    }
)


# Under ``not TYPE_CHECKING``: a checker that saw a module ``__getattr__`` would type every name a host imports from
# here as its return, ``Any``, so a misspelt or removed name would check. It sees the imports above instead.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:  # Any: each name is whatever its defining module holds
        """Serve each name of ``__all__`` from the module defining it, on first read.

        :param name: The attribute asked for.
        :return: The exported object.
        :raises AttributeError: When ``name`` is not one this root exports.
        """
        if (defined_in := _LAZY_EXPORTS.get(name)) is not None:
            return getattr(import_module(defined_in), name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Every name the module holds, the lazily served ones among them."""
    return sorted({*globals(), *_LAZY_EXPORTS})
