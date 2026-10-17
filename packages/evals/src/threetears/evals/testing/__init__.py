"""What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

It holds three kits and two checks. The store conformance kit: every rule of the
:class:`~threetears.evals.schema.DocumentStore` port as a :class:`StoreConformanceCase`, collected
in :data:`STORE_CONFORMANCE_CASES`. An adapter's test suite hands each case a fresh, empty store and
parametrises over the tuple; a broken rule raises :class:`StoreConformanceFailure` naming the case and
the rule. The reader conformance kit: every promise a host's readers make — JSON-safe, deterministic,
order-independent, side-effect free, an open family's members its own, every run deriving a variant key —
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
(:mod:`threetears.evals.testing.recorded_completions`). The negotiation check: :func:`check_refusal_negotiation` scripts
three routers that refuse a host's own client (its temperature under enforcement; any enforced request; its output
cap and effort word, naming what is taken), and raises :class:`NegotiationConformanceFailure` naming each mistake
unless the client resends in order, reports what the served request carried, and remembers what it proved and
nothing else (:mod:`threetears.evals.testing.negotiation_conformance`). All of it is plain Python and imports no
test runner. See :mod:`threetears.evals.testing.store_conformance` and
:mod:`threetears.evals.testing.reader_conformance` for an example under pytest.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.testing.completion_conformance import (
    CompletionConformanceFailure,
    check_completion_conformance,
)
from threetears.evals.testing.jobs_conformance import (
    DEFAULT_CALL_RETURN_TIMEOUT_S,
    DEFAULT_JOB_CONFORMANCE_TIMEOUT_S,
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
from threetears.evals.testing.store_conformance import (
    STORE_CONFORMANCE_CASES,
    StoreConformanceCase,
    StoreConformanceFailure,
)

__all__ = [
    "DEFAULT_CALL_RETURN_TIMEOUT_S",
    "DEFAULT_JOB_CONFORMANCE_TIMEOUT_S",
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
    "STORE_CONFORMANCE_CASES",
    "CompletionConformanceFailure",
    "ReaderConformanceCase",
    "ReaderConformanceFailure",
    "ReaderSample",
    "RecordedCompletions",
    "RecordedPrompt",
    "RecordingRefused",
    "ReplayedCompletion",
    "StoreConformanceCase",
    "StoreConformanceFailure",
    "check_completion_conformance",
]
