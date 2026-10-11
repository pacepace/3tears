"""What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

It holds three kits and two checks. The store conformance kit: every rule of the
:class:`~threetears.evals.schema.DocumentStore` port as a :class:`StoreConformanceCase`, collected
in :data:`STORE_CONFORMANCE_CASES`. An adapter's test suite hands each case a fresh, empty store and
parametrises over the tuple; a broken rule raises :class:`StoreConformanceFailure` naming the case and
the rule. The reader conformance kit: every promise a host's readers make — JSON-safe, deterministic,
order-independent, side-effect free, an open family's members its own, every run deriving a variant key —
as a :class:`ReaderConformanceCase` over a :class:`ReaderSample` (the profile and runs it produced), in
:data:`READER_CONFORMANCE_CASES`. The import check: :func:`nonpublic_evals_imports` reads a host's
own source tree and reports every reach into ``threetears.evals`` that goes below a public root's declared
names — static and dynamic imports, attribute chains, private names, patches — as a :class:`BoundaryFinding`
naming the :data:`BoundaryRule` it breaks. The completion check: :func:`check_completion_conformance` takes an instance of a
host's completion type and raises :class:`CompletionConformanceFailure` naming every attribute eval reads off a
completion that it lacks — the usage ledger reads its own defensively, so a rename there is otherwise
silent. The jobs conformance kit: every action in a catalogue, the host's included, as a :class:`JobsConformanceCase`
from :func:`jobs_conformance_cases` — each action's call makes no completion request, and each spend action's happy
path returns ``JobsStarted`` and its job makes at least one — watched through a :class:`CompletionGuard` the host
wraps its client factory in (:mod:`threetears.evals.testing.jobs_conformance`). All five are plain Python and import
no test runner. See
:mod:`threetears.evals.testing.store_conformance` and :mod:`threetears.evals.testing.reader_conformance` for
an example under pytest.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.testing.completion_conformance import (
    CompletionConformanceFailure,
    check_completion_conformance,
)
from threetears.evals.testing.import_boundary import (
    BOUNDARY_RULES,
    BoundaryFinding,
    BoundaryRule,
    nonpublic_evals_imports,
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
from threetears.evals.testing.reader_conformance import (
    READER_CONFORMANCE_CASES,
    ReaderConformanceCase,
    ReaderConformanceFailure,
    ReaderSample,
)
from threetears.evals.testing.store_conformance import (
    STORE_CONFORMANCE_CASES,
    StoreConformanceCase,
    StoreConformanceFailure,
)

__all__ = [
    "BOUNDARY_RULES",
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
    "READER_CONFORMANCE_CASES",
    "STORE_CONFORMANCE_CASES",
    "BoundaryFinding",
    "BoundaryRule",
    "CompletionConformanceFailure",
    "ReaderConformanceCase",
    "ReaderConformanceFailure",
    "ReaderSample",
    "StoreConformanceCase",
    "StoreConformanceFailure",
    "check_completion_conformance",
    "nonpublic_evals_imports",
]
