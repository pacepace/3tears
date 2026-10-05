"""What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

It holds two kits. The store conformance kit: every rule of the
:class:`~threetears.evals.contracts.DocumentStore` port as a :class:`StoreConformanceCase`, collected
in :data:`STORE_CONFORMANCE_CASES`. An adapter's test suite hands each case a fresh, empty store and
parametrises over the tuple; a broken rule raises :class:`StoreConformanceFailure` naming the case and
the rule. The reader conformance kit: every promise a host's readers make — JSON-safe, deterministic,
order-independent, side-effect free, an open family's members its own, every run deriving a variant key —
as a :class:`ReaderConformanceCase` over a :class:`ReaderSample` (the profile and runs it produced), in
:data:`READER_CONFORMANCE_CASES`. Both are plain Python and import no test runner. See
:mod:`threetears.evals.testing.store_conformance` and :mod:`threetears.evals.testing.reader_conformance` for
an example under pytest.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

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
    "READER_CONFORMANCE_CASES",
    "STORE_CONFORMANCE_CASES",
    "ReaderConformanceCase",
    "ReaderConformanceFailure",
    "ReaderSample",
    "StoreConformanceCase",
    "StoreConformanceFailure",
]
