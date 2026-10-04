"""What the engine ships for an adopter's own test suite: conformance kits a host runs against itself.

Today it holds the store conformance kit: every rule of the
:class:`~threetears.evals.contracts.DocumentStore` port as a :class:`StoreConformanceCase`, collected
in :data:`STORE_CONFORMANCE_CASES`. An adapter's test suite hands each case a fresh, empty store and
parametrises over the tuple; a broken rule raises :class:`StoreConformanceFailure` naming the case and
the rule. It is plain Python and imports no test runner. See
:mod:`threetears.evals.testing.store_conformance` for an example under pytest.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.testing.store_conformance import (
    STORE_CONFORMANCE_CASES,
    StoreConformanceCase,
    StoreConformanceFailure,
)

__all__ = [
    "STORE_CONFORMANCE_CASES",
    "StoreConformanceCase",
    "StoreConformanceFailure",
]
