"""Names an adopter's own suite reaches, exported from the public root that owns them.

A host may import only from the public roots (``nonpublic_evals_imports`` reports anything else), so a
helper an adopter needs and reaches through a module below a root is one it cannot use honestly. Each
is pinned here as the defining object, re-exported — not a copy that could drift.
"""

from __future__ import annotations

import threetears.evals.contracts as contracts
from threetears.evals.contracts import identity, usage_capture


def test_the_variant_derivation_and_the_substituted_delivery_count_are_on_the_contracts_root():
    assert contracts.derive_variant_identity is identity.derive_variant_identity
    assert contracts.count_substituted_deliveries is usage_capture.count_substituted_deliveries
    assert {"derive_variant_identity", "count_substituted_deliveries"} <= set(contracts.__all__)
