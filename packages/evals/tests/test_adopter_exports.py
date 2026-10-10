"""Names an adopter's own suite reaches, exported from the public root that owns them.

A host may import only from the public roots (``nonpublic_evals_imports`` reports anything else), so a
helper an adopter needs and reaches through a module below a root is one it cannot use honestly. Each
is pinned here as the defining object, re-exported — not a copy that could drift.
"""

from __future__ import annotations

import threetears.evals.kernel as kernel
from threetears.evals.kernel import dsl, identity, usage_capture


def test_the_variant_derivation_and_the_substituted_delivery_count_are_on_the_kernel_root():
    assert kernel.derive_variant_identity is identity.derive_variant_identity
    assert kernel.count_substituted_deliveries is usage_capture.count_substituted_deliveries
    assert {"derive_variant_identity", "count_substituted_deliveries"} <= set(kernel.__all__)


def test_the_substituted_delivery_count_reads_a_delivery_record_before_any_result_exists():
    """A kind computes its measures from its cell's deliveries; it has no result to build just to ask."""
    from threetears.evals.schema import AsyncDelivery

    from packages.evals.tests.factories import make_eval_result

    live = AsyncDelivery(tool="scout", status="delivered", acknowledged_turn=1, delivered_turn=2, substituted=False)
    seeded = AsyncDelivery(tool="scout", status="delivered", acknowledged_turn=3, delivered_turn=3, substituted=True)

    assert kernel.count_substituted is usage_capture.count_substituted
    assert "count_substituted" in kernel.__all__
    assert [kernel.count_substituted(record) for record in (None, [], [live], [live, seeded, seeded])] == [
        0,
        0,
        0,
        2,
    ]
    # The result form is the record form, over the result's record: one predicate, not two.
    for record in (None, [live, seeded]):
        result = make_eval_result(async_deliveries=record)
        assert kernel.count_substituted_deliveries(result) == kernel.count_substituted(record)


def test_the_not_established_marker_is_on_the_kernel_root():
    """A host's own suite asserts that no keyless verdict came out "not established"; it reads the engine's marker."""
    assert kernel.NOT_ESTABLISHED is dsl.NOT_ESTABLISHED
    assert "NOT_ESTABLISHED" in kernel.__all__
