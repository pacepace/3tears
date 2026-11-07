"""An analysis generation asks the host for its client, and a host that supplies no client factory is refused.

The report writer's client is built from the host's factory for the ``analyst`` role, not handed in by the caller.
Pinned here: a host built with no client factory is refused by the generation's estimate before anything is built,
the refusal naming what needed a client.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.writer_host_support import CALLER, writer_host
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.ops import analysis_estimate


async def test_a_host_with_no_client_factory_is_refused_before_anything_is_built() -> None:
    held = off_loop(writer_host, clients=None)

    with pytest.raises(ValidationFailedError, match="supplies no completion clients, and an analysis generation calls"):
        await analysis_estimate(held.host, held.campaign.id, CALLER.scope_id)
