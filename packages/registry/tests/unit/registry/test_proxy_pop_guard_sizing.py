"""the proxy refuses a pop replay guard sized for a smaller future tolerance than it accepts.

The guard refuses, after a wipe of its bucket, any proof issued within its sized tolerance of the
bucket's creation. The proxy accepts proofs whose iat leads its clock by up to its pop leeway. A
guard sized below that leeway would admit a replayed proof stamped at the leeway's edge after a
broker restart, and nothing would say so -- so the mismatch is a construction-time failure.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.registry.catalog import ToolCatalog
from threetears.registry.proxy import POP_LEEWAY_SECONDS

from ._dispatch_auth import make_proxy


def _guard(tolerance: timedelta) -> ReplayGuard:
    """a real guard over an unused client; construction never touches the bucket.

    :param tolerance: the verifier future tolerance to size it for
    :ptype tolerance: timedelta
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(MagicMock(), bucket_name="pop_nonces", ttl_seconds=120, verifier_future_tolerance=tolerance)


def test_a_guard_sized_for_the_pop_leeway_is_accepted() -> None:
    make_proxy(ToolCatalog(), pop_replay_guard=_guard(timedelta(seconds=POP_LEEWAY_SECONDS)))


def test_a_guard_sized_below_the_pop_leeway_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="verifier_future_tolerance"):
        make_proxy(ToolCatalog(), pop_replay_guard=_guard(timedelta(seconds=POP_LEEWAY_SECONDS - 1)))
