"""the proxy refuses a pop replay guard sized for a smaller future tolerance than it accepts.

The guard refuses, after a wipe of its bucket, any proof issued within its sized tolerance of the
bucket's creation. The proxy accepts proofs whose iat leads its clock by up to the platform's
issue-time future tolerance. A guard sized below that would admit a replayed proof stamped at the
tolerance's edge after a broker restart, and nothing would say so -- so the mismatch is a
construction-time failure.

The tolerance is the platform's one small number, not the minute a proof may be OLD: a guard sized
for it refuses for ten seconds after a broker restart, where one sized for the old symmetric
leeway refused every tool call for 65.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from threetears.core.coordination import ReplayGuard
from threetears.core.security import ISSUE_TIME_FUTURE_TOLERANCE
from threetears.registry.catalog import ToolCatalog

from .dispatch_auth import make_proxy


def _guard(tolerance: timedelta) -> ReplayGuard:
    """a real guard over an unused client; construction never touches the bucket.

    :param tolerance: the verifier future tolerance to size it for
    :ptype tolerance: timedelta
    :return: the guard
    :rtype: ReplayGuard
    """
    return ReplayGuard(MagicMock(), bucket_name="pop_nonces", ttl_seconds=120, verifier_future_tolerance=tolerance)


def test_a_guard_sized_for_the_platform_future_tolerance_is_accepted() -> None:
    # five seconds, not sixty: the proxy no longer needs a guard that refuses for a minute.
    assert timedelta(seconds=5) == ISSUE_TIME_FUTURE_TOLERANCE
    make_proxy(ToolCatalog(), pop_replay_guard=_guard(ISSUE_TIME_FUTURE_TOLERANCE))


def test_a_guard_sized_below_the_future_tolerance_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="verifier_future_tolerance"):
        make_proxy(ToolCatalog(), pop_replay_guard=_guard(ISSUE_TIME_FUTURE_TOLERANCE - timedelta(seconds=1)))
