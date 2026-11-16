"""What a test declares for an L1 table of its own: every L1 table declares its eviction policy, a test's too.

A test's table is a cache the test fills and reads: ``EVICTABLE``, with the default budget (half the L1 directory's
file system), so nothing a test caches is evicted unless the test sets a budget of its own.
"""

from __future__ import annotations

from typing import Final

from threetears.core.cache.l1_lru import L1_EVICTION_INFO
from threetears.core.eviction import Declared, Policy

__all__ = ["L1_EVICTION_INFO", "TEST_L1_KEEP", "TEST_L1_TABLE"]

#: the policy of a table a test declares for itself
TEST_L1_TABLE: Final = Declared(Policy.EVICTABLE, "a test's own table: a cache it fills and reads back")
#: the policy of a test's table a collection writes behind (its rows wait in L1 for L3), or one it needs whole
TEST_L1_KEEP: Final = Declared(Policy.KEEP, "a test's own table, written behind: its rows wait in L1 for L3")
