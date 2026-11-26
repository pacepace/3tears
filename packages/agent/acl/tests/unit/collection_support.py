"""shared builder for the rbac Collection unit tests.

one place builds a Collection over a mocked registry and config, so every
test module that exercises a Collection method constructs it the same way
instead of importing a private helper out of a sibling test module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, NonCallableMock

from threetears.core.collections.scan_cache import ScanCache

__all__ = ["make_collection"]


async def _answers_postgres(self: object) -> bool:
    """the pool double's database: Postgres, never YugabyteDB.

    :param self: the double
    :ptype self: object
    :return: False
    :rtype: bool
    """
    return False


def make_collection(
    cls: type,
    *,
    l3_pool: object = None,
) -> Any:
    """build a Collection instance with mocked registry + config.

    :param cls: Collection class to instantiate
    :ptype cls: type
    :param l3_pool: the pool the registry hands out: a mock, a store double, or a real in-process store
    :ptype l3_pool: object
    :return: Collection instance with mocks wired in
    :rtype: Any
    """
    if isinstance(l3_pool, NonCallableMock):
        # the registry's L3 is a SqlL3Backend, which says its database without a statement of its own
        # (answers_from_yugabyte): this double does as well, a Postgres one, so a keyed read's index hint asks it no
        # SELECT version() through the canned fetch its test set up. Set on the mock's own class (each mock has one),
        # where threetears.core.backends.dialect looks
        type(l3_pool).answers_from_yugabyte = _answers_postgres
    mock_registry = MagicMock()
    mock_registry.get_l1_backend.return_value = None
    mock_registry.get_l3_pool.return_value = l3_pool
    mock_registry.register.return_value = None
    # no L1: the scan cache stores nothing, as in a process that has none
    mock_registry.scan_cache = ScanCache(None)

    mock_config = MagicMock()
    mock_config.collection_flush = "ALWAYS"
    mock_config.collection_flush_tables = ""

    return cls(registry=mock_registry, config=mock_config)
