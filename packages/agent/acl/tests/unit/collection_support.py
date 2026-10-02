"""shared builder for the rbac Collection unit tests.

one place builds a Collection over a mocked registry and config, so every
test module that exercises a Collection method constructs it the same way
instead of importing a private helper out of a sibling test module.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

__all__ = ["make_collection"]


def make_collection(
    cls: type,
    *,
    l3_pool: AsyncMock | None = None,
) -> Any:
    """build a Collection instance with mocked registry + config.

    :param cls: Collection class to instantiate
    :ptype cls: type
    :param l3_pool: optional mocked pool
    :ptype l3_pool: AsyncMock | None
    :return: Collection instance with mocks wired in
    :rtype: Any
    """
    mock_registry = MagicMock()
    mock_registry.get_l1_backend.return_value = None
    mock_registry.get_l3_pool.return_value = l3_pool
    mock_registry.register.return_value = None

    mock_config = MagicMock()
    mock_config.collection_flush = "ALWAYS"
    mock_config.collection_flush_tables = ""

    return cls(registry=mock_registry, config=mock_config)
