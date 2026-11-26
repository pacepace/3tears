"""the collections tests' shared fixtures."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from threetears.core.testing.published_copy import close_published_copies


@pytest.fixture(autouse=True)
async def _published_copies_closed() -> AsyncIterator[None]:
    """every copy a test published (``testing.published_copy``) closed after it: its query workers and its store."""
    yield
    await close_published_copies()
