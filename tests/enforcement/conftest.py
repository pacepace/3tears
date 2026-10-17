"""hooks shared by every gate in this directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common import clear_parse_cache

_HERE = Path(__file__).parent


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """give back the gates' parsed trees once the run leaves this directory.

    every gate reads the workspace through ``parse_python_path``, which keeps each tree for the
    next gate. a full workspace's trees hold hundreds of megabytes, and this suite shares its
    process with every package's tests. a package-scoped fixture would not do it: this directory
    is not a package, so pytest holds such a fixture to the end of the session.

    :param item: the test that just ran
    :ptype item: pytest.Item
    :param nextitem: the test that runs next, or ``None`` after the last
    :ptype nextitem: pytest.Item | None
    :return: nothing
    :rtype: None
    """
    if nextitem is None or not nextitem.path.is_relative_to(_HERE):
        clear_parse_cache()
