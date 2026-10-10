"""the SearXNG test container runs one pinned image, and refuses one that does not serve its static dir."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest


from threetears.core.testing.fixtures import SEARXNG_IMAGE, SEARXNG_STATIC_DIR, require_searxng_static_dir


def test_the_image_is_pinned_by_tag_and_digest() -> None:
    assert re.fullmatch(r"searxng/searxng:[0-9][^:@]*@sha256:[0-9a-f]{64}", SEARXNG_IMAGE), SEARXNG_IMAGE
    assert ":latest" not in SEARXNG_IMAGE


class _Container:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.ran: list[list[str]] = []

    def exec(self, command: list[str]) -> SimpleNamespace:
        self.ran.append(command)
        return SimpleNamespace(exit_code=self.exit_code, output=b"")


def test_an_image_without_the_static_dir_fails_naming_the_path_and_the_image() -> None:
    container = _Container(exit_code=1)
    with pytest.raises(pytest.fail.Exception) as failed:
        require_searxng_static_dir(container)
    assert SEARXNG_STATIC_DIR in str(failed.value)
    assert SEARXNG_IMAGE in str(failed.value)
    # it checks a directory the image ships, not the static dir a mount would create
    assert container.ran == [["test", "-d", f"{SEARXNG_STATIC_DIR}/themes"]]


def test_an_image_that_serves_it_passes() -> None:
    require_searxng_static_dir(_Container(exit_code=0))
