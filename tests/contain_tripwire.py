"""pytest plugin: a test fails when ``contain()`` absorbed a programming error.

:func:`threetears.observe.contain` is for failures a boundary expects and survives: an
upstream refusing, a peer gone, a vendor exporter raising. a ``TypeError``,
``AttributeError``, ``NameError`` or ``AssertionError`` reaching it is almost always a bug in
our own code that the broad ``except`` is about to hide -- the undefined name that reaches a
running container because every caller's ``except Exception`` caught it. in production that
stays contained, logged and counted; under test it fails the test that triggered it, in CI,
before it ships.

a test that drives a programming error into a boundary on purpose says so::

    @pytest.mark.expect_contained(TypeError, why="a plugin returning the wrong type is skipped, not fatal")
    def test_a_malformed_plugin_does_not_stop_the_loop(): ...

``why`` is required, and is also what keeps pytest from reading a lone class argument as the
thing the mark decorates.

it reads ``contain()``'s own log records -- a WARNING-or-above record whose ``extra_data``
carries ``contained_site`` -- from a handler on the root logger, so it needs nothing from
the code under test.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

#: exception types that, absorbed by a boundary, mean a bug rather than a failure
PROGRAMMING_ERRORS: tuple[type[Exception], ...] = (AttributeError, NameError, TypeError, AssertionError)


class ContainedProgrammingErrors(logging.Handler):
    """collect every programming error ``contain()`` logs while installed."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.found: list[tuple[str, BaseException]] = []

    def emit(self, record: logging.LogRecord) -> None:
        data = getattr(record, "extra_data", None)
        if not isinstance(data, dict) or "contained_site" not in data:
            return
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, PROGRAMMING_ERRORS):
            self.found.append((str(data["contained_site"]), exc))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "expect_contained(*types, why): the test drives these programming-error types into contain() on purpose",
    )


@pytest.fixture(autouse=True)
def _fail_on_a_contained_programming_error(request: pytest.FixtureRequest) -> Iterator[None]:
    """fail the test if ``contain()`` absorbed a programming error it did not expect."""
    handler = ContainedProgrammingErrors()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
    markers = list(request.node.iter_markers("expect_contained"))
    for marker in markers:
        if not str(marker.kwargs.get("why", "")).strip():
            pytest.fail("@pytest.mark.expect_contained needs why=<the reason the test drives it there>", pytrace=False)
    expected = tuple(kind for marker in markers for kind in marker.args)
    unexpected = [(site, exc) for site, exc in handler.found if not isinstance(exc, expected)]
    if unexpected:
        lines = "\n".join(f"  {site}: {type(exc).__name__}: {exc}" for site, exc in unexpected)
        pytest.fail(
            "contain() absorbed a programming error -- a bug the broad except would hide in production:\n"
            f"{lines}\n"
            "fix the bug; if the test drives it there on purpose, mark it "
            "@pytest.mark.expect_contained(<type>, why=<reason>)",
            pytrace=False,
        )
