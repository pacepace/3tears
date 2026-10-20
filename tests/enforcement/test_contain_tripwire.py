"""the contain tripwire fails a test whose code contained a programming error, and only then."""

from __future__ import annotations

import pytest

_TESTS = """
import pytest

from threetears.observe import contain, get_logger

_log = get_logger(__name__)


def _absorb(exc):
    try:
        raise exc
    except Exception as caught:
        contain(caught, log=_log, what="boundary absorbed it")


def test_an_absorbed_attribute_error_fails_the_test():
    _absorb(AttributeError("'NoneType' object has no attribute 'id'"))


@pytest.mark.expect_contained(AttributeError, why="the test drives it there")
def test_an_expected_one_passes():
    _absorb(AttributeError("driven here on purpose"))


@pytest.mark.expect_contained(AttributeError, TypeError)
def test_an_expectation_without_a_reason_fails():
    _absorb(AttributeError("driven here on purpose"))


def test_an_ordinary_failure_passes():
    _absorb(ConnectionError("the peer went away"))
"""


# the inner session runs in this process, so this test's own tripwire sees its records too
@pytest.mark.expect_contained(AttributeError, why="the inner session drives one there on purpose")
def test_only_the_unexpected_programming_error_fails(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.contain_tripwire"]')
    pytester.makepyfile(test_boundary=_TESTS)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    # a teardown error is reported on top of the passed call, so every test also counts as passed
    result.assert_outcomes(passed=4, errors=2)
    result.stdout.fnmatch_lines(
        [
            "*contain() absorbed a programming error*",
            "*test_boundary:_absorb: AttributeError: 'NoneType' object has no attribute 'id'*",
            "*expect_contained needs why=*",
        ],
        consecutive=False,
    )
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_an_absorbed_attribute_error_fails_the_test*",
        ]
    )
