"""a value a test has just written, read back through an API that answers ``None`` for a miss."""

from __future__ import annotations


def present[T](value: T | None) -> T:
    """the value, refusing a miss where the test expects a hit.

    :param value: what the read answered
    :ptype value: T | None
    :return: the value
    :rtype: T
    :raises AssertionError: when the read answered ``None``
    """
    assert value is not None, "expected a value, the read answered None"
    return value
