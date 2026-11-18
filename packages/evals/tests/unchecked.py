"""A call made with arguments its signature refuses, on purpose: the refusal is what the test exercises.

A test that proves a constructor or function refuses an ill-typed value, a retired keyword, or a missing
argument has to make the very call the checker rejects. :func:`unchecked` hands back the same callable
typed ``Callable[..., T]``, so that one call is made without its signature checked while its result keeps
its type, and no ``type: ignore`` is needed at the call.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["unchecked"]


def unchecked[T](function: Callable[..., T]) -> Callable[..., T]:
    """``function`` itself, typed so a call with arguments its signature refuses type-checks.

    :param function: The callable the test calls wrongly on purpose.
    :ptype function: Callable[..., T]
    :return: The same callable, its parameters unchecked and its return type kept.
    :rtype: Callable[..., T]
    """
    return function
