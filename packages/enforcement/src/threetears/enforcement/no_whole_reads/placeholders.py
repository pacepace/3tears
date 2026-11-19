"""a statement's ``$n`` placeholders as the keyed builders read them, for the walk and the rewriter.

The keyed builders read a statement a token at a time (:data:`threetears.core.keyed.SQL_TOKEN`): a ``$n`` inside a
quoted literal, a quoted name, a dollar-quoted string or a comment is text, never a placeholder (a race named
'Proposition: 40 - Impose one-time wealth tax over $1 billion'). Every reader of placeholders here goes through
these, so the walk judges, and the rewriter writes, the statement the builders send.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.core.keyed import SQL_TOKEN, each_placeholder

__all__ = ["placeholder_numbers", "renumbered_with", "without_hidden_dollars"]


def placeholder_numbers(sql: str) -> list[int]:
    """the number of each placeholder ``sql`` names, a use each, in order.

    :param sql: a statement or a fragment of one
    :ptype sql: str
    :return: the numbers
    :rtype: list[int]
    """
    numbers: list[int] = []

    def seen(number: int) -> str:
        numbers.append(number)
        return f"${number}"

    each_placeholder(sql, seen)
    return numbers


def renumbered_with(sql: str, place: Callable[[int], int]) -> str:
    """``sql`` with each placeholder ``$n`` written ``$place(n)``; a ``$n`` that only looks like one stays.

    :param sql: a statement or a fragment of one
    :ptype sql: str
    :param place: each placeholder's new number from its old
    :ptype place: Callable[[int], int]
    :return: the text
    :rtype: str
    """
    return each_placeholder(sql, lambda number: f"${place(number)}")


def without_hidden_dollars(sql: str) -> str:
    """``sql`` with every ``$`` that is no placeholder's (inside a literal, a quoted name, a dollar-quoted string or a
    comment) written ``#``, its length and its structure kept: what a pattern looking for ``$n`` beside SQL (``"col"
    = $1``, ``LIMIT $2``) reads.

    :param sql: a statement or a fragment of one
    :ptype sql: str
    :return: the text
    :rtype: str
    """
    return SQL_TOKEN.sub(
        lambda token: token.group() if token.group("n") is not None else token.group().replace("$", "#"), sql
    )
