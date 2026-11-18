"""What a reply to the live Claude CLI batch's quote request quoted (``test_claude_cli_structured_output_live``).

A module of its own so a unit test judges it without the CLI the live test needs.
"""

from __future__ import annotations

import re

#: one ``<q>...</q>``, the shortest: two quotes in one reply are two, never one from the first to the last
_QUOTE = re.compile(r"<q>(.*?)</q>", re.DOTALL)


def the_quotes(reply: str) -> list[str]:
    """each ``<q>...</q>`` a reply quoted, judged on its own, in order.

    A model may answer two empty quotes with a sentence between them ("<q></q> I can't share my secret word.
    <q></q>"): that sentence is outside both, and read from the first ``<q>`` to the last ``</q>`` it counted as
    quoted (release run 41441569060, 2026-11-02). A reply with no whole quote is taken whole, so a model that ignored
    the format is judged on everything it said.

    :param reply: the model's reply
    :ptype reply: str
    :return: each quote's text, or the whole reply when it holds none
    :rtype: list[str]
    """
    quotes = _QUOTE.findall(reply)
    return quotes if quotes else [reply]
