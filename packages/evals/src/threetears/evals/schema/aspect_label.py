"""What an aspect named after its campaign's behavior is called: the one rule, so no caller keeps a copy.

An aspect's label is the first part of each floor key on it (``billing/rate_drop/*``), which ``/`` separates, so a
label never holds one. A behavior may (``tools/search``), and an aspect named after it is spelled with ``-`` in its
place. Every place the engine names an aspect after a behavior calls :func:`aspect_label_for`: the quick path's one
aspect, a sweep's, a stored v15 campaign read as one aspect, and an aspect a create or an update declares without a
label.
"""

from __future__ import annotations

__all__ = ["aspect_label_for"]


def aspect_label_for(behavior: str) -> str:
    """The label of an aspect named after ``behavior``: each ``/`` spelled ``-``, since ``/`` separates a floor
    key's parts.

    :param behavior: The campaign's behavior, e.g. ``tools/search``.
    :ptype behavior: str
    :return: The label, e.g. ``tools-search``.
    :rtype: str
    """
    return behavior.replace("/", "-")
