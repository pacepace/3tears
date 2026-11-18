"""A candidate's text facet, read as the string the extractor writes there."""

from __future__ import annotations

from threetears.search.contracts.candidate import Candidate

__all__ = ["facet_text"]


def facet_text(candidate: Candidate, key: str) -> str:
    """The text ``candidate`` carries under facet ``key``; a facet that is not text fails the test here.

    :param candidate: the candidate the extractor returned
    :ptype candidate: Candidate
    :param key: the facet's name, such as the extraction reason or next step
    :ptype key: str
    :return: the facet's text
    :rtype: str
    """
    value = candidate.facets[key]
    assert isinstance(value, str), f"facet {key!r} is {value!r}, not text"
    return value
