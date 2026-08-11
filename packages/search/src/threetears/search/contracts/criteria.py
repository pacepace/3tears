"""Criteria -- one open vocabulary, stated once, whoever satisfies it (SR-B1, P6).

A criterion is a key/value pair. Well-known criteria ship as typed
constructors on :class:`Criterion`; anything else travels under a
*namespaced* key (``<namespace>:<name>``). The vocabulary is deliberately
NOT a closed enum: a plain (un-namespaced) key must be one this module
declares well-known, and everything unknown must say whose vocabulary it
belongs to -- so an unrecognised criterion is identifiable, reportable as
``ignored-unknown``, and can never be mistaken for a typo'd well-known key.

The response answers per criterion (SR-B2, SR-B3, P8): each criterion is
met by provider pushdown, by local application, or not at all -- and the
caller is told which, per criterion, in the typed response. An
unsatisfiable criterion is named, never silently dropped (the RES-T4M9
precedent: Tavily 400s when ``time_range`` accompanies absolute dates;
the fix was stated precedence, not silent suppression).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from typing import Final, Literal

from pydantic import JsonValue, field_validator

from threetears.search.contracts._base import ContractModel

__all__ = [
    "CRITERION_CARRIER",
    "CRITERION_DOMAINS_EXCLUDE",
    "CRITERION_DOMAINS_INCLUDE",
    "CRITERION_LANGUAGE",
    "CRITERION_MAX_RESULTS",
    "CRITERION_MIN_RESOLUTION",
    "CRITERION_RIGHTS_CLASS",
    "CRITERION_TIME_RANGE",
    "WELL_KNOWN_CRITERIA",
    "Criterion",
    "CriterionDisposition",
    "Disposition",
]

#: absolute publication-time window; value ``{"start": iso8601?, "end": iso8601?}``.
CRITERION_TIME_RANGE: Final[str] = "time-range"
#: restrict results to these domains; value is a list of domain strings.
CRITERION_DOMAINS_INCLUDE: Final[str] = "domains-include"
#: exclude results from these domains; value is a list of domain strings.
CRITERION_DOMAINS_EXCLUDE: Final[str] = "domains-exclude"
#: BCP 47 language tag; value is the tag string.
CRITERION_LANGUAGE: Final[str] = "language"
#: carrier scoping, keyed by the ``media-contracts`` carrier taxonomy
#: (``MediaInfo.media_category``: "image", "audio", "video", "document", ...).
#: A criterion, not a second tool and not a closed union (D17, SR-C1).
CRITERION_CARRIER: Final[str] = "carrier"
#: minimum pixel dimensions; value ``{"width": int, "height": int}``.
CRITERION_MIN_RESOLUTION: Final[str] = "min-resolution"
#: rights / licensing class; value is an open string (facet vocabulary
#: pins to ``media-contracts`` when its rights facet lands -- SR-C3).
CRITERION_RIGHTS_CLASS: Final[str] = "rights-class"
#: cap on returned candidates; value is a positive int. NOTE (SR-E5):
#: for per-request-priced providers this changes what you see, not what
#: you pay.
CRITERION_MAX_RESULTS: Final[str] = "max-results"

#: every plain (un-namespaced) key this contract version understands.
#: Additive within a family minor (D13); never a closed enum -- unknown
#: criteria ride namespaced keys instead of extending this set ad hoc.
WELL_KNOWN_CRITERIA: Final[frozenset[str]] = frozenset(
    {
        CRITERION_TIME_RANGE,
        CRITERION_DOMAINS_INCLUDE,
        CRITERION_DOMAINS_EXCLUDE,
        CRITERION_LANGUAGE,
        CRITERION_CARRIER,
        CRITERION_MIN_RESOLUTION,
        CRITERION_RIGHTS_CLASS,
        CRITERION_MAX_RESULTS,
    }
)

#: how one criterion was satisfied (or honestly not) -- SR-B2/SR-B3/P8.
Disposition = Literal["pushdown", "local", "unsatisfied", "ignored-unknown"]

_NAMESPACED_KEY: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]*:[a-z0-9][a-z0-9._-]*$")


class Criterion(ContractModel):
    """One constraint on what comes back.

    Build well-known criteria through the typed constructors; build
    foreign ones with :meth:`namespaced`. Values are JSON values by type,
    so a criterion can never smuggle a callable or a port into a payload
    (SR-L4).
    """

    #: a well-known key from :data:`WELL_KNOWN_CRITERIA`, or a namespaced
    #: ``<namespace>:<name>`` key for vocabulary this contract does not own.
    key: str
    #: the constraint's value; shape is defined per key.
    value: JsonValue

    @field_validator("key")
    @classmethod
    def _key_is_well_known_or_namespaced(cls, key: str) -> str:
        """Reject plain keys outside the well-known set.

        :param key: the candidate criterion key
        :ptype key: str
        :return: the validated key
        :rtype: str
        :raises ValueError: when a plain key is not well-known and carries
            no namespace
        """
        if key in WELL_KNOWN_CRITERIA:
            return key
        if _NAMESPACED_KEY.match(key):
            return key
        raise ValueError(
            f"criterion key {key!r} is neither a well-known key ({sorted(WELL_KNOWN_CRITERIA)}) "
            f"nor a namespaced '<namespace>:<name>' key; unknown criteria must be namespaced"
        )

    @classmethod
    def time_range(cls, *, start: datetime | None = None, end: datetime | None = None) -> Criterion:
        """Constrain publication time to an absolute window.

        :param start: earliest publication time (inclusive), if bounded below
        :ptype start: datetime | None
        :param end: latest publication time (inclusive), if bounded above
        :ptype end: datetime | None
        :return: the criterion
        :rtype: Criterion
        :raises ValueError: when neither bound is supplied
        """
        if start is None and end is None:
            raise ValueError("time_range requires at least one of start=, end=")
        value: dict[str, JsonValue] = {}
        if start is not None:
            value["start"] = start.isoformat()
        if end is not None:
            value["end"] = end.isoformat()
        return cls(key=CRITERION_TIME_RANGE, value=value)

    @classmethod
    def domains_include(cls, domains: Sequence[str]) -> Criterion:
        """Restrict results to the given domains.

        :param domains: domains results must come from
        :ptype domains: Sequence[str]
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_DOMAINS_INCLUDE, value=list(domains))

    @classmethod
    def domains_exclude(cls, domains: Sequence[str]) -> Criterion:
        """Exclude results from the given domains.

        :param domains: domains results must not come from
        :ptype domains: Sequence[str]
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_DOMAINS_EXCLUDE, value=list(domains))

    @classmethod
    def language(cls, tag: str) -> Criterion:
        """Constrain result language.

        :param tag: BCP 47 language tag (e.g. ``en``, ``pt-BR``)
        :ptype tag: str
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_LANGUAGE, value=tag)

    @classmethod
    def carrier(cls, media_category: str) -> Criterion:
        """Scope results to one carrier, in the ``media-contracts`` taxonomy.

        :param media_category: carrier name per ``MediaInfo.media_category``
            (``image``, ``audio``, ``video``, ``document``, ...); the
            taxonomy is open
        :ptype media_category: str
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_CARRIER, value=media_category)

    @classmethod
    def min_resolution(cls, *, width: int, height: int) -> Criterion:
        """Require at least the given pixel dimensions.

        :param width: minimum width in pixels
        :ptype width: int
        :param height: minimum height in pixels
        :ptype height: int
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_MIN_RESOLUTION, value={"width": width, "height": height})

    @classmethod
    def rights_class(cls, rights: str) -> Criterion:
        """Constrain results to a rights / licensing class.

        :param rights: rights-class label (open vocabulary; pins to the
            ``media-contracts`` rights facet when it lands)
        :ptype rights: str
        :return: the criterion
        :rtype: Criterion
        """
        return cls(key=CRITERION_RIGHTS_CLASS, value=rights)

    @classmethod
    def max_results(cls, count: int) -> Criterion:
        """Cap the number of candidates returned.

        :param count: maximum candidates wanted (positive)
        :ptype count: int
        :return: the criterion
        :rtype: Criterion
        :raises ValueError: when ``count`` is not positive
        """
        if count < 1:
            raise ValueError(f"max_results requires a positive count, got {count}")
        return cls(key=CRITERION_MAX_RESULTS, value=count)

    @classmethod
    def namespaced(cls, namespace: str, name: str, value: JsonValue) -> Criterion:
        """Build a criterion in a foreign vocabulary.

        :param namespace: whose vocabulary the key belongs to (e.g. a
            provider or consumer name)
        :ptype namespace: str
        :param name: the criterion name within that vocabulary
        :ptype name: str
        :param value: the constraint's value (JSON value)
        :ptype value: JsonValue
        :return: the criterion, keyed ``<namespace>:<name>``
        :rtype: Criterion
        """
        return cls(key=f"{namespace}:{name}", value=value)


class CriterionDisposition(ContractModel):
    """The response's per-criterion answer: how was this criterion met?

    One entry per criterion in the request, always (SR-B2). ``unsatisfied``
    and ``ignored-unknown`` are honest answers, not errors -- the caller
    decides whether a partially-honoured request is usable (P8).
    """

    #: the criterion key this disposition answers for.
    criterion_key: str
    #: how the criterion was handled: pushed down to the provider, applied
    #: locally, named as unsatisfiable, or ignored as unrecognised.
    disposition: Disposition
    #: human-readable specifics -- e.g. *why* a criterion is unsatisfiable,
    #: or which precedence rule applied (the RES-T4M9 teaching text).
    detail: str | None = None
