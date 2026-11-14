"""The listing actions' pages: how big a page is, and the cursor a caller passes back for the next one.

A listing (``runs_list``, ``campaigns_list``) reads its scope newest first, by ``(created_at, id)``, a page at a time
(:func:`~threetears.evals.schema.store_port.page_by_doc_type`), and answers each page with ``next``: the cursor for the
page after it, or ``None`` when no document remains. A caller reads every page by passing ``next`` back as ``cursor``
until ``next`` is ``None``.

**The cursor is the package's own, and opaque.** It is a short string that names the listing it came from and the
place in that listing's order the next page starts past: the last document's ``created_at`` and ``id``, as the store
holds them. It depends on no host's codec, so a host's surface hands it through unchanged, and a caller never builds
one. A cursor from another listing, or one that does not decode, is refused as caller input, never read as the first
page: a caller that lost its place is told so, rather than handed the top of the listing again.

A cursor holds a document's id and time, nothing a caller of the listing could not already read off the page it came
with, so it is not signed.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Final

from threetears.evals.kernel.errors import ValidationFailedError

__all__ = [
    "LIST_PAGE_MAX",
    "LIST_PAGE_SIZE",
    "NEXT_DESCRIPTION",
    "check_page_limit",
    "decode_list_cursor",
    "encode_list_cursor",
]

#: How many rows a listing page reads when its caller names no ``limit``.
LIST_PAGE_SIZE: Final[int] = 100

#: The most rows one listing page reads; a caller wanting more reads the next page.
LIST_PAGE_MAX: Final[int] = 1000

#: What a listing page's ``next`` is, said once for every listing that pages.
NEXT_DESCRIPTION: Final[str] = (
    "The cursor for the next page: pass it back as cursor. None when no row remains past this page. Read every "
    "page by calling again with cursor=next until next is None."
)

#: The format's version, the cursor's first characters: a later format is told apart rather than misread.
_FORMAT: Final[str] = "c1."

#: What a caller does with a cursor that is refused.
_REMEDY: Final[str] = "pass back the next a page of the same listing returned, or no cursor to read the first page"


def encode_list_cursor(listing: str, created_at: Any, doc_id: str) -> str:
    """The cursor for the page of ``listing`` that starts past the document ``(created_at, doc_id)``.

    :param listing: The listing's name (``"runs_list"``), which :func:`decode_list_cursor` checks.
    :ptype listing: str
    :param created_at: The last document's ``created_at``, as the store holds it (``None`` where it holds none).
    :ptype created_at: Any
    :param doc_id: The last document's id.
    :ptype doc_id: str
    :return: The cursor.
    :rtype: str
    """
    packed = json.dumps([listing, created_at, doc_id], separators=(",", ":"), ensure_ascii=False)
    return _FORMAT + base64.urlsafe_b64encode(packed.encode("utf-8")).decode("ascii").rstrip("=")


def decode_list_cursor(listing: str, cursor: str) -> tuple[Any, str]:
    """The place a cursor from :func:`encode_list_cursor` names in ``listing``'s order: ``(created_at, id)``.

    :param listing: The listing reading it.
    :ptype listing: str
    :param cursor: The cursor, as a page of the listing returned it.
    :ptype cursor: str
    :return: ``(created_at, id)``, for :meth:`~threetears.evals.schema.DocumentStore.by_doc_type`'s ``after``.
    :rtype: tuple[Any, str]
    :raises ValidationFailedError: The cursor does not decode, or came from another listing.
    """
    text = cursor.strip()
    if not text.startswith(_FORMAT):
        raise ValidationFailedError(
            f"cursor {cursor!r} is not a page cursor this build returned", remedy=_REMEDY, parameters=["cursor"]
        )
    body = text.removeprefix(_FORMAT)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8"))
    except ValueError as unread:  # bad base64 (binascii.Error), bad UTF-8, bad JSON: each a ValueError
        raise ValidationFailedError(
            f"cursor {cursor!r} is not a page cursor this build returned ({unread})",
            remedy=_REMEDY,
            parameters=["cursor"],
        ) from unread
    if (
        not isinstance(decoded, list)
        or len(decoded) != 3
        or not isinstance(decoded[2], str)
        or not (decoded[1] is None or isinstance(decoded[1], str | int | float))
    ):
        raise ValidationFailedError(
            f"cursor {cursor!r} is not a page cursor this build returned", remedy=_REMEDY, parameters=["cursor"]
        )
    if decoded[0] != listing:
        raise ValidationFailedError(
            f"cursor {cursor!r} is a page of {decoded[0]!r}, not of {listing!r}", remedy=_REMEDY, parameters=["cursor"]
        )
    return decoded[1], decoded[2]


def check_page_limit(limit: int) -> int:
    """A listing's page size, checked: 1 to :data:`LIST_PAGE_MAX`.

    :param limit: The size asked for.
    :ptype limit: int
    :return: ``limit``.
    :rtype: int
    :raises ValidationFailedError: It is out of range.
    """
    if not 1 <= limit <= LIST_PAGE_MAX:
        raise ValidationFailedError(
            f"limit {limit} is not a page size: a page reads 1 to {LIST_PAGE_MAX} rows",
            remedy=f"ask for 1 to {LIST_PAGE_MAX} rows a page, and read the next page with next",
            parameters=["limit"],
        )
    return limit
