"""A fetched carrier read to text: the CPU half of Extract (:mod:`threetears.search.extract`), plain data in and out.

**Why a module of its own.** Decoding a page (the charset detector on a page that names none), its article
(trafilatura), its title and date, and a thin page's frames are CPU that grows with the page, up to the 2 MiB a page
may be. On an event loop they hold it: the agent tools' web fetch held the loop 0.334 s reading a small fixture page,
38 of 39 samples in trafilatura (its first read loading justext's stoplists; the loop-stall guard on the testament
pool, 2026-11-26). So every job here is a module function taking the body's bytes and plain values and answering
plain values -- text, numbers, ``None``, tuples of them -- so a host runs it in a worker process
(:class:`~threetears.search.extract.PageWorkers`), and the fetch stays on the loop. A host that passes no workers
(an embedded consumer with no worker pool) runs the same functions on its own thread, as before.

This module imports only what the read needs, never the leaf's contracts or transports, so the worker that imports
it stays small.
"""

from __future__ import annotations

import codecs
import json
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urljoin, urlsplit

try:
    import trafilatura
    import trafilatura.metadata

    TRAFILATURA_INSTALLED = True
except ImportError:  # NOSILENT: the extract extra is optional; extract() refuses with its name
    TRAFILATURA_INSTALLED = False

try:
    import charset_normalizer

    _CHARSET_DETECTOR = True
except ImportError:  # NOSILENT: no detector; UTF-8 with replacement is the stated last resort
    _CHARSET_DETECTOR = False

__all__ = [
    "POINTER_UNREADABLE",
    "TRAFILATURA_INSTALLED",
    "WEB_SCHEMES",
    "page_article",
    "page_facts",
    "page_pointer",
    "page_written",
]

#: The schemes a frame or a pointer can be read at; any other is not an article's address.
WEB_SCHEMES: Final[tuple[str, ...]] = ("http", "https")

#: what :func:`page_pointer` raises on markup or an address it cannot read: the stdlib HTML parser's
#: ``AssertionError`` on a declaration it cannot make sense of, and ``urljoin``'s ``ValueError``.
#: Anything else is a bug in the frame scan and propagates.
POINTER_UNREADABLE: Final[tuple[type[Exception], ...]] = (AssertionError, ValueError)

#: ``charset=`` in a ``Content-Type`` header.
_HEADER_CHARSET = re.compile(r"charset\s*=\s*[\"']?([A-Za-z0-9._:-]+)", re.I)
#: ``<meta charset=...>`` or ``<meta http-equiv=content-type content="...; charset=...">``.
_META_CHARSET = re.compile(rb"<meta[^>]+charset\s*=\s*[\"']?([A-Za-z0-9._:-]+)", re.I)
#: How far into a page a ``<meta>`` charset is looked for (the HTML standard's prescan reads 1,024).
_PRESCAN_BYTES: Final[int] = 4096


def page_written(body: bytes, content_type: str, as_json: bool) -> tuple[str, str, bool]:
    """A textual carrier, or a page's markup, as it came: decoded, and JSON that parses shown indented.

    :param body: the carrier's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :param as_json: whether it says it is JSON, so it is shown indented when it parses
    :ptype as_json: bool
    :return: the text, the charset it was decoded with, and whether it is JSON that does not parse
    :rtype: tuple[str, str, bool]
    """
    decoded, charset = _decode(body, content_type)
    if not as_json:
        return decoded, charset, False
    try:
        return json.dumps(json.loads(decoded), indent=2, ensure_ascii=False), charset, False
    except ValueError:
        # NOSILENT: JSON that does not parse is still the text the site sent, and is shown as sent (the caller notes it)
        return decoded, charset, True


def page_article(body: bytes, content_type: str) -> tuple[str, str]:
    """A page's article, as trafilatura reads it from the decoded markup.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: the article's text (``""`` when the extractor found none) and the charset the page was decoded with
    :rtype: tuple[str, str]
    """
    markup, charset = _decode(body, content_type)
    return trafilatura.extract(markup) or "", charset


def page_facts(body: bytes, content_type: str) -> tuple[str | None, str | None]:
    """A page's own title and publication day, as trafilatura reads them from its markup.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: the title and the day (``YYYY-MM-DD``, midnight UTC), each None when the page does not say
    :rtype: tuple[str | None, str | None]
    :raises Exception: whatever trafilatura raised reading the page's header; the caller reads
        the page without them and says so, since "the page does not say" is a different answer
    """
    markup, _ = _decode(body, content_type)
    found = trafilatura.metadata.extract_metadata(markup)
    if found is None:
        return None, None
    title = (getattr(found, "title", None) or "").strip() or None
    day = _day(getattr(found, "date", None))
    return title, None if day is None else day.isoformat()


def page_pointer(body: bytes, content_type: str, base: str) -> tuple[str | None, int, bool]:
    """Where a thin page says its article is, how many frames it holds, and whether it said so outright.

    ``citation_pdf_url`` is the page saying outright where its article is, so
    it wins; otherwise the page's first ``iframe``, which is followed only
    when it is the page's one frame. An address that is not a web address
    (``about:blank``, ``javascript:``, ``data:``) holds no article: a frame with
    one is a placeholder an ad or an embed fills in later, so it is neither the
    pointer nor counted among the frames, and a ``citation_pdf_url`` with one is
    passed over for the frames. A frame address that will not split is dropped the
    same way, so it cannot hide a readable frame beside it; only when no frame
    address on the page splits is that an answer of its own.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :param base: the URL the page answered from, to resolve relative addresses
    :ptype base: str
    :return: the absolute URL (None when the page points nowhere), the frame count, and
        whether the URL came from ``citation_pdf_url``
    :rtype: tuple[str | None, int, bool]
    :raises AssertionError: from the stdlib HTML parser, which reports a declaration it cannot
        make sense of that way; the caller says the frames could not be looked for, since
        "points nowhere" is a different answer
    :raises ValueError: from ``urljoin``, on a ``citation_pdf_url`` that will not split, or when
        no frame address on the page splits (``http://[::1``)
    """
    markup, _ = _decode(body, content_type)
    parser = _FramesAndPointers()
    parser.feed(markup)
    parser.close()
    frames: list[str] = []
    resolved = False
    unsplit: ValueError | None = None
    for src in parser.frames:
        try:
            url = urljoin(base, src)
        except ValueError as exc:
            unsplit = unsplit or exc
            continue
        resolved = True
        if urlsplit(url).scheme in WEB_SCHEMES:
            frames.append(url)
    if unsplit is not None and not resolved:
        raise unsplit
    cited = urljoin(base, parser.citation_pdf) if parser.citation_pdf else None
    if cited is not None and urlsplit(cited).scheme in WEB_SCHEMES:
        return cited, len(frames), True
    if frames:
        return frames[0], len(frames), False
    return None, 0, False


def _day(value: object) -> datetime | None:
    """A ``YYYY-MM-DD`` string as midnight UTC on that day, or None.

    :param value: what the page's metadata held
    :ptype value: object
    :return: the day, timezone-aware
    :rtype: datetime | None
    """
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        # NOSILENT: a page date that is not a day is a page with no date, the documented None
        return None


def _decode(body: bytes, content_type: str) -> tuple[str, str]:
    """Decode a page by what it says it is, then what it looks like.

    The header's ``charset`` first, then a byte-order mark, then a ``<meta>``
    charset near the top, then UTF-8 if the bytes are valid UTF-8, then the
    detector trafilatura already depends on. Read as UTF-8 regardless, a page
    in Windows-1252 or Shift JIS came back as replacement characters.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: the text and the charset it was decoded with
    :rtype: tuple[str, str]
    """
    for name in _declared_charsets(body, content_type):
        try:
            return body.decode(name), codecs.lookup(name).name
        except LookupError, UnicodeDecodeError:
            # NOSILENT: a charset the page names wrongly or that Python lacks; the next candidate is tried
            continue
    try:
        return body.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # NOSILENT: not UTF-8, so the detector below decides
        pass
    if _CHARSET_DETECTOR:
        best = charset_normalizer.from_bytes(body).best()
        if best is not None and best.encoding:
            return str(best), codecs.lookup(best.encoding).name
    return body.decode("utf-8", errors="replace"), "utf-8"


def _declared_charsets(body: bytes, content_type: str) -> list[str]:
    """The charsets a page declares, most authoritative first.

    :param body: the page's bytes
    :ptype body: bytes
    :param content_type: the response's ``Content-Type`` header
    :ptype content_type: str
    :return: candidate charset names
    :rtype: list[str]
    """
    names: list[str] = []
    header = _HEADER_CHARSET.search(content_type or "")
    if header:
        names.append(header.group(1))
    for mark, name in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if body.startswith(mark):
            names.insert(0, name)
    meta = _META_CHARSET.search(body[:_PRESCAN_BYTES])
    if meta:
        names.append(meta.group(1).decode("ascii", errors="ignore"))
    return names


class _FramesAndPointers(HTMLParser):
    """Collects a page's ``iframe`` sources and its ``citation_pdf_url``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.frames: list[str] = []
        self.citation_pdf: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        named = {key.lower(): (value or "") for key, value in attrs}
        if tag == "iframe" and named.get("src", "").strip():
            self.frames.append(named["src"].strip())
        elif tag == "meta" and named.get("name", "").lower() == "citation_pdf_url" and named.get("content"):
            self.citation_pdf = self.citation_pdf or named["content"].strip()
