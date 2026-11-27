"""A fetched candidate page's structure, read in a worker process: the CPU half of the page finder's verification.

**Why a worker.** Parsing the body is CPU that grows with the page, up to the 2 MiB verification cap: on the event
loop BeautifulSoup's ``html.parser`` held it 2.4 s for a 2 MiB page (``_verify_candidate_page``, the loop-stall guard
on the testament pool, 2026-11-26). A thread would hold the interpreter lock just as long, so the parse runs in a
worker process (:class:`~threetears.core.process_workers.ProcessWorkers`), the fetch staying on the loop.

This module imports only what the parse needs, so the worker that imports it stays small: never the page finder's
agent, model and search stack.
"""

from __future__ import annotations

import json

from bs4 import BeautifulSoup

__all__ = ["page_structure"]

#: a link ending in one of these is a document the scrape pipeline can extract
_DOCUMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xlsx", ".csv")


def page_structure(
    raw: bytes, content_type: str, encoding: str, truncated: bool, max_bytes: int
) -> tuple[bool, str, str]:
    """Whether a fetched body has real extractable structure: a JSON list, an HTML table, a document link.

    A worker process's job: plain data in and out.

    :param raw: the body bytes, already capped at ``max_bytes``
    :ptype raw: bytes
    :param content_type: the response's ``content-type`` header, ``""`` when absent
    :ptype content_type: str
    :param encoding: a codec Python knows, to decode the body with (never strictly)
    :ptype encoding: str
    :param truncated: whether the page was longer than ``max_bytes``
    :ptype truncated: bool
    :param max_bytes: the verification cap, for the note
    :ptype max_bytes: int
    :return: (verified, driver_backend guess, human-readable note on what was found)
    :rtype: tuple[bool, str, str]
    """
    if "json" in content_type:
        # A truncated body is not parseable JSON, so this correctly declines to
        # call a cut-off document an API rather than guessing at the missing half.
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        has_list = isinstance(body, list) or (
            isinstance(body, dict) and any(isinstance(v, list) for v in body.values())
        )
        if has_list:
            return True, "api", "response is JSON containing a list -- looks like a real API"

    # Decoded never strictly: a page whose bytes do not match its declared
    # charset (or that got cut mid-character at the cap) must still be
    # inspectable for structure. It always is, because every marker below --
    # <table>, <tr>, href -- is ASCII, so structure detection survives text
    # this cannot decode.
    soup = BeautifulSoup(raw.decode(encoding, errors="replace"), "html.parser")

    # Table checked before document link: a real page can carry an incidental PDF link
    # (privacy policy, a related-regulations reference) alongside its actual notices table --
    # live-verified against Maryland's real WARN page, which has exactly this shape. A real
    # table is the stronger, more specific signal of "this is the actual data source."
    for table in soup.find_all("table"):
        if len(table.find_all("tr")) >= 2:
            return True, "nodriver", "found a real HTML table with multiple rows"

    for link in soup.find_all("a", href=True):
        href = str(link["href"]).lower()
        if href.endswith(_DOCUMENT_EXTENSIONS):
            return True, "document", f"found a document link ({href})"

    # "Nothing in the part I read" is a weaker claim than "nothing on the page",
    # and a note that conflated them would send the next reader looking for a
    # structure bug that is really a size cap.
    if truncated:
        return (
            False,
            "nodriver",
            f"no table, document link, or JSON list in the first {max_bytes} bytes "
            "of the page, which was longer than the verification cap",
        )
    return False, "nodriver", "no table, document link, or JSON list found on the fetched page"
