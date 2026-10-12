"""The SSRF guard: which addresses a guarded fetch may reach, checked on every request it sends.

:class:`~threetears.scrape.tool.ScrapeTool` fetches a caller-supplied URL, so it refuses one whose
host resolves to a non-public address before any fetch. Checking only that first URL is not
enough: every HTTP client this package builds follows redirects, so a public URL answering
``302 Location: http://169.254.169.254/...`` sent the fetch to the metadata endpoint after the
check had passed. Drivers that read links out of a page (``listing_detail``'s detail rows,
``multi_document``'s documents) send requests to URLs the page chose, which the first check
never saw either.

So the check runs on every REQUEST, not on the URL a caller handed over. :func:`refuse_private_hosts`
is an httpx request hook, installed on each client the drivers and the robots gate build, and
httpx calls request hooks once per hop of a redirect chain as well as for each new request.

**Whether it refuses is the tool's decision, carried by context.** ``block_private_hosts`` is a
:class:`~threetears.scrape.tool.ScrapeTool` setting, and drivers are built and shared
independently of any one tool, so the setting cannot be a driver constructor argument without
two places to configure one property. The tool opens :func:`refusing_private_hosts` around its
own fetches; the hook reads it. Outside that scope -- a driver driven directly by a scheduler,
or a tool built with ``block_private_hosts=False`` -- the hook does nothing, which is exactly the
behaviour those callers had before.

**What it does not cover**, so nobody reads more into it than is there:

- A browser-rendered fetch (``camoufox``, ``nodriver`` and ``nodriver_download`` through the
  sidecar). The browser follows redirects and loads subresources itself, and none of that passes
  through an httpx client. Only the first URL is checked for those backends.
- A client or fetcher the caller injects (``ApiDriver(client=...)``, ``RobotsGate(fetch=...)``,
  and the other drivers' ``client``). It is used exactly as given, the same rule those drivers
  already apply to egress: rebinding someone else's client would override a decision they made.
- DNS rebinding. The check resolves the host, then httpx resolves it again to connect, so a name
  that answers differently the second time is not caught. Pinning the connection to the checked
  address is the fix for that, and it is not built.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from threetears.observe import get_logger

if TYPE_CHECKING:
    # Annotation only: `robots.py` imports this module and stays importable without httpx.
    import httpx

__all__ = ["PrivateHostRefusedError", "refuse_private_hosts", "refusing_private_hosts", "ssrf_block_reason"]

log = get_logger(__name__)

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

#: Whether the fetch running in this context refuses non-public hosts. A ``ContextVar`` because
#: the setting belongs to the tool and the clients belong to the drivers; it is copied into any
#: task or thread a driver starts, so a fetch fanned out by a driver is still inside the scope.
_REFUSING: ContextVar[bool] = ContextVar("_REFUSING", default=False)


class PrivateHostRefusedError(Exception):
    """A guarded fetch was about to send a request to a host the SSRF guard refuses.

    Not an ``httpx`` error, on purpose: the drivers turn ``httpx`` transport errors into their
    own "transport" failures, and this is not one. Nothing was sent to the refused host.

    :param url: the request URL that was refused
    :ptype url: str
    :param reason: why, as :func:`ssrf_block_reason` put it
    :ptype reason: str
    """

    def __init__(self, url: str, reason: str) -> None:
        """
        :param url: the request URL that was refused
        :ptype url: str
        :param reason: why, as :func:`ssrf_block_reason` put it
        :ptype reason: str
        """
        super().__init__(f"refused {url!r}: {reason}")
        self.url = url
        self.reason = reason


def ssrf_block_reason(url: str) -> str | None:
    """Return a reason to REFUSE fetching *url*, or ``None`` if it is allowed.

    Refuses any non-http(s) scheme and any host that RESOLVES to a private, loopback,
    link-local, reserved, multicast or unspecified address. Every address the name resolves to
    is checked, so a public-looking hostname with an internal address among its records is
    refused even when its other records are public.

    It does not catch DNS rebinding: this resolves the name once, and the client resolves it
    again when it connects, so a name that answers with a public address here and an internal
    one there gets through. Blocking DNS, so an async caller runs it in a thread.

    :param url: the URL about to be fetched
    :ptype url: str
    :return: a human-readable refusal reason, or ``None`` when the URL is a public http(s)
        target safe to fetch
    :rtype: str | None
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        return f"scheme {parsed.scheme!r} is not allowed (only http/https)"
    host = parsed.hostname
    if not host:
        return "URL has no host"
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return f"host {host!r} does not resolve ({exc})"
    for info in infos:
        ip_text = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            # NOSILENT: a getaddrinfo entry that isn't a parseable IP literal can't be
            # range-classified; skip it and check the remaining resolved addresses.
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return f"host {host!r} resolves to non-public address {ip_text} (private/loopback/link-local/reserved)"
    return None


@contextmanager
def refusing_private_hosts(enabled: bool) -> Iterator[None]:
    """Run the enclosed fetches with the SSRF guard on or off.

    Sets the value either way rather than only turning it on: a tool built with
    ``block_private_hosts=False`` is saying how ITS fetches behave, and that holds even when
    the tool is itself called from inside a guarded scope.

    :param enabled: whether requests sent inside the block are checked
    :ptype enabled: bool
    :return: a context manager scoping the setting to the block
    :rtype: Iterator[None]
    """
    token = _REFUSING.set(enabled)
    try:
        yield
    finally:
        _REFUSING.reset(token)


async def refuse_private_hosts(request: httpx.Request) -> None:
    """httpx request hook: refuse a request to a non-public host while the guard is on.

    httpx runs request hooks before every request a client sends, each redirect hop included,
    so this sees the hop the first check never did. A refusal is logged here, where it is
    decided, as the security event it is -- some callers degrade on it (a skipped detail row,
    an unreadable ``robots.txt``) and would otherwise leave no trace that something on the
    other end tried to send a fetch inward.

    :param request: the request httpx is about to send
    :ptype request: httpx.Request
    :return: nothing
    :rtype: None
    :raises PrivateHostRefusedError: the guard is on and the request's host is refused
    """
    if not _REFUSING.get():
        return
    url = str(request.url)
    reason = await asyncio.to_thread(ssrf_block_reason, url)
    if reason is None:
        return
    log.warning(
        "scrape: SSRF guard refused a request to %r -- %s",
        url,
        reason,
        extra={"extra_data": {"url": url, "reason": reason}},
    )
    raise PrivateHostRefusedError(url, reason)
