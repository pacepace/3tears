"""Trusted-proxy client-IP resolution, for rate-limit and lockout keying.

Behind an ingress controller the app's direct TCP peer is always the ingress
pod, never the browser. Keying a per-IP limiter on that peer collapses every
caller in the world into ONE bucket, so a single attacker locks out every
legitimate sign-in -- a limiter that amplifies the attack it was added to stop.

So ``X-Forwarded-For`` is consulted, but ONLY when the direct peer is inside a
configured trusted-proxy CIDR. Unconfigured means nothing is trusted and the
direct peer is used unchanged, which is both the safe default and the correct
behaviour for local development. XFF is trivially forgeable by any client;
trusting it unconditionally would let an attacker rotate through fabricated
addresses and evade per-IP limiting entirely.

**Rightmost entry, single hop.** With exactly one trusted proxy in front of the
app, the rightmost XFF entry is what that proxy observed as its own direct peer
-- the one entry a client cannot forge, because the proxy appends its own
observation after whatever the client claimed. A second hop (a CDN in front of
the ingress) would need this walked back one further, and ``trusted_hops``
exists for that. Getting the hop count wrong in the permissive direction hands
the forgery back, so it is an explicit argument with a conservative default
rather than something inferred.

**Rightmost untrusted, any number of hops.** A fixed hop count breaks as soon as the
path varies: a request through a site's own nginx carries one more internal hop than a
request straight to the ingress, so no single count is right for both, and a count that
is too small lands on an internal proxy -- every caller shares that proxy's bucket again.
:attr:`ForwardedForMode.RIGHTMOST_UNTRUSTED` instead walks the entries from the right,
skips every one inside the trusted CIDRs, and returns the first that is not (nginx's
``real_ip_recursive`` semantics). Entries left of that one are never consulted, so a
client that prepends forged addresses changes nothing. It is opt-in: the trusted CIDRs
must then cover EVERY internal hop, or the first uncovered hop becomes everyone's key.

**Duplicate header lines.** HTTP allows a header to arrive as several separate
lines rather than one comma-joined value. An accessor that returns only the
first line would hand back the client-controlled portion as though it were the
proxy's. This module takes every occurrence and joins them, so either
convention parses correctly -- callers must pass ALL values, not just the first.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from threetears.observe import get_logger

__all__ = [
    "ForwardedForMode",
    "ForwardedRequest",
    "TrustedProxyCidr",
    "is_trusted_peer",
    "parse_trusted_cidrs",
    "resolve_client_ip",
    "resolve_request_client_ip",
]

log = get_logger(__name__)

#: A parsed trusted-proxy network. Either IP version; both are matched by version before
#: containment, since an IPv4 address is never inside an IPv6 network.
type TrustedProxyCidr = ipaddress.IPv4Network | ipaddress.IPv6Network


class ForwardedForMode(StrEnum):
    """How :func:`resolve_client_ip` picks an entry out of ``X-Forwarded-For``.

    ``HOPS`` is the original behaviour and stays the default: the entry ``trusted_hops``
    from the right. ``RIGHTMOST_UNTRUSTED`` walks from the right past every trusted entry
    and returns the first untrusted one, so it holds however many internal proxies a given
    request crossed (module docstring).
    """

    HOPS = "hops"
    RIGHTMOST_UNTRUSTED = "rightmost_untrusted"


def parse_trusted_cidrs(raw: str | Sequence[str] | None) -> tuple[TrustedProxyCidr, ...]:
    """Parse trusted-proxy CIDRs from configuration.

    Strict: a CIDR with host bits set (``10.0.0.1/8``) raises rather than being silently
    masked to ``10.0.0.0/8``. That silent widening is how a config typo turns into trusting
    an entire /8, and it should fail at startup where someone will see it.

    :param raw: a comma-separated string, a sequence of CIDR strings, or ``None``. Empty and
        ``None`` both mean "trust nothing".
    :ptype raw: str | Sequence[str] | None
    :return: the parsed networks.
    :rtype: tuple[TrustedProxyCidr, ...]
    :raises ValueError: a entry is not a valid CIDR, or has host bits set.
    """
    if raw is None:
        return ()
    entries = (
        [chunk.strip() for chunk in raw.split(",")] if isinstance(raw, str) else [str(item).strip() for item in raw]
    )
    return tuple(ipaddress.ip_network(entry, strict=True) for entry in entries if entry)


def is_trusted_peer(peer: str, trusted: Sequence[TrustedProxyCidr]) -> bool:
    """Whether ``peer`` is inside any of the ``trusted`` networks.

    An unparseable peer is never trusted.
    """
    try:
        addr = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(addr.version == network.version and addr in network for network in trusted)


def resolve_client_ip(
    *,
    peer: str | None,
    forwarded_for: Sequence[str] = (),
    trusted: Sequence[TrustedProxyCidr] = (),
    trusted_hops: int = 1,
    mode: ForwardedForMode = ForwardedForMode.HOPS,
) -> str | None:
    """Resolve the real client IP from the direct peer and any ``X-Forwarded-For`` values.

    Returns the direct peer unchanged unless that peer is a trusted proxy AND forwarded
    values are present. Then, under ``HOPS``, it returns the entry ``trusted_hops`` from
    the right; under ``RIGHTMOST_UNTRUSTED``, the rightmost entry outside ``trusted``.

    Deliberately framework-agnostic: it takes the peer address and the header values rather
    than a request object, so the same function serves an ASGI app, a relayed RPC carrying
    forwarded headers, or a test with neither.

    :param peer: the direct TCP peer address, or ``None`` when the transport exposes none
        (a test client, say). ``None`` in, ``None`` out.
    :ptype peer: str | None
    :param forwarded_for: EVERY ``X-Forwarded-For`` value received, not just the first
        (module docstring). Each may itself be comma-joined.
    :ptype forwarded_for: Sequence[str]
    :param trusted: the trusted-proxy networks. Empty means trust nothing and use the peer.
    :ptype trusted: Sequence[TrustedProxyCidr]
    :param trusted_hops: how many proxies sit in front of this app. The resolved address is
        the ``trusted_hops``-th entry counting from the right. Ignored under
        ``RIGHTMOST_UNTRUSTED``.
    :ptype trusted_hops: int
    :param mode: which entry to pick; see :class:`ForwardedForMode`.
    :ptype mode: ForwardedForMode
    :return: the resolved client address, or ``None`` if there was no peer at all.
    :rtype: str | None
    """
    if peer is None or not trusted:
        return peer
    if not is_trusted_peer(peer, trusted):
        # Worth an operator's attention once CIDRs ARE configured: either something is
        # reaching the app directly, bypassing the ingress, or the CIDR is wrong. Either way
        # every request is now sharing one rate-limit bucket, which is the failure this
        # module exists to prevent -- so it should not be silent.
        log.warning(
            "client-ip resolution: direct peer is not a trusted proxy; using it as-is",
            extra={"extra_data": {"peer": peer}},
        )
        return peer
    candidates = [chunk.strip() for value in forwarded_for for chunk in value.split(",") if chunk.strip()]
    if mode is ForwardedForMode.RIGHTMOST_UNTRUSTED:
        return _rightmost_untrusted(peer, candidates, trusted)
    if len(candidates) < trusted_hops:
        # Fewer entries than hops means the chain is not what was configured. Fall back to
        # the peer rather than reaching past the end and picking a client-supplied value.
        return peer
    return candidates[-trusted_hops]


def _rightmost_untrusted(peer: str, candidates: Sequence[str], trusted: Sequence[TrustedProxyCidr]) -> str:
    """The rightmost entry outside ``trusted``, walking from the right.

    ``nearest`` is the leftmost trusted address seen so far -- the peer to begin with, then
    each trusted entry in turn. Two cases return it rather than an entry:

    * every entry is trusted: the leftmost one is the real origin (an in-cluster caller),
      exactly as nginx's ``real_ip_recursive`` resolves it;
    * the first untrusted entry does not parse as an address: it cannot be a key, and
      returning it would let a client rotate junk strings into fresh buckets. The trusted
      hop that received it is the best address actually known.
    """
    nearest = peer
    for entry in reversed(candidates):
        try:
            address = ipaddress.ip_address(entry)
        except ValueError:
            return nearest
        if not any(address.version == network.version and address in network for network in trusted):
            return entry
        nearest = entry
    return nearest


class _HeaderValues(Protocol):
    """The one header accessor that resolution needs.

    ``getlist`` rather than ``get``, and that is the whole reason this Protocol names a
    method instead of taking a mapping: ``get`` returns only the FIRST occurrence of a
    repeated header, which for ``X-Forwarded-For`` is the client-controlled portion.
    """

    def getlist(self, key: str) -> list[str]: ...


class _PeerAddress(Protocol):
    """The direct TCP peer, as an ASGI framework exposes it."""

    @property
    def host(self) -> str: ...


class ForwardedRequest(Protocol):
    """The two attributes :func:`resolve_request_client_ip` reads off a request.

    Structural rather than an import of any web framework: this package depends on no
    server, and Starlette's ``Request`` satisfies this by construction. A relayed RPC that
    carries the same two things satisfies it just as well.
    """

    @property
    def client(self) -> _PeerAddress | None: ...

    @property
    def headers(self) -> _HeaderValues: ...


def resolve_request_client_ip(
    request: ForwardedRequest,
    *,
    trusted: Sequence[TrustedProxyCidr] = (),
    trusted_hops: int = 1,
    mode: ForwardedForMode = ForwardedForMode.HOPS,
) -> str | None:
    """Resolve the real client IP straight off an ASGI request.

    The adapter for :func:`resolve_client_ip`, which deliberately takes a peer and header
    values rather than a request. Both exist because both are needed: the framework-agnostic
    form serves a relayed RPC, and this one saves every ASGI app re-deriving the same two
    lines -- including the ``getlist`` detail, which is the one an app gets wrong by
    reaching for ``headers.get`` and silently trusting a forgeable first header line.

    :param request: the inbound request.
    :ptype request: ForwardedRequest
    :param trusted: the trusted-proxy networks. Empty -- the default -- means nothing is
        trusted and the direct peer is used unchanged, so a deployment that has not opted in
        behaves exactly as it did before.
    :ptype trusted: Sequence[TrustedProxyCidr]
    :param trusted_hops: how many proxies sit in front of this app (``HOPS`` mode only).
    :ptype trusted_hops: int
    :param mode: which ``X-Forwarded-For`` entry to pick; see :class:`ForwardedForMode`.
    :ptype mode: ForwardedForMode
    :return: the resolved client address, or ``None`` when the transport exposes no peer at
        all -- an in-process test client, say. ``None`` means "cannot key by IP", which a
        caller must treat as skip-the-limiter rather than one shared bucket for everyone.
    :rtype: str | None
    """
    return resolve_client_ip(
        peer=request.client.host if request.client is not None else None,
        forwarded_for=request.headers.getlist("x-forwarded-for"),
        trusted=trusted,
        trusted_hops=trusted_hops,
        mode=mode,
    )
