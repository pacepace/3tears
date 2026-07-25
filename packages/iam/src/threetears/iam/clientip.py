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

**Duplicate header lines.** HTTP allows a header to arrive as several separate
lines rather than one comma-joined value. An accessor that returns only the
first line would hand back the client-controlled portion as though it were the
proxy's. This module takes every occurrence and joins them, so either
convention parses correctly -- callers must pass ALL values, not just the first.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence

from threetears.observe import get_logger

__all__ = ["TrustedProxyCidr", "is_trusted_peer", "parse_trusted_cidrs", "resolve_client_ip"]

log = get_logger(__name__)

#: A parsed trusted-proxy network. Either IP version; both are matched by version before
#: containment, since an IPv4 address is never inside an IPv6 network.
type TrustedProxyCidr = ipaddress.IPv4Network | ipaddress.IPv6Network


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
) -> str | None:
    """Resolve the real client IP from the direct peer and any ``X-Forwarded-For`` values.

    Returns the direct peer unchanged unless that peer is a trusted proxy AND forwarded
    values are present, in which case it returns the entry ``trusted_hops`` from the right.

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
        the ``trusted_hops``-th entry counting from the right.
    :ptype trusted_hops: int
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
    if len(candidates) < trusted_hops:
        # Fewer entries than hops means the chain is not what was configured. Fall back to
        # the peer rather than reaching past the end and picking a client-supplied value.
        return peer
    return candidates[-trusted_hops]
