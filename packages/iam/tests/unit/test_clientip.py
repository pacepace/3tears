"""Trusted-proxy client-IP resolution."""

from __future__ import annotations

import pytest

from threetears.iam.clientip import parse_trusted_cidrs, resolve_client_ip

_TRUSTED = parse_trusted_cidrs("10.0.0.0/8,fd00::/8")


def test_no_peer_resolves_to_none() -> None:
    assert resolve_client_ip(peer=None, forwarded_for=["1.2.3.4"], trusted=_TRUSTED) is None


def test_unconfigured_trust_uses_the_direct_peer() -> None:
    # The safe default: with nothing trusted, a forgeable header changes nothing.
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=["1.2.3.4"]) == "10.0.0.5"


def test_untrusted_peer_ignores_a_forged_header() -> None:
    # The attack this module exists to stop: a client that sets its own XFF while
    # connecting directly must not get to choose its rate-limit bucket.
    assert resolve_client_ip(peer="203.0.113.9", forwarded_for=["1.2.3.4"], trusted=_TRUSTED) == "203.0.113.9"


def test_trusted_peer_takes_the_rightmost_entry() -> None:
    # "1.2.3.4" is client-supplied and forgeable; "198.51.100.7" is what the proxy itself
    # observed and appended, so it is the only trustworthy entry.
    assert (
        resolve_client_ip(peer="10.0.0.5", forwarded_for=["1.2.3.4, 198.51.100.7"], trusted=_TRUSTED) == "198.51.100.7"
    )


def test_duplicate_header_lines_are_joined_before_parsing() -> None:
    # A proxy emitting XFF as separate lines must not let the client-controlled first line
    # win. All values are joined, then the rightmost is taken.
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=["1.2.3.4", "198.51.100.7"], trusted=_TRUSTED) == (
        "198.51.100.7"
    )


def test_trusted_peer_with_no_forwarded_header_uses_the_peer() -> None:
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=[], trusted=_TRUSTED) == "10.0.0.5"


def test_two_hops_walks_back_one_further() -> None:
    chain = ["1.2.3.4, 198.51.100.7, 10.0.0.9"]
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=chain, trusted=_TRUSTED, trusted_hops=2) == "198.51.100.7"


def test_short_chain_falls_back_to_the_peer() -> None:
    # Fewer entries than configured hops means the topology is not what was declared.
    # Reaching past the end would hand back a client-supplied value.
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=["1.2.3.4"], trusted=_TRUSTED, trusted_hops=3) == "10.0.0.5"


def test_ipv6_peer_matches_an_ipv6_network() -> None:
    assert resolve_client_ip(peer="fd00::1", forwarded_for=["2001:db8::5"], trusted=_TRUSTED) == "2001:db8::5"


def test_ipv4_peer_is_not_matched_by_an_ipv6_network() -> None:
    v6_only = parse_trusted_cidrs("fd00::/8")
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=["1.2.3.4"], trusted=v6_only) == "10.0.0.5"


def test_unparseable_peer_is_never_trusted() -> None:
    assert resolve_client_ip(peer="not-an-address", forwarded_for=["1.2.3.4"], trusted=_TRUSTED) == "not-an-address"


def test_blank_forwarded_entries_are_discarded() -> None:
    assert resolve_client_ip(peer="10.0.0.5", forwarded_for=[" , ,198.51.100.7, "], trusted=_TRUSTED) == "198.51.100.7"


def test_parse_trusted_cidrs_accepts_strings_and_sequences() -> None:
    assert parse_trusted_cidrs(None) == ()
    assert parse_trusted_cidrs("") == ()
    assert len(parse_trusted_cidrs("10.0.0.0/8, 192.168.0.0/16")) == 2
    assert len(parse_trusted_cidrs(["10.0.0.0/8", "192.168.0.0/16"])) == 2


def test_parse_trusted_cidrs_rejects_host_bits() -> None:
    # Silently masking 10.0.0.1/8 to 10.0.0.0/8 is how a typo becomes "trust an entire /8".
    with pytest.raises(ValueError, match="host bits"):
        parse_trusted_cidrs("10.0.0.1/8")


def test_parse_trusted_cidrs_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        parse_trusted_cidrs("not-a-cidr")
