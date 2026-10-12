"""k-anonymized breach-password screening (the Pwned Passwords model).

Hash the candidate, split the hex digest into a 5-character PREFIX and the
remaining SUFFIX, look candidates up by prefix ONLY, and compare the suffix
locally. Whatever sits behind the lookup boundary -- an in-memory index, a
refreshed range-file cache, or a remote range API -- sees the prefix and
nothing else. It never sees the full hash, and it certainly never sees the
plaintext.

**The default is local, and that is a security property, not a convenience.**
A live HTTP call on the authentication path makes an outage of someone else's
service an outage of your login, and makes every login attempt observable to a
third party in real time. :class:`BreachCorpus` therefore resolves against an
in-memory index by default, refreshed out of band. :class:`RangeApiBreachCorpus`
exists for callers that have weighed those two costs and chosen the remote
lookup deliberately; it is opt-in, and it fails OPEN.

Failing open is the right call for this specific check and only this one:
breach screening is a defence-in-depth control layered on top of length policy
and rate limiting, so a lookup outage that blocked every password change would
do more damage than the small window of unscreened passwords it prevents. That
reasoning does not transfer to any other check in this package.

SHA-1 appears here because it is the corpus format's fixed hash function, not
because anything is being secured with it. Credential storage is argon2id --
see :mod:`threetears.iam.passwords`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import Final

import httpx

from threetears.observe import get_logger

__all__ = ["BreachCorpus", "RangeApiBreachCorpus", "sha1_prefix_suffix"]

log = get_logger(__name__)

#: Pwned Passwords convention: 5 hex characters of the SHA-1 digest.
_PREFIX_LEN: Final[int] = 5

#: The public Pwned Passwords range endpoint. A prefix is appended to this path.
DEFAULT_RANGE_API_URL: Final[str] = "https://api.pwnedpasswords.com/range"


def sha1_prefix_suffix(password: str) -> tuple[str, str]:
    """Split a password's uppercase SHA-1 hex digest into ``(prefix, suffix)``.

    The prefix is the only part that may cross a lookup boundary.

    :param password: the candidate plaintext.
    :ptype password: str
    :return: the 5-character prefix and the remaining 35-character suffix.
    :rtype: tuple[str, str]
    """
    digest = hashlib.sha1(password.encode("utf-8"), usedforsecurity=False).hexdigest().upper()
    return digest[:_PREFIX_LEN], digest[_PREFIX_LEN:]


def _build_index(passwords: Iterable[str]) -> dict[str, frozenset[str]]:
    """Index plaintexts by SHA-1 prefix. The plaintexts themselves are not retained."""
    by_prefix: dict[str, set[str]] = {}
    for password in passwords:
        prefix, suffix = sha1_prefix_suffix(password)
        by_prefix.setdefault(prefix, set()).add(suffix)
    return {prefix: frozenset(suffixes) for prefix, suffixes in by_prefix.items()}


class BreachCorpus:
    """Local k-anonymized breach-password check.

    :meth:`is_breached` hashes and prefix-splits the candidate, then delegates to the
    ``lookup_by_prefix`` boundary, passing ONLY the prefix.

    The default index is built from ``seed_passwords``, which is EMPTY unless supplied. An
    empty corpus screens nothing and reports nothing as breached -- deliberately, so a
    caller that has not wired up a real corpus gets a working no-op rather than a false
    sense of coverage from a token handful of hardcoded passwords. Load a real corpus
    (e.g. the downloadable Pwned Passwords range files) via ``seed_passwords``, or supply
    ``lookup_by_prefix`` to resolve against a cache you refresh out of band.
    """

    def __init__(
        self,
        *,
        seed_passwords: Iterable[str] = (),
        lookup_by_prefix: Callable[[str], frozenset[str]] | None = None,
    ) -> None:
        """
        :param seed_passwords: plaintexts to index at construction. Hashed immediately; the
            plaintexts are not retained. Ignored when ``lookup_by_prefix`` is supplied.
        :ptype seed_passwords: Iterable[str]
        :param lookup_by_prefix: override the k-anonymity boundary -- given a 5-character
            prefix, return the known suffixes under it.
        :ptype lookup_by_prefix: Callable[[str], frozenset[str]] | None
        """
        self._index: dict[str, frozenset[str]] = {} if lookup_by_prefix is not None else _build_index(seed_passwords)
        self._lookup_by_prefix: Callable[[str], frozenset[str]] = lookup_by_prefix or self._local_lookup

    def _local_lookup(self, prefix: str) -> frozenset[str]:
        return self._index.get(prefix, frozenset())

    def is_breached(self, password: str) -> bool:
        """Whether ``password`` matches a known-breached corpus entry.

        The plaintext and its full digest never leave this method -- only the prefix is
        passed to the lookup boundary.

        :param password: the candidate plaintext.
        :ptype password: str
        :return: whether the password appears in the corpus.
        :rtype: bool
        """
        prefix, suffix = sha1_prefix_suffix(password)
        return suffix in self._lookup_by_prefix(prefix)


class RangeApiBreachCorpus:
    """Breach screening against a remote k-anonymity range API. **Opt-in; fails open.**

    Read the module docstring before reaching for this: it puts a third-party HTTP call on
    the password path. The prefix is all that is transmitted, so the API cannot tell which
    password was checked -- but it can tell that a check happened, and when.

    Not a :class:`BreachCorpus` subclass: this one is async, because it does I/O.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = DEFAULT_RANGE_API_URL,
    ) -> None:
        """
        :param client: the HTTP client to issue range requests through. Supplied by the
            caller rather than constructed here, so its timeouts, limits, and lifecycle stay
            under the caller's control.
        :ptype client: httpx.AsyncClient
        :param base_url: the range endpoint; the prefix is appended as a path segment.
        :ptype base_url: str
        """
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def is_breached(self, password: str) -> bool:
        """Whether ``password`` appears in the remote corpus.

        Returns ``False`` -- not breached -- on any transport or protocol failure, logging a
        warning. See the module docstring for why this control specifically fails open.

        :param password: the candidate plaintext.
        :ptype password: str
        :return: whether the password appears in the corpus, or ``False`` if the lookup failed.
        :rtype: bool
        """
        prefix, suffix = sha1_prefix_suffix(password)
        try:
            response = await self._client.get(f"{self._base_url}/{prefix}")
            response.raise_for_status()
            body = response.text
        except httpx.HTTPError as exc:
            log.warning(
                "breach-corpus range lookup failed; treating the password as unscreened",
                extra={"extra_data": {"error": type(exc).__name__}},
            )
            return False
        return suffix in _parse_range_response(body)


def _parse_range_response(body: str) -> frozenset[str]:
    """Parse a range-API body (``SUFFIX:COUNT`` per line) into the set of suffixes.

    Unparseable lines are skipped rather than failing the whole lookup: a single malformed
    line should not turn a successful screen into an unscreened one.
    """
    suffixes: set[str] = set()
    for line in body.splitlines():
        candidate = line.split(":", 1)[0].strip().upper()
        if candidate:
            suffixes.add(candidate)
    return frozenset(suffixes)
