"""configuration for the sender-posture enforcement domain.

the domain enforces one contract: every place a process starts receiving NATS messages -- a
subscribe, a JetStream consumer, a result waiter, a KV watch, an owner-forward handler --
declares how it authenticates whoever sent them, with a ``# sender-posture:`` marker. a
subject proves nothing about its sender: any connection can have its own bytes delivered on
any subject through a push consumer's ``deliver_subject`` or a request's reply subject. a
receiver that acts on what arrives therefore has to say what it checked.

the postures, and what each one claims:

- ``verified-token``: the handler verifies a token carried in the message (signature, issuer,
  expiry) and takes the sender's identity from it, never from the body.
- ``signed-envelope``: the handler verifies a signature over the message itself, bound to its
  body, audience and expiry, with a single-use id.
- ``own-consumer``: a JetStream delivery accepted only when its ack reply names the
  receiver's own stream and consumer.
- ``trigger-only``: the message is a nudge; the handler refetches from the authority and acts
  on nothing in the body.
- ``read-only``: the handler answers a read and changes nothing, and its reply is safe to
  hand whoever can name the subject.
- ``unauthenticated``: none of the above. allowed only while its audit row stays open in the
  repo's allowlist.
- ``primitive``: the call implements a subscribe primitive whose callers declare their own
  posture. allowed only in the modules a repo names as its primitives.

any posture may also carry ``open <ROW>[,<ROW>]``: audit rows still open at that site, each of
which must have a line in the allowlist, and every allowlist line must still have a site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_SITE_CALLEES",
    "POSTURES",
    "SenderPostureConfig",
]

#: every posture a marker may name; see this module's docstring for what each claims.
POSTURES: Final[frozenset[str]] = frozenset(
    {
        "verified-token",
        "signed-envelope",
        "own-consumer",
        "trigger-only",
        "read-only",
        "unauthenticated",
        "primitive",
    }
)

#: callee names that start receiving messages. matched on the final name of a call, method or
#: bare function, so ``nc.subscribe(...)`` and ``serve_owner(...)`` are both sites. a callee
#: named here that is not a receiver in some repo is still declared there, which costs one
#: marker and keeps the walker from guessing receiver types it cannot see.
DEFAULT_SITE_CALLEES: Final[frozenset[str]] = frozenset(
    {
        "subscribe",
        "subscribe_typed",
        "subscribe_on",
        "jetstream_subscribe_durable",
        "jetstream_pull_subscribe",
        "jetstream_result_waiter",
        "pull_subscribe",
        "watch_key",
        "watch_prefix",
        "renew_on_request",
        "abandon_on_refusal",
        "serve_owner",
        "serve_pipe",
        "open_pipe",
        "attach_pipe",
    }
)


@dataclass(frozen=True)
class SenderPostureConfig:
    """per-repo config for the sender-posture enforcement domain.

    :ivar repo_root: absolute path of the repo root; paths in findings and in the allowlist are
        relative to it
    :ivar src_roots: production source trees to walk
    :ivar allowlist_path: the repo's shrinking allowlist of open audit rows. each line reads
        ``<repo-relative path> :: <row> :: <chunk> :: <why it is still open>``
    :ivar site_callees: callee names treated as receiving sites
    :ivar primitive_modules: repo-relative posix paths of the modules allowed to declare
        ``primitive``
    """

    repo_root: Path
    src_roots: tuple[Path, ...]
    allowlist_path: Path
    site_callees: frozenset[str] = DEFAULT_SITE_CALLEES
    primitive_modules: frozenset[str] = frozenset()
