"""Remediation text for the NATS failures that arrive without saying what is wrong.

**One failure class motivates this module: a missing KV grant.** It is the worst
shape a permission error can take, because it does not present as a permission
error at all.

A KV operation is a JetStream API request. When the connection's user JWT does
not cover the bucket, the server refuses the *publish* and simply never answers
the request. ``nats-py`` has nothing to raise -- no reply arrived, so the call
blocks until the wrapper's own deadline fires and reports a timeout. A timeout
is what an unreachable broker looks like too, so the operator reads
"broker unreachable", goes to the network, and finds a healthy connection
happily carrying every other subject. ``tests/enforcement/test_kv_bucket_grant_naming.py``
records the same shape from the grant side: *"the op then blocks to its deadline
rather than raising, indistinguishable from an unreachable broker."*

The server does say so, once, on a channel nobody was reading: a
``permissions violation`` ``-ERR`` frame goes to the client's error callback and
-- unlike an authorization violation -- **does not close the connection**. So the
useful diagnosis is available at the moment of denial, seconds before the
deadline the caller will eventually blame.

Everything here exists to put the fix in the log line rather than in a release
note:

- :func:`permissions_violation_remedy` reads the server's own refusal and, when
  it names a ``$KV`` subject, converts it to the grant that is missing.
- :func:`kv_timeout_remedy` is for the sites where only a deadline is available,
  and states a missing grant as a leading cause rather than leaving the reader
  with "the broker did not answer".
- :func:`kv_grant_remedy` is the shared remediation sentence both of those end
  with, so the instruction cannot drift between them.

**Every subject in a server ``-ERR`` frame arrives lower-cased**, because
``nats-py``'s protocol parser lower-cases the whole frame before dispatching it.
Bucket names survive that intact in practice (they are lower-case by
convention), but the match has to be case-insensitive and the reported name is
whatever the server echoed, not necessarily the caller's spelling.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["kv_grant_remedy", "kv_timeout_remedy", "permissions_violation_remedy"]

#: The server's refusal text, as ``nats-py`` surfaces it (lower-cased by its parser).
_PERMISSIONS_VIOLATION: Final = "permissions violation"

#: A KV data subject inside a refusal: ``$KV.{bucket}.{key...}``.
#:
#: The bucket is the segment after ``$KV.`` -- keys may contain dots, buckets may not.
_KV_SUBJECT: Final = re.compile(r"\$kv\.(?P<bucket>[^.\s\"']+)\.", re.IGNORECASE)

#: A JetStream control subject naming a KV-backed stream: ``$JS.API.<verb>.KV_{bucket}``.
#:
#: The control plane is refused separately from the data plane, and it is the one
#: that fails FIRST -- opening a bucket is ``STREAM.CREATE`` before any key exists.
#:
#: ``kv_`` must open a subject TOKEN (anchored on the preceding dot), not merely a word.
#: A ``\b`` would also match inside a stream named ``events-kv_v2``, and reporting a
#: KV bucket that does not exist is worse than reporting none: it sends the reader to
#: add a grant for a name nothing will ever open.
_KV_STREAM_SUBJECT: Final = re.compile(r"\$js\.api\.[^\s\"']*?\.kv_(?P<bucket>[^.\s\"']+)", re.IGNORECASE)


def kv_grant_remedy(bucket: str, *, certain: bool = True) -> str:
    """The grant a principal needs for ``bucket``, as an actionable sentence.

    Names the declaration site rather than only the wire subjects, because the
    wire subjects are derived: ``mint_user_jwt`` turns one ``JsResource`` record
    into all of them, and hand-adding the subjects leaves the declaration -- the
    thing the next person reads -- still wrong.

    **The old text told the reader to reopen a hole.** It said one entry expands
    into "pub+sub on ``$KV.{bucket}.>``". Both halves of that are now wrong, and
    following it would undo the fix: ``$KV.`` is PUBLISH-ONLY (a subscribe grant
    confers no read at all and leaks every write's full value), and a bucket
    whose keys carry a principal scope gets ``$KV.{bucket}.{scope}.>`` rather
    than the whole subtree.

    :param bucket: fully-qualified bucket name, prefix included
    :ptype bucket: str
    :param certain: ``True`` where the server named this bucket in a refusal, so
        the missing grant IS the cause. ``False`` where a grant is only the
        leading candidate -- a bind that fails against a bucket nobody created
        looks the same, and an error that asserts the wrong cause sends the
        reader further away than one that admits it is guessing.
    :ptype certain: bool
    :return: remediation text naming the declaration and the grants it mints
    :rtype: str
    """
    opening = (
        f"FIX: grant this principal the KV bucket {bucket!r}."
        if certain
        else f"MOST LIKELY FIX (the other candidate is a bucket that was never created): grant "
        f"this principal the KV bucket {bucket!r}."
    )
    return (
        f"{opening} Add a `JsResource.kv(...)` entry for it to the principal's "
        f"`js_resources` in `threetears.nats.subject_permissions`, deciding its key scope and its "
        f"write intent, then re-mint the user JWT and reconnect. `mint_user_jwt` expands that one "
        f'entry into a PUBLISH-ONLY data grant ("$KV.{bucket}.>", or "$KV.{bucket}.<scope>.>" for a '
        f'scoped bucket) plus JetStream control over stream "KV_{bucket}" at the capability the '
        f"entry declares. Do NOT hand-add `$KV` to the subscribe list: nothing subscribes it, and "
        f"it leaks every write's full value. If this deployment declares grants anywhere else as "
        f"well (the static NATS users in `nats.conf`), add it there too."
    )


def permissions_violation_remedy(exc: Exception) -> str | None:
    """Turn a server permissions refusal into a log line that names the fix.

    Returns ``None`` for anything that is not a permissions violation, so the
    caller keeps its ordinary error path for ordinary errors.

    A refusal that names no recognisable KV subject still gets a remedy: the
    connection was denied a subject it tried to use, which is worth stating
    plainly even when this module cannot name the bucket -- the alternative is
    the bland line that sent the last reader to the network.

    :param exc: the exception ``nats-py`` handed to the error callback
    :ptype exc: Exception
    :return: the loud, actionable message, or ``None`` when ``exc`` is unrelated
    :rtype: str | None
    """
    text = str(exc)
    if _PERMISSIONS_VIOLATION not in text.lower():
        return None

    bucket = _bucket_in(text)
    if bucket is None:
        return (
            "NATS PERMISSIONS VIOLATION -- the server REFUSED this connection a subject it "
            "used, and did not close the connection, so everything else keeps working and "
            "this is the only warning. Requests on the refused subject go unanswered and will "
            "surface later as timeouts that look like an unreachable broker. "
            "FIX: add the refused subject to this principal's allow-list in "
            f"`threetears.nats.subject_permissions`, re-mint the user JWT, and reconnect. Server said: {text}"
        )

    return (
        f"NATS PERMISSIONS VIOLATION on KV bucket {bucket!r} -- this connection's user JWT does "
        f"NOT grant it. The connection stays up and every other subject keeps working, so this "
        f"is the only warning you get: KV calls against {bucket!r} will hang until their deadline "
        f"and report a timeout, which reads as an unreachable broker. It is not the network. "
        f"{kv_grant_remedy(bucket)} Server said: {text}"
    )


def kv_timeout_remedy(bucket: str) -> str:
    """What to check when a KV operation on ``bucket`` blew its deadline.

    A deadline cannot distinguish "the broker is gone" from "this bucket is
    ungranted" -- a refused JetStream request is simply never answered. Both are
    named here, missing grant first, because the network is the one the reader
    checks unprompted.

    :param bucket: fully-qualified bucket name, prefix included
    :ptype bucket: str
    :return: remediation text covering both causes
    :rtype: str
    """
    return (
        f"A KV operation on {bucket!r} timed out. TWO causes look identical here, because a "
        f"JetStream request the server REFUSES is never answered at all: (1) the connection's "
        f"user JWT does not grant this bucket, and (2) the broker is unreachable. Rule out (1) "
        f"first -- it is silent, it is per-bucket, and the rest of this connection will look "
        f"perfectly healthy while it holds. {kv_grant_remedy(bucket)} "
        f"Only if the grant is already present is this a broker or network problem."
    )


def _bucket_in(text: str) -> str | None:
    """Extract the KV bucket a refusal names, from either plane.

    The control subject is tried first: opening a bucket is a ``STREAM.CREATE``,
    so a fully-ungranted bucket is refused there before any ``$KV`` data subject
    is ever published.

    :param text: the refusal message
    :ptype text: str
    :return: the bucket name, or ``None`` when the refusal names no KV subject
    :rtype: str | None
    """
    for pattern in (_KV_STREAM_SUBJECT, _KV_SUBJECT):
        match = pattern.search(text)
        if match is not None:
            return match.group("bucket")
    return None
