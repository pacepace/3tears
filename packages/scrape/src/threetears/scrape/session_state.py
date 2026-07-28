"""Keep a human's solve, sealed, and hand it back to the unattended fetcher.

A person clears a challenge once. Without this, they clear it again on the next poll, and the
whole human-in-the-loop path costs an operator's attention per fetch rather than per outage.
What makes the solve reusable is the browser state it produced -- the cookies a challenge
system sets to say "this one is fine" -- carried forward into a later unattended render.

**Those cookies are live session credentials, and this module treats them as such.** They are
sealed with :func:`threetears.core.security.encryption.seal` under an operator-supplied master
key before anything persists them, they are never written in the clear, never logged, and
never placed in a ``repr``. The sidecar that exports them holds no key and never seals
anything; that boundary is deliberate, because the container that drives a browser for
arbitrary targets is the one you least want holding a decryption key.

**An expiry, and it is advisory in the safe direction.** Past it the state is ignored and the
target needs a human again. The failure mode of a wrong expiry is therefore "ask for help
sooner than strictly necessary", never "send a dead cookie and record a wall as a fresh
observation".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr
from threetears.core.security.encryption import DecryptionError, open_secret, seal
from threetears.observe import get_logger

from .health import _merge_health

if TYPE_CHECKING:
    from .health import ScrapeTargetHealth, ScrapeTargetHealthCollection

__all__ = [
    "DEFAULT_SESSION_STATE_TTL",
    "SealedSessionState",
    "open_session_state",
    "record_session_state",
    "seal_session_state",
    "usable_session_state",
]

# `record_session_state` writes a health row and therefore looks like it belongs in `health.py`
# beside the other health writers, which share that module's `_merge_health`. It lives here
# because the write is the LAST step of sealing and is meaningless without the sealing functions
# above it: a caller that has a sealed blob and no way to store it has been handed half an
# operation. Splitting them would put the encryption in one module and its only destination in
# another, and the seam between them is exactly where a future change drops the expiry or
# stores plaintext.

log = get_logger(__name__)

#: How long a human's solve is trusted by default. Twelve hours rather than a day: a challenge
#: system's own session cookies commonly outlive this, but the thing being bounded is not the
#: cookie's lifetime -- it is how long we are willing to keep re-sending a credential a person
#: earned without checking whether it still means anything. Overridable per call.
DEFAULT_SESSION_STATE_TTL = timedelta(hours=12)


@dataclass(frozen=True)
class SealedSessionState:
    """A sealed solve and the moment it stops being trusted.

    ``__repr__`` and ``__str__`` are overridden rather than inherited. A dataclass repr prints
    every field, and this one holds ciphertext that a debug log, an exception rendering or an
    error tracker would then carry off the machine. Ciphertext is not plaintext, but a
    credential's ciphertext sitting in a log aggregator is still a credential sitting in a log
    aggregator, and there is no reason to put it there.
    """

    sealed: str
    expires_at: datetime

    def __repr__(self) -> str:
        """Describe the state without disclosing it."""
        return f"SealedSessionState(sealed=<{len(self.sealed)} chars redacted>, expires_at={self.expires_at!r})"

    def __str__(self) -> str:
        """Same redaction as :meth:`__repr__`, for f-strings and ``%s``."""
        return self.__repr__()


def seal_session_state(
    state: dict[str, Any],
    key: SecretStr,
    *,
    ttl: timedelta = DEFAULT_SESSION_STATE_TTL,
    now: datetime | None = None,
) -> SealedSessionState:
    """Seal an exported browser state for storage.

    :param state: the raw cookie/storage export from the sidecar
    :ptype state: dict[str, Any]
    :param key: operator master key, resolved via ``secret_refs``
    :ptype key: SecretStr
    :param ttl: how long the solve is trusted
    :ptype ttl: timedelta
    :param now: current time; injected by tests
    :ptype now: datetime | None
    :return: the sealed token and its expiry
    :rtype: SealedSessionState
    """
    moment = now or datetime.now(UTC)
    # sort_keys so the same state seals to the same plaintext, which makes a test able to say
    # "these two are the same solve" without depending on dict ordering. The ciphertext still
    # differs per call: seal() uses a fresh nonce, deliberately, so equal secrets do not
    # produce equal tokens.
    plaintext = json.dumps(state, sort_keys=True, separators=(",", ":"))
    return SealedSessionState(sealed=seal(plaintext, key), expires_at=moment + ttl)


def open_session_state(sealed: str, key: SecretStr, *, target_id: str | None = None) -> dict[str, Any] | None:
    """Recover a sealed state, or ``None`` when it cannot be trusted.

    ``None`` rather than an exception for every failure mode a caller can do nothing about: a
    wrong key, a tampered token, and a state written by a build that seals differently all
    mean the same thing operationally, which is "this target needs a human again". Raising
    would turn a recoverable degradation into a failed fetch.

    The reason is logged without the token, because a token that will not open is exactly the
    kind of thing someone pastes into an issue.

    :param sealed: the stored ciphertext
    :ptype sealed: str
    :param key: operator master key
    :ptype key: SecretStr
    :param target_id: which target this state belongs to, for log correlation only. Optional
        because this function is usable without one, and never part of what is decrypted. Every
        other log line in this package carries it, and without it these three say a solve was
        discarded while naming no target -- true, unactionable, and indistinguishable from the
        same line about any other target in the fleet.
    :ptype target_id: str | None
    :return: the state, or ``None`` when it could not be opened or parsed
    :rtype: dict[str, Any] | None
    """
    try:
        plaintext = open_secret(sealed, key)
    except DecryptionError as exc:
        log.warning(
            "scrape session state: stored state could not be opened (%s); this target needs a human again",
            exc,
            extra={"extra_data": {"target_id": target_id}},
        )
        return None
    try:
        state = json.loads(plaintext.get_secret_value())
    except json.JSONDecodeError:
        # Opened but not parseable: the key was right, so this is a format change rather than
        # a tamper. Same operational answer, different cause, and worth not conflating.
        log.warning(
            "scrape session state: stored state opened but is not valid JSON; ignoring it",
            extra={"extra_data": {"target_id": target_id}},
        )
        return None
    if not isinstance(state, dict):
        log.warning(
            "scrape session state: stored state is not an object; ignoring it",
            extra={"extra_data": {"target_id": target_id}},
        )
        return None
    return state


def usable_session_state(
    health: ScrapeTargetHealth | None,
    key: SecretStr | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """The state to send with the next unattended fetch, or ``None``.

    One place answers "should this fetch carry a human's solve", so a caller cannot honour the
    expiry in one path and forget it in another. Every reason to decline returns ``None``: no
    row, nothing stored, no key configured, or an expiry that has passed.

    An expired state is not deleted here. Deleting is a write, and this runs on the read path
    of every poll; the row is harmless while it sits there, and it is replaced by the next
    solve.

    :param health: the target's health row, if it has one
    :ptype health: ScrapeTargetHealth | None
    :param key: operator master key, or ``None`` when the deployment configured none
    :ptype key: SecretStr | None
    :param now: current time; injected by tests
    :ptype now: datetime | None
    :return: the state to apply, or ``None``
    :rtype: dict[str, Any] | None
    """
    if health is None or key is None:
        return None
    sealed = health.session_state_sealed
    if not sealed:
        return None
    expires_at = health.session_state_expires_at
    moment = now or datetime.now(UTC)
    if expires_at is None or moment >= expires_at:
        # A stored state with no expiry is treated as expired rather than as eternal. The
        # writer always sets one, so its absence means a hand-edited or half-written row, and
        # the safe reading of "I do not know when this stops being valid" is "now".
        log.info(
            "scrape session state: stored state has expired; this target needs a human again",
            extra={"extra_data": {"target_id": health.target_id}},
        )
        return None
    return open_session_state(sealed, key, target_id=health.target_id)


async def record_session_state(
    health_collection: ScrapeTargetHealthCollection,
    *,
    target_id: str,
    state: SealedSessionState | None,
) -> None:
    """Persist a sealed solve on the target's health row, or clear it.

    ``None`` clears both columns together. Clearing one without the other would leave either a
    token nothing knows the lifetime of, or an expiry guarding nothing -- the same pairing
    argument ``record_circuit_state`` makes for the columns a trip writes together.

    :param health_collection: where the durable state lives
    :ptype health_collection: ScrapeTargetHealthCollection
    :param target_id: the target this solve belongs to
    :ptype target_id: str
    :param state: the sealed state, or ``None`` to clear
    :ptype state: SealedSessionState | None
    :return: nothing
    :rtype: None
    """

    await _merge_health(
        health_collection,
        target_id=target_id,
        changes={
            "session_state_sealed": state.sealed if state is not None else None,
            "session_state_expires_at": state.expires_at if state is not None else None,
        },
    )
