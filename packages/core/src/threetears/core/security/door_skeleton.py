"""what every door that admits a signed message shares: one window, one key lookup, one refusal account.

Two doors admit a message only when what it carries verifies: :class:`~threetears.core.security.
service_envelope.ServiceEnvelopeDoor` (a service's envelope over its message) and
:class:`~threetears.core.security.door_request.DoorRequestDoor` (a caller's proof of possession of
its token's key). Each judges a message without its credential by the door's dated window, reads
the sender's key set and fetches it afresh once for a key it lacks, treats an empty set as
unavailable rather than as a forgery, parses the bytes it was sent as one JSON object, and counts
and logs every refusal by cause. They share these pieces from here, so the two cannot drift apart
again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.security.door_proof import DoorProofSwitch, ProofOutcome

if TYPE_CHECKING:
    from threetears.core.security.service_envelope import EnvelopeJwksSource

__all__ = ["DoorAccounting", "KeysUnavailableError", "json_object", "keys_naming", "window_admits"]

log = get_logger(__name__)


class KeysUnavailableError(Exception):
    """the sender's key set is empty: nothing can verify yet, which is not a forgery."""


class DoorAccounting:
    """one door's refusals: a counter labelled by door and cause, and a warning naming why.

    :param door: the door's name, the counter's ``door`` label
    :ptype door: str
    :param counter_name: the counter's metric name
    :ptype counter_name: str
    :param description: the counter's description
    :ptype description: str
    :param message: the warning's message
    :ptype message: str
    """

    def __init__(self, *, door: str, counter_name: str, description: str, message: str) -> None:
        self._door = door
        self._counter_name = counter_name
        self._description = description
        self._message = message

    @property
    def door(self) -> str:
        """the door's name.

        :return: the name
        :rtype: str
        """
        return self._door

    def refused(self, cause: str, reason: str, **fields: str) -> None:
        """count and log one refusal.

        :param cause: the refusal's class, the counter's ``cause`` label
        :ptype cause: str
        :param reason: what failed
        :ptype reason: str
        :param fields: more for the log line (an operation, say); never a credential
        :ptype fields: str
        :return: nothing
        :rtype: None
        """
        counter(self._counter_name, description=self._description, label_names=("door", "cause")).labels(
            door=self._door, cause=cause
        ).inc()
        log.warning(
            self._message, extra={"extra_data": {"door": self._door, **fields, "cause": cause, "reason": reason}}
        )


def window_admits(switch: DoorProofSwitch, *, claimed_sender: str | None) -> bool:
    """whether the door's window still accepts a message without its credential, counting it.

    :param switch: the door's window
    :ptype switch: DoorProofSwitch
    :param claimed_sender: who the message says it is from, unverified, for the count
    :ptype claimed_sender: str | None
    :return: whether it is accepted
    :rtype: bool
    """
    return switch.judge(presented=False, claimed_sender=claimed_sender) is ProofOutcome.ACCEPTED_WITHOUT_PROOF


def json_object(data: bytes) -> dict[str, Any] | None:
    """the message ``data`` as the JSON object it was sent as, or ``None`` when it is not one.

    :param data: the message bytes
    :ptype data: bytes
    :return: the object, or ``None``
    :rtype: dict[str, Any] | None
    """
    try:
        body = json.loads(data)
    except ValueError, UnicodeDecodeError:
        body = None
    return body if isinstance(body, dict) else None


async def keys_naming(source: EnvelopeJwksSource, kid: Any) -> Mapping[str, Any]:
    """the sender's key set, fetched afresh once when it lacks ``kid`` (a rotated key).

    :param source: where the door reads the sender's key set
    :ptype source: EnvelopeJwksSource
    :param kid: the key id the message's credential names
    :ptype kid: Any
    :return: the key set
    :rtype: Mapping[str, Any]
    :raises KeysUnavailableError: when the set is empty even after the refresh
    """
    jwks = await source.get_jwks()
    if not any(isinstance(key, Mapping) and key.get("kid") == kid for key in jwks.get("keys", ())):
        jwks = await source.get_jwks(refresh=True)
    if not jwks.get("keys"):
        raise KeysUnavailableError("the sender's key set is not loaded")
    return jwks
