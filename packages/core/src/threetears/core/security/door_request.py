"""a door request carries its credential and a proof that the presenter holds the credential's key.

A platform door (``l3.*``, ``datasource.*``, object resolve, namespace discovery, engagement reads,
approval record, audit anonymize) reads its caller off the ``identity_token`` in the request body.
Alone that is a bearer token: whoever holds it can use it, at any door, for anything, until it
expires. A caller now also sends ``identity_proof``, a proof of possession
(:mod:`threetears.core.security.pop`) signed with the credential's key over this request's body and
this token. A tool pod presents the call's delegated token
(:mod:`threetears.core.security.delegated_token`) and proves the pod's own key; an agent presents
its own token and proves the key its ``cnf`` names. The door verifies both with
:func:`verify_door_request`, and spends the proof's ``jti`` in its replay ledger.

The body hash is over the body's canonical JSON with the proof field left out, the same canonical
form a service envelope's body hash uses, so the sender and the door hash one text. The body must
already be JSON values.

Rolling: a caller with no key to prove sends the token alone, as before; each door's dated window
decides how long it still accepts that.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, TypeVar

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel

from threetears.nats.errors import KvError
from threetears.observe import get_logger
from threetears.observe.metrics import counter

from threetears.core.security.delegated_token import (
    DelegatedTokenError,
    DoorCaller,
    ProofAbsentError,
    is_delegated_token,
    verify_door_caller,
)
from threetears.core.security.door_proof import DoorProofSwitch, ProofOutcome
from threetears.core.security.freshness import DEFAULT_PROOF_MAX_AGE, ISSUE_TIME_FUTURE_TOLERANCE
from threetears.core.security.pop import access_token_hash, make_pop_proof
from threetears.core.security.service_envelope import EnvelopeLedger

__all__ = [
    "DOOR_PROOF_FIELD",
    "DOOR_PROOF_REPLAY_TTL",
    "DOOR_TOKEN_FIELD",
    "DoorCredential",
    "DoorRequestDoor",
    "DoorVerdict",
    "door_request_hash",
    "verify_door_request",
]

_M = TypeVar("_M", bound=BaseModel)

log = get_logger(__name__)

_REFUSED_COUNTER: Final[str] = "door_request_refused"

#: how long a door's ledger remembers a proof's ``jti``: as long as the proof could still be
#: accepted -- its age limit plus the clock tolerance ahead -- and a margin.
DOOR_PROOF_REPLAY_TTL: Final[timedelta] = DEFAULT_PROOF_MAX_AGE + ISSUE_TIME_FUTURE_TOLERANCE + timedelta(seconds=5)

#: the body field a door reads its caller's credential from.
DOOR_TOKEN_FIELD: Final[str] = "identity_token"

#: the body field carrying the proof of possession over the rest of the body.
DOOR_PROOF_FIELD: Final[str] = "identity_proof"


def door_request_hash(body: Mapping[str, Any]) -> str:
    """the hash a door request's proof signs: sha-256 over the body's canonical JSON, the proof left out.

    :param body: the request body, as JSON values
    :ptype body: Mapping[str, Any]
    :return: the base64url digest, unpadded
    :rtype: str
    :raises TypeError: when the body holds a value that is not JSON
    """
    content = {key: value for key, value in body.items() if key != DOOR_PROOF_FIELD}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # str(bytes, "ascii"), as canonical_call_hash: the alg-pin enforcement reads every ``.decode(``
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


@dataclass(frozen=True, slots=True)
class DoorCredential:
    """what a caller presents at a door: a token, and the key that proves it when the caller holds one.

    :ivar token: the delegated token (a tool pod) or the caller's own identity token (an agent)
    :ivar holder_key: the key the token binds -- the pod's own key for a delegated token, the
        agent's holder key for its own token -- or ``None`` for a caller that predates proving
    """

    token: str
    holder_key: Ed25519PrivateKey | None = None

    @classmethod
    def for_tool_call(
        cls,
        *,
        delegated_token: str | None,
        identity_token: str | None,
        pod_key: Ed25519PrivateKey | None,
        own_key: Ed25519PrivateKey | None = None,
    ) -> DoorCredential | None:
        """the credential a tool server presents for one call.

        A tool pod: the call's delegated token proven with the pod's key when the registry minted
        one and the pod holds its key; otherwise the caller's forwarded token, bearer, as before,
        which the doors accept until their windows close. An agent's in-process tool server is the
        caller itself: it passes ``own_key``, the key its own token binds, and proves that token.

        :param delegated_token: the call context's delegated token, if the registry minted one
        :ptype delegated_token: str | None
        :param identity_token: the caller's forwarded identity token, if any
        :ptype identity_token: str | None
        :param pod_key: the pod's own identity key, if the pod holds it
        :ptype pod_key: Ed25519PrivateKey | None
        :param own_key: the key ``identity_token`` itself binds, held only by the agent it names;
            never a tool pod's key, which no forwarded token binds
        :ptype own_key: Ed25519PrivateKey | None
        :return: the credential, or ``None`` when the call carries no token at all
        :rtype: DoorCredential | None
        """
        credential: DoorCredential | None = None
        if delegated_token and pod_key is not None:
            credential = cls(token=delegated_token, holder_key=pod_key)
        elif identity_token:
            credential = cls(token=identity_token, holder_key=own_key)
        return credential

    def present(self, body: Mapping[str, Any], *, now: int | None = None) -> dict[str, Any]:
        """``body`` with this credential's token and, when it holds a key, the proof over both.

        :param body: the request body, as JSON values, without the token or proof
        :ptype body: Mapping[str, Any]
        :param now: unix-seconds issue time; the current time when ``None``
        :ptype now: int | None
        :return: the body to send
        :rtype: dict[str, Any]
        :raises TypeError: when the body holds a value that is not JSON
        """
        sent = {key: value for key, value in body.items() if key != DOOR_PROOF_FIELD}
        sent[DOOR_TOKEN_FIELD] = self.token
        if self.holder_key is not None:
            sent[DOOR_PROOF_FIELD] = make_pop_proof(
                holder_key=self.holder_key,
                access_token_hash=access_token_hash(self.token),
                body_hash=door_request_hash(sent),
                nonce=secrets.token_urlsafe(16),
                iat=int(time.time()) if now is None else now,
            )
        return sent

    def presented(self, message: _M) -> _M:
        """``message`` with the proof in its ``identity_proof`` field, when this credential holds a key.

        For a wire model that already carries this credential's token in ``identity_token``: the
        proof signs ``model_dump(mode="json")``, which is what ``model_dump_json`` puts on the wire.

        :param message: the wire model, its ``identity_proof`` unset
        :ptype message: BaseModel
        :return: a copy carrying the proof, or ``message`` itself for a credential with no key
        :rtype: BaseModel
        :raises ValueError: when the model has no ``identity_proof`` field, or carries another token
        """
        if DOOR_PROOF_FIELD not in type(message).model_fields:
            raise ValueError(f"{type(message).__name__} has no {DOOR_PROOF_FIELD} field to carry a proof")
        proven = message
        if self.holder_key is not None:
            body = message.model_dump(mode="json")
            if body.get(DOOR_TOKEN_FIELD) != self.token:
                raise ValueError(f"{type(message).__name__} does not carry this credential's token")
            proven = message.model_copy(update={DOOR_PROOF_FIELD: self.present(body)[DOOR_PROOF_FIELD]})
        return proven


def verify_door_request(
    body: Mapping[str, Any],
    *,
    jwks: Mapping[str, Any],
    hub_issuer: str,
    door: str,
) -> DoorCaller:
    """verify a door request's credential and its proof over the body as received.

    :param body: the parsed request body
    :ptype body: Mapping[str, Any]
    :param jwks: the hub JWKS
    :ptype jwks: Mapping[str, Any]
    :param hub_issuer: the hub's identity-token issuer
    :ptype hub_issuer: str
    :param door: this door, which a delegated token must name
    :ptype door: str
    :return: the proven caller; the door spends ``caller.proof.jti`` in its ledger
    :rtype: DoorCaller
    :raises ProofAbsentError: when the body carries a token but no proof (the door's window decides)
    :raises DelegatedTokenError: when the token is absent, or the token or proof does not verify
    """
    token = body.get(DOOR_TOKEN_FIELD)
    proof = body.get(DOOR_PROOF_FIELD)
    if not isinstance(token, str) or not token:
        raise DelegatedTokenError("the request carries no credential")
    return verify_door_caller(
        token,
        proof if isinstance(proof, str) else None,
        jwks=jwks,
        hub_issuer=hub_issuer,
        door=door,
        body_hash=door_request_hash(body),
    )


@dataclass(frozen=True, slots=True)
class DoorVerdict:
    """what a door decided about one request's credential.

    :ivar admitted: whether the door may go on to serve the request
    :ivar caller: the proven caller when the request carried a proof that verified; ``None`` for a
        bearer request the door's window still accepts, which the door verifies its own way, as before
    """

    admitted: bool
    caller: DoorCaller | None = None


class DoorRequestDoor:
    """one platform door's check of a request's proof of possession, inside its dated window.

    A request carrying ``identity_proof`` is served only when the proof verifies over the body and
    its ``jti`` is spent in the ledger. A request without one is the window's to judge -- accepted,
    and counted per claimed sender, while it is open; refused from its date -- and its token is
    then verified by the door exactly as before. The per-door count of bearer callers is the fleet's
    rollout signal, never a switch. The door never raises: its answer is a :class:`DoorVerdict`.

    :param switch: the door's window, from the process's door registry; its door name is the
        audience a delegated token must name
    :ptype switch: DoorProofSwitch
    :param jwks: answers the hub JWKS -- the hub's identity keys and the registry's delegated-token
        key -- or ``None`` while it is not loaded
    :ptype jwks: Callable[[], Mapping[str, Any] | None]
    :param hub_issuer: the hub's identity-token issuer
    :ptype hub_issuer: str
    :param replay_guard: the door's single-use ledger of proof ids
    :ptype replay_guard: EnvelopeLedger
    :param accepts_delegated: whether a tool pod may act for its caller here with the call's
        delegated token; a door no tool pod reaches as its caller refuses one
    :ptype accepts_delegated: bool
    :raises ValueError: when the ledger was not sized for the proof's clock tolerance
    """

    def __init__(
        self,
        *,
        switch: DoorProofSwitch,
        jwks: Callable[[], Mapping[str, Any] | None],
        hub_issuer: str,
        replay_guard: EnvelopeLedger,
        accepts_delegated: bool,
    ) -> None:
        replay_guard.require_covers(ISSUE_TIME_FUTURE_TOLERANCE)
        self._switch = switch
        self._jwks = jwks
        self._hub_issuer = hub_issuer
        self._replay_guard = replay_guard
        self._accepts_delegated = accepts_delegated

    @property
    def door(self) -> str:
        """the door's name, its window's.

        :return: the name
        :rtype: str
        """
        return self._switch.door

    async def admit(self, body: Mapping[str, Any], *, claimed_sender: str | None = None) -> DoorVerdict:
        """decide ``body``, the request as received, its proof field included.

        :param body: the parsed request
        :ptype body: Mapping[str, Any]
        :param claimed_sender: who the request says it is from, unverified, for the bearer count
        :ptype claimed_sender: str | None
        :return: the verdict
        :rtype: DoorVerdict
        """
        token = body.get(DOOR_TOKEN_FIELD)
        proof = body.get(DOOR_PROOF_FIELD)
        if isinstance(proof, str) and proof:
            verdict = await self._verified(body)
        elif isinstance(token, str) and is_delegated_token(token):
            # a delegated token is useless without its pod's key; there is no bearer form of one
            verdict = self._refused("absent", "a delegated token arrived without its proof")
        else:
            outcome = self._switch.judge(presented=False, claimed_sender=claimed_sender)
            verdict = (
                DoorVerdict(admitted=True)
                if outcome is ProofOutcome.ACCEPTED_WITHOUT_PROOF
                else self._refused("absent", "the door requires a proof of possession")
            )
        return verdict

    async def admit_message(self, data: bytes, *, claimed_sender: str | None = None) -> DoorVerdict:
        """decide the request ``data``, hashed exactly as sent.

        The door to use when the door's own request model is a tolerant reader, which would hash
        less than the caller signed for a caller a release ahead.

        :param data: the request bytes, a JSON object
        :ptype data: bytes
        :param claimed_sender: who the request says it is from, unverified, for the bearer count
        :ptype claimed_sender: str | None
        :return: the verdict; a body that is not a JSON object is refused
        :rtype: DoorVerdict
        """
        try:
            body = json.loads(data)
        except ValueError, UnicodeDecodeError:
            body = None
        verdict = self._refused("malformed", "the request is not a JSON object")
        if isinstance(body, dict):
            verdict = await self.admit(body, claimed_sender=claimed_sender)
        return verdict

    async def _verified(self, body: Mapping[str, Any]) -> DoorVerdict:
        """verify the proof and spend its id.

        :param body: the parsed request, carrying a proof
        :ptype body: Mapping[str, Any]
        :return: the verdict
        :rtype: DoorVerdict
        """
        jwks = self._jwks()
        verdict: DoorVerdict
        if jwks is None:
            verdict = self._refused("keys_unavailable", "the hub key set is not loaded")
        else:
            try:
                caller = verify_door_request(body, jwks=jwks, hub_issuer=self._hub_issuer, door=self.door)
                fresh = await self._replay_guard.record_unique(caller.proof.jti, issued_at=caller.proof.issued_at)
            except ProofAbsentError as exc:
                verdict = self._refused("absent", f"{exc}")
            except DelegatedTokenError as exc:
                verdict = self._refused("unverified", f"{exc}")
            except KvError as exc:
                # the ledger cannot say whether the proof was used: refuse rather than admit a replay
                verdict = self._refused("ledger_unavailable", type(exc).__name__)
            else:
                if not fresh:
                    verdict = self._refused("replayed", "the proof was used before")
                elif caller.pod_id is not None and not self._accepts_delegated:
                    verdict = self._refused("delegated", "this door does not take a tool pod acting for its caller")
                else:
                    verdict = DoorVerdict(admitted=True, caller=caller)
        return verdict

    def _refused(self, cause: str, reason: str) -> DoorVerdict:
        """log and count a refusal, and answer it.

        :param cause: the refusal's class, the counter's ``cause`` label
        :ptype cause: str
        :param reason: what failed
        :ptype reason: str
        :return: a refusing verdict
        :rtype: DoorVerdict
        """
        counter(
            _REFUSED_COUNTER,
            description="door requests refused for their proof of possession",
            label_names=("door", "cause"),
        ).labels(door=self.door, cause=cause).inc()
        log.warning("door request refused", extra={"extra_data": {"door": self.door, "cause": cause, "reason": reason}})
        return DoorVerdict(admitted=False)
