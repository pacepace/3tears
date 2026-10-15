"""a door request carries its credential and a proof that the presenter holds the credential's key.

A platform door (``l3.*``, ``datasource.*``, object resolve, namespace discovery, engagement reads,
approval record, audit anonymize) reads its caller off the ``identity_token`` in the request body.
Alone that is a bearer token: whoever holds it can use it, at any door, for anything, until it
expires. A caller now also sends ``identity_proof``, a proof of possession
(:mod:`threetears.core.security.pop`) signed with the credential's key over this token, this
request's body, and the door and operation it is sent to. A tool pod presents the call's delegated
token (:mod:`threetears.core.security.delegated_token`) and proves the pod's own key; an agent
presents its own token and proves the key its ``cnf`` names. The door verifies both with
:class:`DoorRequestDoor`, and spends the proof's ``jti`` in its replay ledger.

The proof's body hash is over the door and operation and the body's canonical JSON with the proof
field left out -- the canonical form a service envelope's body hash uses, so the sender and the door
hash one text. Binding the operation is what keeps a proven request from being replayed on another
subject that takes the same body: ``l3.tx.rollback`` and ``l3.tx.commit`` send identical bodies.
The body must already be JSON values.

A credential carries a key only when the token binds it (its ``cnf`` names the key's thumbprint):
a key the token does not bind would have the door refuse a request its window would still accept
as bearer, so such a credential is a bearer credential.

Rolling: a caller with no key to prove sends the token alone, as before; each door's dated window
decides how long it still accepts that.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Final, Protocol, TypeVar

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, SerializerFunctionWrapHandler, model_serializer

from threetears.nats.errors import KvError
from threetears.nats.subjects import Subject
from threetears.observe import get_logger

from threetears.core.security.delegated_token import (
    L3_DOOR,
    DelegatedTokenError,
    DoorCaller,
    ProofAbsentError,
    is_delegated_token,
    verify_door_caller,
)
from threetears.core.security.door_proof import DoorProofSwitch
from threetears.core.security.door_skeleton import (
    DoorAccounting,
    KeysUnavailableError,
    json_object,
    keys_naming,
    window_admits,
)
from threetears.core.security.freshness import DEFAULT_PROOF_MAX_AGE, ISSUE_TIME_FUTURE_TOLERANCE
from threetears.core.security.identity_token import jwk_thumbprint
from threetears.core.security.pop import access_token_hash, make_pop_proof
from threetears.core.security.service_envelope import EnvelopeJwksSource

__all__ = [
    "ProvenDoorRequest",
    "omit_absent_proof",
    "APPROVAL_RECORD_DOOR",
    "APPROVAL_RECORD_OP",
    "AUDIT_ANONYMIZE_OP",
    "DOOR_PROOF_FIELD",
    "DOOR_PROOF_REPLAY_TTL",
    "DOOR_SENDER_FIELDS",
    "DOOR_TOKEN_FIELD",
    "DoorCredential",
    "DoorProofLedger",
    "DoorRequestDoor",
    "DoorVerdict",
    "ENGAGEMENT_SCOPE_OP",
    "NAMESPACE_DISCOVER_OP",
    "OBJECT_RESOLVE_OP",
    "PopKey",
    "RegistrationKey",
    "claimed_sender_of",
    "datasource_query_op",
    "door_request_hash",
    "l3_request_op",
    "verify_door_request",
]

_M = TypeVar("_M", bound=BaseModel)

log = get_logger(__name__)

_REFUSED_COUNTER: Final[str] = "door_request_refused"

#: how long a door's ledger must remember a proof's ``jti``: as long as the proof could still be
#: accepted -- its age limit plus the clock tolerance ahead -- and a margin.
DOOR_PROOF_REPLAY_TTL: Final[timedelta] = DEFAULT_PROOF_MAX_AGE + ISSUE_TIME_FUTURE_TOLERANCE + timedelta(seconds=5)

#: the body field a door reads its caller's credential from.
DOOR_TOKEN_FIELD: Final[str] = "identity_token"

#: the request fields a door reads its unverified sender from, for the count of callers without a
#: proof: the caller's own token, then the user assertion a request may carry in its place.
DOOR_SENDER_FIELDS: Final[tuple[str, ...]] = (DOOR_TOKEN_FIELD, "user_identity_token")

#: the body field carrying the proof of possession over the rest of the body.
DOOR_PROOF_FIELD: Final[str] = "identity_proof"

#: the approval record door, which no delegated token names: only an agent records its own pause.
APPROVAL_RECORD_DOOR: Final[str] = "hub.approval.record"

# each single-operation door's operation, as the proof binds it; an ``l3.*`` request's is its
# subject's own part (``query``, ``tx.commit``), a datasource query's :func:`datasource_query_op`
OBJECT_RESOLVE_OP: Final[str] = "resolve"
ENGAGEMENT_SCOPE_OP: Final[str] = "scope"
NAMESPACE_DISCOVER_OP: Final[str] = "discover"
AUDIT_ANONYMIZE_OP: Final[str] = "anonymize"
APPROVAL_RECORD_OP: Final[str] = "record"


def l3_request_op(subject: str) -> str:
    """the operation an ``l3.*`` request is proven for: its subject's own part (``query``, ``tx.commit``).

    The one derivation the L3 client signs with and the broker verifies with, from the subject each
    sends or receives on.

    :param subject: the request's concrete subject, ``{ns}.l3.<op>``
    :ptype subject: str
    :return: the operation
    :rtype: str
    :raises ValueError: when the subject is not an ``l3`` subject
    """
    _, separator, op = subject.rpartition(f".{L3_DOOR}.")
    if not separator or not op:
        raise ValueError(f"{subject!r} is not an l3 request subject")
    return op


def datasource_query_op(subject: Subject) -> str:
    """the operation a query to one datasource is proven for, read off the subject it is sent on.

    From :meth:`~threetears.nats.Subjects.datasource_query`'s subject, ``{ns}.datasource.<name>.query``:
    the client and the hub door both derive it from the subject itself, so they sign one text and a
    proof never reaches another datasource.

    :param subject: the datasource query subject
    :ptype subject: Subject
    :return: the operation, ``<name>.query``
    :rtype: str
    :raises ValueError: when the subject is not a datasource query subject
    """
    tokens = subject.path.split(".")
    if len(tokens) < 4 or tokens[-3] != "datasource" or tokens[-1] != "query" or not tokens[-2]:
        raise ValueError(f"{subject.path!r} is not a datasource query subject")
    return ".".join(tokens[-2:])


def omit_absent_proof(data: dict[str, Any]) -> dict[str, Any]:
    """leave a request's proof off the wire when it carries none.

    A caller holding no key the token binds sends the token alone. A hub from before the door
    proofs (0.66.0 to 0.68.0) reads some of these requests with ``extra="forbid"``, so a body
    carrying ``"identity_proof": null`` would be refused there: the two must upgrade in either
    order. The proof is left out of what it signs (:func:`door_request_hash`), so omitting a null
    one changes no hash. :class:`ProvenDoorRequest` calls it from its serializer, and a subclass that
    replaces that serializer calls it from its own.

    :param data: the serialized request
    :ptype data: dict[str, Any]
    :return: the same mapping, without a null proof
    :rtype: dict[str, Any]
    """
    if data.get(DOOR_PROOF_FIELD, "") is None:
        del data[DOOR_PROOF_FIELD]
    return data


class ProvenDoorRequest(BaseModel):
    """a request a hub door takes with its caller's proof of possession: the one place the proof field lives.

    Every request model that carries ``identity_proof`` subclasses this, so the field and the rule
    that a request without a proof carries no proof key (:func:`omit_absent_proof`) come together and
    cannot be copied apart. A subclass that needs a serializer of its own replaces this one and must
    end it with :func:`omit_absent_proof`; ``tests/enforcement/test_proof_carrying_requests_omit_an_absent_proof.py``
    fails a model that declares the field by hand or puts a null proof on the wire.

    :param identity_proof: the presenter's proof of possession of its token's key over this request
        (:class:`DoorCredential`); ``None`` sends the token alone, and the key is then left off the wire
    :ptype identity_proof: str | None
    """

    identity_proof: str | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_proof(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """leave ``identity_proof`` off the wire when there is none, for a hub that predates it.

        :param handler: pydantic's serializer for the fields
        :ptype handler: SerializerFunctionWrapHandler
        :return: the serialized request
        :rtype: dict[str, Any]
        """
        return omit_absent_proof(handler(self))


def door_request_hash(body: Mapping[str, Any], *, door: str, op: str) -> str:
    """the hash a door request's proof signs: sha-256 over the door, the operation and the body.

    :param body: the request body, as JSON values; its proof field is left out
    :ptype body: Mapping[str, Any]
    :param door: the door the request is sent to
    :ptype door: str
    :param op: the operation within the door -- the subject's own part, so two subjects that take
        one body shape never share a hash
    :ptype op: str
    :return: the base64url digest, unpadded
    :rtype: str
    :raises TypeError: when the body holds a value that is not JSON
    """
    content = {key: value for key, value in body.items() if key != DOOR_PROOF_FIELD}
    # length-prefixed, so no door or operation, whatever it holds, can be read as another split
    frame = f"{len(door)}:{door}{len(op)}:{op}\n"
    text = frame + json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # str(bytes, "ascii"), as canonical_call_hash: the alg-pin enforcement reads every ``.decode(``
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


def _unverified_payload(token: str) -> Mapping[str, Any] | None:
    """the claims ``token`` carries, read WITHOUT verifying it -- for routing and counting only.

    :param token: a compact JWS
    :ptype token: str
    :return: its payload, or ``None`` when it does not parse as one
    :rtype: Mapping[str, Any] | None
    """
    payload: Any = None
    parts = token.split(".")
    if len(parts) == 3:
        segment = parts[1]
        try:
            payload = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
        except ValueError, TypeError:
            payload = None
    return payload if isinstance(payload, dict) else None


def _bound_thumbprint(token: str) -> str | None:
    """the key thumbprint ``token`` binds (its ``cnf``), read without verifying it.

    Unverified on purpose: the sender only decides whether its key is the one the token names; the
    door verifies everything. An identity token carries ``{"jkt": ...}``, a delegated token the
    thumbprint itself.

    :param token: a compact JWS
    :ptype token: str
    :return: the thumbprint, or ``None`` when the token binds none or does not parse
    :rtype: str | None
    """
    payload = _unverified_payload(token)
    cnf = payload.get("cnf") if payload is not None else None
    bound = cnf.get("jkt") if isinstance(cnf, dict) else cnf
    return bound if isinstance(bound, str) and bound else None


def claimed_sender_of(body: Mapping[str, Any]) -> str | None:
    """who a request says it is from, unverified, for a door's count of callers without a proof.

    The ``sub`` of the request's own token, else of its user assertion. The count is the fleet's
    rollout signal -- it names the principals still sending a token without its proof -- and never
    an authorization input.

    :param body: the request as received
    :ptype body: Mapping[str, Any]
    :return: the claimed sender, or ``None`` when neither token carries one
    :rtype: str | None
    """
    sender: str | None = None
    for name in DOOR_SENDER_FIELDS:
        token = body.get(name)
        payload = _unverified_payload(token) if isinstance(token, str) else None
        claimed = payload.get("sub") if payload is not None else None
        if sender is None and isinstance(claimed, str) and claimed:
            sender = claimed
    return sender


@dataclass(frozen=True, slots=True)
class PopKey:
    """the key a principal's own hub identity token binds (its ``cnf``): what proves that token.

    An agent's or a tool pod's handshake proof-of-possession key. Its own type, so it can never be
    handed where a :class:`RegistrationKey` is meant -- the two are different keys of one pod.

    :ivar key: the Ed25519 private key
    """

    key: Ed25519PrivateKey = field(repr=False)


@dataclass(frozen=True, slots=True)
class RegistrationKey:
    """a tool pod's registration identity key, which the registry binds each call's delegated token to.

    The key a pod's self-signed connect token and registration manifest are signed with; its public
    half is the pod's row in the hub. Not the pod's :class:`PopKey`, which its own hub identity token
    binds.

    :ivar key: the Ed25519 private key
    """

    key: Ed25519PrivateKey = field(repr=False)


@dataclass(frozen=True, slots=True)
class DoorCredential:
    """what a caller presents at a door: a token, and the key that proves it when the token binds one.

    A key the token does not bind is dropped at construction, so the credential is bearer: the door
    would refuse a proof under a key the token does not name, where its window still admits the
    token alone. Nothing can pair a token with a key it does not bind.

    :ivar token: the delegated token (a tool pod) or the caller's own identity token (an agent)
    :ivar holder_key: the key the token binds, or ``None`` for a bearer credential
    """

    token: str
    holder_key: Ed25519PrivateKey | None = None

    def __post_init__(self) -> None:
        """drop a key the token does not bind.

        :return: nothing
        :rtype: None
        """
        key = self.holder_key
        if key is not None and _bound_thumbprint(self.token) != jwk_thumbprint(key.public_key()):
            object.__setattr__(self, "holder_key", None)

    @property
    def proves(self) -> bool:
        """whether requests made with this credential carry a proof.

        :return: whether it holds the key its token binds
        :rtype: bool
        """
        return self.holder_key is not None

    @classmethod
    def for_tool_call(
        cls,
        *,
        delegated_token: str | None,
        identity_token: str | None,
        key: PopKey | RegistrationKey | None,
    ) -> DoorCredential | None:
        """the credential a tool server presents for one call.

        A tool pod holding its :class:`RegistrationKey`: the call's delegated token proven with it,
        when the registry minted one bound to it. An agent's in-process server holding its
        :class:`PopKey`: the call's identity token -- the agent's own -- proven with it. Otherwise
        the call's identity token alone, bearer, which the doors accept until their windows close.
        The key's type says which token it may prove, so neither can prove the other's.

        :param delegated_token: the call context's delegated token, if the registry minted one
        :ptype delegated_token: str | None
        :param identity_token: the call's forwarded identity token, if any
        :ptype identity_token: str | None
        :param key: the server's own key, if it holds one
        :ptype key: PopKey | RegistrationKey | None
        :return: the credential, or ``None`` when the call carries no token at all
        :rtype: DoorCredential | None
        """
        credential: DoorCredential | None = None
        delegated = (
            cls(token=delegated_token, holder_key=key.key)
            if delegated_token and isinstance(key, RegistrationKey)
            else None
        )
        if delegated is not None and delegated.proves:
            credential = delegated
        elif identity_token:
            credential = cls(token=identity_token, holder_key=key.key if isinstance(key, PopKey) else None)
        return credential

    @classmethod
    def for_request(cls, identity_token: str, credential: DoorCredential | None) -> DoorCredential:
        """the credential a request carrying ``identity_token`` is sent with: ``credential`` when given.

        :param identity_token: the token the request carries
        :ptype identity_token: str
        :param credential: the caller's credential, if it passed one
        :ptype credential: DoorCredential | None
        :return: the credential to present
        :rtype: DoorCredential
        :raises ValueError: when ``credential`` carries another token than the request
        """
        if credential is not None and credential.token != identity_token:
            raise ValueError("the credential carries another token than the request")
        return credential if credential is not None else cls(token=identity_token)

    @classmethod
    def own(cls, token: str, key: PopKey | None) -> DoorCredential:
        """a principal's own token, proven with its :class:`PopKey` when that is the key it binds.

        :param token: the principal's hub identity token
        :ptype token: str
        :param key: the principal's proof-of-possession key, if it holds one
        :ptype key: PopKey | None
        :return: the credential
        :rtype: DoorCredential
        """
        return cls(token=token, holder_key=key.key if key is not None else None)

    def present(self, body: Mapping[str, Any], *, door: str, op: str, now: int | None = None) -> dict[str, Any]:
        """``body`` with this credential's token and, when it holds a key, the proof over both.

        :param body: the request body, as JSON values, without the token or proof
        :ptype body: Mapping[str, Any]
        :param door: the door the request is sent to
        :ptype door: str
        :param op: the operation within the door
        :ptype op: str
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
                body_hash=door_request_hash(sent, door=door, op=op),
                nonce=secrets.token_urlsafe(16),
                iat=int(time.time()) if now is None else now,
            )
        return sent

    def presented(self, message: _M, *, door: str, op: str) -> _M:
        """``message`` with the proof in its ``identity_proof`` field, when this credential holds a key.

        For a wire model that already carries this credential's token in ``identity_token``: the
        proof signs ``model_dump(mode="json")``, which is what ``model_dump_json`` puts on the wire.

        :param message: the wire model, its ``identity_proof`` unset
        :ptype message: BaseModel
        :param door: the door the request is sent to
        :ptype door: str
        :param op: the operation within the door
        :ptype op: str
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
            proof = self.present(body, door=door, op=op)[DOOR_PROOF_FIELD]
            proven = message.model_copy(update={DOOR_PROOF_FIELD: proof})
        return proven


def verify_door_request(
    body: Mapping[str, Any],
    *,
    jwks: Mapping[str, Any],
    hub_issuer: str,
    door: str,
    op: str,
) -> DoorCaller:
    """verify a door request's credential and its proof over the body as received, at ``door`` for ``op``.

    :param body: the parsed request body
    :ptype body: Mapping[str, Any]
    :param jwks: the hub JWKS
    :ptype jwks: Mapping[str, Any]
    :param hub_issuer: the hub's identity-token issuer
    :ptype hub_issuer: str
    :param door: this door, which a delegated token must name
    :ptype door: str
    :param op: the operation the door is about to perform
    :ptype op: str
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
        body_hash=door_request_hash(body, door=door, op=op),
    )


class DoorProofLedger(Protocol):
    """where a door spends proof ids: a shared ``ReplayGuard``, or one process's own ledger."""

    @property
    def retention(self) -> timedelta:
        """how long a spent id is remembered.

        :return: the retention
        :rtype: timedelta
        """
        ...

    def require_covers(self, future_tolerance: timedelta) -> None:
        """refuse a verifier whose future tolerance the ledger was not sized for.

        :param future_tolerance: the verifier's tolerance
        :ptype future_tolerance: timedelta
        :raises ValueError: when it exceeds the ledger's sizing
        """
        ...

    async def record_unique(self, nonce: str, *, issued_at: Any) -> bool:
        """spend ``nonce``; ``False`` when it was spent before (or must be refused).

        :param nonce: the id
        :ptype nonce: str
        :param issued_at: the proof's issue time, timezone-aware
        :ptype issued_at: datetime
        :return: whether it was fresh
        :rtype: bool
        """
        ...


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

    A request carrying ``identity_proof`` is served only when the proof verifies over the body, the
    door and the operation, and its ``jti`` is spent in the ledger. A request without one is the
    window's to judge -- accepted, and counted per claimed sender, while it is open; refused from its
    date -- and its token is then verified by the door exactly as before. The per-door count of
    bearer callers is the fleet's rollout signal, never a switch. The door never raises: its answer
    is a :class:`DoorVerdict`.

    Its skeleton is :class:`~threetears.core.security.service_envelope.ServiceEnvelopeDoor`'s: the
    sender's key set read through an :class:`EnvelopeJwksSource` and refreshed once when the token
    names a key it lacks (a hub re-key), an empty key set refused as unavailable, ids spent under a
    namespace of their own, every refusal counted by cause.

    :param switch: the door's window, from the process's door registry; its door name is the
        audience a delegated token must name
    :ptype switch: DoorProofSwitch
    :param jwks: the hub JWKS -- the hub's identity keys and the registry's delegated-token key
    :ptype jwks: EnvelopeJwksSource
    :param hub_issuer: the hub's identity-token issuer
    :ptype hub_issuer: str
    :param replay_guard: the door's single-use ledger of proof ids
    :ptype replay_guard: DoorProofLedger
    :param accepts_delegated: whether a tool pod may act for its caller here with the call's
        delegated token; a door no tool pod reaches as its caller refuses one
    :ptype accepts_delegated: bool
    :raises ValueError: when the ledger was not sized for the proof's clock tolerance, or forgets an
        id before :data:`DOOR_PROOF_REPLAY_TTL`, while its proof could still be accepted
    """

    def __init__(
        self,
        *,
        switch: DoorProofSwitch,
        jwks: EnvelopeJwksSource,
        hub_issuer: str,
        replay_guard: DoorProofLedger,
        accepts_delegated: bool,
    ) -> None:
        replay_guard.require_covers(ISSUE_TIME_FUTURE_TOLERANCE)
        if replay_guard.retention < DOOR_PROOF_REPLAY_TTL:
            raise ValueError(
                f"a door-proof ledger must remember an id for {DOOR_PROOF_REPLAY_TTL}, "
                f"not {replay_guard.retention}: a proof is accepted that long"
            )
        self._switch = switch
        self._jwks = jwks
        self._accounting = DoorAccounting(
            door=switch.door,
            counter_name=_REFUSED_COUNTER,
            description="door requests refused for their proof of possession",
            message="door request refused",
        )
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

    async def admit(self, body: Mapping[str, Any], *, op: str) -> DoorVerdict:
        """decide ``body``, the request as received, its proof field included, for ``op``.

        A request without a proof is counted under its claimed sender (:func:`claimed_sender_of`),
        read here off the token the door already reads, so no door attributes it by hand.

        :param body: the parsed request
        :ptype body: Mapping[str, Any]
        :param op: the operation the door is about to perform, as the sender signed it
        :ptype op: str
        :return: the verdict
        :rtype: DoorVerdict
        """
        token = body.get(DOOR_TOKEN_FIELD)
        proof = body.get(DOOR_PROOF_FIELD)
        if isinstance(proof, str) and proof:
            verdict = await self._verified(body, op=op)
        elif isinstance(token, str) and is_delegated_token(token):
            # a delegated token is useless without its pod's key; there is no bearer form of one
            verdict = self._refused("absent", "a delegated token arrived without its proof", op=op)
        else:
            verdict = (
                DoorVerdict(admitted=True)
                if window_admits(self._switch, claimed_sender=claimed_sender_of(body))
                else self._refused("absent", "the door requires a proof of possession", op=op)
            )
        return verdict

    async def admit_message(self, data: bytes, *, op: str) -> DoorVerdict:
        """decide the request ``data`` for ``op``, hashed exactly as sent.

        The door to use when the door's own request model is a tolerant reader, which would hash
        less than the caller signed for a caller a release ahead.

        :param data: the request bytes, a JSON object
        :ptype data: bytes
        :param op: the operation the door is about to perform
        :ptype op: str
        :return: the verdict; a body that is not a JSON object is refused
        :rtype: DoorVerdict
        """
        body = json_object(data)
        verdict = (
            await self.admit(body, op=op)
            if body is not None
            else self._refused("malformed", "the request is not a JSON object", op=op)
        )
        return verdict

    async def _verified(self, body: Mapping[str, Any], *, op: str) -> DoorVerdict:
        """verify the proof and spend its id.

        :param body: the parsed request, carrying a proof
        :ptype body: Mapping[str, Any]
        :param op: the operation
        :ptype op: str
        :return: the verdict
        :rtype: DoorVerdict
        """
        verdict: DoorVerdict
        try:
            jwks = await self._jwks_naming(body.get(DOOR_TOKEN_FIELD))
            caller = verify_door_request(body, jwks=jwks, hub_issuer=self._hub_issuer, door=self.door, op=op)
            fresh = await self._replay_guard.record_unique(
                f"door-proof|{caller.agent_id}|{caller.proof.jti}", issued_at=caller.proof.issued_at
            )
        except KeysUnavailableError as exc:
            verdict = self._refused("keys_unavailable", f"{exc}", op=op)
        except ProofAbsentError as exc:
            verdict = self._refused("absent", f"{exc}", op=op)
        except DelegatedTokenError as exc:
            verdict = self._refused("unverified", f"{exc}", op=op)
        except KvError as exc:
            # the ledger cannot say whether the proof was used: refuse rather than admit a replay
            verdict = self._refused("ledger_unavailable", type(exc).__name__, op=op)
        else:
            if not fresh:
                verdict = self._refused("replayed", "the proof was used before", op=op)
            elif caller.pod_id is not None and not self._accepts_delegated:
                verdict = self._refused("delegated", "this door does not take a tool pod acting for its caller", op=op)
            else:
                verdict = DoorVerdict(admitted=True, caller=caller)
        return verdict

    async def _jwks_naming(self, token: Any) -> Mapping[str, Any]:
        """the hub key set, fetched afresh once when it lacks the token's kid (a hub re-key).

        :param token: the presented token
        :ptype token: Any
        :return: the key set
        :rtype: Mapping[str, Any]
        :raises KeysUnavailableError: when no key set is loaded
        """
        kid: Any = None
        if isinstance(token, str):
            try:
                kid = jwt.get_unverified_header(token).get("kid")
            except jwt.PyJWTError:
                kid = None
        return await keys_naming(self._jwks, kid)

    def _refused(self, cause: str, reason: str, *, op: str) -> DoorVerdict:
        """log and count a refusal, and answer it.

        :param cause: the refusal's class, the counter's ``cause`` label
        :ptype cause: str
        :param reason: what failed
        :ptype reason: str
        :param op: the operation refused
        :ptype op: str
        :return: a refusing verdict
        :rtype: DoorVerdict
        """
        self._accounting.refused(cause, reason, op=op)
        return DoorVerdict(admitted=False)
