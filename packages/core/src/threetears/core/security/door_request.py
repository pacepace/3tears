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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from threetears.core.security.delegated_token import DelegatedTokenError, DoorCaller, verify_door_caller
from threetears.core.security.pop import access_token_hash, make_pop_proof

__all__ = [
    "DOOR_PROOF_FIELD",
    "DOOR_TOKEN_FIELD",
    "DoorCredential",
    "door_request_hash",
    "verify_door_request",
]

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
    ) -> DoorCredential | None:
        """the credential a tool pod presents for one call.

        The call's delegated token proven with the pod's key when the registry minted one and the
        pod holds its key; otherwise the caller's forwarded token, bearer, as before, which the
        doors accept until their windows close.

        :param delegated_token: the call context's delegated token, if the registry minted one
        :ptype delegated_token: str | None
        :param identity_token: the caller's forwarded identity token, if any
        :ptype identity_token: str | None
        :param pod_key: the pod's own identity key, if the pod holds it
        :ptype pod_key: Ed25519PrivateKey | None
        :return: the credential, or ``None`` when the call carries no token at all
        :rtype: DoorCredential | None
        """
        credential: DoorCredential | None = None
        if delegated_token and pod_key is not None:
            credential = cls(token=delegated_token, holder_key=pod_key)
        elif identity_token:
            credential = cls(token=identity_token)
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
