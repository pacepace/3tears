"""a forwarded request proves it comes from the repository's own deployment.

A NATS subject proves nothing about who sent a message: any connection that can publish on the git-store forward
family could otherwise have its change set committed (the receiver was declared ``unauthenticated``; Pace chose the
signed form, 2026-10-17). So every request a process forwards to a repository's writer travels sealed in 3tears'
service envelope (:mod:`threetears.core.security.service_envelope`): an Ed25519 signature over exactly the body,
addressed to this repository's door and this handler, alive for at most a minute, with a single-use id the writer
spends in a shared :class:`~threetears.core.coordination.ReplayGuard`. The writer's door is born requiring it: an
unsigned, wrongly signed, stale or replayed request is refused, logged and counted, and the sender gets a typed
refusal.

The processes of one deployment share one signing seed (``GitRepoConfig.forward_signing_seed``, 32 bytes, base64);
its public half is pinned at the writer (:class:`~threetears.core.security.PinnedEnvelopeJwks`), so no key server is
needed. A deployment that runs the 3tears hub can move to its published keys behind the same door.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from threetears.core.coordination import ReplayAnchor, ReplayGuard
from threetears.core.security import (
    SERVICE_ENVELOPE_LEEWAY,
    SERVICE_ENVELOPE_REPLAY_TTL,
    PinnedEnvelopeJwks,
    ServiceEnvelopeDoor,
    ServiceEnvelopeSigner,
)

if TYPE_CHECKING:
    from pydantic import SecretStr

    from threetears.nats import NatsClient

__all__ = ["DEFAULT_ISSUER", "ForwardEnvelope", "ForwardRefusedError", "GitDirReplayAnchor"]

#: the sending service a writer takes forwarded requests from, unless the configuration names another
DEFAULT_ISSUER: Final[str] = "git-store"


class ForwardRefusedError(PermissionError):
    """the writer refused a forwarded request: unsigned, signed by another key, stale, or a replay."""


class GitDirReplayAnchor:
    """the replay ledger's first-existence moment, kept in the repository's git directory.

    The ledger (a memory-backed KV bucket) dies with the NATS broker, so its guard can't tell a new bucket from a
    wiped one unless something durable remembers when the ledger first existed (3tears' ``ReplayAnchor``). Without
    one the guard refused every request issued within its reach of the bucket's creation: on a fresh deployment and
    after every NATS restart, every forward for the first seconds (found adopting it in samhla). The repository's
    git directory is this store's own durable place: the first process to anchor creates the file exclusively and
    every later one reads its moment back. Processes that share a working tree share it; a clone of its own (scriob)
    keeps its own, so a writer on a new host anchors when it first serves. A consumer with database-backed
    coordination tables can pass 3tears' ``CollectionReplayAnchor`` instead.

    :param directory: a folder inside the repository's git directory (``Git.git_path("git-store")``)
    :ptype directory: Path
    """

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    async def first_existed(self, purpose: str, *, now: datetime) -> datetime:
        """claim or read ``purpose``'s first-existence moment (ReplayAnchor).

        :param purpose: the ledger, the guard's bucket name
        :ptype purpose: str
        :param now: this process's clock, recorded when nothing is
        :ptype now: datetime
        :return: the aware UTC moment the ledger first existed
        :rtype: datetime
        :raises OSError: on a storage failure (the guard then falls back to refusing, as it should)
        :raises ValueError: when the recorded moment can't be read
        """
        return await asyncio.to_thread(self._claim_or_read, purpose, now)

    def _claim_or_read(self, purpose: str, now: datetime) -> datetime:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"replay-anchor-{purpose}"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            moment = datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
            if moment.tzinfo is None:
                raise ValueError(f"{path}: the anchor has no time zone") from None
            return moment
        stamp = now.astimezone(UTC)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(stamp.isoformat() + "\n")
            f.flush()
            os.fsync(f.fileno())
        return stamp


class ForwardEnvelope:
    """seals what this process forwards to a repository's writer, and admits only sealed requests there.

    :param nats: the NATS client (the replay guard's bucket)
    :ptype nats: NatsClient
    :param seed: the deployment's signing seed: 32 raw Ed25519 bytes, base64 or base64url; never logged
    :ptype seed: SecretStr
    :param audience: the writer's door, one per repository and branch (the lease key)
    :ptype audience: str
    :param issuer: the sending service's name, the same in every process of the deployment
    :ptype issuer: str
    :param bucket: the replay guard's KV bucket
    :ptype bucket: str
    :param anchor: where the ledger's first existence is remembered (:class:`GitDirReplayAnchor`, or a consumer's
        ``CollectionReplayAnchor``)
    :ptype anchor: ReplayAnchor
    """

    def __init__(
        self,
        nats: NatsClient,
        *,
        seed: SecretStr,
        audience: str,
        issuer: str = DEFAULT_ISSUER,
        bucket: str,
        anchor: ReplayAnchor,
    ) -> None:
        self._signer = ServiceEnvelopeSigner.from_seed_text(
            seed.get_secret_value(), issuer=issuer, what="GitRepoConfig.forward_signing_seed"
        )
        self._audience = audience
        self._guard = ReplayGuard(
            nats,
            bucket_name=bucket,
            ttl_seconds=int(SERVICE_ENVELOPE_REPLAY_TTL.total_seconds()),
            verifier_future_tolerance=SERVICE_ENVELOPE_LEEWAY,
            anchor=anchor,
        )
        self._door = ServiceEnvelopeDoor.required(
            door=f"git-store:{audience}",
            issuer=issuer,
            audience=audience,
            jwks=PinnedEnvelopeJwks(self._signer.signing_key.public_key()),
            replay_guard=self._guard,
        )

    async def bind(self) -> None:
        """open the replay guard's bucket; before serving anything.

        :return: None
        :rtype: None
        """
        await self._guard.bind()

    def seal(self, name: str, payload: bytes) -> bytes:
        """the request ``payload`` for handler ``name``, sealed for this repository's writer.

        :param name: the handler, the envelope's operation
        :ptype name: str
        :param payload: the request
        :ptype payload: bytes
        :return: the message to send
        :rtype: bytes
        """
        body = {"name": name, "payload": base64.b64encode(payload).decode("ascii")}
        sealed = self._signer.seal(audience=self._audience, op=name, body=body)
        return json.dumps(sealed, separators=(",", ":")).encode()

    async def open(self, name: str, data: bytes) -> bytes:
        """the request inside ``data``, once the writer's door admits it for handler ``name``.

        :param name: the handler it arrived for
        :ptype name: str
        :param data: the message as received
        :ptype data: bytes
        :return: the request
        :rtype: bytes
        :raises ForwardRefusedError: when the door refuses it (logged and counted by the door)
        """
        if not await self._door.admit_message(data, op=name):
            raise ForwardRefusedError(f"the writer of {self._audience} refused an unsigned or invalid {name} request")
        body = json.loads(data)
        if body.get("name") != name or not isinstance(body.get("payload"), str):
            raise ForwardRefusedError(f"the {name} request names another handler or carries no payload")
        return base64.b64decode(body["payload"])
