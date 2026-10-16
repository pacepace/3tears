"""A reusable live fixture for the door-refusal regression: a forged message, sent as a real pod, is
refused by its receiver with no side effect, while the genuine sender still succeeds.

This generalises the two patterns already in the tree:

- ``test_a_pod_can_bounce_its_bytes_to_any_subject_residual_live.py`` stands up a real
  ``nats:2.12.6`` server, mints a real tool-pod JWT as the server's config permissions, and shows a
  pod can put bytes of its choosing on a subject it may not publish, two ways: a push consumer whose
  ``deliver_subject`` is the target, and a request whose reply subject is the target. That is the
  *delivery* a receiver cannot stop at the bus.
- ``packages/core/tests/unit/security/`` (``test_door_proof``, ``test_door_request``,
  ``test_nats_proxy_door_proof``) shows each door's receiver refuses a request whose proof is
  missing, replayed, or signed by a key the door does not trust. That is the *refusal* the receiver
  owns.

A door "passes" the harness only when all three hold, for a forgery arriving by each delivery:

- **no side effect** — no database row, KV write, onward publish, or membership change;
- **one structured refusal** on the receiver's log;
- **the legitimate path is unaffected** — the real sender's genuine message still succeeds.

The harness is the fixture and these helpers; each door is one case that supplies its forged body,
its genuine body, and a probe for the side effect. The cases live in
``test_forged_door_delivery_live.py``.

Gated on docker: a checkout without docker skips cleanly. Credentials are ephemeral testcontainer
values; nothing here reaches a real cluster. Against cobalt dev the same cases run through the
cluster's own NATS with credentials read by reference (see ``scripts/run-forger-harness.sh``);
nothing is printed.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from threetears.nats import NatsClient
from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

#: the local stack's nats-server, so a forgery is proved on the version that runs
NATS_IMAGE = "nats:2.12.6"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential


@dataclass
class ForgeryOutcome:
    """what a receiver did with a message the harness delivered, read after each send.

    :ivar side_effect: True if the receiver acted on the message (a row, a write, an onward publish)
    :ivar refusal_logged: True if exactly one structured refusal names the message
    :ivar delivered: True if the message reached the receiver's subject at all
    """

    side_effect: bool
    refusal_logged: bool
    delivered: bool


@dataclass
class DoorCase:
    """one door's live forgery case.

    :ivar door: the door's name, for the test id and the refusal assertion
    :ivar target_subject: the receiver's subject the forgery is aimed at
    :ivar forged_body: a body a real pod has no valid proof for (missing, replayed, or wrong-key)
    :ivar genuine_body: the body the real sender sends, which must still succeed
    :ivar read_outcome: reads the receiver's state and log into a ForgeryOutcome
    """

    door: str
    target_subject: str
    forged_body: bytes
    genuine_body: bytes
    read_outcome: Callable[[], Awaitable[ForgeryOutcome]]
    genuine_subject: str | None = None
    #: extra subjects the harness subscribes so a bounced delivery is observed
    watch_subjects: list[str] = field(default_factory=list)


def minted_allow_lists(permissions: PrincipalPermissions) -> tuple[list[str], list[str]]:
    """mint a real user JWT for ``permissions`` and return its (publish allow, subscribe allow) lists.

    :param permissions: the resolved allow-list to mint
    :ptype permissions: PrincipalPermissions
    :return: the JWT's publish allow-list and subscribe allow-list
    :rtype: tuple[list[str], list[str]]
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UFORGER",
        permissions=permissions,
        name="forger-harness-live",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    return payload["nats"]["pub"]["allow"], payload["nats"]["sub"]["allow"]


def server_config(pub_allow: list[str], sub_allow: list[str]) -> str:
    """a JetStream nats-server config with a full admin and the pod user under its minted allow-lists.

    :param pub_allow: the minted publish allow-list
    :ptype pub_allow: list[str]
    :param sub_allow: the minted subscribe allow-list
    :ptype sub_allow: list[str]
    :return: the nats-server configuration text
    :rtype: str
    """
    authorization = {
        "users": [
            {
                "user": "admin",
                "password": _ADMIN_PW,
                "permissions": {"publish": ">", "subscribe": ">", "allow_responses": True},
            },
            {
                "user": "pod",
                "password": _POD_PW,
                "permissions": {
                    "publish": {"allow": pub_allow},
                    "subscribe": {"allow": sub_allow},
                    "allow_responses": True,
                },
            },
        ]
    }
    return f"jetstream {{ store_dir: /tmp/js-store }}\nport: 4222\nauthorization {json.dumps(authorization)}\n"


@contextlib.contextmanager
def nats_with_auth(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a JetStream nats-server with a custom ``authorization`` config; yield its URI.

    :param config_text: the server configuration
    :ptype config_text: str
    :param conf_dir: a directory to mount the configuration from
    :ptype conf_dir: Path
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415

    (conf_dir / "nats.conf").write_text(config_text)
    container = (
        nats_server(NATS_IMAGE)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield nats_server_uri(container)
    finally:
        container.stop()


async def pod_client(uri: str, namespace: str, pod_id: str) -> tuple[NatsClient, PrincipalPermissions]:
    """a NatsClient connected as the forging tool pod, under its own real minted grant.

    :param uri: the server URI
    :ptype uri: str
    :param namespace: the subject namespace
    :ptype namespace: str
    :param pod_id: the pod's id
    :ptype pod_id: str
    :return: the connected client and the permissions it was minted under
    :rtype: tuple[NatsClient, PrincipalPermissions]
    """
    permissions = build_permissions(Principal.TOOL_POD, pod_id=pod_id, conn_id=pod_id, object_store=True)
    client = await NatsClient.connect(
        nats_url=uri,
        nats_subject_namespace=namespace,
        client_name="pod",
        user="pod",
        password=_POD_PW,
        inbox_prefix=permissions.inbox_prefix,
    )
    return client, permissions


@contextlib.asynccontextmanager
async def admin_client(uri: str, namespace: str) -> Any:
    """a NatsClient connected as the server admin (stands in for the receiver), shut down on exit.

    :param uri: the server URI
    :ptype uri: str
    :param namespace: the subject namespace
    :ptype namespace: str
    :return: the connected client, yielded
    :rtype: Any
    """
    client = await NatsClient.connect(
        nats_url=uri, nats_subject_namespace=namespace, client_name="admin", user="admin", password=_ADMIN_PW
    )
    try:
        yield client
    finally:
        await client.shutdown(drain_timeout=timedelta(seconds=1))


async def collect(client: NatsClient, subjects: list[str], seconds: float = 5.0) -> dict[str, list[bytes]]:
    """subscribe ``subjects`` on ``client`` and return what each received within ``seconds``.

    The receiver stands in for a service whose door the forgery targets: anything delivered here is a
    message the bus let through, which the door must still refuse to act on.

    :param client: the receiving client
    :ptype client: NatsClient
    :param subjects: the subjects to watch
    :ptype subjects: list[str]
    :param seconds: how long to wait for deliveries
    :ptype seconds: float
    :return: subject to the list of bodies it received
    :rtype: dict[str, list[bytes]]
    """
    received: dict[str, list[bytes]] = {subject: [] for subject in subjects}
    for subject in subjects:

        async def note(msg: Any, subject: str = subject) -> None:
            received[subject].append(bytes(msg.data))

        await client.raw.subscribe(subject, cb=note)
    deadline = asyncio.get_running_loop().time() + seconds
    while not all(received.values()) and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
    return received
