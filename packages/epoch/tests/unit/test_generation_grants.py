"""who may read a table's write generation, and who may write it.

The grants in ``threetears.nats.subject_permissions`` are paired here with the keys
``threetears.epoch`` actually opens, on the MINTED grant, because the grant is what the broker
enforces: a grant that names any other spelling of the bucket or the key is a JetStream call
that blocks to its deadline rather than a refusal anyone reads.

The contract this pins:

- an agent pod and a tool pod may FOLLOW: bind the epoch bucket, read a generation key, and watch
  one through a named consumer filtered on that key;
- neither may write a generation key. A pod that could would fake an advance every follower acts
  on; the hub's L3 broker advances for a pod's write;
- the hub and the gateway may write one.
"""

from __future__ import annotations

import base64
import json

import pytest

from threetears.epoch import generation_kv_key
from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.subjects import set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

_NS = "grantprobe"
_BUCKET = f"{_NS}-epochs"
_STREAM = f"KV_{_BUCKET}"

#: the tables a pod's per-caller access cache is derived from, and so follows.
_ACCESS_TABLES = ("groups", "group_members", "roles", "role_assignments")


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace(_NS)


def _pattern_admits(pattern: str, subject: str) -> bool:
    """NATS subject matching: ``*`` spans one token, ``>`` one or more trailing tokens."""
    pattern_tokens = pattern.split(".")
    subject_tokens = subject.split(".")
    result = len(pattern_tokens) == len(subject_tokens)
    for index, token in enumerate(pattern_tokens):
        if token == ">":
            result = index < len(subject_tokens)
            break
        if index >= len(subject_tokens) or (token != "*" and token != subject_tokens[index]):
            result = False
            break
    return result


def _publish_allow(permissions: PrincipalPermissions) -> list[str]:
    """the publish allow-list a real minted user JWT carries for ``permissions``."""
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTESTUSERPUBLICKEY",
        permissions=permissions,
        name="generation-grants",
        expires_in_seconds=300,
    )
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    allow: list[str] = claims["nats"]["pub"]["allow"]
    return allow


def _may(permissions: PrincipalPermissions, subject: str) -> bool:
    return any(_pattern_admits(pattern, subject) for pattern in _publish_allow(permissions))


def _permissions(principal: str) -> PrincipalPermissions:
    built = {
        "agent_pod": lambda: build_permissions(
            Principal.AGENT_POD,
            agent_id="019470a8-b5c3-7def-8123-000000000001",
            pod_id="01947100-0000-7000-8000-000000000001",
        ),
        "tool_pod": lambda: build_permissions(Principal.TOOL_POD, pod_id="01947100-0000-7000-8000-000000000002"),
        "hub": lambda: build_permissions(Principal.HUB, conn_id="conn-1"),
        "gateway": lambda: build_permissions(Principal.GATEWAY, conn_id="conn-2"),
    }
    return built[principal]()


@pytest.mark.parametrize("principal", ["agent_pod", "tool_pod"])
@pytest.mark.parametrize("table", _ACCESS_TABLES)
class TestAPodMayFollowAndMayNotAdvance:
    def test_it_may_bind_the_bucket(self, principal: str, table: str) -> None:
        assert _may(_permissions(principal), f"$JS.API.STREAM.INFO.{_STREAM}")

    def test_it_may_watch_the_generation_key_through_a_named_consumer(self, principal: str, table: str) -> None:
        key = generation_kv_key(table)
        assert key == f"{_NS}.collections.{table}.epoch"
        assert _may(_permissions(principal), f"$JS.API.CONSUMER.CREATE.{_STREAM}.kw-0123abcd.$KV.{_BUCKET}.{key}")

    def test_it_may_read_the_generation_key(self, principal: str, table: str) -> None:
        # the epoch bucket is not declared for direct gets, so a read is the stream's message get
        assert _may(_permissions(principal), f"$JS.API.STREAM.MSG.GET.{_STREAM}")

    def test_it_may_not_write_the_generation_key(self, principal: str, table: str) -> None:
        assert not _may(_permissions(principal), f"$KV.{_BUCKET}.{generation_kv_key(table)}")

    def test_it_may_not_manage_the_bucket(self, principal: str, table: str) -> None:
        permissions = _permissions(principal)
        for verb in ("CREATE", "UPDATE", "DELETE", "PURGE"):
            assert not _may(permissions, f"$JS.API.STREAM.{verb}.{_STREAM}")


@pytest.mark.parametrize("principal", ["hub", "gateway"])
def test_the_principals_that_commit_writes_may_advance(principal: str) -> None:
    assert _may(_permissions(principal), f"$KV.{_BUCKET}.{generation_kv_key('role_assignments')}")
