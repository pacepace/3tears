"""pinned pair: the key a collection WRITES is matched by the grant its principal HOLDS.

The L2 key scope exists twice. ``threetears.core.collections.base.BaseCollection.l2_key`` puts it
at the head of every key it writes into ``{ns}-collections``; ``threetears.nats.user_jwt`` puts the
same token into the ``$KV.`` publish grant and the ``$JS.API.DIRECT.GET`` read grant the auth
callout mints for that principal. **Nothing at runtime compares the two.** If they diverge, every
read misses, falls through to L3, and nothing logs it: correct answers, a dead cache -- and every
write lands on a subject the principal is not granted, which NATS answers with silence rather than
an error.

So this file composes the two halves and asserts they meet: it builds a real
:class:`CollectionRegistry` scoped through :func:`kv_key_scope_for`, takes the literal key its
collection writes, mints the real user JWT for the same principal, and asserts the minted subjects
MATCH that key under NATS subject-matching rules.

**What this proves, and what it does not -- state both, because the first invites over-trust.**

It proves both sides call :func:`kv_key_scope_for`, that the composed subjects are literally
token-compatible with the composed key (the ``$KV`` / ``{bucket}`` two-token shape in the direct-read
subject is the specific thing hand-derivation got wrong twice), and that one principal's grant does
NOT match another principal's key.

It CANNOT prove the two sides feed the helper the same INPUT. For an infra principal there is no
input beyond the enum member, so the pair is total there. For a POD principal the grant side takes
``agent_id`` / ``pod_id`` from the authenticated identity the callout verified, while the key side
takes it from whatever the process was configured with -- and a test that passes the same variable
to both is asserting its own setup. That residual closes only by sourcing the key-side input from
the authenticated identity, not by widening this file.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from typing import Any, Final

import pytest

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats.subject_permissions import (
    JsResource,
    JsResourceKind,
    Principal,
    build_permissions,
    kv_key_scope_for,
)
from threetears.nats.subjects import set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

#: every resolver interpolates the subject namespace, so it must be set before the principal set
#: below can be derived -- and that derivation happens at COLLECTION time, which is why this call
#: is at module scope rather than in the autouse fixture that keeps each test deterministic.
_NS: Final[str] = "pinnedpair"
set_default_namespace(_NS)

_BUCKET_SUFFIX: Final[str] = BaseCollection.L2_BUCKET_SUFFIX
_TABLE: Final[str] = "widgets"
_ENTITY_ID: Final[str] = "e1"

#: a stable id for the pod principals, threaded into BOTH sides on purpose -- see the module
#: docstring for why that is a setup fact rather than a proof.
_POD_UUID: Final[uuid.UUID] = uuid.UUID("019470a8-b5c3-7def-8123-456789abcdef")


class _StubEntity(BaseEntity):
    """minimal single-pk entity for the key composition."""

    primary_key_field = "id"


class _StubCollection(BaseCollection[_StubEntity]):
    """the smallest collection that can compose an L2 key.

    L3 is a dict, because nothing here reads or writes a store -- the subject of the test is
    :meth:`BaseCollection.l2_key`, which touches only the registry's scope and the table name.
    """

    @property
    def table_name(self) -> str:
        """the table segment of every key this collection writes.

        :return: table name
        :rtype: str
        """
        return _TABLE

    @property
    def entity_class(self) -> type[_StubEntity]:
        """the entity type this collection hydrates.

        :return: entity class
        :rtype: type[_StubEntity]
        """
        return _StubEntity

    async def fetch_from_store(self, entity_id: object) -> dict[str, Any] | None:
        """no L3 behind this collection.

        :param entity_id: pk value
        :ptype entity_id: object
        :return: always ``None``
        :rtype: dict[str, Any] | None
        """
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        """no L3 behind this collection.

        :param data: row payload
        :ptype data: dict[str, Any]
        :param original_timestamp: optimistic-concurrency stamp
        :ptype original_timestamp: datetime | None
        :return: rows written
        :rtype: int
        """
        return 0

    async def delete_from_store(self, entity_id: object) -> None:
        """no L3 behind this collection.

        :param entity_id: pk value
        :ptype entity_id: object
        :return: nothing
        :rtype: None
        """
        return None

    def serialize(self, data: dict[str, Any]) -> bytes:
        """encode a row for L2.

        :param data: row payload
        :ptype data: dict[str, Any]
        :return: encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=str).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """decode a row from L2.

        :param data: encoded bytes
        :ptype data: bytes
        :return: row payload
        :rtype: dict[str, Any]
        """
        decoded: dict[str, Any] = json.loads(data)
        return decoded


def _scope_inputs(principal: Principal) -> dict[str, Any]:
    """the identifying ids one principal's resolvers require.

    :param principal: the connection identity class
    :ptype principal: Principal
    :return: keyword arguments for :func:`build_permissions`
    :rtype: dict[str, Any]
    """
    if principal is Principal.AGENT_POD:
        return {"agent_id": str(_POD_UUID), "pod_id": str(_POD_UUID), "conn_id": str(_POD_UUID)}
    if principal is Principal.TOOL_POD:
        return {"pod_id": str(_POD_UUID), "conn_id": str(_POD_UUID)}
    return {"conn_id": "conn-1"}


def _key_side_scope(principal: Principal) -> str:
    """the scope the KEY side derives for one principal.

    :param principal: the connection identity class
    :ptype principal: Principal
    :return: the scope token
    :rtype: str
    """
    if principal is Principal.AGENT_POD:
        return kv_key_scope_for(principal, agent_id=str(_POD_UUID))
    if principal is Principal.TOOL_POD:
        return kv_key_scope_for(principal, pod_id=str(_POD_UUID))
    return kv_key_scope_for(principal)


def _l2_key_for(scope: str) -> str:
    """the literal key a scoped registry's collection writes for :data:`_ENTITY_ID`.

    Built through the production path -- a real :class:`CollectionRegistry` configured with the
    scope, and a real :class:`BaseCollection` composing the key -- so a change to the key shape
    lands here rather than in a hand-written mirror of it.

    :param scope: the registry's key scope
    :ptype scope: str
    :return: the composed L2 key
    :rtype: str
    """
    registry = CollectionRegistry()
    registry.configure(kv_key_scope=scope)
    collection = _StubCollection(registry, DefaultCoreConfig())
    return collection.l2_key(_ENTITY_ID)


def _minted_publish_subjects(principal: Principal) -> list[str]:
    """the ``pub.allow`` list the auth callout would mint for one principal.

    Read back out of a REAL minted JWT rather than recomposed from the resolver, so the assertion
    covers ``mint_user_jwt``'s own composition (the ``$KV.{bucket}.{scope}.>`` tail and the
    ``js_api_grants_for_stream`` fan-out) and not merely the record it reads.

    :param principal: the connection identity class
    :ptype principal: Principal
    :return: every subject the principal may publish to
    :rtype: list[str]
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTESTUSERPUBLICKEY",
        permissions=build_permissions(principal, **_scope_inputs(principal)),
        name=principal.value,
        expires_in_seconds=300,
    )
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    allow: list[str] = claims["nats"]["pub"]["allow"]
    return allow


def _collections_resource(principal: Principal) -> JsResource | None:
    """the collections-bucket declaration on one principal, when it holds one.

    :param principal: the connection identity class
    :ptype principal: Principal
    :return: the resource record, or ``None`` when the principal declares no collections bucket
    :rtype: JsResource | None
    """
    permissions = build_permissions(principal, **_scope_inputs(principal))
    found: JsResource | None = None
    for resource in permissions.js_resources:
        if resource.kind is JsResourceKind.KV_BUCKET and resource.name.endswith(f"-{_BUCKET_SUFFIX}"):
            found = resource
    return found


def _subject_matches(pattern: str, subject: str) -> bool:
    """NATS subject matching: ``*`` spans one token, ``>`` spans one or more trailing tokens.

    :param pattern: the granted subject pattern
    :ptype pattern: str
    :param subject: the concrete subject under test
    :ptype subject: str
    :return: whether the pattern admits the subject
    :rtype: bool
    """
    pattern_tokens = pattern.split(".")
    subject_tokens = subject.split(".")
    result = len(pattern_tokens) == len(subject_tokens)
    for index, token in enumerate(pattern_tokens):
        if token == ">":
            result = index < len(subject_tokens)
            break
        if index >= len(subject_tokens):
            result = False
            break
        if token != "*" and token != subject_tokens[index]:
            result = False
            break
    return result


#: every principal that declares the shared collections bucket. derived, never listed: a principal
#: added to the resolvers joins this test automatically, which is the whole point of pinning the
#: pair rather than pinning a table of strings.
_COLLECTIONS_PRINCIPALS: Final[tuple[Principal, ...]] = tuple(
    principal for principal in Principal if _collections_resource(principal) is not None
)


@pytest.fixture(autouse=True)
def _pin_the_namespace() -> None:
    """re-pin the process-wide subject namespace for every test in this module.

    the default namespace is a process-wide module global that sibling test modules set and reset,
    so a module that derived its expectations at import time would otherwise assert against
    whatever ran last.

    :return: nothing
    :rtype: None
    """
    set_default_namespace(_NS)


class TestTheGrantMatchesTheKey:
    """the pair itself, per principal that holds the shared bucket."""

    @pytest.mark.parametrize("principal", _COLLECTIONS_PRINCIPALS, ids=lambda p: p.value)
    def test_the_write_subject_is_granted(self, principal: Principal) -> None:
        """the ``$KV.`` publish subject the collection's write produces is in the mint's allow list."""
        resource = _collections_resource(principal)
        assert resource is not None
        key = _l2_key_for(_key_side_scope(principal))
        write_subject = f"$KV.{resource.name}.{key}"
        allow = _minted_publish_subjects(principal)
        assert any(_subject_matches(pattern, write_subject) for pattern in allow), (
            f"{principal.value} writes {write_subject!r} and holds no grant that matches it; "
            f"the key side and the grant side have diverged. granted: "
            f"{[p for p in allow if p.startswith('$KV.')]}"
        )

    @pytest.mark.parametrize("principal", _COLLECTIONS_PRINCIPALS, ids=lambda p: p.value)
    def test_the_direct_read_subject_is_granted(self, principal: Principal) -> None:
        """the ``$JS.API.DIRECT.GET`` subject nats-py issues for that key is granted too.

        ``$KV`` and the bucket name are SEPARATE TOKENS after the stream name -- nats-py appends
        the whole ``$KV.<bucket>.<key>`` subject. A grant written as
        ``$JS.API.DIRECT.GET.{stream}.{scope}.>`` matches nothing, and a subject that matches
        nothing is never answered: the caller sees a ten-second deadline, not a refusal.
        """
        resource = _collections_resource(principal)
        assert resource is not None
        key = _l2_key_for(_key_side_scope(principal))
        read_subject = f"$JS.API.DIRECT.GET.{resource.stream_name}.$KV.{resource.name}.{key}"
        allow = _minted_publish_subjects(principal)
        assert any(_subject_matches(pattern, read_subject) for pattern in allow), (
            f"{principal.value} reads {read_subject!r} and holds no grant that matches it. "
            f"granted: {[p for p in allow if p.startswith('$JS.API.DIRECT')]}"
        )


class TestTheGrantDoesNotMatchAnotherPrincipalsKey:
    """the isolation half: a matched pair that matched everything would prove nothing."""

    def test_the_registry_cannot_reach_the_hubs_key(self) -> None:
        """the registry's grants admit neither the hub's write subject nor its read subject."""
        resource = _collections_resource(Principal.REGISTRY)
        assert resource is not None
        foreign_key = _l2_key_for(_key_side_scope(Principal.HUB))
        allow = _minted_publish_subjects(Principal.REGISTRY)
        write_subject = f"$KV.{resource.name}.{foreign_key}"
        read_subject = f"$JS.API.DIRECT.GET.{resource.stream_name}.$KV.{resource.name}.{foreign_key}"
        assert not any(_subject_matches(pattern, write_subject) for pattern in allow)
        assert not any(_subject_matches(pattern, read_subject) for pattern in allow)


class TestThePairIsNotVacuous:
    """a pair over an empty principal set would pass by matching nothing."""

    def test_the_registry_is_under_test(self) -> None:
        """the process this shard wires must be one of the principals the pair covers."""
        assert Principal.REGISTRY in _COLLECTIONS_PRINCIPALS

    def test_several_principals_hold_the_shared_bucket(self) -> None:
        """the bucket is shared; a single-holder reading means a resolver stopped declaring it."""
        assert len(_COLLECTIONS_PRINCIPALS) >= 4

    def test_the_key_carries_the_scope_as_its_leading_token(self) -> None:
        """the key shape the whole pair rests on, asserted directly."""
        scope = _key_side_scope(Principal.REGISTRY)
        assert _l2_key_for(scope) == f"{scope}.{_TABLE}.{_ENTITY_ID}"

    def test_the_matcher_rejects_a_neighbouring_scope(self) -> None:
        """the subject matcher itself, proven to discriminate rather than to always agree."""
        assert _subject_matches("$KV.b.registry.>", "$KV.b.registry.widgets.e1")
        assert not _subject_matches("$KV.b.registry.>", "$KV.b.hub.widgets.e1")
        assert not _subject_matches("$KV.b.registry.>", "$KV.b.registry")
