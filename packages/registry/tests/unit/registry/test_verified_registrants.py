"""unit -- a copy of a tool is admitted from VERIFIED identity, and every refusal is in the reply.

Registration trusted the manifest's ``pod_id`` on every path. A token-bearing pod could name a
peer's pod id; a tokenless manifest could name anybody's, including the shared built-in pod's,
and rewrite what that pod's tools said they were. And a refused tool was logged on the registry
and never told to the pod that offered it.

Now:

* a token proves a Tool Pod's identity, and the manifest must name THAT pod or the whole
  manifest is refused (``POD_ID_MISMATCH``);
* an agent's in-process manifest carries the agent's OWN identity token, which must verify and
  must name the agent the pod id names;
* a copy that serves everyone under no provider node needs the platform (``NOT_PLATFORM_SHARED``);
* a tokenless manifest serves nobody but, during the 0.55.0 rollout, its own agent;
* every refusal is named in ``RegistrationResponse.refused_tools``, success or not;
* a verified pod's refused tool withdraws that pod's prior copy, and nobody else's.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from threetears.agent.tools.server import RegistrationManifest, RegistrationResponse, ToolManifestEntry
from threetears.nats import IncomingMessage, Subjects, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import CopyStatus, ToolCatalog
from threetears.registry.registration import RegistrationHandler

__all__: list[str] = []

_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_B = UUID("01948a00-aaaa-7000-8000-00000000000b")
_A_POD = Subjects.agent_inprocess_pod_id(_AGENT_A, "inst-1")
_B_POD = Subjects.agent_inprocess_pod_id(_AGENT_B, "inst-1")
_SHARED_POD = "builtin-tool-server"
_STRAY_POD = "stray-tool-pod"
_PENTEST_POD = "pentest-tool-server"
_CALC = "threetears.calculator@1.0.0"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind a deterministic subject namespace for the probe subjects."""
    set_default_namespace("test")


class _Directory:
    """a ``ToolPodAuthenticator`` double: named pod tokens, named agent tokens, one graph.

    ``pods`` maps a token to the ToolPodAuth it verifies as; ``agents`` maps a token to the agent
    id it verifies as. Anything else fails verification.
    """

    def __init__(
        self,
        *,
        pods: dict[str, ToolPodAuth] | None = None,
        agents: dict[str, UUID] | None = None,
        nodes: tuple[str, ...] = ("tools.pentest",),
    ) -> None:
        self._pods = pods or {}
        self._agents = agents or {}
        self._nodes = nodes
        self.seen_pod_tokens: list[str] = []
        self.seen_agent_tokens: list[str] = []

    def set_pod(self, token: str, auth: ToolPodAuth) -> None:
        self._pods[token] = auth

    def set_nodes(self, nodes: tuple[str, ...]) -> None:
        self._nodes = nodes

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        self.seen_pod_tokens.append(token)
        return self._pods.get(token)

    async def verify_agent(self, token: str) -> UUID | None:
        self.seen_agent_tokens.append(token)
        return self._agents.get(token)

    async def provider_nodes(self) -> tuple[str, ...]:
        return self._nodes


def _default_directory() -> _Directory:
    """the platform's cast: a shared pod, a pentest owner, a stray, and two agents.

    :return: the directory
    :rtype: _Directory
    """
    return _Directory(
        pods={
            "shared-token": ToolPodAuth(
                pod_entity_id=_SHARED_POD, name="builtin-tool-server", owned_namespaces=[], platform_shared=True
            ),
            "pentest-token": ToolPodAuth(
                pod_entity_id=_PENTEST_POD, name="pentest-tool-server", owned_namespaces=["pentest"]
            ),
            "stray-token": ToolPodAuth(pod_entity_id=_STRAY_POD, name="stray", owned_namespaces=[]),
        },
        agents={"agent-a-token": _AGENT_A, "agent-b-token": _AGENT_B},
    )


def _nc() -> AsyncMock:
    """a NATS double answering every reachability probe.

    :return: the double
    :rtype: AsyncMock
    """

    async def _probe(*, subject: Any, message: Any, response_type: Any, timeout: Any) -> Any:
        del message, timeout
        return response_type(pod_id=subject.path.rsplit(".", 1)[-1], ready=True)

    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=_probe)
    return nc


def _manifest(pod_id: str, *tools: tuple[str, str], token: str | None = None, gated: bool = False) -> bytes:
    """a manifest offering ``(name, description)`` tools at version 1.0.0.

    :param pod_id: the registering pod's id
    :ptype pod_id: str
    :param tools: the tools, as name and description
    :ptype tools: tuple[str, str]
    :param token: the credential, or ``None`` for a tokenless manifest
    :ptype token: str | None
    :param gated: whether each tool requires confirmation
    :ptype gated: bool
    :return: the serialized manifest
    :rtype: bytes
    """
    offered = tools or (("threetears.calculator", "calculator"),)
    manifest = RegistrationManifest(
        pod_id=pod_id,
        tools=[
            ToolManifestEntry(
                name=name,
                version="1.0.0",
                description=description,
                input_schema={"type": "object", "properties": {}},
                requires_confirmation=gated,
            )
            for name, description in offered
        ],
        bootstrap_token=token,
    )
    return manifest.model_dump_json().encode("utf-8")


async def _started(directory: _Directory | None = None) -> tuple[RegistrationHandler, ToolCatalog, AsyncMock]:
    """a started handler over a fresh catalog.

    :param directory: the authenticator, or ``None`` for open mode
    :ptype directory: _Directory | None
    :return: the handler, its catalog and its NATS double
    :rtype: tuple[RegistrationHandler, ToolCatalog, AsyncMock]
    """
    catalog = ToolCatalog()
    handler = RegistrationHandler(catalog, namespace="test", authenticator=directory)
    nc = _nc()
    await handler.start(nc)
    return handler, catalog, nc


async def _register(handler: RegistrationHandler, nc: AsyncMock, data: bytes) -> RegistrationResponse:
    """drive one registration and return its reply.

    :param handler: the handler under test
    :ptype handler: RegistrationHandler
    :param nc: its NATS double
    :ptype nc: AsyncMock
    :param data: the serialized manifest
    :ptype data: bytes
    :return: the reply
    :rtype: RegistrationResponse
    """
    await handler.handle_registration(
        IncomingMessage(data=data, reply_subject="reply.to", subject="test.tools.register")
    )
    reply = nc.publish_reply.await_args.kwargs["message"]
    assert isinstance(reply, RegistrationResponse)
    return reply


def _descriptions(catalog: ToolCatalog, full_name: str, pod_id: str) -> list[str]:
    """every description one pod's copy holds.

    :param catalog: the catalog
    :ptype catalog: ToolCatalog
    :param full_name: the tool
    :ptype full_name: str
    :param pod_id: the pod
    :ptype pod_id: str
    :return: the descriptions, empty when the pod holds no copy
    :rtype: list[str]
    """
    entry = catalog.get(full_name)
    copy = entry.get_endpoint(pod_id) if entry is not None else None
    return [] if copy is None else [d.definition.description for d in copy.definitions.values()]


class TestAVerifiedTokenMustNameItsOwnPod:
    """the token path compares the manifest's pod id with the verified one."""

    @pytest.mark.asyncio
    async def test_a_token_naming_another_pods_id_refuses_the_whole_manifest(self) -> None:
        """the stray's valid token under the shared pod's id: nothing written, incumbent intact."""
        handler, catalog, nc = await _started(_default_directory())
        await _register(handler, nc, _manifest(_SHARED_POD, ("threetears.calculator", "REAL"), token="shared-token"))

        reply = await _register(
            handler, nc, _manifest(_SHARED_POD, ("threetears.calculator", "STRAY"), token="stray-token")
        )

        assert reply.success is False
        assert reply.error_code == "POD_ID_MISMATCH"
        assert [(r.name, r.code) for r in reply.refused_tools] == [("threetears.calculator", "POD_ID_MISMATCH")]
        assert _descriptions(catalog, _CALC, _SHARED_POD) == ["REAL"]

    @pytest.mark.asyncio
    async def test_the_same_token_under_its_own_id_is_admitted(self) -> None:
        """the paired twin: the pentest owner under its own id registers its own node."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(
            handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "sqlmap"), token="pentest-token")
        )
        assert reply.success is True
        assert reply.refused_tools == []
        assert _descriptions(catalog, "pentest.sqlmap@1.0.0", _PENTEST_POD) == ["sqlmap"]


class TestOnlyThePlatformServesEveryoneOutsideAProvider:
    """a token-bearing pod owning no node that is not the shared pod is now refused."""

    @pytest.mark.asyncio
    async def test_a_stray_token_pod_is_refused_and_the_incumbent_untouched(self) -> None:
        """NOT_PLATFORM_SHARED, and the shared pod's copy and gate are exactly as they were."""
        handler, catalog, nc = await _started(_default_directory())
        await _register(
            handler, nc, _manifest(_SHARED_POD, ("threetears.calculator", "REAL"), token="shared-token", gated=True)
        )

        reply = await _register(
            handler, nc, _manifest(_STRAY_POD, ("threetears.calculator", "STRAY"), token="stray-token")
        )

        assert reply.success is False
        assert [(r.name, r.version, r.code) for r in reply.refused_tools] == [
            ("threetears.calculator", "1.0.0", "NOT_PLATFORM_SHARED")
        ]
        assert _descriptions(catalog, _CALC, _SHARED_POD) == ["REAL"]
        assert _descriptions(catalog, _CALC, _STRAY_POD) == []
        entry = catalog.get(_CALC)
        assert entry is not None
        assert entry.select_copies(None).requires_confirmation is True

    @pytest.mark.asyncio
    async def test_the_shared_pod_is_admitted_for_the_same_name(self) -> None:
        """the twin: the verified platform pod may serve the name to everyone."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(handler, nc, _manifest(_SHARED_POD, token="shared-token"))
        assert reply.success is True
        assert reply.registered_tools == [_CALC]

    @pytest.mark.asyncio
    async def test_refusals_are_named_even_when_the_registration_succeeds(self) -> None:
        """the pentest owner offering one of its own and one unowned name: success, and one refusal."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(
            handler,
            nc,
            _manifest(
                _PENTEST_POD,
                ("pentest.sqlmap", "sqlmap"),
                ("threetears.calculator", "not mine"),
                token="pentest-token",
            ),
        )
        assert reply.success is True
        assert reply.registered_tools == ["pentest.sqlmap@1.0.0"]
        assert [(r.name, r.code) for r in reply.refused_tools] == [("threetears.calculator", "NOT_PLATFORM_SHARED")]
        assert reply.refused_tools[0].reason
        assert catalog.get(_CALC) is None


class TestATokenlessManifestServesNobodyElse:
    """no credential, no copy that anybody but its own agent may be served."""

    @pytest.mark.asyncio
    async def test_a_tokenless_manifest_claiming_the_shared_pods_id_is_refused(self) -> None:
        """UNVERIFIED_PUBLISHER; the shared pod stays available with its own definition."""
        handler, catalog, nc = await _started(_default_directory())
        await _register(handler, nc, _manifest(_SHARED_POD, ("threetears.calculator", "REAL"), token="shared-token"))

        reply = await _register(handler, nc, _manifest(_SHARED_POD, ("threetears.calculator", "IMPOSTOR")))

        assert reply.success is False
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert [(r.name, r.code) for r in reply.refused_tools] == [("threetears.calculator", "UNVERIFIED_PUBLISHER")]
        assert _descriptions(catalog, _CALC, _SHARED_POD) == ["REAL"]
        entry = catalog.get(_CALC)
        assert entry is not None
        assert entry.copy_status(_SHARED_POD) is CopyStatus.AVAILABLE

    @pytest.mark.asyncio
    async def test_an_invalid_token_is_refused_as_unverified(self) -> None:
        """a token nothing verifies is a refusal, never a downgrade to tokenless."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(handler, nc, _manifest(_SHARED_POD, token="forged"))
        assert reply.success is False
        assert reply.error == "invalid bootstrap token"
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert catalog.get(_CALC) is None

    @pytest.mark.asyncio
    async def test_an_unsigned_agent_manifest_is_refused(self) -> None:
        """0.57.0 ended the rollout concession: an agent registers signed, or not at all."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(handler, nc, _manifest(_A_POD, ("threetears.calculator", "A ONLY")))

        assert reply.success is False
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert catalog.get(_CALC) is None


class TestAnAgentSignsWithItsOwnIdentity:
    """an agent's in-process manifest is verified against the agent the pod id names."""

    @pytest.mark.asyncio
    async def test_a_signed_copy_is_visible_only_to_its_agent(self) -> None:
        """admitted as a verified copy, served to agent A and nobody else."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(
            handler, nc, _manifest(_A_POD, ("threetears.calculator", "A ONLY"), token="agent-a-token")
        )

        assert reply.success is True
        assert reply.owned_namespaces == [f"agents.{_AGENT_A}"]
        entry = catalog.get(_CALC)
        assert entry is not None
        copy = entry.get_endpoint(_A_POD)
        assert copy is not None and copy.verified_publisher is True
        assert entry.available_to(_AGENT_A) is True
        assert entry.available_to(_AGENT_B) is False

    @pytest.mark.asyncio
    async def test_an_agent_claiming_another_agents_pod_id_is_refused(self) -> None:
        """agent A's valid token under agent B's pod id: POD_ID_MISMATCH, B's copy untouched."""
        handler, catalog, nc = await _started(_default_directory())
        await _register(handler, nc, _manifest(_B_POD, ("threetears.calculator", "B REAL"), token="agent-b-token"))

        reply = await _register(
            handler, nc, _manifest(_B_POD, ("threetears.calculator", "FROM A"), token="agent-a-token")
        )

        assert reply.success is False
        assert reply.error_code == "POD_ID_MISMATCH"
        assert _descriptions(catalog, _CALC, _B_POD) == ["B REAL"]

    @pytest.mark.asyncio
    async def test_a_signature_that_fails_is_refused_not_downgraded(self) -> None:
        """a bad token on an agent pod is refused; it does not fall back to the unsigned path."""
        handler, catalog, nc = await _started(_default_directory())
        reply = await _register(handler, nc, _manifest(_A_POD, token="not-a-real-token"))
        assert reply.success is False
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert catalog.get(_CALC) is None

    @pytest.mark.asyncio
    async def test_an_agent_token_is_never_asked_of_the_tool_pod_verifier(self) -> None:
        """the pod id decides which verifier is asked, so an agent token cannot pass as a pod's."""
        directory = _default_directory()
        handler, _catalog, nc = await _started(directory)
        await _register(handler, nc, _manifest(_A_POD, token="agent-a-token"))
        assert directory.seen_pod_tokens == []
        assert directory.seen_agent_tokens == ["agent-a-token"]

    @pytest.mark.asyncio
    async def test_an_unsigned_manifest_cannot_follow_a_signed_one_under_the_same_pod_id(self) -> None:
        """once a pod id has registered signed, an unsigned manifest under it is an impersonation."""
        handler, catalog, nc = await _started(_default_directory())
        await _register(handler, nc, _manifest(_B_POD, ("threetears.calculator", "B REAL"), token="agent-b-token"))

        reply = await _register(handler, nc, _manifest(_B_POD, ("threetears.calculator", "UNSIGNED")))

        assert reply.success is False
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert _descriptions(catalog, _CALC, _B_POD) == ["B REAL"]


class TestARefusedToolWithdrawsOnlyItsPublishersCopy:
    """re-registration goes through admission every time, and a refusal takes the prior copy."""

    @pytest.mark.asyncio
    async def test_a_verified_pod_that_loses_ownership_loses_its_copy(self) -> None:
        """admitted while it owned the node; refused and withdrawn once it does not."""
        directory = _default_directory()
        handler, catalog, nc = await _started(directory)
        await _register(handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "sqlmap"), token="pentest-token"))
        assert catalog.get("pentest.sqlmap@1.0.0") is not None

        directory.set_pod(
            "pentest-token", ToolPodAuth(pod_entity_id=_PENTEST_POD, name="pentest-tool-server", owned_namespaces=[])
        )
        reply = await _register(
            handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "sqlmap"), token="pentest-token")
        )

        assert [(r.name, r.code) for r in reply.refused_tools] == [("pentest.sqlmap", "OWNED_ELSEWHERE")]
        assert catalog.get("pentest.sqlmap@1.0.0") is None

    @pytest.mark.asyncio
    async def test_an_unverified_refusal_withdraws_nothing(self) -> None:
        """an unsigned manifest cannot prove it IS the pod, so it cannot take that pod's copy.

        Agent A holds a signed copy. An unsigned manifest under A's pod id -- which might be an
        impersonator -- is refused, and A's copy stays exactly as it was.
        """
        handler, catalog, nc = await _started(_default_directory())
        signed = await _register(handler, nc, _manifest(_A_POD, ("threetears.calculator", "A"), token="agent-a-token"))
        assert signed.success is True

        reply = await _register(handler, nc, _manifest(_A_POD, ("threetears.calculator", "IMPOSTOR")))

        assert reply.success is False
        assert reply.error_code == "UNVERIFIED_PUBLISHER"
        assert _descriptions(catalog, _CALC, _A_POD) == ["A"]


class TestOpenMode:
    """no authenticator: nothing enforced, said once at startup."""

    @pytest.mark.asyncio
    async def test_open_mode_admits_a_tokenless_tool_pod_and_says_so_at_startup(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """admitted, serving everyone, and the one startup line names the mode."""
        with caplog.at_level(logging.WARNING, logger="threetears.registry.registration"):
            handler, catalog, nc = await _started(None)
            reply = await _register(handler, nc, _manifest(_STRAY_POD))
            await _register(handler, nc, _manifest(_STRAY_POD))
        assert reply.success is True
        entry = catalog.get(_CALC)
        assert entry is not None and entry.available_to(None) is True
        open_mode = [r for r in caplog.records if "open mode" in r.getMessage()]
        assert len(open_mode) == 1


class TestAnAuthenticatorMustImplementTheWholeProtocol:
    """a host whose authenticator predates agent verification is refused at construction."""

    def test_an_authenticator_without_verify_agent_is_refused(self) -> None:
        """caught where it is wired, not on the first agent registration."""

        class _Old:
            async def verify_pod(self, token: str) -> ToolPodAuth | None:
                del token
                return None

            async def provider_nodes(self) -> tuple[str, ...]:
                return ()

        with pytest.raises(TypeError, match="verify_agent"):
            RegistrationHandler(ToolCatalog(), namespace="test", authenticator=_Old())  # type: ignore[arg-type]


class TestTheLogLinesOperatorsMatchOn:
    """scripts outside this repo match these phrases and keys; they are a contract."""

    @pytest.mark.asyncio
    async def test_completed_and_authorized_keep_their_phrases_and_keys(self, caplog: pytest.LogCaptureFixture) -> None:
        """``registration completed`` and ``tool pod registration authorized`` with the old keys."""
        handler, _catalog, nc = await _started(_default_directory())
        with caplog.at_level(logging.INFO, logger="threetears.registry.registration"):
            await _register(handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "s"), token="pentest-token"))
        messages = {r.getMessage(): r for r in caplog.records}
        assert "registration completed" in messages
        authorized = messages["tool pod registration authorized"]
        data = getattr(authorized, "extra_data", None)
        assert data is not None
        assert {"tools_accepted", "tools_rejected"} <= set(data)

    @pytest.mark.asyncio
    async def test_a_rejection_still_starts_with_registration_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        """the refusal line keeps its prefix; its code rides under a new key."""
        handler, _catalog, nc = await _started(_default_directory())
        with caplog.at_level(logging.WARNING, logger="threetears.registry.registration"):
            await _register(handler, nc, _manifest(_SHARED_POD, token="stray-token"))
        rejected = [r for r in caplog.records if r.getMessage().startswith("registration rejected")]
        assert rejected
        data = getattr(rejected[-1], "extra_data", None)
        assert data is not None and data.get("error_code") == "POD_ID_MISMATCH"


class TestATransientRefusalWithdrawsNothing:
    """only an admission verdict takes a copy away; a graph that cannot be read does not."""

    @pytest.mark.asyncio
    async def test_an_unreadable_graph_leaves_a_verified_pods_copy_in_place(self) -> None:
        """the pod is refused for this manifest and retries; its copy keeps serving meanwhile."""

        class _Flaky(_Directory):
            def __init__(self) -> None:
                super().__init__(
                    pods={
                        "pentest-token": ToolPodAuth(
                            pod_entity_id=_PENTEST_POD, name="pentest", owned_namespaces=["pentest"]
                        )
                    }
                )
                self.broken = False

            async def provider_nodes(self) -> tuple[str, ...]:
                if self.broken:
                    raise RuntimeError("the broker is down")
                return await super().provider_nodes()

        directory = _Flaky()
        handler, catalog, nc = await _started(directory)
        await _register(handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "sqlmap"), token="pentest-token"))
        directory.broken = True

        reply = await _register(
            handler, nc, _manifest(_PENTEST_POD, ("pentest.sqlmap", "sqlmap"), token="pentest-token")
        )

        assert reply.error_code == "OWNERSHIP_GRAPH_UNAVAILABLE"
        assert _descriptions(catalog, "pentest.sqlmap@1.0.0", _PENTEST_POD) == ["sqlmap"]
