"""unit -- a registration the host cannot judge is ANSWERED, with a temporary refusal, never dropped.

Found on a live bring-up: the host's authenticator raised ``DataLayerUnavailableError`` (its store
refused a read), the exception escaped ``handle_registration``, and the manifest's reply subject was
never answered. Every agent's in-process server saw only a ten-second ``RequestTimeoutError`` --
"could not read the registration reply" -- retried forever, and never became ready, while the one
line naming the cause sat in the registry's log as a generic callback failure.

Now a failure of the host's store is a refusal the pod can read: a code that is not in
``FINAL_REFUSAL_CODES`` (so the pod waits it out on its heartbeat), logged once at ERROR on the
registry with the cause.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from threetears.agent.tools.server import (
    FINAL_REFUSAL_CODES,
    RegistrationManifest,
    RegistrationResponse,
    ToolManifestEntry,
    refusal_is_final,
)
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.nats import IncomingMessage, KvError, Subjects, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import CatalogEntry, ToolCatalog
from threetears.registry.ownership import RefusalCode
from threetears.registry.registration import RegistrationHandler

__all__: list[str] = []

_AGENT = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_POD = Subjects.agent_inprocess_pod_id(_AGENT, "inst-1")
_TOOL_POD = "pentest-tool-server"
_CAUSE = "L3 query failed: NAMESPACE_ACCESS_DENIED: carve-out read names table(s) this principal may not read: agents"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind a deterministic subject namespace for the probe subjects."""
    set_default_namespace("test")


class _UnreadableStore:
    """a ``ToolPodAuthenticator`` whose principal store cannot be read.

    ``verify_agent`` and ``verify_pod`` raise what a proxy-backed collection raises when the broker
    refuses the read; ``provider_nodes`` answers, so the failure under test is verification alone.
    """

    async def verify_pod(self, token: str) -> ToolPodAuth | None:
        """raise the store failure.

        :param token: the presented token
        :ptype token: str
        :return: never returns
        :rtype: ToolPodAuth | None
        :raises DataLayerUnavailableError: always
        """
        del token
        raise DataLayerUnavailableError(_CAUSE)

    async def verify_agent(self, token: str) -> UUID | None:
        """raise the store failure.

        :param token: the presented token
        :ptype token: str
        :return: never returns
        :rtype: UUID | None
        :raises DataLayerUnavailableError: always
        """
        del token
        raise DataLayerUnavailableError(_CAUSE)

    async def provider_nodes(self) -> tuple[str, ...]:
        """the ownership graph, readable.

        :return: one provider node
        :rtype: tuple[str, ...]
        """
        return ("tools.pentest",)


class _CancelledStore(_UnreadableStore):
    """an authenticator whose read is cancelled -- a shutdown, not a store failure."""

    async def verify_agent(self, token: str) -> UUID | None:
        """raise cancellation.

        :param token: the presented token
        :ptype token: str
        :return: never returns
        :rtype: UUID | None
        :raises asyncio.CancelledError: always
        """
        del token
        raise asyncio.CancelledError


class _AdmitsTheAgent(_UnreadableStore):
    """an authenticator that verifies the agent, so the failure under test is the catalog's."""

    async def verify_agent(self, token: str) -> UUID | None:
        """verify every token as the agent.

        :param token: the presented token
        :ptype token: str
        :return: the agent
        :rtype: UUID | None
        """
        del token
        return _AGENT


class _UnwritableCatalog(ToolCatalog):
    """a real catalog whose durable write fails, as its KV bucket does when unreachable."""

    async def register(self, entry: CatalogEntry) -> None:
        """raise the KV failure a persist raises.

        :param entry: the entry offered
        :ptype entry: CatalogEntry
        :return: never returns
        :rtype: None
        :raises KvError: always
        """
        del entry
        raise KvError("catalog bucket unreachable")


def _manifest(pod_id: str, token: str) -> bytes:
    """a one-tool manifest.

    :param pod_id: the registering pod
    :ptype pod_id: str
    :param token: its credential
    :ptype token: str
    :return: the serialized manifest
    :rtype: bytes
    """
    return (
        RegistrationManifest(
            pod_id=pod_id,
            tools=[
                ToolManifestEntry(
                    name="survey.lookup",
                    version="1.0.0",
                    description="lookup",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            bootstrap_token=token,
        )
        .model_dump_json()
        .encode("utf-8")
    )


async def _register(authenticator: Any, data: bytes, catalog: ToolCatalog | None = None) -> RegistrationResponse:
    """drive one registration through a started handler and return the reply it published.

    :param authenticator: the host's authenticator
    :ptype authenticator: Any
    :param data: the serialized manifest
    :ptype data: bytes
    :param catalog: the catalog, a fresh one when ``None``
    :ptype catalog: ToolCatalog | None
    :return: the reply
    :rtype: RegistrationResponse
    """
    handler = RegistrationHandler(catalog or ToolCatalog(), namespace="test", authenticator=authenticator)
    nc = AsyncMock()
    await handler.start(nc)
    await handler.handle_registration(
        IncomingMessage(data=data, reply_subject="reply.to", subject="test.tools.register")
    )
    nc.publish_reply.assert_awaited_once()
    reply = nc.publish_reply.await_args.kwargs["message"]
    assert isinstance(reply, RegistrationResponse)
    return reply


class TestTheCodesAreTemporary:
    """a pod waits out what these codes name; ending its readiness over one would be wrong."""

    @pytest.mark.parametrize("code", [RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE, RefusalCode.CATALOG_UNAVAILABLE])
    def test_a_store_failure_is_not_final(self, code: RefusalCode) -> None:
        """the next heartbeat is a real retry.

        :param code: the refusal code
        :ptype code: RefusalCode
        :return: nothing
        :rtype: None
        """
        assert code.value not in FINAL_REFUSAL_CODES
        assert refusal_is_final(code.value) is False


class TestAnUnreadablePrincipalStoreIsAnswered:
    """the live defect: the authenticator raised, and nothing answered."""

    @pytest.mark.asyncio
    async def test_an_agent_publisher_gets_a_temporary_refusal(self, caplog: pytest.LogCaptureFixture) -> None:
        """one reply naming the code on every tool, one ERROR on the registry naming the cause.

        :param caplog: the log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        with caplog.at_level(logging.WARNING, logger="threetears.registry.registration"):
            reply = await _register(_UnreadableStore(), _manifest(_AGENT_POD, "agent-token"))

        assert reply.success is False
        assert reply.error_code == RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE.value
        assert [(tool.name, tool.code) for tool in reply.refused_tools] == [
            ("survey.lookup", RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE.value)
        ]
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(errors) == 1
        assert _CAUSE in str(errors[0].__dict__["extra_data"])

    @pytest.mark.asyncio
    async def test_a_tool_pod_publisher_gets_the_same_refusal(self) -> None:
        """the tool-pod verifier reads through the same store, and fails the same way.

        :return: nothing
        :rtype: None
        """
        reply = await _register(_UnreadableStore(), _manifest(_TOOL_POD, "pod-token"))

        assert reply.success is False
        assert reply.error_code == RefusalCode.PUBLISHER_VERIFICATION_UNAVAILABLE.value

    @pytest.mark.asyncio
    async def test_cancellation_is_not_turned_into_a_refusal(self) -> None:
        """a shutdown cancelling the read propagates; it is not a store failure to report.

        :return: nothing
        :rtype: None
        """
        handler = RegistrationHandler(ToolCatalog(), namespace="test", authenticator=_CancelledStore())
        nc = AsyncMock()
        await handler.start(nc)
        with pytest.raises(asyncio.CancelledError):
            await handler.handle_registration(
                IncomingMessage(data=_manifest(_AGENT_POD, "t"), reply_subject="r", subject="test.tools.register")
            )
        nc.publish_reply.assert_not_awaited()


class TestAnUnwritableCatalogIsAnswered:
    """the same silence, one step later: the catalog write raised after admission."""

    @pytest.mark.asyncio
    async def test_the_admitted_tools_are_refused_temporarily(self, caplog: pytest.LogCaptureFixture) -> None:
        """the verdict was admit; the write failed; the pod is told, and retries on its heartbeat.

        :param caplog: the log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        with caplog.at_level(logging.ERROR, logger="threetears.registry.registration"):
            reply = await _register(_AdmitsTheAgent(), _manifest(_AGENT_POD, "agent-token"), _UnwritableCatalog())

        assert reply.success is False
        assert reply.error_code == RefusalCode.CATALOG_UNAVAILABLE.value
        assert [(tool.name, tool.code) for tool in reply.refused_tools] == [
            ("survey.lookup", RefusalCode.CATALOG_UNAVAILABLE.value)
        ]
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "catalog bucket unreachable" in str(errors[0].__dict__["extra_data"])
