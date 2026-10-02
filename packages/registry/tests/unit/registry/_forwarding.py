"""drive one call through the proxy's front door and capture the envelope the pod receives.

The registry->pod envelope is built inside :class:`~threetears.registry.proxy.CallProxy`, and the
properties the forwarding tests pin -- an unset optional never reaches the wire, a caller's deadline
is clamped to the proxy's own wait, the durable path names its delivery subject -- are properties of
the BYTES a pod is handed. So these helpers route a real authenticated call through
``CallProxy.handle_call`` against a one-tool catalog and decode what the transport was asked to send,
rather than calling the envelope builder directly.

not a ``test_*`` module, so pytest does not collect it; the registry test package makes it
importable as a sibling.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.server import CallAccepted
from threetears.nats import IncomingMessage, Subject, set_default_namespace
from threetears.registry.catalog import ToolCatalog
from threetears.registry.proxy import ProxyCallRequest, ProxyCallResponse

from ._copies import definition, endpoint, entry
from ._dispatch_auth import make_proxy

__all__ = [
    "FORWARDING_NAMESPACE",
    "PROXY_TIMEOUT_SECONDS",
    "forwarded_envelope",
]

#: the subject namespace the forwarding helpers bind.
FORWARDING_NAMESPACE = "test"

#: the proxy's own default wait, applied when the routed tool declares no timeout.
PROXY_TIMEOUT_SECONDS = 30.0


class _ResultWaiter:
    """a durable-result waiter that hands back one successful pod answer."""

    def __init__(self, subject: Subject) -> None:
        """
        binds the waiter to the subject the proxy opened it on.

        :param subject: the delivery subject
        :ptype subject: Subject
        """
        self.subject = subject

    async def wait(self, *, timeout: timedelta) -> bytes:
        """
        answers with a successful pod response.

        :param timeout: the wait budget, unused
        :ptype timeout: timedelta
        :return: serialized pod answer
        :rtype: bytes
        """
        del timeout
        return ProxyCallResponse(success=True, content="ok", context=CallContext()).model_dump_json().encode("utf-8")

    async def close(self) -> None:
        """
        releases nothing; the waiter holds no resource.
        """


class _RecordingTransport:
    """the proxy's NATS client, recording every envelope it is asked to forward to a pod."""

    def __init__(self) -> None:
        """
        starts with no forwarded envelope.
        """
        self.payloads: list[bytes] = []

    async def subscribe(self, **kwargs: Any) -> Any:
        """
        accepts the proxy's subscriptions without delivering anything.

        :param kwargs: subscription arguments, unused
        :ptype kwargs: Any
        :return: an opaque subscription handle
        :rtype: Any
        """
        del kwargs
        return object()

    async def unsubscribe(self, sub: Any) -> None:
        """
        accepts an unsubscribe.

        :param sub: the subscription handle, unused
        :ptype sub: Any
        """
        del sub

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        """
        records the forwarded envelope and answers as a healthy pod would.

        a durable dispatch (the envelope names a ``result_subject``) is acknowledged with an accept;
        a synchronous one is answered directly.

        :param subject: the pod's internal subject, unused
        :ptype subject: Subject
        :param payload: the envelope the pod receives
        :ptype payload: bytes
        :param timeout: the wait, unused
        :ptype timeout: timedelta
        :return: the pod's reply
        :rtype: bytes
        """
        del subject, timeout
        self.payloads.append(payload)
        result: bytes
        if "result_subject" in json.loads(payload):
            result = CallAccepted(accepted=True, pod_id="pod-1", result_subject="x").model_dump_json().encode("utf-8")
        else:
            result = (
                ProxyCallResponse(success=True, content="ok", context=CallContext()).model_dump_json().encode("utf-8")
            )
        return result

    async def jetstream_result_waiter(self, *, subject: Subject, stream: str, wait_budget: timedelta) -> _ResultWaiter:
        """
        opens a waiter that delivers one answer.

        :param subject: the delivery subject
        :ptype subject: Subject
        :param stream: the result stream, unused
        :ptype stream: str
        :param wait_budget: the wait budget, unused
        :ptype wait_budget: timedelta
        :return: the waiter
        :rtype: _ResultWaiter
        """
        del stream, wait_budget
        return _ResultWaiter(subject)

    async def jetstream_publish(self, *, subject: Subject, payload: bytes) -> None:
        """
        accepts a durable publish.

        :param subject: unused
        :ptype subject: Subject
        :param payload: unused
        :ptype payload: bytes
        """
        del subject, payload

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        """
        accepts the proxy's answer to its caller.

        :param reply_subject: unused
        :ptype reply_subject: str
        :param message: unused
        :ptype message: Any
        """
        del reply_subject, message


async def forwarded_envelope(request: ProxyCallRequest, *, tool_timeout_seconds: float | None) -> dict[str, Any]:
    """
    routes one call through the proxy's front door and returns the envelope the pod received.

    :param request: an authenticated call for ``threetears.calculator@1.0.0``
    :ptype request: ProxyCallRequest
    :param tool_timeout_seconds: the timeout the tool declares, or ``None`` for the proxy default
        (:data:`PROXY_TIMEOUT_SECONDS`);
        a value past the synchronous reply budget sends the call down the durable path
    :ptype tool_timeout_seconds: float | None
    :return: the decoded envelope
    :rtype: dict[str, Any]
    :raises AssertionError: when the proxy forwarded anything other than exactly one envelope
    """
    set_default_namespace(FORWARDING_NAMESPACE)
    catalog = ToolCatalog()
    await catalog.register(
        entry(
            request.tool_name,
            request.tool_version,
            endpoint("pod-1", tool_definition=definition(timeout_seconds=tool_timeout_seconds)),
        )
    )
    proxy = make_proxy(catalog, namespace=FORWARDING_NAMESPACE, timeout=PROXY_TIMEOUT_SECONDS)
    transport = _RecordingTransport()
    await proxy.start(transport)
    await proxy.handle_call(
        IncomingMessage(
            data=request.model_dump_json().encode("utf-8"),
            reply_subject="_INBOX.forwarding",
            subject=f"{FORWARDING_NAMESPACE}.tools.call",
        )
    )
    await proxy.stop()
    assert len(transport.payloads) == 1, f"expected one forwarded envelope, got {len(transport.payloads)}"
    decoded: dict[str, Any] = json.loads(transport.payloads[0])
    return decoded
