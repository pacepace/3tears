"""unit tests for :mod:`threetears.agent.tools.builtin.http_api_tool`.

covers gu-task-04 success criteria: path-template fill + query/body split,
missing-path-param guard, secret-ref resolution into the outbound header (and
never into metadata / logs), passthrough 2xx/3xx/4xx results, retry-exhausted
``UpstreamHttpError`` and tripped ``CircuitOpenError`` conversion to a failed
``ToolResult``, schema derivation, and the "no raw httpx in the module" guard.

the injected transport is a subclass of the production
:class:`threetears.core.http_client.TracedHttpClient` (subclass = parity
declaration for ``test_fake_protocol_parity``); it records the built request and
returns a canned :class:`httpx.Response` or raises a canned exception.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from threetears.agent.tools.builtin.http_api_tool import (
    HttpApiTool,
    HttpOperationDescriptor,
)
from threetears.core.http_client import TracedHttpClient, UpstreamHttpError
from threetears.core.security.secret_refs import SecretResolutionError


class _FakeTracedHttpClient(TracedHttpClient):
    """records the outbound request and returns a canned response / raises.

    subclasses the production transport so the fake-parity walker treats the
    subclass declaration as the parity contract. it does NOT call
    ``super().__init__`` -- no real httpx client is opened; only
    :meth:`request` is exercised.
    """

    def __init__(
        self,
        *,
        response: httpx.Response | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        """capture the canned response / exception; open no real client.

        :param response: canned response returned from :meth:`request`
        :ptype response: httpx.Response | None
        :param raise_exc: exception raised from :meth:`request` instead
        :ptype raise_exc: BaseException | None
        :return: nothing
        :rtype: None
        """
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._raise_exc = raise_exc

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: Any = None,
        params: Any = None,
        content: Any = None,
        json: Any = None,
    ) -> httpx.Response:
        """record the request and return the canned response / raise.

        :param method: HTTP verb
        :ptype method: str
        :param path: filled request path
        :ptype path: str
        :param headers: outbound headers
        :ptype headers: Any
        :param params: query params
        :ptype params: Any
        :param content: raw body
        :ptype content: Any
        :param json: JSON body
        :ptype json: Any
        :return: canned response
        :rtype: httpx.Response
        """
        self.calls.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers) if headers else None,
                "params": dict(params) if params else None,
                "content": content,
                "json": json,
            }
        )
        if self._raise_exc is not None:
            raise self._raise_exc
        assert self._response is not None
        return self._response


class CircuitOpenError(RuntimeError):
    """stand-in for ``threetears.models.circuit_breaker.CircuitOpenError``.

    the tool cannot import the real class (agent-tools does not depend on
    models); it duck-types on ``type(exc).__name__``. the class name here
    matches (``CircuitOpenError``) so the tool's classifier fires.
    """


def _get_descriptor(**overrides: Any) -> HttpOperationDescriptor:
    """build a GET ``/users/{id}`` descriptor for tests.

    :param overrides: field overrides
    :ptype overrides: Any
    :return: operation descriptor
    :rtype: HttpOperationDescriptor
    """
    base: dict[str, Any] = {
        "method": "GET",
        "path_template": "/users/{id}",
        "param_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "verbose": {"type": "boolean"},
            },
            "required": ["id"],
        },
        "credentials_ref": None,
        "name": "example.get_user",
        "version": "1.0",
        "description": "get a user by id",
    }
    base.update(overrides)
    return HttpOperationDescriptor(**base)


def _response(status: int, text: str, headers: dict[str, str] | None = None) -> httpx.Response:
    """build a canned httpx.Response.

    :param status: HTTP status code
    :ptype status: int
    :param text: response body text
    :ptype text: str
    :param headers: response headers
    :ptype headers: dict[str, str] | None
    :return: response
    :rtype: httpx.Response
    """
    return httpx.Response(status_code=status, text=text, headers=headers or {})


class TestPathAndArguments:
    """path-template fill + query/body split."""

    async def test_get_fills_path_and_maps_remainder_to_query(self) -> None:
        """GET ``/users/{id}`` -> path filled, remaining args -> query."""
        client = _FakeTracedHttpClient(response=_response(200, "ok"))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        await tool.execute(id="42", verbose=True)

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["method"] == "GET"
        assert call["path"] == "/users/42"
        assert call["params"] == {"verbose": True}
        assert call["json"] is None

    async def test_post_maps_remainder_to_json_body(self) -> None:
        """POST -> non-path args become the JSON body, not query params."""
        descriptor = _get_descriptor(
            method="POST",
            path_template="/users/{id}/notes",
            name="example.add_note",
        )
        client = _FakeTracedHttpClient(response=_response(201, "created"))
        tool = HttpApiTool(descriptor=descriptor, http_client=client)

        await tool.execute(id="42", body="hello")

        call = client.calls[0]
        assert call["method"] == "POST"
        assert call["path"] == "/users/42/notes"
        assert call["json"] == {"body": "hello"}
        assert call["params"] is None

    async def test_missing_required_path_param_returns_error_no_request(self) -> None:
        """missing path placeholder -> failed ToolResult, no request sent."""
        client = _FakeTracedHttpClient(response=_response(200, "ok"))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        result = await tool.execute(verbose=True)

        assert result.success is False
        assert result.error is not None
        assert "id" in result.error
        assert client.calls == []


class TestCredentialResolution:
    """secret-ref resolution into the outbound header, never elsewhere."""

    async def test_env_ref_becomes_bearer_header_and_not_in_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env ref -> Authorization: Bearer <value>; value absent from metadata."""
        monkeypatch.setenv("UPSTREAM_TOKEN", "s3cr3t-value")
        descriptor = _get_descriptor(credentials_ref="env://UPSTREAM_TOKEN")
        client = _FakeTracedHttpClient(response=_response(200, "ok", {"content-type": "application/json"}))
        tool = HttpApiTool(descriptor=descriptor, http_client=client)

        result = await tool.execute(id="42")

        call = client.calls[0]
        assert call["headers"]["Authorization"] == "Bearer s3cr3t-value"
        # the resolved secret must not appear in the returned metadata.
        assert "s3cr3t-value" not in json.dumps(result.metadata)
        # nor stored on the instance.
        assert "s3cr3t-value" not in json.dumps({k: str(v) for k, v in vars(tool).items()})

    async def test_unresolvable_ref_returns_failure_naming_reference(self) -> None:
        """valid-shape but unset env ref -> failure ToolResult naming the ref."""
        descriptor = _get_descriptor(credentials_ref="env://DEFINITELY_UNSET_VAR")
        client = _FakeTracedHttpClient(response=_response(200, "ok"))
        tool = HttpApiTool(descriptor=descriptor, http_client=client)

        result = await tool.execute(id="42")

        assert result.success is False
        assert result.error is not None
        assert "DEFINITELY_UNSET_VAR" in result.error
        assert client.calls == []

    def test_malformed_ref_fails_at_construction(self) -> None:
        """a malformed / unknown-scheme ref fails at tool-build time."""
        descriptor = _get_descriptor(credentials_ref="not-a-scheme")
        client = _FakeTracedHttpClient(response=_response(200, "ok"))

        with pytest.raises(SecretResolutionError):
            HttpApiTool(descriptor=descriptor, http_client=client)


class TestPassthroughResult:
    """passthrough content + metadata shape across statuses."""

    async def test_2xx_is_success_with_http_metadata(self) -> None:
        """200 -> success, content == body, metadata['http'] populated."""
        client = _FakeTracedHttpClient(response=_response(200, "the-body", {"content-type": "text/plain"}))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        result = await tool.execute(id="42")

        assert result.success is True
        assert result.content == "the-body"
        assert result.metadata is not None
        http_meta = result.metadata["http"]
        assert http_meta["status"] == 200
        assert http_meta["content_type"] == "text/plain"
        assert http_meta["headers"]["content-type"] == "text/plain"

    async def test_4xx_is_passthrough_failure_no_exception(self) -> None:
        """404 -> failure, body preserved in content, status in metadata."""
        client = _FakeTracedHttpClient(response=_response(404, "not found"))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        result = await tool.execute(id="42")

        assert result.success is False
        assert result.content == "not found"
        assert result.metadata is not None
        assert result.metadata["http"]["status"] == 404


class TestUpstreamFailures:
    """retry-exhausted + circuit-open become failed ToolResults, not raises."""

    async def test_upstream_http_error_becomes_failure(self) -> None:
        """retry-exhausted UpstreamHttpError -> failed ToolResult, body kept."""
        exc = UpstreamHttpError("boom", status_code=503, body=b"unavailable")
        client = _FakeTracedHttpClient(raise_exc=exc)
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        result = await tool.execute(id="42")

        assert result.success is False
        assert result.content == "unavailable"
        assert result.metadata is not None
        assert result.metadata["http"]["status"] == 503

    async def test_circuit_open_becomes_failure(self) -> None:
        """tripped CircuitOpenError (duck-typed) -> failed ToolResult."""
        client = _FakeTracedHttpClient(raise_exc=CircuitOpenError("open"))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        result = await tool.execute(id="42")

        assert result.success is False
        assert result.error is not None
        assert "circuit" in result.error.lower()

    async def test_unknown_exception_is_reraised(self) -> None:
        """an exception the tool cannot classify propagates out."""
        client = _FakeTracedHttpClient(raise_exc=ValueError("mystery"))
        tool = HttpApiTool(descriptor=_get_descriptor(), http_client=client)

        with pytest.raises(ValueError, match="mystery"):
            await tool.execute(id="42")


class TestSchemaDerivation:
    """mcp_schema / mcp_name / mcp_version derive from the descriptor."""

    def test_input_schema_is_descriptor_param_schema(self) -> None:
        """mcp_schema().input_schema IS the descriptor's param_schema object."""
        descriptor = _get_descriptor()
        client = _FakeTracedHttpClient(response=_response(200, "ok"))
        tool = HttpApiTool(descriptor=descriptor, http_client=client)

        schema = tool.mcp_schema()

        assert schema.input_schema is descriptor.param_schema
        assert schema.name == "example.get_user"
        assert schema.description == "get a user by id"

    def test_mcp_name_and_version_from_descriptor(self) -> None:
        """mcp_name / mcp_version return the descriptor's values."""
        descriptor = _get_descriptor()
        client = _FakeTracedHttpClient(response=_response(200, "ok"))
        tool = HttpApiTool(descriptor=descriptor, http_client=client)

        assert tool.mcp_name() == "example.get_user"
        assert tool.mcp_version() == "1.0"


class TestNoRawHttpx:
    """the module opens no raw httpx client (only the injected traced client)."""

    def test_module_constructs_no_httpx_client(self) -> None:
        """AST-assert no ``httpx.Client`` / ``httpx.AsyncClient`` construction."""
        module_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "threetears"
            / "agent"
            / "tools"
            / "builtin"
            / "http_api_tool.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"Client", "AsyncClient"}
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "httpx"
                ):
                    pytest.fail("http_api_tool.py constructs a raw httpx client")
