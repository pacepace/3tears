"""entry point for running built-in tool server.

starts ToolServer with all available built-in TearsTool instances
and serves them via NATS. intended for use as tool pod main process.
"""

from __future__ import annotations

import os

from threetears.agent.tools.bootstrap import ToolServerBootstrap
from threetears.agent.tools.server import ToolServer
from threetears.core.security import IdentityMinter
from threetears.core.security.secret_refs import resolve_secret
from threetears.observe import get_logger

__all__ = [
    "main",
]

_logger = get_logger(__name__)

#: identity-JWT lifetime for the self-minting connect path. re-minted on every (re)connect + for
#: each registration manifest (the provider), so a short TTL only means more transparent re-mints.
_IDENTITY_TTL_SECONDS = 3600

#: the ``customer_id`` claim sentinel for a platform-shared tool pod (not a per-customer principal).
#: the 3tears identity-token schema requires the claim; a shared tool pod's authorization derives
#: from its authenticated pod id alone, not this claim (the Hub tool-pod verifier ignores it).
_PLATFORM_CUSTOMER_SENTINEL = "3tears-platform"


def _register_builtin_tools(server: ToolServer) -> None:
    """register all available built-in tools on server.

    tools with missing optional dependencies are skipped with
    warning log. tools requiring external configuration (e.g.
    WebSearchTool, AnalyzeMediaTool) or session-scoped state
    (workflow, todo) are only registered when required environment
    variables or host-application context are present.

    :param server: tool server to register tools on
    :ptype server: ToolServer
    """
    registered_count = 0
    skipped_count = 0
    skipped_reasons: list[str] = []

    from threetears.agent.tools.builtin.calculator import CalculatorTool
    from threetears.agent.tools.builtin.current_date import CurrentDateTool
    from threetears.agent.tools.builtin.dictionary import DictionaryTool

    server.register(CalculatorTool())
    server.register(CurrentDateTool())
    server.register(DictionaryTool())
    registered_count += 3

    try:
        from threetears.agent.tools.builtin.unit_converter import UnitConverterTool

        server.register(UnitConverterTool())
        registered_count += 1
    except ImportError:
        skipped_count += 1
        skipped_reasons.append("unit_converter: missing pint dependency")
        _logger.warning(
            "skipping unit_converter (missing pint dependency)",
            extra={"extra_data": {"tool": "threetears.unit_converter"}},
        )

    try:
        from threetears.agent.tools.builtin.timezone_converter import TimezoneConverterTool

        server.register(TimezoneConverterTool())
        registered_count += 1
    except ImportError:
        skipped_count += 1
        skipped_reasons.append("timezone_converter: missing dependency")
        _logger.warning(
            "skipping timezone_converter (missing dependency)",
            extra={"extra_data": {"tool": "threetears.timezone_converter"}},
        )

    try:
        from threetears.agent.tools.builtin.web_fetch import WebFetchTool

        server.register(WebFetchTool())
        registered_count += 1
    except ImportError:
        skipped_count += 1
        skipped_reasons.append("web_fetch: missing trafilatura dependency")
        _logger.warning(
            "skipping web_fetch (missing trafilatura dependency)",
            extra={"extra_data": {"tool": "threetears.web_fetch"}},
        )

    searxng_url = os.environ.get("THREETEARS_SEARXNG_URL")
    if searxng_url:
        try:
            from threetears.agent.tools.builtin.web_search import WebSearchTool

            server.register(WebSearchTool(base_url=searxng_url))
            registered_count += 1
        except ImportError:
            skipped_count += 1
            skipped_reasons.append("web_search: missing dependency")
            _logger.warning(
                "skipping web_search (missing dependency)",
                extra={"extra_data": {"tool": "threetears.web_search"}},
            )

    try:
        from threetears.agent.tools.document import ParseDocumentTool

        server.register(ParseDocumentTool())
        registered_count += 1
    except ImportError:
        skipped_count += 1
        skipped_reasons.append("parse_document: missing dependency")
        _logger.warning(
            "skipping parse_document (missing dependency)",
            extra={"extra_data": {"tool": "threetears.parse_document"}},
        )

    try:
        from threetears.agent.tools.builtin.image_prep import ImagePrepTool

        server.register(ImagePrepTool())
        registered_count += 1
    except ImportError:
        skipped_count += 1
        skipped_reasons.append("image_prep: missing Pillow dependency")
        _logger.warning(
            "skipping image_prep (missing Pillow dependency)",
            extra={"extra_data": {"tool": "threetears.image_prep"}},
        )

    # -- session-scoped tools (not registerable in shared tool server) --------

    skipped_count += 1
    skipped_reasons.append(
        "analyze_media: requires MediaStorage and analyzer configuration (host-application-provided)"
    )
    _logger.info(
        "skipping analyze_media (requires MediaStorage and analyzer configuration, host-application-provided)",
        extra={
            "extra_data": {
                "tool": "threetears.analyze_media",
                "hint": (
                    "register in your agent host via "
                    "AnalyzeMediaTool(storage=<MediaStorage>, "
                    "analyzers=<dict[str, AnalyzerConfig]>). "
                    "see threetears.agent.tools.builtin.analyze_media"
                ),
            }
        },
    )

    skipped_count += 1
    skipped_reasons.append("workflow tools: requires session-scoped ToolContextManager (host-application-provided)")
    _logger.info(
        "skipping workflow tools (requires session-scoped ToolContextManager, host-application-provided)",
        extra={
            "extra_data": {
                "tools": [
                    "threetears.set_variable",
                    "threetears.get_variable",
                    "threetears.declare_workflow",
                ],
                "hint": (
                    "register per-session via "
                    "load_workflow_tools(tool_context=<ToolContextManager>). "
                    "see threetears.agent.tools.workflow"
                ),
            }
        },
    )

    skipped_count += 1
    skipped_reasons.append("todo: requires session-scoped TodoStorage (host-application-provided)")
    _logger.info(
        "skipping todo (requires session-scoped TodoStorage, host-application-provided)",
        extra={
            "extra_data": {
                "tool": "threetears.todo",
                "hint": (
                    "register per-session via "
                    "load_todo_tools(storage=<TodoStorage>, "
                    "conversation_id=<UUID>, user_id=<UUID>). "
                    "see threetears.agent.tools.todo"
                ),
            }
        },
    )

    _logger.info(
        "built-in tool registration complete",
        extra={
            "extra_data": {
                "registered": registered_count,
                "skipped": skipped_count,
                "skipped_reasons": skipped_reasons,
            }
        },
    )


class _BuiltinToolBootstrap(ToolServerBootstrap):
    """``ToolServerBootstrap`` subclass that hosts built-in TearsTool pods.

    standalone platform-only pod (no host-application HubClient
    lifecycle). reads NATS connection details from
    ``THREETEARS_NATS_URL`` and ``THREETEARS_NATS_SUBJECT_NAMESPACE``
    environment variables. ``namespace_collection`` is suppressed because
    this entrypoint serves calculator / dictionary / current-date / etc.
    from a standalone process and does not participate in the agent-side
    three-tier stack -- :class:`NamespaceCollection` wiring is the agent
    bootstrap's responsibility when agent-owned tools spin up.
    """

    async def build_server(self) -> ToolServer:
        """build standalone ``ToolServer`` from environment variables.

        per-key identity ONLY (v0.14.1): the pod self-mints a short-lived identity JWT from its own
        Ed25519 key (``kid`` = ``THREETEARS_TOOL_POD_ID``, issuer =
        ``THREETEARS_TOOL_POD_CONNECT_ISSUER``) and presents it to NATS AND on its registration
        manifest. this is the auth-callout path -- the CALLER (this entrypoint) builds the minter;
        ``ToolServer`` stays issuer-agnostic. the key itself is never read directly:
        ``THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF`` is a ``scheme://locator`` reference
        (:mod:`threetears.core.security.secret_refs`), resolved here. there is NO static-credential
        fallback: a pod with no signing-key ref FAILS LOUD rather than connect unauthenticated (a
        deploy without one is a config error that must crash startup, never register ZERO tools on
        an anonymous connect).

        :return: configured ToolServer presenting a self-minted identity token
        :rtype: ToolServer
        :raises ValueError: when ``THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF`` (or, downstream,
            its required companion vars) is missing
        :raises threetears.core.security.secret_refs.SecretResolutionError: when the ref is
            malformed or cannot be resolved
        """
        identity_signing_key_ref = os.environ.get("THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF") or None
        if identity_signing_key_ref is None:
            raise ValueError(
                "THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF is required: the built-in tool pod "
                "self-mints its NATS connect + registration identity from a per-pod Ed25519 key "
                "(referenced, never the raw key, via a scheme://locator); there is no "
                "static-credential fallback (v0.14.1 per-key cutover)"
            )
        identity_signing_key = resolve_secret(identity_signing_key_ref).get_secret_value()
        nats_url = os.environ.get("THREETEARS_NATS_URL", "nats://localhost:4222")
        namespace = os.environ.get("THREETEARS_NATS_SUBJECT_NAMESPACE", "3tears")
        pod_id = os.environ.get("THREETEARS_TOOL_POD_ID") or None
        return self._build_identity_server(nats_url, namespace, pod_id, identity_signing_key)

    def _build_identity_server(
        self,
        nats_url: str,
        namespace: str,
        pod_id: str | None,
        identity_signing_key: str,
    ) -> ToolServer:
        """build a ``ToolServer`` on the self-minted per-key-identity connect path.

        FAILS LOUD when the identity key is set but a required companion var is missing -- a partial
        identity config is a deploy error that must crash startup, never silently fall back to an
        unauthenticated connect.

        :param nats_url: NATS server URL
        :ptype nats_url: str
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param pod_id: the pod id (the minter ``kid``); REQUIRED on this path
        :ptype pod_id: str | None
        :param identity_signing_key: the pod's Ed25519 signing-key PEM
        :ptype identity_signing_key: str
        :return: configured ToolServer presenting a self-minted identity token
        :rtype: ToolServer
        :raises ValueError: when ``THREETEARS_TOOL_POD_ID`` / ``THREETEARS_TOOL_POD_CONNECT_ISSUER``
            is missing while an identity signing key is set
        """
        if not pod_id:
            raise ValueError(
                "THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF is set but THREETEARS_TOOL_POD_ID is "
                "missing (the minter kid must be the pod id)"
            )
        issuer = os.environ.get("THREETEARS_TOOL_POD_CONNECT_ISSUER") or None
        if not issuer:
            raise ValueError(
                "THREETEARS_TOOL_POD_IDENTITY_SIGNING_KEY_REF is set but "
                "THREETEARS_TOOL_POD_CONNECT_ISSUER is missing (the issuer the Hub verifier pins)"
            )
        customer_id = os.environ.get("THREETEARS_TOOL_POD_CUSTOMER_ID") or _PLATFORM_CUSTOMER_SENTINEL
        # the CALLER builds the minter; ToolServer stays issuer/minter-agnostic. the provider
        # re-mints on every (re)connect AND for each registration manifest, so neither hop ever
        # re-presents an expired token.
        minter = IdentityMinter.from_pem(
            identity_signing_key,
            kid=pod_id,
            issuer=issuer,
            ttl_seconds=_IDENTITY_TTL_SECONDS,
        )
        return ToolServer(
            nats_url=nats_url,
            namespace=namespace,
            pod_id=pod_id,
            auth_token=lambda: minter.mint(pod_id, customer_id=customer_id),
            namespace_collection=None,
        )

    async def register_tools(self, server: ToolServer) -> None:
        """register every built-in tool that has its dependencies wired."""
        _register_builtin_tools(server)


def main() -> None:
    """run built-in tool server.

    reads NATS connection URL from ``THREETEARS_NATS_URL`` env var
    (defaults to ``nats://localhost:4222``). registers all available
    built-in tools and serves them until interrupted. the lifecycle
    plumbing (logging configuration, signal handlers, serve loop) is
    owned by :class:`ToolServerBootstrap`.

    :return: None
    :rtype: None
    """
    _BuiltinToolBootstrap("builtin-tool-server").run()


if __name__ == "__main__":
    main()
