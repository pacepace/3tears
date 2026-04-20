"""entry point for running built-in tool server.

starts ToolServer with all available built-in TearsTool instances
and serves them via NATS. intended for use as tool pod main process.
"""

from __future__ import annotations

import asyncio
import os
import signal

from threetears.agent.tools.server import ToolServer
from threetears.observe import get_logger

__all__ = [
    "main",
]

_logger = get_logger(__name__)


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

    searxng_url = os.environ.get("FOURTEENAIBOTS_SEARXNG_URL")
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


def main() -> None:
    """run built-in tool server.

    reads NATS connection URL from FOURTEENAIBOTS_NATS_URL environment
    variable (defaults to nats://localhost:4222). registers all available
    built-in tools and serves them until interrupted.
    """
    nats_url = os.environ.get("FOURTEENAIBOTS_NATS_URL", "nats://localhost:4222")
    namespace = os.environ.get("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "aibots")

    # the built-in tool entrypoint is a platform-only pod that serves
    # calculator / dictionary / current-date / etc. from a standalone
    # process; it does not participate in the agent-side three-tier
    # stack so namespace emission is deliberately suppressed.
    # :class:`NamespaceCollection` wiring is the agent bootstrap's
    # responsibility when agent-owned tools spin up.
    server = ToolServer(
        nats_url=nats_url,
        namespace=namespace,
        namespace_collection=None,
    )
    _register_builtin_tools(server)

    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        """schedule graceful shutdown on signal."""
        loop.create_task(server.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    _logger.info(
        "starting built-in tool server",
        extra={
            "extra_data": {
                "nats_url": nats_url,
                "namespace": namespace,
            }
        },
    )

    loop.run_until_complete(server.serve())
    loop.close()


if __name__ == "__main__":
    main()
