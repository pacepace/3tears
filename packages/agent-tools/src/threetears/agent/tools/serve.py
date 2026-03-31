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

_logger = get_logger(__name__)


def _register_builtin_tools(server: ToolServer) -> None:
    """register all available built-in tools on server.

    tools with missing optional dependencies are skipped with
    warning log. tools requiring external configuration (e.g.
    WebSearchTool, AnalyzeMediaTool) are only registered when
    required environment variables are present.

    :param server: tool server to register tools on
    :ptype server: ToolServer
    """
    from threetears.agent.tools.builtin.calculator import CalculatorTool
    from threetears.agent.tools.builtin.current_date import CurrentDateTool
    from threetears.agent.tools.builtin.dictionary import DictionaryTool

    server.register(CalculatorTool())
    server.register(CurrentDateTool())
    server.register(DictionaryTool())

    try:
        from threetears.agent.tools.builtin.unit_converter import UnitConverterTool
        server.register(UnitConverterTool())
    except ImportError:
        _logger.warning(
            "skipping unit_converter (missing pint dependency)",
            extra={"extra_data": {"tool": "threetears.unit_converter"}},
        )

    try:
        from threetears.agent.tools.builtin.timezone_converter import TimezoneConverterTool
        server.register(TimezoneConverterTool())
    except ImportError:
        _logger.warning(
            "skipping timezone_converter (missing dependency)",
            extra={"extra_data": {"tool": "threetears.timezone_converter"}},
        )

    try:
        from threetears.agent.tools.builtin.web_fetch import WebFetchTool
        server.register(WebFetchTool())
    except ImportError:
        _logger.warning(
            "skipping web_fetch (missing trafilatura dependency)",
            extra={"extra_data": {"tool": "threetears.web_fetch"}},
        )

    searxng_url = os.environ.get("FOURTEENAIBOTS_SEARXNG_URL")
    if searxng_url:
        try:
            from threetears.agent.tools.builtin.web_search import WebSearchTool
            server.register(WebSearchTool(base_url=searxng_url))
        except ImportError:
            _logger.warning(
                "skipping web_search (missing dependency)",
                extra={"extra_data": {"tool": "threetears.web_search"}},
            )


def main() -> None:
    """run built-in tool server.

    reads NATS connection URL from FOURTEENAIBOTS_NATS_URL environment
    variable (defaults to nats://localhost:4222). registers all available
    built-in tools and serves them until interrupted.
    """
    nats_url = os.environ.get("FOURTEENAIBOTS_NATS_URL", "nats://localhost:4222")
    namespace = os.environ.get("FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "aibots")

    server = ToolServer(nats_url=nats_url, namespace=namespace)
    _register_builtin_tools(server)

    loop = asyncio.new_event_loop()

    def _signal_handler() -> None:
        """schedule graceful shutdown on signal."""
        loop.create_task(server.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    _logger.info(
        "starting built-in tool server",
        extra={"extra_data": {
            "nats_url": nats_url,
            "namespace": namespace,
        }},
    )

    loop.run_until_complete(server.serve())
    loop.close()


if __name__ == "__main__":
    main()
