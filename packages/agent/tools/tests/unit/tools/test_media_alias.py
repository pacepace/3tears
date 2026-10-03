"""the ``media`` alias names only tools something can serve.

an agent's ``access.tools`` grant of ``media`` expands to the names in
:data:`threetears.agent.tools.aliases.MEDIA_TOOLS`, and the agent's readiness
gate logs an ERROR for every expanded name no tool in the registry answers to.
a tool reaches the registry only as a :class:`TearsTool` a ``ToolServer``
registers -- the built-in pod's ``image_prep`` and ``parse_document``, or the
host's ``media_analyze``. a name in the alias with no such tool behind it is an
ERROR at every boot of every agent granted ``media``: that is what
``threetears.image_generation`` was, a LangChain factory needing host backends
and host persistence that no pod serves.
"""

from __future__ import annotations

from threetears.agent.tools.aliases import MEDIA_TOOLS, expand_selectors
from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.tools.builtin.analyze_media import AnalyzeMediaTool
from threetears.agent.tools.builtin.image_prep import ImagePrepTool
from threetears.agent.tools.document import ParseDocumentTool


# parity-exempt: construction-only stand-in; AnalyzeMediaTool needs a storage object to exist, never calls it here
class _FakeUnusedStorage:
    """a storage the tool is built with and never reads from."""


def _servable_media_tools() -> list[TearsTool]:
    """every media TearsTool 3tears ships, built the way a ToolServer host builds it.

    :return: the media tools a ToolServer can register
    :rtype: list[TearsTool]
    """
    return [
        AnalyzeMediaTool(storage=_FakeUnusedStorage()),  # type: ignore[arg-type]
        ImagePrepTool(),
        ParseDocumentTool(),
    ]


class TestMediaAliasNamesOnlyServableTools:
    """a grant of ``media`` expands to names a registry can answer."""

    def test_every_media_name_is_a_tool_a_tool_server_can_register(self) -> None:
        """the alias and the servable media tools are the same set, name for name."""
        servable = {tool.mcp_name() for tool in _servable_media_tools()}

        expanded = set(expand_selectors(["media"]))

        assert expanded - servable == set(), (
            f"the media alias names {sorted(expanded - servable)}, which no TearsTool serves: "
            "every agent granted media logs an ERROR at boot for each"
        )
        assert servable - expanded == set(), f"the media alias omits servable tools {sorted(servable - expanded)}"
        assert expanded, "the media alias expanded to nothing"

    def test_image_generation_is_not_in_the_media_alias(self) -> None:
        """the name live boots reported as matching no tool is gone from the alias."""
        assert "threetears.image_generation" not in MEDIA_TOOLS
        assert "threetears.image_generation" not in expand_selectors(["media"])
