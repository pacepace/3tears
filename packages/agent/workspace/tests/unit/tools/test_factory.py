"""tests for :mod:`threetears.agent.workspace.factory` registry and builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import threetears.agent.workspace.tools  # noqa: F401  -- registers builders
from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.workspace.factory import build_workspace_tools
from packages.agent.workspace.tests._helpers.factory_deps import minimal_tool_deps

#: the repo root, where ``packages.`` resolves as a namespace package for the subprocess probe.
_REPO_ROOT = Path(__file__).resolve().parents[6]


def test_tool_builders_registry_has_nineteen_after_history_tools() -> None:
    """importing the tools subpackage must register all nineteen tools, each once.

    six meta + lifecycle (shards 09+10) plus four fs_* tools (shard 11)
    plus three doc_* tools (shard 12) plus four history tools (shard 13:
    history, diff, checkpoint, rollback_to) plus the refresh_from_disk
    live-sync tool that landed alongside bind's watcher, plus the
    flush_to_disk one-shot that projects L3 back onto disk.
    """
    names = [t.mcp_name() for t in build_workspace_tools(**minimal_tool_deps())]
    assert len(names) == 19
    assert len(set(names)) == 19


def test_build_workspace_tools_returns_nineteen_tools() -> None:
    """build_workspace_tools instantiates every registered builder."""
    tools = build_workspace_tools(**minimal_tool_deps())

    assert len(tools) == 19
    assert all(isinstance(t, TearsTool) for t in tools)


def test_build_workspace_tools_includes_each_expected_mcp_name() -> None:
    """built tools include exactly the nineteen expected mcp_name strings."""
    tools = build_workspace_tools(**minimal_tool_deps())

    names = {t.mcp_name() for t in tools}
    assert names == {
        "threetears.workspace.list",
        "threetears.workspace.use",
        "threetears.workspace.current",
        "threetears.workspace.create",
        "threetears.workspace.reset",
        "threetears.workspace.delete",
        "threetears.workspace.fs_read",
        "threetears.workspace.fs_write",
        "threetears.workspace.fs_list",
        "threetears.workspace.fs_edit",
        "threetears.workspace.doc_get",
        "threetears.workspace.doc_set",
        "threetears.workspace.doc_merge",
        "threetears.workspace.history",
        "threetears.workspace.diff",
        "threetears.workspace.checkpoint",
        "threetears.workspace.rollback_to",
        "threetears.workspace.refresh_from_disk",
        "threetears.workspace.flush_to_disk",
    }


def test_build_workspace_tools_returns_fresh_instances() -> None:
    """each call returns new instances; tools are not singletons."""
    first = build_workspace_tools(**minimal_tool_deps())
    second = build_workspace_tools(**minimal_tool_deps())

    first_ids = {id(t) for t in first}
    second_ids = {id(t) for t in second}
    assert first_ids.isdisjoint(second_ids)


def test_build_workspace_tools_tolerates_missing_optional_deps() -> None:
    """unused deps default to None so callers can pass only what tools need."""
    deps = minimal_tool_deps()
    tools = build_workspace_tools(**deps)

    assert len(tools) == 19


#: registers a sentinel builder in a fresh interpreter, builds, and reports what came back. the
#: registry is process-wide with no way to remove a builder, so the probe runs where adding one
#: cannot leak into any other test's count.
_REGISTRATION_PROBE = """
import json
from typing import Any

import threetears.agent.workspace.tools  # registers the shipped builders
from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.workspace.factory import build_workspace_tools, register_tool_builder
from packages.agent.workspace.tests._helpers.factory_deps import minimal_tool_deps


class SentinelTool(TearsTool):
    async def execute(self, **kwargs: Any) -> Any:
        return None

    def mcp_schema(self) -> Any:
        return None

    def mcp_name(self) -> str:
        return "threetears.workspace.sentinel"

    def mcp_version(self) -> str:
        return "0.0"


calls: list[list[str]] = []


def build(**kwargs: Any) -> SentinelTool:
    calls.append(sorted(kwargs))
    return SentinelTool()


register_tool_builder(build)
register_tool_builder(build)
names = [t.mcp_name() for t in build_workspace_tools(**minimal_tool_deps())]
print(json.dumps({"names": names, "calls": calls}))
"""


def test_register_tool_builder_appends_to_registry() -> None:
    """register_tool_builder appends the builder so it is emitted on next build, once.

    the sentinel's mcp_name uses the canonical dotted
    ``threetears.workspace.<segment>`` shape so the enforcement test's
    ``_NAMESPACE_PREFIX`` check would accept it; registering the same
    builder object twice is a no-op, so it is built exactly once.
    """
    probe = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_PROBE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    report = json.loads(probe.stdout.strip().splitlines()[-1])
    assert report["names"].count("threetears.workspace.sentinel") == 1
    assert len(report["names"]) == 20
    # the builder was handed the whole canonical dependency bundle.
    [received] = report["calls"]
    assert "acl_cache" in received and "workspace_collection" in received


@pytest.mark.parametrize(
    "expected",
    [
        "threetears.workspace.list",
        "threetears.workspace.use",
        "threetears.workspace.current",
        "threetears.workspace.create",
        "threetears.workspace.reset",
        "threetears.workspace.delete",
        "threetears.workspace.fs_read",
        "threetears.workspace.fs_write",
        "threetears.workspace.fs_list",
        "threetears.workspace.fs_edit",
        "threetears.workspace.doc_get",
        "threetears.workspace.doc_set",
        "threetears.workspace.doc_merge",
        "threetears.workspace.history",
        "threetears.workspace.diff",
        "threetears.workspace.checkpoint",
        "threetears.workspace.rollback_to",
        "threetears.workspace.refresh_from_disk",
        "threetears.workspace.flush_to_disk",
    ],
)
def test_each_expected_mcp_name_present(expected: str) -> None:
    """each of the nineteen required mcp_name strings is emitted."""
    tools = build_workspace_tools(**minimal_tool_deps())

    names = [t.mcp_name() for t in tools]
    assert expected in names
