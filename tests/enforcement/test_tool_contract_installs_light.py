"""enforcement: the TearsTool contract and the MCP server install without NATS, Postgres drivers, LangGraph or LangChain.

**Why.** A host that serves its tools locally -- over MCP, with no platform behind it -- takes ``3tears-mcp`` (its
base, not the ``grants`` extra), ``3tears-tool-schema``, ``3tears-observe`` and the ``TearsTool`` contract, and
nothing else. 3tears-agent-tools' base pulls a NATS client, Postgres drivers, LangGraph and LangChain, which is why
the contract is a package of its own (``3tears-tool-contract``). One innocent dependency, here or in anything these
reach, would put the platform back into every such install, and no other check would notice: the workspace installs
everything for everyone.

**How.** ``uv export --frozen`` names, from ``uv.lock``, every distribution an install of a package pulls with the
extras it names -- the resolver's own answer, read from the lock, so the check needs no network and the pool runs it.
Each install is checked for the forbidden distributions, whatever platform marker a line carries (a forbidden
package is forbidden on every platform). The check is shown able to fail: agent-tools' install holds them.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: the distributions a local install must not pull: NATS clients, Postgres drivers, LangGraph and LangChain
_FORBIDDEN = re.compile(
    r"^(nats-py|nkeys|3tears-nats|asyncpg|psycopg|psycopg-binary|psycopg-c|psycopg-pool|psycopg2|psycopg2-binary"
    r"|pg8000|langgraph(-.+)?|langchain(-.+)?)$"
)

#: a requirement line of ``uv export``: ``name==version`` (with a marker), or ``-e ./<path>``
_PINNED = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==")
_EDITABLE = re.compile(r"^-e \./(?P<path>\S+)$")

#: the light installs: (package, extras)
_LIGHT = (
    ("3tears-tool-contract", ()),
    ("3tears-tool-contract", ("mcp",)),
    ("3tears-mcp", ()),
    ("3tears-tool-schema", ()),
    ("3tears-observe", ()),
)


def _normalized(name: str) -> str:
    """a distribution name as the index compares it.

    :param name: the name
    :ptype name: str
    :return: lower case, runs of ``-``, ``_`` and ``.`` as one ``-``
    :rtype: str
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed(package: str, extras: tuple[str, ...]) -> frozenset[str]:
    """every distribution the lock installs for ``package`` with ``extras``, by normalized name.

    :param package: the workspace member
    :ptype package: str
    :param extras: its extras
    :ptype extras: tuple[str, ...]
    :return: the distributions, the member itself and the workspace members it pulls among them
    :rtype: frozenset[str]
    """
    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH: the gate reads the lock with it"
    argv = [uv, "export", "--frozen", "--package", package, "--no-dev", "--no-hashes"]
    for extra in extras:
        argv += ["--extra", extra]
    done = subprocess.run(argv, cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"{' '.join(argv)} failed ({done.returncode}):\n{done.stdout}\n{done.stderr}"
    names: set[str] = set()
    for line in done.stdout.splitlines():
        line = line.strip()
        if editable := _EDITABLE.match(line):
            project = tomllib.loads((_REPO_ROOT / editable["path"] / "pyproject.toml").read_text(encoding="utf-8"))
            names.add(_normalized(project["project"]["name"]))
        elif pinned := _PINNED.match(line):
            names.add(_normalized(pinned["name"]))
    return frozenset(names)


@pytest.mark.parametrize(
    ("package", "extras"), _LIGHT, ids=[f"{package}[{','.join(extras)}]" for package, extras in _LIGHT]
)
def test_a_light_install_pulls_no_nats_postgres_langgraph_or_langchain(package: str, extras: tuple[str, ...]) -> None:
    installed = _installed(package, extras)
    assert _normalized(package) in installed, f"the export of {package} does not hold the package itself"
    forbidden = sorted(name for name in installed if _FORBIDDEN.match(name))
    assert not forbidden, f"an install of {package}{list(extras) or ''} pulls {forbidden}"


def test_the_contracts_install_is_the_contract_observe_and_tool_schema() -> None:
    """the contract's whole install, so a new dependency of it is a decision here, not a drift."""
    assert _installed("3tears-tool-contract", ()) == {"3tears-tool-contract", "3tears-observe", "3tears-tool-schema"}


def test_the_check_can_fail_agent_tools_pulls_what_it_forbids() -> None:
    """the full tool framework does pull them: a matcher that matched nothing would pass every install."""
    forbidden = {name for name in _installed("3tears-agent-tools", ()) if _FORBIDDEN.match(name)}
    assert {"langchain-core", "langgraph", "asyncpg", "nats-py"} <= forbidden, sorted(forbidden)
